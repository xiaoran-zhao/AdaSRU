from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .data import build_normalized_adj, load_split
from .eval import evaluate_ranking, forget_score, forget_detailed_score
from .model import LightGCN
from .utils import device_from_arg, ensure_dir, save_json, set_seed


# ============================================================
# 1. 命令行参数
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # 基础路径参数
    p.add_argument('--run_dir', type=str, required=True)
    p.add_argument('--output_dir', type=str, default='runs/unlearned')

    # 随机种子与设备
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='auto')

    # unlearning 训练参数
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=2048)
    p.add_argument('--steps_per_epoch', type=int, default=64)
    p.add_argument('--lr', type=float, default=5e-4)

    # UPUP 约束强度
    # 这里 eps 使用相对尺度：
    # eps_scaled = eps * |g_r^T g_u|
    p.add_argument('--eps', type=float, default=1e-2)
    p.add_argument('--eps_list', type=str, default='')

    # 正则项
    p.add_argument('--reg_weight', type=float, default=0.0)
    p.add_argument('--anchor_weight', type=float, default=0.0)

    # 拉格朗日乘子最大值
    p.add_argument('--max_lam', type=float, default=1e5)

    # 日志间隔
    p.add_argument('--log_interval', type=int, default=16)

    # Adam 二阶矩归一化目标均值
    # 该值控制 Adam diag 作为曲率代理时的整体尺度
    p.add_argument('--adam_target_mean', type=float, default=1e-3)

    # 为了兼容旧命令保留，当前一阶 UPUP 版本不使用
    p.add_argument('--curv_constraint_scale', type=float, default=1.0)
    p.add_argument('--bisection_steps', type=int, default=40)

    # reverse BPR 的 margin
    p.add_argument('--reverse_margin', type=float, default=0.0)

    # 压低 forget positive item 分数的权重
    p.add_argument('--forget_score_weight', type=float, default=0.0)

    # hard negative 采样参数
    p.add_argument('--hard_neg_topk', type=int, default=0)
    p.add_argument('--hard_neg_pool', type=int, default=1000)

    return p.parse_args()


# ============================================================
# 2. Forget pair 采样器
# ============================================================

class PairDataset:
    """
    从 forget_pairs 中采样三元组：

        (user, forget_positive_item, negative_item)

    其中 negative_item 不能是该用户历史正样本，也不能等于当前 forget item。
    """

    def __init__(
        self,
        pairs: List[Tuple[int, int]],
        num_items: int,
        all_pos: Dict[int, set],
    ):
        self.pairs = pairs
        self.num_items = num_items
        self.all_pos = all_pos
        self.idx = np.arange(len(pairs))

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        choice = np.random.choice(self.idx, size=batch_size, replace=True)
        batch = [self.pairs[i] for i in choice]

        users, pos, neg = [], [], []

        for u, pos_item in batch:
            users.append(u)
            pos.append(pos_item)

            neg_item = np.random.randint(0, self.num_items)

            while neg_item in self.all_pos[u] or neg_item == pos_item:
                neg_item = np.random.randint(0, self.num_items)

            neg.append(neg_item)

        return torch.tensor(users), torch.tensor(pos), torch.tensor(neg)


# ============================================================
# 3. Unlearning loss：reverse BPR loss
# ============================================================

def unlearning_reverse_bpr_loss(
    model: LightGCN,
    users: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
    norm_adj: torch.Tensor,
    reg_weight: float = 0.0,
    reverse_margin: float = 0.0,
    forget_score_weight: float = 0.0,
) -> torch.Tensor:
    """
    UPUP 对齐版本中的 unlearning loss：

        l_u(theta)

    它是 loss objective，越小越好。

    推荐遗忘的目标是让 forget positive item 的分数低于 negative item：

        s_neg - s_pos > margin

    因此定义 reverse BPR loss：

        l_u = -log sigmoid(s_neg - s_pos - margin)

    同时加入 score suppression loss：

        + beta * s_pos

    因为后续是最小化 l_u，所以 +s_pos 会压低 forget positive item 的分数。

    最终：

        l_u(theta)
        =
        -log sigmoid(s_neg - s_pos - margin)
        + beta * s_pos
    """

    user_emb, item_emb = model.computer(norm_adj)

    u_e = user_emb[users]
    p_e = item_emb[pos]
    n_e = item_emb[neg]

    pos_scores = torch.sum(u_e * p_e, dim=1)
    neg_scores = torch.sum(u_e * n_e, dim=1)

    # reverse BPR loss：越小表示 neg_scores 越大于 pos_scores
    reverse_bpr_loss = -F.logsigmoid(
        neg_scores - pos_scores - reverse_margin
    ).mean()

    # 压低 forget positive item 分数
    score_suppression_loss = pos_scores.mean()

    loss = reverse_bpr_loss + forget_score_weight * score_suppression_loss

    if reg_weight > 0:
        u0 = model.user_embedding(users)
        p0 = model.item_embedding(pos)
        n0 = model.item_embedding(neg)

        reg = 0.5 * (
            u0.norm(2).pow(2)
            + p0.norm(2).pow(2)
            + n0.norm(2).pow(2)
        ) / users.shape[0]

        # loss objective 中，正则项是加号
        loss = loss + reg_weight * reg

    return loss


# ============================================================
# 4. Forget-side original BPR loss
# ============================================================

def forget_bpr_train_loss(
    model: LightGCN,
    users: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
    norm_adj: torch.Tensor,
    reg_weight: float = 0.0,
) -> torch.Tensor:
    """
    forget-side training loss：

        l_f(theta) = L_BPR(D_f; theta)

    它是原始推荐训练目标在 forget data 上的 BPR loss。

    用于构造 retain loss gradient proxy：

        g_r_hat
        =
        -rho * grad l_f(theta_t)
        + kappa * H_D * (theta_t - theta_star)

    其中：

        rho = m / (n - m)
        kappa = n / (n - m)
    """

    user_emb, item_emb = model.computer(norm_adj)

    u_e = user_emb[users]
    p_e = item_emb[pos]
    n_e = item_emb[neg]

    pos_scores = torch.sum(u_e * p_e, dim=1)
    neg_scores = torch.sum(u_e * n_e, dim=1)

    # 原始 BPR loss：希望 positive item 分数高于 negative item
    loss = -F.logsigmoid(pos_scores - neg_scores).mean()

    if reg_weight > 0:
        u0 = model.user_embedding(users)
        p0 = model.item_embedding(pos)
        n0 = model.item_embedding(neg)

        reg = 0.5 * (
            u0.norm(2).pow(2)
            + p0.norm(2).pow(2)
            + n0.norm(2).pow(2)
        ) / users.shape[0]

        loss = loss + reg_weight * reg

    return loss


# ============================================================
# 5. Hard negative 采样
# ============================================================

@torch.no_grad()
def sample_hard_negatives(
    model: LightGCN,
    users: torch.Tensor,
    pos: torch.Tensor,
    norm_adj: torch.Tensor,
    all_pos: Dict[int, set],
    num_items: int,
    hard_neg_topk: int = 100,
    hard_neg_pool: int = 1000,
) -> torch.Tensor:
    """
    为每个 user 从随机候选池中选一个高分负样本。
    """

    model.eval()

    user_emb, item_emb = model.computer(norm_adj)

    hard_negs = []

    users_cpu = users.detach().cpu().tolist()
    pos_cpu = pos.detach().cpu().tolist()

    for u, pos_item in zip(users_cpu, pos_cpu):
        pool = []

        while len(pool) < hard_neg_pool:
            item = np.random.randint(0, num_items)
            if item not in all_pos[u] and item != pos_item:
                pool.append(item)

        pool_tensor = torch.tensor(pool, device=users.device, dtype=torch.long)

        u_e = user_emb[u]
        i_e = item_emb[pool_tensor]

        scores = torch.sum(u_e.unsqueeze(0) * i_e, dim=1)

        topk = min(hard_neg_topk, len(pool))
        top_indices = torch.topk(scores, k=topk).indices

        chosen_idx = top_indices[
            torch.randint(0, topk, size=(1,), device=users.device)
        ].item()

        hard_negs.append(pool[chosen_idx])

    model.train()

    return torch.tensor(hard_negs, device=users.device, dtype=torch.long)


# ============================================================
# 6. Adam diag 读取与归一化
# ============================================================

def name_to_adam_diag(
    model: LightGCN,
    checkpoint: dict,
) -> Dict[str, torch.Tensor]:
    """
    从 checkpoint 中读取训练阶段保存的 Adam 二阶矩。

    期望 checkpoint 中存在：

        checkpoint['adam_diag_by_name']

    如果无法按照 name 对齐，则尝试按照参数顺序对齐。
    """

    raw_diag = checkpoint.get('adam_diag_by_name', {})
    named: Dict[str, torch.Tensor] = {}

    state_dict = model.state_dict()

    # 优先按照参数名匹配
    for name, _ in state_dict.items():
        if name in raw_diag:
            named[name] = raw_diag[name].detach().clone()

    if named:
        return named

    # 如果 checkpoint 中没有名字，则尝试按照顺序匹配
    diag_values = list(raw_diag.values())

    if len(diag_values) == len(list(model.named_parameters())):
        for (name, p), d in zip(model.named_parameters(), diag_values):
            if tuple(d.shape) == tuple(p.shape):
                named[name] = d.detach().clone()

    return named


def normalize_adam_diag(
    adam_diag: Dict[str, torch.Tensor],
    target_mean: float = 1e-3,
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    """
    将 Adam diag 的全局均值缩放到 target_mean。

    缩放方式：

        scale = target_mean / mean(abs(adam_diag))

        normalized_diag = abs(adam_diag) * scale

    target_mean 是曲率代理的尺度超参数。

    常用候选：

        1e-3, 1e-2, 1e-1, 1.0
    """

    normalized = {}

    all_vals = torch.cat([
        v.detach().abs().reshape(-1).cpu()
        for v in adam_diag.values()
    ])

    mean_val = all_vals.mean().item()
    scale = target_mean / (mean_val + eps)

    for name, v in adam_diag.items():
        normalized[name] = v.detach().abs().clone() * scale

    print(
        f"[Adam diag normalize] mean={mean_val:.3e}, "
        f"scale={scale:.3e}, target_mean={target_mean}"
    )

    return normalized


# ============================================================
# 7. 参数向量展开与还原
# ============================================================

def flatten_named_tensors(named_tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    将 name -> tensor 字典展平成一个长向量。
    """

    if not named_tensors:
        raise ValueError("No tensors to flatten. Please check whether gradients exist.")

    return torch.cat([v.reshape(-1) for v in named_tensors.values()])


def unflatten_to_named(
    flat: torch.Tensor,
    template: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    将长向量按照 template 的 tensor 形状还原为 name -> tensor 字典。
    """

    out: Dict[str, torch.Tensor] = {}
    offset = 0

    for name, t in template.items():
        numel = t.numel()
        out[name] = flat[offset: offset + numel].reshape_as(t)
        offset += numel

    return out


def get_grad_dict(model: LightGCN) -> Dict[str, torch.Tensor]:
    """
    提取当前 model 中所有参数的梯度。
    """

    grads: Dict[str, torch.Tensor] = {}

    for name, p in model.named_parameters():
        if p.grad is not None:
            grads[name] = p.grad.detach().clone()

    if not grads:
        raise ValueError("No gradients found. Did you call loss.backward()?")

    return grads


def compute_delta_norm(
    model: LightGCN,
    theta_star: Dict[str, torch.Tensor],
) -> float:
    """
    计算当前参数和原始 fulltrain 参数 theta_star 的距离。
    """

    total = 0.0

    with torch.no_grad():
        for name, p in model.named_parameters():
            delta = p.detach() - theta_star[name].to(p.device)
            total += delta.norm().item()

    return float(total)


# ============================================================
# 8. UPUP 一阶约束方向
# ============================================================

def build_proxy_direction(
    model: LightGCN,
    theta_star: Dict[str, torch.Tensor],
    adam_diag: Dict[str, torch.Tensor],
    grads_u: Dict[str, torch.Tensor],
    grads_f: Dict[str, torch.Tensor],
    rho: float,
    kappa: float,
    eps: float,
    anchor_weight: float,
    max_lam: float,
) -> tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """
    UPUP loss-objective 一阶约束版本。

    ------------------------------------------------------------
    参数更新：
    ------------------------------------------------------------

        theta_{t+1} = theta_t - alpha_t d_t

    ------------------------------------------------------------
    UPUP 一阶子问题：
    ------------------------------------------------------------

        max_d  g_u^T d - 1/2 ||d||^2

        s.t.   g_r_hat^T d >= -eps_t

    其中：

        g_u = grad l_u(theta_t)

    l_u 是 reverse BPR unlearning loss，越小越好。

    ------------------------------------------------------------
    retain loss gradient proxy：
    ------------------------------------------------------------

        g_r_hat
        =
        -rho * grad l_f(theta_t)
        + kappa * H_D * (theta_t - theta_star)

    其中：

        l_f(theta_t) = L_BPR(D_f; theta_t)

        rho  = m / (n - m)
        kappa = n / (n - m)

    ------------------------------------------------------------
    闭式解：
    ------------------------------------------------------------

    如果无约束方向 d = g_u 满足：

        g_r_hat^T g_u >= -eps_scaled

    则：

        d = g_u

    否则，投影到半空间边界：

        d = g_u + lambda * g_r_hat

        lambda =
            (-eps_scaled - g_r_hat^T g_u)
            / ||g_r_hat||^2

    注意：
        这个一阶 UPUP 子问题中不出现 lr。
        lr 只出现在真正的参数更新：

            theta <- theta - lr * d
    """

    proxy_r: Dict[str, torch.Tensor] = {}
    hessian_diag: Dict[str, torch.Tensor] = {}
    curv_terms: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------
    # 1. 构造 retain loss gradient proxy
    # ------------------------------------------------------------
    for name, p in model.named_parameters():
        if name not in grads_u:
            continue

        if name not in grads_f:
            raise ValueError(f"Missing forget BPR gradient for parameter: {name}")

        # 当前参数和原始 fulltrain 参数之间的位移
        delta = p.detach() - theta_star[name].to(p.device)

        # 读取 Adam diag，作为 H_D 的对角近似
        hdiag = adam_diag.get(
            name,
            torch.zeros_like(p.detach().cpu())
        ).to(p.device).abs()

        # H_r ≈ kappa * H_D
        h = kappa * hdiag

        # 曲率修正项：
        # kappa * H_D * (theta_t - theta_star)
        curv = h * delta

        # loss-objective 视角：
        # g_r_hat = -rho * grad l_f + kappa * H_D * delta
        proxy_r[name] = -rho * grads_f[name] + curv

        hessian_diag[name] = h
        curv_terms[name] = curv

    # ------------------------------------------------------------
    # 2. 展平成向量，方便计算内积
    # ------------------------------------------------------------
    g_u_flat = flatten_named_tensors(grads_u)
    g_r_flat = flatten_named_tensors(proxy_r)
    h_flat = flatten_named_tensors(hessian_diag)
    curv_flat = flatten_named_tensors(curv_terms)

    norm_gu = torch.norm(g_u_flat)
    norm_gr = torch.norm(g_r_flat)

    # g_r_hat^T g_u
    gr_gu = torch.dot(g_r_flat, g_u_flat)

    # ------------------------------------------------------------
    # 3. eps 缩放
    # ------------------------------------------------------------
    # 严格 UPUP 形式中 eps_t 是直接给定的。
    # 这里采用相对 eps：
    #
    #     eps_scaled = eps * |g_r_hat^T g_u|
    #
    # 这样 eps 是无量纲比例，更方便做 sweep。
    # eps_scaled = eps * (
    #     torch.abs(gr_gu.detach()) + 1e-12
    # )

    # ------------------------------------------------------------
    # 3. UPUP absolute epsilon
    # ------------------------------------------------------------
    # 与 UPUP 原文保持一致：
    #
    #     g_r_hat^T d >= -epsilon_t
    #
    # 这里命令行中的 --eps 就直接作为 epsilon_t，
    # 不再乘以 |g_r_hat^T g_u|。
    eps_scaled = torch.tensor(float(eps), device=g_u_flat.device)

    # ------------------------------------------------------------
    # 4. 求解一阶 UPUP 方向
    # ------------------------------------------------------------
    if gr_gu >= -eps_scaled:
        # 无约束方向 d = g_u 已经满足 retain 约束
        lam = torch.tensor(0.0, device=g_u_flat.device)
        raw_lam = lam
        d_flat = g_u_flat
        constraint_value = gr_gu
    else:
        # 无约束方向会让 retain loss 上升过多，需要修正
        denom = torch.dot(g_r_flat, g_r_flat) + 1e-12

        raw_lam = (-eps_scaled - gr_gu) / denom
        lam = torch.clamp(raw_lam, min=0.0, max=max_lam)

        # 投影到半空间边界：
        # d = g_u + lambda * g_r_hat
        d_flat = g_u_flat + lam * g_r_flat

        constraint_value = torch.dot(g_r_flat, d_flat)

    direction = unflatten_to_named(d_flat, grads_u)

    # ------------------------------------------------------------
    # 5. 可选 anchor 项
    # ------------------------------------------------------------
    # anchor 的作用是把参数拉回 theta_star 附近，防止过度漂移。
    # 默认 anchor_weight=0，不启用。
    if anchor_weight > 0:
        for name, p in model.named_parameters():
            if name not in direction:
                continue

            delta = p.detach() - theta_star[name].to(p.device)

            hdiag = adam_diag.get(
                name,
                torch.zeros_like(p.detach().cpu())
            ).to(p.device).abs()

            # 更新是 theta <- theta - lr * direction
            # 若希望把 theta 拉回 theta_star，
            # direction 中应该加上 + H delta。
            direction[name] = direction[name] + anchor_weight * hdiag * delta

    # ------------------------------------------------------------
    # 6. 日志统计
    # ------------------------------------------------------------
    dir_flat = flatten_named_tensors(direction)
    norm_dir = torch.norm(dir_flat)

    gr_dot_d = torch.dot(g_r_flat, dir_flat)
    h_d_quad = torch.sum(h_flat * dir_flat.pow(2))

    cos_dt_gu = torch.dot(dir_flat, g_u_flat) / (norm_dir * norm_gu + 1e-12)
    cos_dt_gr = torch.dot(dir_flat, g_r_flat) / (norm_dir * norm_gr + 1e-12)

    stats = {
        'raw_lam': float(raw_lam.item()),
        'lam': float(lam.item()),

        # g_r_hat^T g_u
        'proj_gr_gu': float(gr_gu.item()),

        'norm_gu': float(norm_gu.item()),
        'norm_gr': float(norm_gr.item()),
        'norm_dir': float(norm_dir.item()),

        # 曲率修正项 ||kappa H_D delta||
        'curv_norm': float(torch.norm(curv_flat).item()),

        'eps': float(eps),
        'eps_scaled': float(eps_scaled.item()),

        # g_r_hat^T d
        'constraint_gr_dt': float(gr_dot_d.item()),

        # 当前是一阶 UPUP 约束，linear 和 second_order 字段保留用于日志兼容
        'constraint_linear': float(gr_dot_d.item()),
        'constraint_quad': 0.0,
        'constraint_second_order': float(constraint_value.item()),

        # 检查：
        # g_r_hat^T d >= -eps_scaled
        'constraint_satisfied': float(
            (constraint_value >= -eps_scaled - 1e-12).item()
        ),

        # 仅作为曲率诊断，不进入一阶约束
        'h_d_quad': float(h_d_quad.item()),

        # 兼容旧日志字段；当前不再是严格 cancel factor
        'cancel_factor': float(1.0 + lam.item() * rho),

        'cos_dt_gu': float(cos_dt_gu.item()),
        'cos_dt_gr': float(cos_dt_gr.item()),
    }

    return direction, stats


# ============================================================
# 9. 参数更新
# ============================================================

def apply_direction(
    model: LightGCN,
    direction: Dict[str, torch.Tensor],
    lr: float,
) -> None:
    """
    执行 UPUP 对齐的参数更新：

        theta <- theta - lr * direction

    注意：
        direction 是 loss-objective 下的下降方向。
    """

    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in direction:
                p.add_(-lr * direction[name])


# ============================================================
# 10. 单个 eps 实验
# ============================================================

def run_single_eps(args: argparse.Namespace, eps: float) -> dict:
    set_seed(args.seed)
    device = device_from_arg(args.device)

    run_dir = Path(args.run_dir)
    out_dir = ensure_dir(Path(args.output_dir) / f'eps_{eps:.4g}')

    # ------------------------------------------------------------
    # 1. 读取数据划分和 checkpoint
    # ------------------------------------------------------------
    split = load_split(run_dir / 'split.json')

    ckpt = torch.load(run_dir / 'best_model.pt', map_location='cpu')

    print('checkpoint keys:', ckpt.keys())
    print('has adam diag:', 'adam_diag_by_name' in ckpt)

    train_args = ckpt['args']

    # ------------------------------------------------------------
    # 2. 构造 retain graph
    # ------------------------------------------------------------
    # unlearning 阶段使用 retain 图：
    # 也就是从训练图中去掉 forget interactions。
    retain_user_items = split.retain_user_items

    norm_adj = build_normalized_adj(
        split.num_users,
        split.num_items,
        retain_user_items
    ).to(device)

    # ------------------------------------------------------------
    # 3. 初始化模型，并加载 fulltrain 参数 theta_star
    # ------------------------------------------------------------
    model = LightGCN(
        split.num_users,
        split.num_items,
        emb_dim=train_args['emb_dim'],
        num_layers=train_args['layers']
    ).to(device)

    model.load_state_dict(ckpt['model_state'])

    # theta_star 是原始 fulltrain 模型参数
    theta_star = {
        name: p.detach().cpu().clone()
        for name, p in model.named_parameters()
    }

    # ------------------------------------------------------------
    # 4. 读取 Adam diag 作为曲率代理
    # ------------------------------------------------------------
    adam_diag = name_to_adam_diag(model, ckpt)

    if not adam_diag:
        # 如果没有保存 Adam diag，则曲率修正项退化为 0
        adam_diag = {
            name: torch.zeros_like(p.detach().cpu())
            for name, p in model.named_parameters()
        }
        print("[Warning] No adam_diag_by_name found. Curvature proxy is zero.")
    else:
        adam_diag = normalize_adam_diag(
            adam_diag,
            target_mean=args.adam_target_mean
        )

    # ------------------------------------------------------------
    # 5. 构造 forget sampler
    # ------------------------------------------------------------
    forget_dataset = PairDataset(
        split.forget_pairs,
        split.num_items,
        split.all_pos
    )

    # ------------------------------------------------------------
    # 6. 计算 rho 和 kappa
    # ------------------------------------------------------------
    m = max(len(split.forget_pairs), 1)
    n = len(split.retain_pairs) + len(split.forget_pairs)

    rho = m / max(n - m, 1)
    kappa = n / max(n - m, 1)

    history = []

    # ============================================================
    # 7. Unlearning 主循环
    # ============================================================
    for epoch in range(1, args.epochs + 1):
        model.train()

        epoch_raw_lams = []
        epoch_lams = []
        epoch_proj = []
        epoch_dir_norm = []
        epoch_cancel = []
        epoch_constraint = []
        epoch_constraint_ok = []
        epoch_eps_scaled = []
        epoch_cos_gu = []
        epoch_cos_gr = []
        epoch_constraint_linear = []
        epoch_constraint_quad = []
        epoch_constraint_second_order = []
        epoch_h_d_quad = []
        epoch_curv_norm = []
        epoch_unlearn_loss = []
        epoch_bpr_loss = []

        prog = tqdm(
            range(args.steps_per_epoch),
            desc=f'Unlearn eps={eps:.4g} epoch={epoch}',
            leave=False
        )

        for step in prog:
            # ----------------------------------------------------
            # 7.1 采样 forget batch
            # ----------------------------------------------------
            users, pos, neg = forget_dataset.sample(args.batch_size)

            users = users.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            # 可选 hard negative
            if args.hard_neg_topk > 0:
                neg = sample_hard_negatives(
                    model=model,
                    users=users,
                    pos=pos,
                    norm_adj=norm_adj,
                    all_pos=split.all_pos,
                    num_items=split.num_items,
                    hard_neg_topk=args.hard_neg_topk,
                    hard_neg_pool=args.hard_neg_pool,
                )

            # ----------------------------------------------------
            # 7.2 计算 unlearning loss gradient:
            #
            #     g_u = grad l_u(theta_t)
            #
            # 注意：
            #     l_u 是 loss，越小越好。
            # ----------------------------------------------------
            model.zero_grad(set_to_none=True)

            unlearn_loss = unlearning_reverse_bpr_loss(
                model=model,
                users=users,
                pos=pos,
                neg=neg,
                norm_adj=norm_adj,
                reg_weight=args.reg_weight,
                reverse_margin=args.reverse_margin,
                forget_score_weight=args.forget_score_weight,
            )

            unlearn_loss.backward()
            grads_u = get_grad_dict(model)

            # ----------------------------------------------------
            # 7.3 计算 forget-side original BPR loss gradient:
            #
            #     g_f = grad l_f(theta_t)
            #
            # 用于构造：
            #
            #     g_r_hat = -rho * g_f + kappa * H_D * delta
            # ----------------------------------------------------
            model.zero_grad(set_to_none=True)

            bpr_loss = forget_bpr_train_loss(
                model=model,
                users=users,
                pos=pos,
                neg=neg,
                norm_adj=norm_adj,
                reg_weight=0.0,
            )

            bpr_loss.backward()
            grads_f = get_grad_dict(model)

            # ----------------------------------------------------
            # 7.4 根据 UPUP 一阶约束求方向
            # ----------------------------------------------------
            direction, stats = build_proxy_direction(
                model=model,
                theta_star=theta_star,
                adam_diag=adam_diag,
                grads_u=grads_u,
                grads_f=grads_f,
                rho=rho,
                kappa=kappa,
                eps=eps,
                anchor_weight=args.anchor_weight,
                max_lam=args.max_lam,
            )

            # ----------------------------------------------------
            # 7.5 执行参数更新：
            #
            #     theta <- theta - lr * direction
            # ----------------------------------------------------
            apply_direction(model, direction, lr=args.lr)

            # ----------------------------------------------------
            # 7.6 记录 step 级统计
            # ----------------------------------------------------
            epoch_raw_lams.append(stats['raw_lam'])
            epoch_lams.append(stats['lam'])
            epoch_proj.append(stats['proj_gr_gu'])
            epoch_dir_norm.append(stats['norm_dir'])
            epoch_cancel.append(stats['cancel_factor'])
            epoch_constraint.append(stats['constraint_gr_dt'])
            epoch_constraint_ok.append(stats['constraint_satisfied'])
            epoch_eps_scaled.append(stats['eps_scaled'])
            epoch_cos_gu.append(stats['cos_dt_gu'])
            epoch_cos_gr.append(stats['cos_dt_gr'])
            epoch_constraint_linear.append(stats['constraint_linear'])
            epoch_constraint_quad.append(stats['constraint_quad'])
            epoch_constraint_second_order.append(stats['constraint_second_order'])
            epoch_h_d_quad.append(stats['h_d_quad'])
            epoch_curv_norm.append(stats['curv_norm'])
            epoch_unlearn_loss.append(float(unlearn_loss.item()))
            epoch_bpr_loss.append(float(bpr_loss.item()))

            if (step + 1) % args.log_interval == 0 or (step + 1) == args.steps_per_epoch:
                prog.set_postfix({
                    'u_loss': f'{unlearn_loss.item():.4f}',
                    'bpr': f'{bpr_loss.item():.4f}',
                    'lam': f'{stats["lam"]:.2e}',
                    'eps_s': f'{stats["eps_scaled"]:.2e}',
                    'cons': f'{stats["constraint_second_order"]:.2e}',
                    'dir': f'{stats["norm_dir"]:.2e}',
                    'cos_gu': f'{stats["cos_dt_gu"]:.3f}',
                    'ok': f'{stats["constraint_satisfied"]:.0f}',
                })

        # ========================================================
        # 8. 每个 epoch 结束后评估
        # ========================================================

        test_metric = evaluate_ranking(
            model,
            norm_adj,
            split.test_pairs,
            split.all_pos,
            split.num_items,
            k=10
        )

        forget_metric = forget_score(
            model,
            norm_adj,
            split.forget_pairs,
            split.all_pos,
            split.num_items
        )

        forget_detail_metric = forget_detailed_score(
            model,
            norm_adj,
            split.forget_pairs,
            split.all_pos,
            split.num_items,
            k=10,
            n_neg=99,
        )

        delta_norm = compute_delta_norm(model, theta_star)

        log = {
            'epoch': epoch,

            # test ranking metrics
            **test_metric,

            # forget score metrics
            **forget_metric,

            # detailed forget ranking metrics
            **forget_detail_metric,

            # constants
            'rho': float(rho),
            'kappa': float(kappa),

            # losses
            'avg_unlearn_loss': float(np.mean(epoch_unlearn_loss)) if epoch_unlearn_loss else 0.0,
            'avg_bpr_loss': float(np.mean(epoch_bpr_loss)) if epoch_bpr_loss else 0.0,

            # lambda and direction stats
            'avg_raw_lam': float(np.mean(epoch_raw_lams)) if epoch_raw_lams else 0.0,
            'avg_lam': float(np.mean(epoch_lams)) if epoch_lams else 0.0,
            'avg_cancel_factor': float(np.mean(epoch_cancel)) if epoch_cancel else 0.0,
            'avg_proj_gr_gu': float(np.mean(epoch_proj)) if epoch_proj else 0.0,
            'avg_dir_norm': float(np.mean(epoch_dir_norm)) if epoch_dir_norm else 0.0,

            # constraint stats
            'avg_eps_scaled': float(np.mean(epoch_eps_scaled)) if epoch_eps_scaled else 0.0,
            'avg_constraint_gr_dt': float(np.mean(epoch_constraint)) if epoch_constraint else 0.0,
            'avg_constraint_linear': float(np.mean(epoch_constraint_linear)) if epoch_constraint_linear else 0.0,
            'avg_constraint_quad': float(np.mean(epoch_constraint_quad)) if epoch_constraint_quad else 0.0,
            'avg_constraint_second_order': float(np.mean(epoch_constraint_second_order)) if epoch_constraint_second_order else 0.0,
            'avg_constraint_satisfied': float(np.mean(epoch_constraint_ok)) if epoch_constraint_ok else 0.0,

            # curvature diagnostics
            'avg_h_d_quad': float(np.mean(epoch_h_d_quad)) if epoch_h_d_quad else 0.0,
            'avg_curv_norm': float(np.mean(epoch_curv_norm)) if epoch_curv_norm else 0.0,

            # cosine diagnostics
            'avg_cos_dt_gu': float(np.mean(epoch_cos_gu)) if epoch_cos_gu else 0.0,
            'avg_cos_dt_gr': float(np.mean(epoch_cos_gr)) if epoch_cos_gr else 0.0,

            # parameter drift
            'delta_norm': delta_norm,
        }

        history.append(log)

        print(f'eps={eps:.4g} epoch={epoch} | {log}')

    # ============================================================
    # 9. 保存结果
    # ============================================================

    final_metric = history[-1]

    save_obj = {
        'model_state': model.state_dict(),
        'history': history,
        'eps': eps,
        'rho': rho,
        'kappa': kappa,
        'anchor_weight': args.anchor_weight,
        'adam_target_mean': args.adam_target_mean,
        'curv_constraint_scale': args.curv_constraint_scale,
        'note': 'UPUP loss-objective first-order constraint with retain loss gradient proxy.',
    }

    torch.save(save_obj, out_dir / 'unlearned.pt')

    save_json(
        {
            'history': history,
            'final': final_metric,
            'eps': eps,
            'rho': rho,
            'kappa': kappa,
            'anchor_weight': args.anchor_weight,
            'adam_target_mean': args.adam_target_mean,
            'curv_constraint_scale': args.curv_constraint_scale,
            'note': 'UPUP loss-objective first-order constraint with retain loss gradient proxy.',
        },
        out_dir / 'metrics.json'
    )

    return {'eps': eps, **final_metric}


# ============================================================
# 11. eps list 解析
# ============================================================

def parse_eps_list(args: argparse.Namespace) -> List[float]:
    if args.eps_list.strip():
        return [float(x) for x in args.eps_list.split(',') if x.strip()]

    return [float(args.eps)]


# ============================================================
# 12. main
# ============================================================

def main() -> None:
    args = parse_args()
    eps_list = parse_eps_list(args)

    if any(e < 0 for e in eps_list):
        raise ValueError('All epsilon values must be non-negative.')

    ensure_dir(Path(args.output_dir))

    summary = []

    for eps in eps_list:
        summary.append(run_single_eps(args, eps))

    save_json(
        {'results': summary},
        Path(args.output_dir) / 'summary.json'
    )

    print('Summary:', summary)


if __name__ == '__main__':
    main()
