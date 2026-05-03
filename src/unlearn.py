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

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    
    p.add_argument('--run_dir', type=str, required=True)
    p.add_argument('--output_dir', type=str, default='runs/unlearned')

    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='auto')

    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=2048)
    p.add_argument('--steps_per_epoch', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)

    p.add_argument('--eps', type=float, default=1e-3)
    p.add_argument('--eps_list', type=str, default='')

    p.add_argument('--reg_weight', type=float, default=0.0)
    p.add_argument('--anchor_weight', type=float, default=0.0)

    p.add_argument('--max_lam', type=float, default=1e5)

    p.add_argument('--log_interval', type=int, default=16)

    p.add_argument('--adam_target_mean', type=float, default=1e-3)

    p.add_argument('--curv_constraint_scale', type=float, default=1.0)
    p.add_argument('--bisection_steps', type=int, default=40)

    p.add_argument('--reverse_margin', type=float, default=0.0)

    p.add_argument('--forget_score_weight', type=float, default=0.0)

    p.add_argument('--hard_neg_topk', type=int, default=0)
    p.add_argument('--hard_neg_pool', type=int, default=1000)

    return p.parse_args()

class PairDataset:

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
    
    l.computer(norm_adj)

    u_e = user_emb[users]
    p_e = item_emb[pos]
    n_e = item_emb[neg]

    pos_scores = torch.sum(u_e * p_e, dim=1)
    neg_scores = torch.sum(u_e * n_e, dim=1)

    reverse_bpr_loss = -F.logsigmoid(
        neg_scores - pos_scores - reverse_margin
    ).mean()

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

        loss = loss + reg_weight * reg

    return loss

def forget_bpr_train_loss(
    model: LightGCN,
    users: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
    norm_adj: torch.Tensor,
    reg_weight: float = 0.0,
) -> torch.Tensor:

    user_emb, item_emb = model.computer(norm_adj)

    u_e = user_emb[users]
    p_e = item_emb[pos]
    n_e = item_emb[neg]

    pos_scores = torch.sum(u_e * p_e, dim=1)
    neg_scores = torch.sum(u_e * n_e, dim=1)

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

def name_to_adam_diag(
    model: LightGCN,
    checkpoint: dict,
) -> Dict[str, torch.Tensor]:

    raw_diag = checkpoint.get('adam_diag_by_name', {})
    named: Dict[str, torch.Tensor] = {}

    state_dict = model.state_dict()

    for name, _ in state_dict.items():
        if name in raw_diag:
            named[name] = raw_diag[name].detach().clone()

    if named:
        return named

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

def flatten_named_tensors(named_tensors: Dict[str, torch.Tensor]) -> torch.Tensor:

    if not named_tensors:
        raise ValueError("No tensors to flatten. Please check whether gradients exist.")

    return torch.cat([v.reshape(-1) for v in named_tensors.values()])


def unflatten_to_named(
    flat: torch.Tensor,
    template: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:

    out: Dict[str, torch.Tensor] = {}
    offset = 0

    for name, t in template.items():
        numel = t.numel()
        out[name] = flat[offset: offset + numel].reshape_as(t)
        offset += numel

    return out


def get_grad_dict(model: LightGCN) -> Dict[str, torch.Tensor]:

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

    total = 0.0

    with torch.no_grad():
        for name, p in model.named_parameters():
            delta = p.detach() - theta_star[name].to(p.device)
            total += delta.norm().item()

    return float(total)
    
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
    
    for name, p in model.named_parameters():
        if name not in grads_u:
            continue

        if name not in grads_f:
            raise ValueError(f"Missing forget BPR gradient for parameter: {name}")

        delta = p.detach() - theta_star[name].to(p.device)

        hdiag = adam_diag.get(
            name,
            torch.zeros_like(p.detach().cpu())
        ).to(p.device).abs()

        h = kappa * hdiag

        curv = h * delta

        proxy_r[name] = -rho * grads_f[name] + curv

        hessian_diag[name] = h
        curv_terms[name] = curv

    g_u_flat = flatten_named_tensors(grads_u)
    g_r_flat = flatten_named_tensors(proxy_r)
    h_flat = flatten_named_tensors(hessian_diag)
    curv_flat = flatten_named_tensors(curv_terms)

    norm_gu = torch.norm(g_u_flat)
    norm_gr = torch.norm(g_r_flat)

    gr_gu = torch.dot(g_r_flat, g_u_flat)

    eps_scaled = torch.tensor(float(eps), device=g_u_flat.device)

    if gr_gu >= -eps_scaled:
        lam = torch.tensor(0.0, device=g_u_flat.device)
        raw_lam = lam
        d_flat = g_u_flat
        constraint_value = gr_gu
    else:
        denom = torch.dot(g_r_flat, g_r_flat) + 1e-12

        raw_lam = (-eps_scaled - gr_gu) / denom
        lam = torch.clamp(raw_lam, min=0.0, max=max_lam)

        # d = g_u + lambda * g_r_hat
        d_flat = g_u_flat + lam * g_r_flat

        constraint_value = torch.dot(g_r_flat, d_flat)

    direction = unflatten_to_named(d_flat, grads_u)

    if anchor_weight > 0:
        for name, p in model.named_parameters():
            if name not in direction:
                continue

            delta = p.detach() - theta_star[name].to(p.device)

            hdiag = adam_diag.get(
                name,
                torch.zeros_like(p.detach().cpu())
            ).to(p.device).abs()

            direction[name] = direction[name] + anchor_weight * hdiag * delta

    dir_flat = flatten_named_tensors(direction)
    norm_dir = torch.norm(dir_flat)

    gr_dot_d = torch.dot(g_r_flat, dir_flat)
    h_d_quad = torch.sum(h_flat * dir_flat.pow(2))

    cos_dt_gu = torch.dot(dir_flat, g_u_flat) / (norm_dir * norm_gu + 1e-12)
    cos_dt_gr = torch.dot(dir_flat, g_r_flat) / (norm_dir * norm_gr + 1e-12)

    stats = {
        'raw_lam': float(raw_lam.item()),
        'lam': float(lam.item()),

        'proj_gr_gu': float(gr_gu.item()),

        'norm_gu': float(norm_gu.item()),
        'norm_gr': float(norm_gr.item()),
        'norm_dir': float(norm_dir.item()),

        'curv_norm': float(torch.norm(curv_flat).item()),

        'eps': float(eps),
        'eps_scaled': float(eps_scaled.item()),

        'constraint_gr_dt': float(gr_dot_d.item()),

        'constraint_linear': float(gr_dot_d.item()),
        'constraint_quad': 0.0,
        'constraint_second_order': float(constraint_value.item()),

        'constraint_satisfied': float(
            (constraint_value >= -eps_scaled - 1e-12).item()
        ),

        'h_d_quad': float(h_d_quad.item()),
        'cancel_factor': float(1.0 + lam.item() * rho),

        'cos_dt_gu': float(cos_dt_gu.item()),
        'cos_dt_gr': float(cos_dt_gr.item()),
    }

    return direction, stats

def apply_direction(
    model: LightGCN,
    direction: Dict[str, torch.Tensor],
    lr: float,
) -> None:

    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in direction:
                p.add_(-lr * direction[name])
                
def run_single_eps(args: argparse.Namespace, eps: float) -> dict:
    set_seed(args.seed)
    device = device_from_arg(args.device)

    run_dir = Path(args.run_dir)
    out_dir = ensure_dir(Path(args.output_dir) / f'eps_{eps:.4g}')

    split = load_split(run_dir / 'split.json')

    ckpt = torch.load(run_dir / 'best_model.pt', map_location='cpu')

    print('checkpoint keys:', ckpt.keys())
    print('has adam diag:', 'adam_diag_by_name' in ckpt)

    train_args = ckpt['args']

    retain_user_items = split.retain_user_items

    norm_adj = build_normalized_adj(
        split.num_users,
        split.num_items,
        retain_user_items
    ).to(device)

    model = LightGCN(
        split.num_users,
        split.num_items,
        emb_dim=train_args['emb_dim'],
        num_layers=train_args['layers']
    ).to(device)

    model.load_state_dict(ckpt['model_state'])

    theta_star = {
        name: p.detach().cpu().clone()
        for name, p in model.named_parameters()
    }

    adam_diag = name_to_adam_diag(model, ckpt)

    if not adam_diag:
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

    forget_dataset = PairDataset(
        split.forget_pairs,
        split.num_items,
        split.all_pos
    )

    m = max(len(split.forget_pairs), 1)
    n = len(split.retain_pairs) + len(split.forget_pairs)

    rho = m / max(n - m, 1)
    kappa = n / max(n - m, 1)

    history = []

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
            users, pos, neg = forget_dataset.sample(args.batch_size)

            users = users.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

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

            apply_direction(model, direction, lr=args.lr)

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
            **test_metric,
            **forget_metric,
            **forget_detail_metric,

            'rho': float(rho),
            'kappa': float(kappa),

            'avg_unlearn_loss': float(np.mean(epoch_unlearn_loss)) if epoch_unlearn_loss else 0.0,
            'avg_bpr_loss': float(np.mean(epoch_bpr_loss)) if epoch_bpr_loss else 0.0,
            
            'avg_raw_lam': float(np.mean(epoch_raw_lams)) if epoch_raw_lams else 0.0,
            'avg_lam': float(np.mean(epoch_lams)) if epoch_lams else 0.0,
            'avg_cancel_factor': float(np.mean(epoch_cancel)) if epoch_cancel else 0.0,
            'avg_proj_gr_gu': float(np.mean(epoch_proj)) if epoch_proj else 0.0,
            'avg_dir_norm': float(np.mean(epoch_dir_norm)) if epoch_dir_norm else 0.0,

            'avg_eps_scaled': float(np.mean(epoch_eps_scaled)) if epoch_eps_scaled else 0.0,
            'avg_constraint_gr_dt': float(np.mean(epoch_constraint)) if epoch_constraint else 0.0,
            'avg_constraint_linear': float(np.mean(epoch_constraint_linear)) if epoch_constraint_linear else 0.0,
            'avg_constraint_quad': float(np.mean(epoch_constraint_quad)) if epoch_constraint_quad else 0.0,
            'avg_constraint_second_order': float(np.mean(epoch_constraint_second_order)) if epoch_constraint_second_order else 0.0,
            'avg_constraint_satisfied': float(np.mean(epoch_constraint_ok)) if epoch_constraint_ok else 0.0,

            'avg_h_d_quad': float(np.mean(epoch_h_d_quad)) if epoch_h_d_quad else 0.0,
            'avg_curv_norm': float(np.mean(epoch_curv_norm)) if epoch_curv_norm else 0.0,

            'avg_cos_dt_gu': float(np.mean(epoch_cos_gu)) if epoch_cos_gu else 0.0,
            'avg_cos_dt_gr': float(np.mean(epoch_cos_gr)) if epoch_cos_gr else 0.0,

            'delta_norm': delta_norm,
        }

        history.append(log)

        print(f'eps={eps:.4g} epoch={epoch} | {log}')

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

def parse_eps_list(args: argparse.Namespace) -> List[float]:
    if args.eps_list.strip():
        return [float(x) for x in args.eps_list.split(',') if x.strip()]

    return [float(args.eps)]

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
