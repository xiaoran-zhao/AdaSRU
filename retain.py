
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

# 从 checkpoint 模块中导入保存训练检查点的函数
from .checkpoint import save_training_checkpoint
# 从 data 模块中导入：
# BPRDataset：BPR 采样数据集
# build_leave_one_out_split：构建 leave-one-out 数据划分
# build_normalized_adj：构建归一化邻接矩阵
# save_split：保存数据划分结果
from .data_old import BPRDataset, build_random_user_ratio_split, build_normalized_adj, save_split
# 从 eval 模块中导入：
# evaluate_ranking：推荐效果评估
# forget_score：遗忘集打分评估
from .eval import evaluate_ranking, forget_score
# 从 model 模块中导入 LightGCN 模型
from .model import LightGCN
# 从 utils 模块中导入：
# device_from_arg：根据参数选择运行设备
# ensure_dir：确保目录存在
# save_json：保存 json 文件
# set_seed：设置随机种子
from .utils import device_from_arg, ensure_dir, save_json, set_seed

# 解析命令行参数
def parse_args() -> argparse.Namespace:
    # 创建参数解析器对象
    p = argparse.ArgumentParser()
    # 数据根目录，默认是 data
    p.add_argument('--data_root', type=str, default='data')
    # 输出目录，默认是 runs/ml1m_base
    p.add_argument('--output_dir', type=str, default='runs/ml1m_base')
    # 随机种子
    p.add_argument('--seed', type=int, default=42)
    # 从训练交互中抽取多少比例作为 forget 数据
    p.add_argument('--forget_ratio', type=float, default=0.1)
    # 正反馈阈值，评分大于等于该值才视为正样本
    p.add_argument('--positive_threshold', type=int, default=4)
    # 嵌入维度
    p.add_argument('--emb_dim', type=int, default=64)
    # LightGCN 的传播层数
    p.add_argument('--layers', type=int, default=3)
    # 训练轮数
    p.add_argument('--epochs', type=int, default=200)
    # 每个 batch 的样本数量
    p.add_argument('--batch_size', type=int, default=4096)
    # 每个 epoch 训练多少个 batch
    p.add_argument('--batches_per_epoch', type=int, default=256)
    # 学习率
    p.add_argument('--lr', type=float, default=1e-3)
    # 优化器的权重衰减系数
    p.add_argument('--weight_decay', type=float, default=0.0)
    # BPR 损失中的 embedding 正则项权重
    p.add_argument('--reg_weight', type=float, default=1e-4)
    # 运行设备，例如 cpu / cuda / auto
    p.add_argument('--device', type=str, default='auto')
    # 返回解析后的参数对象
    return p.parse_args()


# 主函数
def main() -> None:
    # 读取命令行参数
    args = parse_args()
    # 设置随机种子，保证实验可复现
    set_seed(args.seed)
    # 根据参数决定使用 CPU 还是 GPU
    device = device_from_arg(args.device)
    # 确保输出目录存在
    out_dir = ensure_dir(args.output_dir)

    # 构建 leave-one-out 数据划分
    split = build_random_user_ratio_split(
        # 数据根目录
        data_root=args.data_root,
        # 遗忘比例
        forget_ratio=args.forget_ratio,
        # 随机种子
        seed=args.seed,
        # 正反馈阈值
        positive_threshold=args.positive_threshold,
    )

    # ===== 检查整体划分结果 =====
    num_train = sum(len(items) for items in split.train_user_items.values())
    num_val = len(split.val_pairs)
    num_test = len(split.test_pairs)
    total = num_train + num_val + num_test

    print('\n===== Overall Split Stats =====')
    print(f'train interactions: {num_train}')
    print(f'val interactions:   {num_val}')
    print(f'test interactions:  {num_test}')
    print(f'total interactions: {total}')

    if total > 0:
        print(f'train ratio: {num_train / total:.4f}')
        print(f'val ratio:   {num_val / total:.4f}')
        print(f'test ratio:  {num_test / total:.4f}')

        # ===== 检查每个用户的划分情况 =====
    user_val_count = {}
    user_test_count = {}

    for u, _ in split.val_pairs:
        user_val_count[u] = user_val_count.get(u, 0) + 1

    for u, _ in split.test_pairs:
        user_test_count[u] = user_test_count.get(u, 0) + 1

    print('\n===== Per-user Split Examples =====')
    checked_users = list(split.train_user_items.keys())[:10]  # 先看前10个用户

    for u in checked_users:
        n_train_u = len(split.train_user_items[u])
        n_val_u = user_val_count.get(u, 0)
        n_test_u = user_test_count.get(u, 0)
        n_total_u = n_train_u + n_val_u + n_test_u

        print(
            f'user={u}, total={n_total_u}, '
            f'train={n_train_u}, val={n_val_u}, test={n_test_u}, '
            f'ratios=({n_train_u / n_total_u:.2f}, {n_val_u / n_total_u:.2f}, {n_test_u / n_total_u:.2f})'
        )
    # 将划分结果保存到输出目录
    save_split(split, out_dir)

    # 根据训练交互构建归一化邻接矩阵，并放到指定设备上
    norm_adj = build_normalized_adj(split.num_users, split.num_items, split.train_user_items).to(device)
    # 构建 BPR 数据集，用于训练时动态采样 (user, pos_item, neg_item)
    dataset = BPRDataset(split.train_user_items, split.num_items, split.all_pos)

    # 创建 LightGCN 模型，并移动到指定设备
    model = LightGCN(split.num_users, split.num_items, emb_dim=args.emb_dim, num_layers=args.layers).to(device)
    # 创建 Adam 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 记录当前最优验证指标，初始设为一个很小的值
    best_metric = -1.0
    # 用来保存当前最优模型对应的评测结果
    best_eval = {}
    # 开始按 epoch 训练
    for epoch in range(1, args.epochs + 1):
        # 切换到训练模式
        model.train()
        # 创建进度条，显示当前 epoch 内 batch 训练进度
        prog = tqdm(range(args.batches_per_epoch), desc=f'Train {epoch}/{args.epochs}', leave=False)
        # 遍历本 epoch 的所有 batch
        for _ in prog:
            # 从数据集中采样一个 batch 的 (用户, 正样本物品, 负样本物品)
            users, pos, neg = dataset.sample(args.batch_size)
            # 将用户张量移动到设备上
            users = users.to(device)
            # 将正样本张量移动到设备上
            pos = pos.to(device)
            # 将负样本张量移动到设备上
            neg = neg.to(device)
            # 清空优化器中的梯度
            # set_to_none=True 可以略微节省显存并提高效率
            optimizer.zero_grad(set_to_none=True)
            # 计算 BPR 总损失以及统计信息
            loss, stats = model.bpr_loss(users, pos, neg, norm_adj, reg_weight=args.reg_weight)
            # 反向传播，计算梯度
            loss.backward()
            # 用优化器更新模型参数
            optimizer.step()
            # 在进度条中显示当前 batch 的 total loss
            prog.set_postfix(loss=f"{stats['total']:.4f}")

        # 每个 epoch 结束后，在验证集上评估推荐效果
        val = evaluate_ranking(model, norm_adj, split.val_pairs, split.all_pos, split.num_items, k=10)
        # 相加作为选最优模型的综合指标
        val_score = val['recall@10'] + val['ndcg@10']
        # 打印当前 epoch 的验证集结果
        print(f'Epoch {epoch:03d} | val={val}')
        # 如果当前模型优于历史最优模型
        if val_score > best_metric:
            # 更新最优指标
            best_metric = val_score
            # 在测试集上评估推荐效果
            test = evaluate_ranking(model, norm_adj, split.test_pairs, split.all_pos, split.num_items, k=10)
            # 在遗忘集上评估遗忘相关分数
            fg = forget_score(model, norm_adj, split.forget_pairs, split.all_pos, split.num_items)
            # 保存当前最优模型的完整评测结果
            best_eval = {'epoch': epoch, 'val': val, 'test': test, 'forget': fg}
            # 保存训练检查点，包括模型参数、优化器状态、训练参数和评测信息
            save_training_checkpoint(
                Path(out_dir) / 'best_model.pt',
                model,
                optimizer,
                vars(args),
                best_eval,
            )
            # 将最优评测结果单独保存为 json 文件
            save_json(best_eval, Path(out_dir) / 'best_eval.json')
            # 打印提示，说明当前最优 checkpoint 已保存
            print(f'  saved best checkpoint: {best_eval}')


# 如果当前脚本是直接运行，而不是被 import
if __name__ == '__main__':
    # 执行主函数
    main()