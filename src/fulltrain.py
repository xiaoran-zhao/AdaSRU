from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from .checkpoint import save_training_checkpoint
from .data import (
    BPRDataset,
    build_random_user_ratio_split,
    build_normalized_adj,
    save_split,
)
from .eval import evaluate_ranking, forget_score
from .model import LightGCN
from .utils import device_from_arg, ensure_dir, save_json, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument('--data_root', type=str, default='data')
    p.add_argument('--output_dir', type=str, default='runs/ml1m_fulltrain')
    p.add_argument('--seed', type=int, default=42)

    # 从完整训练集中抽取多少比例作为 forget 数据
    # 注意：forget 数据仍然参与 fulltrain 原始模型训练
    p.add_argument('--forget_ratio', type=float, default=0.1)

    # 评分 >= positive_threshold 视为正反馈
    p.add_argument('--positive_threshold', type=int, default=4)

    p.add_argument('--emb_dim', type=int, default=64)
    p.add_argument('--layers', type=int, default=3)
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--batch_size', type=int, default=4096)
    p.add_argument('--batches_per_epoch', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--reg_weight', type=float, default=1e-4)
    p.add_argument('--device', type=str, default='auto')

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = device_from_arg(args.device)
    out_dir = ensure_dir(args.output_dir)

    split = build_random_user_ratio_split(
        data_root=args.data_root,
        forget_ratio=args.forget_ratio,
        seed=args.seed,
        positive_threshold=args.positive_threshold,
    )

    save_split(split, out_dir)

    num_full_train = sum(len(items) for items in split.full_train_user_items.values())
    num_retain = sum(len(items) for items in split.retain_user_items.values())
    num_forget = len(split.forget_pairs)
    num_val = len(split.val_pairs)
    num_test = len(split.test_pairs)

    print('\n===== Split Stats =====')
    print(f'full train interactions: {num_full_train}')
    print(f'retain interactions:     {num_retain}')
    print(f'forget interactions:     {num_forget}')
    print(f'val interactions:        {num_val}')
    print(f'test interactions:       {num_test}')

    if num_full_train > 0:
        print(f'forget ratio in full train: {num_forget / num_full_train:.4f}')
        print(f'retain ratio in full train: {num_retain / num_full_train:.4f}')

    # 原始模型使用完整训练集训练：retain + forget
    norm_adj = build_normalized_adj(
        split.num_users,
        split.num_items,
        split.full_train_user_items,
    ).to(device)

    dataset = BPRDataset(
        split.full_train_user_items,
        split.num_items,
        split.all_pos,
    )

    model = LightGCN(
        split.num_users,
        split.num_items,
        emb_dim=args.emb_dim,
        num_layers=args.layers,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_metric = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()

        prog = tqdm(
            range(args.batches_per_epoch),
            desc=f'FullTrain {epoch}/{args.epochs}',
            leave=False,
        )

        for _ in prog:
            users, pos, neg = dataset.sample(args.batch_size)

            users = users.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            optimizer.zero_grad(set_to_none=True)

            loss, stats = model.bpr_loss(
                users,
                pos,
                neg,
                norm_adj,
                reg_weight=args.reg_weight,
            )

            loss.backward()
            optimizer.step()

            prog.set_postfix(loss=f"{stats['total']:.4f}")

        val = evaluate_ranking(
            model,
            norm_adj,
            split.val_pairs,
            split.all_pos,
            split.num_items,
            k=10,
        )

        val_score = val['recall@10'] + val['ndcg@10']

        print(f'Epoch {epoch:03d} | val={val}')

        if val_score > best_metric:
            best_metric = val_score

            test = evaluate_ranking(
                model,
                norm_adj,
                split.test_pairs,
                split.all_pos,
                split.num_items,
                k=10,
            )

            fg = forget_score(
                model,
                norm_adj,
                split.forget_pairs,
                split.all_pos,
                split.num_items,
            )

            best_eval = {
                'epoch': epoch,
                'val': val,
                'test': test,
                'forget': fg,
                'train_type': 'full_train',
                'used_train_data': 'full_train_user_items',
            }

            save_training_checkpoint(
                Path(out_dir) / 'best_model.pt',
                model,
                optimizer,
                vars(args),
                best_eval,
            )

            save_json(best_eval, Path(out_dir) / 'best_eval.json')

            print(f'  saved best fulltrain checkpoint: {best_eval}')


if __name__ == '__main__':
    main()
