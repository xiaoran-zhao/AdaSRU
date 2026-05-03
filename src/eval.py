from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Sequence, Tuple

import numpy as np
import torch

from .model import LightGCN

def group_eval_pairs(eval_pairs: Sequence[Tuple[int, int]]) -> dict[int, list[int]]:
    user_pos = defaultdict(list)
    for u, i in eval_pairs:
        user_pos[int(u)].append(int(i))
    return dict(user_pos)

def precision_at_k(ranklist: list[int], ground_truth: set[int]) -> np.ndarray:
    hits = np.array([1.0 if item in ground_truth else 0.0 for item in ranklist], dtype=np.float32)
    return np.cumsum(hits) / np.arange(1, len(ranklist) + 1)

def recall_at_k(ranklist: list[int], ground_truth: set[int]) -> np.ndarray:
    hits = np.array([1.0 if item in ground_truth else 0.0 for item in ranklist], dtype=np.float32)
    return np.cumsum(hits) / max(len(ground_truth), 1)

def ndcg_at_k(ranklist: list[int], ground_truth: set[int]) -> np.ndarray:
    len_rank = len(ranklist)
    len_gt = len(ground_truth)
    if len_gt == 0:
        return np.zeros(len_rank, dtype=np.float32)

    gains = np.array(
        [1.0 / math.log2(idx + 2) if item in ground_truth else 0.0 for idx, item in enumerate(ranklist)],
        dtype=np.float32,
    )
    dcg = np.cumsum(gains)

    idcg_len = min(len_gt, len_rank)
    ideal_gains = np.array([1.0 / math.log2(i + 2) for i in range(len_rank)], dtype=np.float32)
    idcg = np.cumsum(ideal_gains)
    idcg[idcg_len:] = idcg[idcg_len - 1]

    return dcg / idcg

@torch.no_grad()
def evaluate_ranking(
    model: LightGCN,
    norm_adj: torch.Tensor,
    eval_pairs: Sequence[Tuple[int, int]],
    all_pos: Dict[int, set],
    num_items: int,
    k: int = 10,
) -> dict[str, float]:

    model.eval()

    user_emb, item_emb = model.get_user_item_embeddings(norm_adj)

    user_test = group_eval_pairs(eval_pairs)

    precisions = []
    recalls = []
    ndcgs = []

    for u, gt_items in user_test.items():
        ground_truth = set(gt_items)

        scores = torch.matmul(item_emb, user_emb[u]).detach().cpu().numpy()

        train_pos = set(all_pos[u]) - ground_truth
        if len(train_pos) > 0:
            scores[list(train_pos)] = -np.inf

        topk_idx = np.argpartition(-scores, kth=min(k, len(scores) - 1))[:k]
        topk_idx = topk_idx[np.argsort(-scores[topk_idx])]
        ranklist = topk_idx.tolist()

        p = precision_at_k(ranklist, ground_truth)
        r = recall_at_k(ranklist, ground_truth)
        n = ndcg_at_k(ranklist, ground_truth)

        precisions.append(p[k - 1] if len(p) >= k else p[-1])
        recalls.append(r[k - 1] if len(r) >= k else r[-1])
        ndcgs.append(n[k - 1] if len(n) >= k else n[-1])

    return {
        f'precision@{k}': float(np.mean(precisions)) if precisions else 0.0,
        f'recall@{k}': float(np.mean(recalls)) if recalls else 0.0,
        f'ndcg@{k}': float(np.mean(ndcgs)) if ndcgs else 0.0,
    }

@torch.no_grad()
def forget_score(
    model: LightGCN,
    norm_adj: torch.Tensor,
    forget_pairs: Sequence[Tuple[int, int]],
    all_pos: Dict[int, set],
    num_items: int,
    n_neg: int = 10,
) -> dict[str, float]:
    model.eval()
    user_emb, item_emb = model.get_user_item_embeddings(norm_adj)

    margins = []
    probs = []

    for u, gt in forget_pairs:
        for _ in range(n_neg):
            neg = np.random.randint(0, num_items)
            while neg in all_pos[u] or neg == gt:
                neg = np.random.randint(0, num_items)

            pos_score = torch.dot(user_emb[u], item_emb[gt]).item()
            neg_score = torch.dot(user_emb[u], item_emb[neg]).item()

            margin = pos_score - neg_score
            margins.append(margin)
            probs.append(1.0 / (1.0 + math.exp(-margin)))

    return {
        'forget_margin': float(np.mean(margins)) if margins else 0.0,
        'forget_prob': float(np.mean(probs)) if probs else 0.0,
    }

@torch.no_grad()
def forget_detailed_score(
    model,
    norm_adj,
    forget_pairs,
    all_pos,
    num_items,
    k: int = 10,
    n_neg: int = 99,
):
    model.eval()

    user_emb, item_emb = model.computer(norm_adj)

    pos_scores_all = []
    neg_scores_all = []
    gap_scores_all = []
    hit_list = []
    ndcg_list = []
    rank_list = []

    for u, pos_item in forget_pairs:
        user_all_pos = all_pos[u]

        negatives = []
        while len(negatives) < n_neg:
            neg_item = np.random.randint(0, num_items)
            if neg_item not in user_all_pos and neg_item != pos_item:
                negatives.append(neg_item)

        items = [pos_item] + negatives

        u_e = user_emb[u]
        i_e = item_emb[items]

        scores = torch.sum(u_e.unsqueeze(0) * i_e, dim=1)

        pos_score = scores[0].item()
        neg_score_mean = scores[1:].mean().item()
        gap = pos_score - neg_score_mean

        rank = int(torch.argsort(scores, descending=True).tolist().index(0)) + 1

        hit = 1.0 if rank <= k else 0.0
        ndcg = 1.0 / np.log2(rank + 1) if rank <= k else 0.0

        pos_scores_all.append(pos_score)
        neg_scores_all.append(neg_score_mean)
        gap_scores_all.append(gap)
        hit_list.append(hit)
        ndcg_list.append(ndcg)
        rank_list.append(rank)

    return {
        'forget_pos_score_mean': float(np.mean(pos_scores_all)),
        'forget_neg_score_mean': float(np.mean(neg_scores_all)),
        'forget_gap_mean': float(np.mean(gap_scores_all)),
        'forget_hit@10': float(np.mean(hit_list)),
        'forget_ndcg@10': float(np.mean(ndcg_list)),
        'forget_rank_mean': float(np.mean(rank_list)),
    }
