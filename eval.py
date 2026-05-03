# from __future__ import annotations
#
# import math
# import random
# from typing import Dict, Iterable, List, Sequence, Tuple
#
# import numpy as np
# import torch
#
# from .model import LightGCN
#
#
# # 为评测阶段采样负样本
# def sample_eval_negatives(num_items: int, all_pos: set[int], target_item: int, n_neg: int = 99) -> list[int]:
#     # 初始化负样本列表
#     negatives = []
#     # 当负样本数量还没有达到要求时，持续采样
#     while len(negatives) < n_neg:
#         # 在 [0, num_items-1] 范围内随机选一个物品 id
#         x = random.randint(0, num_items - 1)
#         # 如果这个物品不是用户的历史正样本，并且不是目标物品 itself，则可作为负样本
#         if x not in all_pos and x != target_item:
#             # 加入负样本列表
#             negatives.append(x)
#     # 返回采样好的负样本列表
#     return negatives
#
#
# # 关闭梯度计算，表示该函数只用于评估，不进行反向传播
# @torch.no_grad()
# def evaluate_ranking(
#     model: LightGCN,
#     norm_adj: torch.Tensor,
#     eval_pairs: Sequence[Tuple[int, int]],
#     all_pos: Dict[int, set],
#     num_items: int,
#     k: int = 10,
#     n_neg: int = 99,
# ) -> dict[str, float]:
#     # 将模型切换到评估模式
#     model.eval()
#     # 根据归一化邻接矩阵，获取所有用户和物品的最终嵌入表示
#     user_emb, item_emb = model.get_user_item_embeddings(norm_adj)
#     # 用于保存每个样本的 Hit 指标
#     hits = []
#     # 用于保存每个样本的 NDCG 指标
#     ndcgs = []
#     # 遍历评测集中的每个 (用户, 真实目标物品) 对
#     for u, gt in eval_pairs:
#         # 候选集合由 1 个真实物品 + n_neg 个负样本物品组成
#         candidates = [gt] + sample_eval_negatives(num_items, all_pos[u], gt, n_neg=n_neg)
#         # 将候选物品列表转为张量，并放到与 item_emb 相同的设备上
#         cand_t = torch.tensor(candidates, device=item_emb.device)
#         # 计算该用户对所有候选物品的打分
#         # item_emb[cand_t] 取出候选物品嵌入
#         # user_emb[u] 是该用户嵌入
#         # torch.matmul 相当于逐个候选物品与用户向量做点积
#         scores = torch.matmul(item_emb[cand_t], user_emb[u])
#         # 取打分最高的 top-k 候选物品索引
#         _, idx = torch.topk(scores, k=min(k, len(candidates)))
#         # 根据索引取出 top-k 排名列表，并转回 python list
#         ranklist = cand_t[idx].detach().cpu().tolist()
#         # 如果真实物品出现在 top-k 排名列表中
#         if gt in ranklist:
#             # Hit 记为 1
#             hits.append(1.0)
#             # 找到真实物品在 ranklist 中的位置，位置从 0 开始
#             rank = ranklist.index(gt)
#             # 按照 NDCG 的定义计算该样本的增益
#             # rank=0 时分数为 1/log2(2)=1
#             ndcgs.append(1.0 / math.log2(rank + 2))
#         else:
#             # 如果真实物品不在 top-k 中，则 Hit 为 0
#             hits.append(0.0)
#             # NDCG 也为 0
#             ndcgs.append(0.0)
#     # 返回所有评测样本上的平均 Hit@k 和平均 NDCG@k
#     return {
#         f'hit@{k}': float(np.mean(hits)),
#         f'ndcg@{k}': float(np.mean(ndcgs)),
#     }
#
#
# # 关闭梯度计算，该函数只用于评估遗忘效果
# @torch.no_grad()
# def forget_score(
#     model: LightGCN,
#     norm_adj: torch.Tensor,
#     forget_pairs: Sequence[Tuple[int, int]],
#     all_pos: Dict[int, set],
#     num_items: int,
#     n_neg: int = 10,
# ) -> dict[str, float]:
#     # 将模型切换到评估模式
#     model.eval()
#     # 获取所有用户和物品的最终嵌入表示
#     user_emb, item_emb = model.get_user_item_embeddings(norm_adj)
#     # 用于保存遗忘样本相对负样本的分数差 margin
#     margins = []
#     # 用于保存由 margin 经过 sigmoid 转换后的概率
#     probs = []
#     # 遍历所有待遗忘交互对 (用户, 目标物品)
#     for u, gt in forget_pairs:
#         # 对每个遗忘样本采样 n_neg 个负样本进行比较
#         for _ in range(n_neg):
#             # 随机采样一个物品作为负样本
#             neg = random.randint(0, num_items - 1)
#             # 如果采到的是该用户的历史正样本，或者恰好等于目标物品，则继续重采
#             while neg in all_pos[u] or neg == gt:
#                 neg = random.randint(0, num_items - 1)
#             # 计算用户 u 与目标物品 gt 的匹配分数
#             pos_score = torch.dot(user_emb[u], item_emb[gt]).item()
#             # 计算用户 u 与负样本物品 neg 的匹配分数
#             neg_score = torch.dot(user_emb[u], item_emb[neg]).item()
#             # margin 表示目标物品分数减去负样本分数
#             margin = pos_score - neg_score
#             # 保存该 margin
#             margins.append(margin)
#             # 将 margin 通过 sigmoid 映射到 (0,1)，表示目标物品优于负样本的概率倾向
#             probs.append(1.0 / (1.0 + math.exp(-margin)))
#     # 返回所有遗忘样本上的平均 margin 和平均概率
#     return {
#         # 如果 margins 非空，则返回其均值；否则返回 0，评估模型对待遗忘交互的记忆强度
#         'forget_margin': float(np.mean(margins)) if margins else 0.0,
#         # 如果 probs 非空，则返回其均值；否则返回 0，越大，说明模型越“记得”这些待遗忘样本；如果遗忘做得好，这两个值通常应该下降
#         'forget_prob': float(np.mean(probs)) if probs else 0.0,
#     }


from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Sequence, Tuple

import numpy as np
import torch

from .model import LightGCN


# 将 [(u, i), (u, j), (v, k)] 这种二元组列表，按用户聚合成 {u: [i, j], v: [k]}
def group_eval_pairs(eval_pairs: Sequence[Tuple[int, int]]) -> dict[int, list[int]]:
    user_pos = defaultdict(list)
    for u, i in eval_pairs:
        user_pos[int(u)].append(int(i))
    return dict(user_pos)


# 计算单个用户在前 k 个位置上的 precision 曲线
def precision_at_k(ranklist: list[int], ground_truth: set[int]) -> np.ndarray:
    hits = np.array([1.0 if item in ground_truth else 0.0 for item in ranklist], dtype=np.float32)
    return np.cumsum(hits) / np.arange(1, len(ranklist) + 1)


# 计算单个用户在前 k 个位置上的 recall 曲线
def recall_at_k(ranklist: list[int], ground_truth: set[int]) -> np.ndarray:
    hits = np.array([1.0 if item in ground_truth else 0.0 for item in ranklist], dtype=np.float32)
    return np.cumsum(hits) / max(len(ground_truth), 1)


# 计算单个用户在前 k 个位置上的 ndcg 曲线
def ndcg_at_k(ranklist: list[int], ground_truth: set[int]) -> np.ndarray:
    len_rank = len(ranklist)
    len_gt = len(ground_truth)
    if len_gt == 0:
        return np.zeros(len_rank, dtype=np.float32)

    # 实际 DCG
    gains = np.array(
        [1.0 / math.log2(idx + 2) if item in ground_truth else 0.0 for idx, item in enumerate(ranklist)],
        dtype=np.float32,
    )
    dcg = np.cumsum(gains)

    # 理想 IDCG
    idcg_len = min(len_gt, len_rank)
    ideal_gains = np.array([1.0 / math.log2(i + 2) for i in range(len_rank)], dtype=np.float32)
    idcg = np.cumsum(ideal_gains)
    idcg[idcg_len:] = idcg[idcg_len - 1]

    return dcg / idcg


# 评测：按用户多正样本方式计算 Precision / Recall / NDCG
@torch.no_grad()
def evaluate_ranking(
    model: LightGCN,
    norm_adj: torch.Tensor,
    eval_pairs: Sequence[Tuple[int, int]],
    all_pos: Dict[int, set],
    num_items: int,
    k: int = 10,
) -> dict[str, float]:
    # 切换到评估模式
    model.eval()

    # 获取最终用户和物品嵌入
    user_emb, item_emb = model.get_user_item_embeddings(norm_adj)

    # 将 (u, i) 列表按用户聚合
    user_test = group_eval_pairs(eval_pairs)

    precisions = []
    recalls = []
    ndcgs = []

    # 逐个用户评测
    for u, gt_items in user_test.items():
        ground_truth = set(gt_items)

        # 对该用户的所有物品打分
        scores = torch.matmul(item_emb, user_emb[u]).detach().cpu().numpy()

        # 将训练集中出现过的正样本物品屏蔽掉，避免参与排序
        # 注意：这里保留 ground_truth 自身，即使它们也在 all_pos 里，仍应作为测试目标参与评测
        train_pos = set(all_pos[u]) - ground_truth
        if len(train_pos) > 0:
            scores[list(train_pos)] = -np.inf

        # 取前 k 个物品索引
        topk_idx = np.argpartition(-scores, kth=min(k, len(scores) - 1))[:k]
        topk_idx = topk_idx[np.argsort(-scores[topk_idx])]
        ranklist = topk_idx.tolist()

        # 计算该用户的 Precision@K、Recall@K、NDCG@K
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

# 评估遗忘强度，这部分保留你原来的定义
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