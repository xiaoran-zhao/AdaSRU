# 启用延迟类型注解，避免类型提示在定义时立即求值
from __future__ import annotations

# 导入 PyTorch
import torch
# 导入 PyTorch 的神经网络模块
import torch.nn as nn
# 导入常用神经网络函数模块，这里主要用到 logsigmoid
import torch.nn.functional as F


# 定义 LightGCN 模型类，并继承 nn.Module
class LightGCN(nn.Module):
    # 初始化函数
    def __init__(self, num_users: int, num_items: int, emb_dim: int = 64, num_layers: int = 3):
        # 调用父类 nn.Module 的初始化方法
        super().__init__()
        # 用户总数
        self.num_users = num_users
        # 物品总数
        self.num_items = num_items
        # 嵌入向量维度
        self.emb_dim = emb_dim
        # 图卷积传播层数
        self.num_layers = num_layers

        # 定义用户嵌入矩阵，大小为 [num_users, emb_dim]
        self.user_embedding = nn.Embedding(num_users, emb_dim)
        # 定义物品嵌入矩阵，大小为 [num_items, emb_dim]
        self.item_embedding = nn.Embedding(num_items, emb_dim)
        # 使用均值为 0、标准差为 0.1 的正态分布初始化用户嵌入
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        # 使用均值为 0、标准差为 0.1 的正态分布初始化物品嵌入
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    # LightGCN 的前向传播核心过程：根据归一化邻接矩阵计算最终用户和物品表示
    def computer(self, norm_adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 将用户嵌入和物品嵌入在第 0 维拼接，形成统一的节点嵌入矩阵
        # 拼接后前 num_users 行是用户，后 num_items 行是物品
        all_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        # 用列表保存每一层传播得到的嵌入，初始层（0 层）先放进去
        embs = [all_emb]
        # 进行 num_layers 次图传播
        for _ in range(self.num_layers):
            # 稀疏矩阵乘法：norm_adj @ all_emb
            # 表示根据图结构，把邻居信息传播到每个节点
            all_emb = torch.sparse.mm(norm_adj, all_emb)
            # 将当前层得到的嵌入保存下来
            embs.append(all_emb)
        # 将每一层的嵌入堆叠起来
        # 原来 embs 是长度为 num_layers+1 的列表
        # stack 后形状大致为 [num_users + num_items, num_layers + 1, emb_dim]
        embs = torch.stack(embs, dim=1)
        # 对所有层的嵌入求平均，作为最终表示
        # 这是 LightGCN 的核心思想之一：保留各层信息并做均值聚合
        final = torch.mean(embs, dim=1)
        # 再把最终节点嵌入拆分回用户嵌入和物品嵌入
        users, items = torch.split(final, [self.num_users, self.num_items], dim=0)
        # 返回最终的用户表示和物品表示
        return users, items

    # 计算 BPR 损失
    def bpr_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        norm_adj: torch.Tensor,
        reg_weight: float = 1e-4,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # 先通过图传播，得到最终的用户嵌入和物品嵌入
        user_emb, item_emb = self.computer(norm_adj)
        # 取出当前 batch 中这些用户的最终嵌入
        u = user_emb[users]
        # 取出当前 batch 中正样本物品的最终嵌入
        p = item_emb[pos_items]
        # 取出当前 batch 中负样本物品的最终嵌入
        n = item_emb[neg_items]
        # 计算用户与正样本物品的匹配分数，逐元素乘后按维度求和，相当于点积
        pos_scores = torch.sum(u * p, dim=1)
        # 计算用户与负样本物品的匹配分数
        neg_scores = torch.sum(u * n, dim=1)
        # BPR 损失
        # 希望 pos_scores > neg_scores
        # 所以优化目标是最大化 log sigmoid(pos_scores - neg_scores)
        # 前面加负号并求均值，变成最小化损失
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()

        # 下面这部分是 L2 正则项，但注意这里使用的是“初始 embedding 参数”
        # 而不是图传播之后的最终 embedding
        u0 = self.user_embedding(users)
        # 取出 batch 中正样本物品的原始嵌入参数
        p0 = self.item_embedding(pos_items)
        # 取出 batch 中负样本物品的原始嵌入参数
        n0 = self.item_embedding(neg_items)
        # 计算正则项
        # norm(2).pow(2) 表示平方 L2 范数
        # 0.5 是常见写法，便于求导
        # 再除以 batch 大小做平均
        reg = 0.5 * (u0.norm(2).pow(2) + p0.norm(2).pow(2) + n0.norm(2).pow(2)) / users.shape[0]
        # 总损失 = BPR 损失 + 正则权重 * 正则项
        total = loss + reg_weight * reg
        # 记录一些训练过程中的统计信息，便于日志打印
        stats = {
            # 纯 BPR 损失
            'bpr': float(loss.detach().cpu()),
            # 正则项
            'reg': float(reg.detach().cpu()),
            # 总损失
            'total': float(total.detach().cpu()),
        }
        # 返回总损失，以及统计信息字典
        return total, stats

    # 不计算梯度，表示这里只是取 embedding 做评估
    @torch.no_grad()
    def get_user_item_embeddings(self, norm_adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 直接调用 computer，返回最终用户和物品嵌入
        return self.computer(norm_adj)