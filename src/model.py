from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class LightGCN(nn.Module):
    def __init__(self, num_users: int, num_items: int, emb_dim: int = 64, num_layers: int = 3):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.emb_dim = emb_dim
        self.num_layers = num_layers

        self.user_embedding = nn.Embedding(num_users, emb_dim)
        self.item_embedding = nn.Embedding(num_items, emb_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def computer(self, norm_adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        all_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        embs = [all_emb]

        for _ in range(self.num_layers):
            all_emb = torch.sparse.mm(norm_adj, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        final = torch.mean(embs, dim=1)
        users, items = torch.split(final, [self.num_users, self.num_items], dim=0)
        return users, items

    def bpr_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        norm_adj: torch.Tensor,
        reg_weight: float = 1e-4,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        user_emb, item_emb = self.computer(norm_adj)
        u = user_emb[users]
        p = item_emb[pos_items]
        n = item_emb[neg_items]
        pos_scores = torch.sum(u * p, dim=1)
        neg_scores = torch.sum(u * n, dim=1)

        loss = -F.logsigmoid(pos_scores - neg_scores).mean()

        u0 = self.user_embedding(users)
        p0 = self.item_embedding(pos_items)
        n0 = self.item_embedding(neg_items)

        reg = 0.5 * (u0.norm(2).pow(2) + p0.norm(2).pow(2) + n0.norm(2).pow(2)) / users.shape[0]
        total = loss + reg_weight * reg
        stats = {
            'bpr': float(loss.detach().cpu()),
            'reg': float(reg.detach().cpu()),
            'total': float(total.detach().cpu()),
        }

        return total, stats


    @torch.no_grad()
    def get_user_item_embeddings(self, norm_adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        return self.computer(norm_adj)
