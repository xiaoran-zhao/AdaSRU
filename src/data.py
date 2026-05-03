from __future__ import annotations

import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from .utils import ensure_dir, save_json


ML1M_URL = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'


@dataclass
class SplitData:
    num_users: int
    num_items: int

    # 完整训练集：retain + forget，用于训练原始模型
    full_train_user_items: Dict[int, List[int]]

    # 保留训练集：去掉 forget，用于 retrain baseline 或遗忘后模型
    retain_user_items: Dict[int, List[int]]

    val_pairs: List[Tuple[int, int]]
    test_pairs: List[Tuple[int, int]]
    forget_pairs: List[Tuple[int, int]]
    retain_pairs: List[Tuple[int, int]]
    all_pos: Dict[int, set]


def download_ml1m(data_root: str | Path) -> Path:
    data_root = ensure_dir(data_root)
    out_dir = data_root / 'ml-1m'

    if (out_dir / 'ratings.dat').exists():
        return out_dir

    zip_path = data_root / 'ml-1m.zip'

    if not zip_path.exists():
        print(f'Downloading MovieLens-1M from {ML1M_URL} ...')
        urlretrieve(ML1M_URL, zip_path)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(data_root)

    return out_dir


def parse_ratings(data_dir: str | Path) -> pd.DataFrame:
    ratings_path = Path(data_dir) / 'ratings.dat'

    df = pd.read_csv(
        ratings_path,
        sep='::',
        engine='python',
        names=['user_id', 'movie_id', 'rating', 'timestamp'],
    )

    return df


def remap_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, int], dict[int, int]]:
    user_ids = sorted(df['user_id'].unique().tolist())
    item_ids = sorted(df['movie_id'].unique().tolist())

    user_map = {u: i for i, u in enumerate(user_ids)}
    item_map = {i: j for j, i in enumerate(item_ids)}

    df = df.copy()
    df['uid'] = df['user_id'].map(user_map)
    df['iid'] = df['movie_id'].map(item_map)

    return df, user_map, item_map


def build_random_user_ratio_split(
    data_root: str | Path,
    forget_ratio: float = 0.05,
    min_user_train_interactions: int = 5,
    seed: int = 42,
    positive_threshold: int = 4,
) -> SplitData:
    random.seed(seed)
    np.random.seed(seed)

    data_dir = download_ml1m(data_root)
    df = parse_ratings(data_dir)

    # 只保留正反馈交互
    df = df[df['rating'] >= positive_threshold].copy()
    df, _, _ = remap_ids(df)

    full_train_user_items: Dict[int, List[int]] = {}
    val_pairs: List[Tuple[int, int]] = []
    test_pairs: List[Tuple[int, int]] = []
    all_pos: Dict[int, set] = {}
    full_train_pairs: List[Tuple[int, int]] = []

    num_users = int(df['uid'].max()) + 1
    num_items = int(df['iid'].max()) + 1

    for uid, group in df.groupby('uid'):
        items = group['iid'].tolist()

        if len(items) < min_user_train_interactions + 2:
            continue

        items = items.copy()
        random.shuffle(items)

        n = len(items)

        train_end = int(n * 0.8)
        val_end = int(n * 0.9)

        if train_end < min_user_train_interactions:
            train_end = min_user_train_interactions

        if val_end <= train_end:
            val_end = train_end + 1

        if val_end >= n:
            val_end = n - 1

        train = items[:train_end]
        val = items[train_end:val_end]
        test = items[val_end:]

        if len(val) == 0:
            val = [train.pop()]

        if len(test) == 0:
            if len(train) > min_user_train_interactions:
                test = [train.pop()]
            else:
                test = [val.pop()]
                val = [train.pop()]

        if len(train) < min_user_train_interactions:
            continue

        uid = int(uid)

        full_train_user_items[uid] = [int(i) for i in train]
        val_pairs.extend((uid, int(i)) for i in val)
        test_pairs.extend((uid, int(i)) for i in test)
        all_pos[uid] = set(int(i) for i in items)
        full_train_pairs.extend((uid, int(i)) for i in train)

    # 从完整训练集中抽取 forget_pairs
    forget_pairs: List[Tuple[int, int]] = []

    for uid, item in full_train_pairs:
        if len(full_train_user_items[uid]) <= min_user_train_interactions:
            continue

        if random.random() < forget_ratio:
            forget_pairs.append((uid, item))

    # 按用户组织 forget 交互
    forget_by_user: Dict[int, set] = {}

    for uid, iid in forget_pairs:
        forget_by_user.setdefault(uid, set()).add(iid)

    # retain_user_items = full_train_user_items - forget_pairs
    retain_user_items: Dict[int, List[int]] = {
        u: list(items)
        for u, items in full_train_user_items.items()
    }

    for uid, forgotten in forget_by_user.items():
        kept = [iid for iid in retain_user_items[uid] if iid not in forgotten]

        if len(kept) < min_user_train_interactions:
            needed = min_user_train_interactions - len(kept)
            give_back = list(forgotten)[:needed]

            kept.extend(give_back)
            forget_by_user[uid] = forgotten.difference(give_back)

        retain_user_items[uid] = kept

    # 根据修正后的 forget_by_user 重新生成 forget_pairs
    forget_pairs = [
        (u, i)
        for u, items in forget_by_user.items()
        for i in items
    ]

    retain_pairs = [
        (u, i)
        for u, items in retain_user_items.items()
        for i in items
    ]

    return SplitData(
        num_users=num_users,
        num_items=num_items,
        full_train_user_items=full_train_user_items,
        retain_user_items=retain_user_items,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        forget_pairs=forget_pairs,
        retain_pairs=retain_pairs,
        all_pos=all_pos,
    )


def save_split(split: SplitData, out_dir: str | Path) -> None:
    out_dir = ensure_dir(out_dir)

    obj = {
        'num_users': split.num_users,
        'num_items': split.num_items,
        'full_train_user_items': {
            str(k): v for k, v in split.full_train_user_items.items()
        },
        'retain_user_items': {
            str(k): v for k, v in split.retain_user_items.items()
        },
        'val_pairs': split.val_pairs,
        'test_pairs': split.test_pairs,
        'forget_pairs': split.forget_pairs,
        'retain_pairs': split.retain_pairs,
        'all_pos': {
            str(k): sorted(list(v)) for k, v in split.all_pos.items()
        },
    }

    save_json(obj, Path(out_dir) / 'split.json')


def load_split(path: str | Path) -> SplitData:
    import json

    with open(path, 'r', encoding='utf-8') as f:
        obj = json.load(f)

    return SplitData(
        num_users=obj['num_users'],
        num_items=obj['num_items'],
        full_train_user_items={
            int(k): list(v) for k, v in obj['full_train_user_items'].items()
        },
        retain_user_items={
            int(k): list(v) for k, v in obj['retain_user_items'].items()
        },
        val_pairs=[tuple(x) for x in obj['val_pairs']],
        test_pairs=[tuple(x) for x in obj['test_pairs']],
        forget_pairs=[tuple(x) for x in obj['forget_pairs']],
        retain_pairs=[tuple(x) for x in obj['retain_pairs']],
        all_pos={int(k): set(v) for k, v in obj['all_pos'].items()},
    )


def build_normalized_adj(
    num_users: int,
    num_items: int,
    user_items: Dict[int, List[int]],
) -> torch.Tensor:
    rows = []
    cols = []
    vals = []

    for u, items in user_items.items():
        for i in items:
            rows.append(u)
            cols.append(num_users + i)
            vals.append(1.0)

            rows.append(num_users + i)
            cols.append(u)
            vals.append(1.0)

    size = num_users + num_items

    mat = sp.coo_matrix(
        (vals, (rows, cols)),
        shape=(size, size),
        dtype=np.float32,
    )

    rowsum = np.array(mat.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(rowsum + 1e-12, -0.5)
    d_mat = sp.diags(d_inv_sqrt)

    norm = d_mat @ mat @ d_mat
    norm = norm.tocoo()

    indices = torch.tensor(
        np.vstack([norm.row, norm.col]),
        dtype=torch.long,
    )

    values = torch.tensor(norm.data, dtype=torch.float32)

    return torch.sparse_coo_tensor(
        indices,
        values,
        torch.Size(norm.shape),
    ).coalesce()


class BPRDataset:
    def __init__(
        self,
        user_items: Dict[int, List[int]],
        num_items: int,
        all_pos: Dict[int, set],
    ):
        self.user_items = user_items
        self.num_items = num_items
        self.all_pos = all_pos
        self.users = [u for u, items in user_items.items() if len(items) > 0]

    def sample(
        self,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        users = np.random.choice(self.users, size=batch_size, replace=True)

        pos_items = []
        neg_items = []

        for u in users:
            u = int(u)

            pos = random.choice(self.user_items[u])

            while True:
                neg = random.randint(0, self.num_items - 1)
                if neg not in self.all_pos[u]:
                    break

            pos_items.append(pos)
            neg_items.append(neg)

        return (
            torch.tensor(users, dtype=torch.long),
            torch.tensor(pos_items, dtype=torch.long),
            torch.tensor(neg_items, dtype=torch.long),
        )
