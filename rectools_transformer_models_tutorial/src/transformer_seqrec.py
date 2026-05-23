from dataclasses import dataclass
import math
import random
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


PAD_TOKEN = 0
MASK_TOKEN = 1
IGNORE_INDEX = -100


@dataclass
class SequenceEncoding:
    """保存 item 序列推荐所需的基础映射。"""

    item_to_token: Dict[str, int]
    token_to_item: Dict[int, str]
    category_to_id: Dict[str, int]
    token_to_category: torch.Tensor
    max_len: int

    @property
    def vocab_size(self) -> int:
        return max(self.token_to_item) + 1

    @property
    def mask_token(self) -> int:
        return MASK_TOKEN


def make_toy_interactions() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """构造一个小型顺序推荐数据集，便于 notebook 离线快速跑通。"""

    user_sequences = {
        "U1": ["I1", "I2", "I5", "I7", "I9", "I12", "I16"],
        "U2": ["I3", "I4", "I8", "I11", "I13", "I17", "I20"],
        "U3": ["I1", "I6", "I10", "I14", "I18", "I22", "I24"],
        "U4": ["I2", "I5", "I9", "I15", "I19", "I21", "I25"],
        "U5": ["I4", "I8", "I11", "I13", "I17", "I20", "I23"],
        "U6": ["I6", "I10", "I14", "I18", "I22", "I24", "I26"],
        "U7": ["I3", "I7", "I12", "I16", "I19", "I21", "I25"],
        "U8": ["I5", "I9", "I15", "I18", "I22", "I24", "I26"],
    }
    categories = ["sports", "finance", "tech", "travel", "health", "movie"]

    item_rows = []
    for idx in range(1, 27):
        item_id = f"I{idx}"
        item_rows.append(
            {
                "item_id": item_id,
                "category": categories[(idx - 1) % len(categories)],
                "title": f"{categories[(idx - 1) % len(categories)]} item {idx}",
            }
        )
    item_features = pd.DataFrame(item_rows)

    rows = []
    timestamp = 0
    for user_id, items in user_sequences.items():
        for item_id in items:
            timestamp += 1
            rows.append({"user_id": user_id, "item_id": item_id, "timestamp": timestamp})
    interactions = pd.DataFrame(rows)
    return interactions, item_features


def build_encoding(item_features: pd.DataFrame, max_len: int) -> SequenceEncoding:
    """把原始 item id 和类别特征映射为模型可用的整数 token。"""

    item_ids = sorted(item_features["item_id"].unique())
    item_to_token = {item_id: idx + 2 for idx, item_id in enumerate(item_ids)}
    token_to_item = {PAD_TOKEN: "<PAD>", MASK_TOKEN: "<MASK>"}
    token_to_item.update({token: item_id for item_id, token in item_to_token.items()})

    categories = sorted(item_features["category"].unique())
    category_to_id = {"<PAD>": 0, "<MASK>": 1}
    category_to_id.update({category: idx + 2 for idx, category in enumerate(categories)})

    token_to_category = torch.zeros(len(token_to_item), dtype=torch.long)
    token_to_category[MASK_TOKEN] = category_to_id["<MASK>"]
    for row in item_features.itertuples(index=False):
        token = item_to_token[row.item_id]
        token_to_category[token] = category_to_id[row.category]

    return SequenceEncoding(
        item_to_token=item_to_token,
        token_to_item=token_to_item,
        category_to_id=category_to_id,
        token_to_category=token_to_category,
        max_len=max_len,
    )


def encode_user_sequences(interactions: pd.DataFrame, encoding: SequenceEncoding) -> Dict[str, List[int]]:
    """按时间排序，把每个用户的交互历史编码成 item token 序列。"""

    ordered = interactions.sort_values(["user_id", "timestamp"])
    sequences = {}
    for user_id, group in ordered.groupby("user_id"):
        sequences[user_id] = [encoding.item_to_token[item_id] for item_id in group["item_id"].tolist()]
    return sequences


def left_pad(values: Sequence[int], max_len: int, pad_value: int = PAD_TOKEN) -> List[int]:
    """只保留最近 max_len 个行为，并在左侧补 PAD。"""

    values = list(values)[-max_len:]
    return [pad_value] * (max_len - len(values)) + values


class SASRecDataset(Dataset):
    """SASRec 训练样本：用当前位置之前的物品预测下一个物品。"""

    def __init__(self, user_sequences: Dict[str, List[int]], max_len: int):
        self.examples = []
        self.max_len = max_len
        for user_id, seq in user_sequences.items():
            if len(seq) < 2:
                continue
            source = left_pad(seq[:-1], max_len)
            target = left_pad(seq[1:], max_len, IGNORE_INDEX)
            self.examples.append((user_id, source, target))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        user_id, source, target = self.examples[index]
        return {
            "user_id": user_id,
            "input_ids": torch.tensor(source, dtype=torch.long),
            "labels": torch.tensor(target, dtype=torch.long),
        }


class BERT4RecDataset(Dataset):
    """BERT4Rec 训练样本：随机遮住历史中的部分物品，再预测被遮住的物品。"""

    def __init__(
        self,
        user_sequences: Dict[str, List[int]],
        max_len: int,
        mask_prob: float = 0.4,
        mask_token: int = MASK_TOKEN,
        seed: int = 42,
    ):
        self.examples = []
        self.max_len = max_len
        rng = random.Random(seed)
        for user_id, seq in user_sequences.items():
            padded = left_pad(seq, max_len)
            input_ids = list(padded)
            labels = [IGNORE_INDEX] * max_len
            valid_positions = [idx for idx, token in enumerate(padded) if token != PAD_TOKEN]
            masked_positions = [idx for idx in valid_positions if rng.random() < mask_prob]
            if valid_positions and not masked_positions:
                masked_positions = [valid_positions[-1]]
            for idx in masked_positions:
                labels[idx] = input_ids[idx]
                input_ids[idx] = mask_token
            self.examples.append((user_id, input_ids, labels))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        user_id, input_ids, labels = self.examples[index]
        return {
            "user_id": user_id,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class SequentialTransformerRec(nn.Module):
    """SASRec / BERT4Rec 共用的轻量 PyTorch Transformer 序列推荐模型。"""

    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        token_to_category: torch.Tensor,
        category_vocab_size: int,
        hidden_dim: int = 48,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_item_features: bool = True,
        causal: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.causal = causal
        self.use_item_features = use_item_features

        self.item_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=PAD_TOKEN)
        self.position_embedding = nn.Embedding(max_len, hidden_dim)
        self.category_embedding = nn.Embedding(category_vocab_size, hidden_dim, padding_idx=PAD_TOKEN)
        self.register_buffer("token_to_category", token_to_category)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.output.weight = self.item_embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        hidden = self.item_embedding(input_ids) + self.position_embedding(positions)
        if self.use_item_features:
            category_ids = self.token_to_category[input_ids]
            hidden = hidden + self.category_embedding(category_ids)

        key_padding_mask = input_ids == PAD_TOKEN
        attn_mask = None
        if self.causal:
            attn_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device), diagonal=1)
        hidden = self.transformer(hidden, mask=attn_mask, src_key_padding_mask=key_padding_mask)
        hidden = self.layer_norm(hidden)
        return self.output(hidden)

    def encode_sequence(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits = self.forward(input_ids)
        valid_lengths = (input_ids != PAD_TOKEN).sum(dim=1).clamp(min=1) - 1
        return logits[torch.arange(input_ids.size(0), device=input_ids.device), valid_lengths]


def full_softmax_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """对所有 item 做交叉熵，忽略 PAD 和未被 mask 的位置。"""

    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=IGNORE_INDEX)


def sampled_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
    num_negatives: int = 5,
) -> torch.Tensor:
    """ sampled BCE：每个正样本只和若干随机负样本比较。"""

    positive_mask = labels != IGNORE_INDEX
    positive_logits = logits[positive_mask]
    positive_labels = labels[positive_mask]
    if positive_labels.numel() == 0:
        return logits.sum() * 0

    negative_ids = torch.randint(2, vocab_size, (positive_labels.numel(), num_negatives), device=logits.device)
    positive_scores = positive_logits.gather(1, positive_labels.unsqueeze(1))
    negative_scores = positive_logits.gather(1, negative_ids)
    scores = torch.cat([positive_scores, negative_scores], dim=1)
    targets = torch.zeros_like(scores)
    targets[:, 0] = 1
    return F.binary_cross_entropy_with_logits(scores, targets)


def bpr_loss(logits: torch.Tensor, labels: torch.Tensor, vocab_size: int, num_negatives: int = 5) -> torch.Tensor:
    """BPR：让正样本分数高于负样本分数。"""

    positive_mask = labels != IGNORE_INDEX
    positive_logits = logits[positive_mask]
    positive_labels = labels[positive_mask]
    if positive_labels.numel() == 0:
        return logits.sum() * 0

    negative_ids = torch.randint(2, vocab_size, (positive_labels.numel(), num_negatives), device=logits.device)
    positive_scores = positive_logits.gather(1, positive_labels.unsqueeze(1))
    negative_scores = positive_logits.gather(1, negative_ids)
    return -F.logsigmoid(positive_scores - negative_scores).mean()


def train_one_model(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    loss_name: str,
    device: torch.device,
    epochs: int = 12,
    num_negatives: int = 5,
) -> List[float]:
    """训练 SASRec 或 BERT4Rec，并返回每轮平均损失。"""

    losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            logits = model(input_ids)
            if loss_name == "softmax":
                loss = full_softmax_loss(logits, labels)
            elif loss_name == "sampled_bce":
                loss = sampled_bce_loss(logits, labels, model.vocab_size, num_negatives=num_negatives)
            elif loss_name == "bpr":
                loss = bpr_loss(logits, labels, model.vocab_size, num_negatives=num_negatives)
            else:
                raise ValueError(f"未知 loss：{loss_name}")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * input_ids.size(0)
            total_count += input_ids.size(0)
        losses.append(total_loss / max(total_count, 1))
    return losses


def leave_one_out_split(user_sequences: Dict[str, List[int]]) -> Tuple[Dict[str, List[int]], Dict[str, int]]:
    """每个用户最后一个 item 做测试目标，其余历史用于训练或推理。"""

    train_sequences = {}
    test_targets = {}
    for user_id, seq in user_sequences.items():
        if len(seq) < 3:
            continue
        train_sequences[user_id] = seq[:-1]
        test_targets[user_id] = seq[-1]
    return train_sequences, test_targets


@torch.no_grad()
def recommend_topk(
    model: SequentialTransformerRec,
    history: Sequence[int],
    max_len: int,
    k: int,
    device: torch.device,
    seen_items: Iterable[int] = (),
    bert_style: bool = False,
) -> List[int]:
    """根据历史行为推荐 Top-K item。"""

    model.eval()
    if bert_style:
        input_ids = left_pad(list(history)[-(max_len - 1) :] + [MASK_TOKEN], max_len)
        score_position = max_len - 1
    else:
        input_ids = left_pad(history, max_len)
        score_position = min(len(history), max_len) - 1
        score_position = max(score_position, 0) + (max_len - min(len(history), max_len))

    tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    logits = model(tensor)[0, score_position].clone()
    logits[:2] = -1e9
    for token in seen_items:
        logits[token] = -1e9
    return torch.topk(logits, k=k).indices.cpu().tolist()


def ranking_metrics(
    model: SequentialTransformerRec,
    train_sequences: Dict[str, List[int]],
    test_targets: Dict[str, int],
    max_len: int,
    k: int,
    device: torch.device,
    bert_style: bool = False,
) -> Dict[str, float]:
    """计算 HitRate@K、NDCG@K、MRR@K。"""

    hits = []
    ndcgs = []
    mrrs = []
    for user_id, history in train_sequences.items():
        target = test_targets[user_id]
        recs = recommend_topk(model, history, max_len, k, device, seen_items=set(history), bert_style=bert_style)
        if target in recs:
            rank = recs.index(target) + 1
            hits.append(1.0)
            ndcgs.append(1.0 / math.log2(rank + 1))
            mrrs.append(1.0 / rank)
        else:
            hits.append(0.0)
            ndcgs.append(0.0)
            mrrs.append(0.0)
    return {
        f"HitRate@{k}": float(np.mean(hits)),
        f"NDCG@{k}": float(np.mean(ndcgs)),
        f"MRR@{k}": float(np.mean(mrrs)),
    }


@torch.no_grad()
def item_to_item_recommendations(
    model: SequentialTransformerRec,
    item_token: int,
    token_to_item: Dict[int, str],
    k: int = 5,
) -> pd.DataFrame:
    """用 item embedding 余弦相似度做 item-to-item 推荐。"""

    embeddings = model.item_embedding.weight.detach().cpu()
    normalized = F.normalize(embeddings, dim=1)
    similarities = normalized @ normalized[item_token]
    similarities[:2] = -1
    similarities[item_token] = -1
    top_tokens = torch.topk(similarities, k=k).indices.tolist()
    return pd.DataFrame(
        {
            "item_token": top_tokens,
            "item_id": [token_to_item[token] for token in top_tokens],
            "cosine_similarity": [float(similarities[token]) for token in top_tokens],
        }
    )

