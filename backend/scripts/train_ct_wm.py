"""Train an action-conditioned Transformer outcome surrogate on KuaiRand-Pure.

Design SSOT: ``docs/level-3-implementation/research/oranbench-neurips-2026/ct-wm-training-plan.md``.
Internal name in the implementation path: CT-WM (Causal Transformer World Model).
**Do not** introduce the "CT-WM" string into the paper text — the paper uses
"action-conditioned Transformer outcome model" / "outcome surrogate" only.

Role boundary (see ct-wm-training-plan.md §1):

- supervised outcome surrogate, *not* a Pearl counterfactual oracle,
- intervention/action indicators are supervised conditioning variables.

Architecture (ct-wm-training-plan.md §3, parameter target <= 10M):

- decoder-only Transformer with causal self-attention (SASRec-style),
- embedding dim 128, 2 layers, 4 heads, max history 50, dropout 0.1,
- 3 binary heads (click / like / long_view) + 1 watch-time regression head on
  ``log1p(reward_watch_time)``.

Data: ``data/kuairand/processed/clicks.parquet`` produced by
``backend/scripts/prepare_kuairand.py``. Split rule is **temporal** on the
sorted ``timestamp`` column — no random row split.

Diagnostic scores (evaluated by ``eval_ct_wm.py``, schema ``e7-v0.2``):

- per-head binary AUC (click, like, long_view),
- watch-time Spearman, MAE, RMSE on ``log1p(reward_watch_time)``,
- per-seed values + cross-seed mean ± std.

Earlier 2026-05-02 wording specified pre-set sanity thresholds (binary AUC
>= 0.65 and watch-time Spearman >= 0.4 from ct-wm-training-plan.md §6); both
were retired 2026-05-07 as engineering sanity values without a published
literature anchor. ``eval_ct_wm.py`` now reports diagnostic scores only.

------------------------------------------------------------------------------
autodl quickstart (no GPU on the dev workstation; train remotely)
------------------------------------------------------------------------------

1. Pick instance: RTX 4090 / A10 single GPU, PyTorch 2.x + CUDA 12.x image.

2. Install deps::

    pip install --upgrade pip
    pip install torch transformers pandas pyarrow scikit-learn scipy

3. Sync code + data to the remote workstation (replace ``<repo>`` with the
   directory you cloned this artifact into on the remote host)::

    scp backend/scripts/train_ct_wm.py backend/scripts/eval_ct_wm.py \\
        backend/causaltwin/paths.py \\
        <user>@<remote_host>:~/<repo>/backend/scripts/
    scp data/kuairand/processed/clicks.parquet \\
        <user>@<remote_host>:~/<repo>/data/kuairand/processed/

4. Train all three seeds::

    cd ~/<repo>
    for seed in 42 137 256; do
      python3 backend/scripts/train_ct_wm.py \\
        --data data/kuairand/processed/clicks.parquet \\
        --out  models/ct_wm/v0.1/seed_${seed} \\
        --max-history 50 --d-model 128 --layers 2 --heads 4 \\
        --epochs 2 --batch-size 512 --seed ${seed}
    done

5. Evaluate + write metrics.json::

    python3 backend/scripts/eval_ct_wm.py \\
        --runs-dir models/ct_wm/v0.1 \\
        --data data/kuairand/processed/clicks.parquet \\
        --out  models/ct_wm/v0.1/metrics.json

Estimated wall time per seed on 4090: 0.5-1.5 h (budget cap 8 GPU-h total).

Outputs per seed run, written under ``--out``:

- ``checkpoint.pt``         best epoch weights (selected by val total loss),
- ``training_log.jsonl``    per-step / per-epoch loss + metric snapshots,
- ``run_config.json``       CLI args + dataset stats + final epoch summary.

------------------------------------------------------------------------------
Note on the "transformers" dep
------------------------------------------------------------------------------

The Phase B P2 prompt mentions "PyTorch + transformers". This script implements
the Transformer in pure PyTorch (``nn.TransformerEncoder`` with a causal mask)
rather than instantiating a Hugging Face model class, because:

- the SSOT architecture is a small custom decoder-only stack with non-standard
  multi-head outputs (3 binary + 1 regression),
- avoiding ``AutoModel`` keeps the parameter count under the 10M cap from
  ct-wm-training-plan.md §3 without surgery on a pretrained backbone,
- ``transformers`` is still installed in the autodl bootstrap above so any
  follow-up sanity comparison against an HF SASRec / encoder baseline can
  reuse the same env.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ``torch`` and friends are imported inside ``main`` so the module can still be
# imported on the dev workstation (no GPU, no torch installed) for static
# inspection.

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from causaltwin.paths import KUAIRAND_PROCESSED, MODEL_DIR  # noqa: E402

DEFAULT_DATA = KUAIRAND_PROCESSED / "clicks.parquet"
DEFAULT_OUT = MODEL_DIR / "ct_wm" / "v0.4" / "seed_42"

PAD_IDX = 0  # reserved id 0 for padding; real action ids are shifted by +1

BINARY_HEADS = ("click", "like", "long_view")
WATCH_HEAD = "watch_time"

# v0.4 meta features (numerical + binary only; categorical skipped for simplicity)
USER_NUMERIC_COLS = ["follow_user_num", "fans_user_num", "friend_user_num", "register_days"]
USER_BINARY_COLS = ["is_lowactive_period", "is_live_streamer", "is_video_author"]
USER_ONEHOT_COLS = [f"onehot_feat{i}" for i in range(18)]
USER_FEATURE_COLS = USER_NUMERIC_COLS + USER_BINARY_COLS + USER_ONEHOT_COLS
USER_FEATURE_DIM = len(USER_FEATURE_COLS)  # 4 + 3 + 18 = 25

VIDEO_NUMERIC_COLS = ["video_duration", "server_width", "server_height"]
VIDEO_BINARY_COLS = ["visible_status"]
VIDEO_FEATURE_COLS = VIDEO_NUMERIC_COLS + VIDEO_BINARY_COLS
VIDEO_FEATURE_DIM = len(VIDEO_FEATURE_COLS)  # 3 + 1 = 4


@dataclass
class TrainConfig:
    data: str
    out: str
    max_history: int
    d_model: int
    layers: int
    heads: int
    dropout: float
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    warmup_steps: int
    val_frac: float
    early_stop_patience: int
    seed: int
    max_users: int
    max_items: int
    log_every: int
    device: str
    pos_weight_balance: bool = False
    pos_weight_mode: str = "sqrt"  # "linear" | "sqrt" | "cap"
    pos_weight_cap: float = 10.0
    watch_time_loss: str = "mse"  # "mse" | "huber" | "ranknet" | "mixed"
    huber_delta: float = 1.0
    rank_alpha: float = 0.5  # weight on RankNet vs Huber when watch_time_loss="mixed"
    user_features: str = ""  # path to user_features.parquet; empty = no user meta tower
    video_features: str = ""  # path to video_features.parquet; empty = no video meta tower
    max_rows_per_user: int = 0  # v0.5+: cap each user's most recent N rows; 0 = all

    def to_json(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Meta features (v0.4+)
# ---------------------------------------------------------------------------


def load_meta_features(
    user_features_path: str,
    video_features_path: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict] | None:
    """Load user/video features parquet, normalize, return id→vec dicts + stats.

    - Numerical cols: z-score on the feature parquet itself (one-shot, before train).
    - Binary cols: cast to float32, no normalization.
    - Missing values: filled with 0 (after normalization, equivalent to mean).
    Returns ``None`` when either path is empty.
    """
    if not user_features_path or not video_features_path:
        return None

    udf = pd.read_parquet(user_features_path)
    udf["user_id"] = udf["user_id"].astype(str)
    user_mat = np.zeros((len(udf), USER_FEATURE_DIM), dtype=np.float32)
    user_stats = {}
    for i, col in enumerate(USER_FEATURE_COLS):
        if col not in udf.columns:
            user_stats[col] = {"mean": 0.0, "std": 1.0, "missing_col": True}
            continue
        x = pd.to_numeric(udf[col], errors="coerce").to_numpy(dtype=np.float64)
        if col in USER_BINARY_COLS:
            user_mat[:, i] = np.nan_to_num(x, nan=0.0).astype(np.float32)
            user_stats[col] = {"mean": 0.0, "std": 1.0, "binary": True}
        else:
            mean = float(np.nanmean(x)) if np.isfinite(x).any() else 0.0
            std = float(np.nanstd(x)) if np.isfinite(x).any() else 1.0
            std = std if std > 1e-6 else 1.0
            user_mat[:, i] = np.nan_to_num((x - mean) / std, nan=0.0).astype(np.float32)
            user_stats[col] = {"mean": mean, "std": std}
    user_dict = {uid: user_mat[i] for i, uid in enumerate(udf["user_id"].tolist())}

    vdf = pd.read_parquet(video_features_path)
    vdf["video_id"] = vdf["video_id"].astype(str)
    video_mat = np.zeros((len(vdf), VIDEO_FEATURE_DIM), dtype=np.float32)
    video_stats = {}
    for i, col in enumerate(VIDEO_FEATURE_COLS):
        if col not in vdf.columns:
            video_stats[col] = {"mean": 0.0, "std": 1.0, "missing_col": True}
            continue
        x = pd.to_numeric(vdf[col], errors="coerce").to_numpy(dtype=np.float64)
        if col in VIDEO_BINARY_COLS:
            video_mat[:, i] = np.nan_to_num(x, nan=0.0).astype(np.float32)
            video_stats[col] = {"mean": 0.0, "std": 1.0, "binary": True}
        else:
            mean = float(np.nanmean(x)) if np.isfinite(x).any() else 0.0
            std = float(np.nanstd(x)) if np.isfinite(x).any() else 1.0
            std = std if std > 1e-6 else 1.0
            video_mat[:, i] = np.nan_to_num((x - mean) / std, nan=0.0).astype(np.float32)
            video_stats[col] = {"mean": mean, "std": std}
    video_dict = {vid: video_mat[i] for i, vid in enumerate(vdf["video_id"].tolist())}

    print(f"[ct-wm-train] user_features: {len(user_dict):,} users × {USER_FEATURE_DIM} dims (z-scored)")
    print(f"[ct-wm-train] video_features: {len(video_dict):,} videos × {VIDEO_FEATURE_DIM} dims (z-scored)")
    return user_dict, video_dict, {"user_stats": user_stats, "video_stats": video_stats}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def build_user_sequences(
    df: pd.DataFrame,
    max_history: int,
    max_users: int = 0,
    max_rows_per_user: int = 0,
    user_features_dict: dict[str, np.ndarray] | None = None,
    video_features_dict: dict[str, np.ndarray] | None = None,
) -> list[dict]:
    """Group rows by ``user_id`` in temporal order; emit sliding windows.

    Each emitted sample is one supervision target step ``t`` together with the
    preceding ``max_history`` action ids and the rewards at step ``t``. When
    ``user_features_dict`` / ``video_features_dict`` are provided, each sample
    additionally carries a ``user_feat`` and ``video_feat`` vector for the
    target-step user / video (v0.4 meta tower).
    """
    df = df.sort_values(["user_id", "timestamp"], kind="stable")
    samples: list[dict] = []
    n_users = 0
    use_meta = user_features_dict is not None and video_features_dict is not None
    user_zero = np.zeros(USER_FEATURE_DIM, dtype=np.float32) if use_meta else None
    video_zero = np.zeros(VIDEO_FEATURE_DIM, dtype=np.float32) if use_meta else None
    for uid, group in df.groupby("user_id", sort=False):
        if max_users and n_users >= max_users:
            break
        n_users += 1
        actions = group["action_id_int"].to_numpy()
        item_ids = group["item_id"].astype(str).tolist() if use_meta else None
        rc = group["reward_click"].to_numpy()
        rl = group["reward_like"].to_numpy()
        rlv = group["reward_long_view"].to_numpy()
        rwt = group["log1p_watch_time"].to_numpy()
        ts_arr = group["timestamp"].to_numpy()
        if max_rows_per_user > 0 and len(actions) > max_rows_per_user:
            actions = actions[-max_rows_per_user:]
            rc = rc[-max_rows_per_user:]
            rl = rl[-max_rows_per_user:]
            rlv = rlv[-max_rows_per_user:]
            rwt = rwt[-max_rows_per_user:]
            ts_arr = ts_arr[-max_rows_per_user:]
            if use_meta:
                item_ids = item_ids[-max_rows_per_user:]
        user_feat = user_features_dict.get(str(uid), user_zero) if use_meta else None
        n = len(actions)
        for t in range(n):
            start = max(0, t - max_history)
            history = actions[start:t]  # may be empty for t == 0
            sample = {
                "history": history.astype(np.int64),
                "action": int(actions[t]),
                "rc": float(rc[t]),
                "rl": float(rl[t]),
                "rlv": float(rlv[t]),
                "rwt": float(rwt[t]),
                "ts": int(ts_arr[t]),
            }
            if use_meta:
                sample["user_feat"] = user_feat
                sample["video_feat"] = video_features_dict.get(item_ids[t], video_zero)
            samples.append(sample)
    return samples


def collate(batch: list[dict], max_history: int, pad_idx: int = PAD_IDX):
    import torch

    bsz = len(batch)
    hist = torch.full((bsz, max_history), pad_idx, dtype=torch.long)
    mask = torch.zeros((bsz, max_history), dtype=torch.bool)
    for i, sample in enumerate(batch):
        h = sample["history"]
        L = min(len(h), max_history)
        if L > 0:
            hist[i, max_history - L :] = torch.from_numpy(h[-L:])
            mask[i, max_history - L :] = True
    actions = torch.tensor([s["action"] for s in batch], dtype=torch.long)
    rc = torch.tensor([s["rc"] for s in batch], dtype=torch.float32)
    rl = torch.tensor([s["rl"] for s in batch], dtype=torch.float32)
    rlv = torch.tensor([s["rlv"] for s in batch], dtype=torch.float32)
    rwt = torch.tensor([s["rwt"] for s in batch], dtype=torch.float32)
    if "user_feat" in batch[0]:
        user_feat = torch.from_numpy(np.stack([s["user_feat"] for s in batch], axis=0))
        video_feat = torch.from_numpy(np.stack([s["video_feat"] for s in batch], axis=0))
        return hist, mask, actions, rc, rl, rlv, rwt, user_feat, video_feat
    return hist, mask, actions, rc, rl, rlv, rwt


def temporal_split(samples: list[dict], val_frac: float, df: pd.DataFrame | None = None):
    """Sort samples by the **target step** timestamp and tail-split.

    Uses each sample's own ``ts`` field (populated by ``build_user_sequences``)
    so it works with row-cap subsets. The ``df`` parameter is kept for API
    backward-compat but no longer required.
    """
    ts = np.array([s["ts"] for s in samples], dtype=np.int64)
    order = np.argsort(ts, kind="stable")
    cutoff = int(len(order) * (1.0 - val_frac))
    train_idx = order[:cutoff]
    val_idx = order[cutoff:]
    train = [samples[i] for i in train_idx]
    val = [samples[i] for i in val_idx]
    return train, val


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def build_model(n_actions: int, cfg: TrainConfig):
    import torch
    import torch.nn as nn

    use_meta = bool(cfg.user_features) and bool(cfg.video_features)

    class ActionConditionedTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.action_emb = nn.Embedding(
                num_embeddings=n_actions + 1,  # +1 for PAD
                embedding_dim=cfg.d_model,
                padding_idx=PAD_IDX,
            )
            self.pos_emb = nn.Embedding(cfg.max_history + 1, cfg.d_model)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.heads,
                dim_feedforward=4 * cfg.d_model,
                dropout=cfg.dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.layers)
            # Heads consume [history_summary || action_emb_at_t].
            head_in = cfg.d_model * 2
            self.head_click = nn.Linear(head_in, 1)
            self.head_like = nn.Linear(head_in, 1)
            self.head_long_view = nn.Linear(head_in, 1)
            self.head_watch = nn.Linear(head_in, 1)
            self.dropout = nn.Dropout(cfg.dropout)
            self.use_meta = use_meta
            if use_meta:
                # User meta tower: USER_FEATURE_DIM → d_model (additive on history summary).
                self.user_proj = nn.Sequential(
                    nn.Linear(USER_FEATURE_DIM, cfg.d_model),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(cfg.d_model, cfg.d_model),
                )
                # Video meta tower: VIDEO_FEATURE_DIM → d_model (additive on action emb).
                self.video_proj = nn.Sequential(
                    nn.Linear(VIDEO_FEATURE_DIM, cfg.d_model),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(cfg.d_model, cfg.d_model),
                )

        def forward(self, hist, mask, action, user_feat=None, video_feat=None):
            B, L = hist.shape
            pos = torch.arange(L, device=hist.device).unsqueeze(0).expand(B, L)
            x = self.action_emb(hist) + self.pos_emb(pos)
            # Causal mask only (upper-triangular True = masked). SASRec-style:
            # we deliberately do NOT pass ``src_key_padding_mask``. With both
            # causal + padding masks active, queries at pad positions end up with
            # zero valid keys (causal blocks later positions, padding blocks
            # earlier ones) → softmax(-inf, ..., -inf) = NaN. PyTorch's attention
            # then propagates that NaN into later layers via ``0 * NaN = NaN``
            # in the weighted-value sum, contaminating even the rightmost real
            # position's output. Pad action embeddings are zero (padding_idx) and
            # add only the positional embedding — small constants the model
            # learns to discount.
            causal = torch.triu(
                torch.ones(L, L, device=hist.device, dtype=torch.bool), diagonal=1
            )
            h = self.encoder(x, mask=causal)
            # Left-padded collate: real history occupies the rightmost L slots,
            # so the rightmost position is always the last real step when any
            # history exists.
            history_summary = h[:, -1, :]
            # Where mask is fully empty (cold-start step), zero out the summary.
            empty = (~mask.any(dim=1)).unsqueeze(-1)
            history_summary = torch.where(empty, torch.zeros_like(history_summary), history_summary)
            a_emb = self.action_emb(action)
            if self.use_meta and user_feat is not None and video_feat is not None:
                # Additive injection: meta projections residual onto attention summary
                # and action embedding. Keeps the feature flag a no-op when meta
                # features are unavailable.
                history_summary = history_summary + self.user_proj(user_feat)
                a_emb = a_emb + self.video_proj(video_feat)
            joined = self.dropout(torch.cat([history_summary, a_emb], dim=-1))
            return {
                "click": self.head_click(joined).squeeze(-1),
                "like": self.head_like(joined).squeeze(-1),
                "long_view": self.head_long_view(joined).squeeze(-1),
                "watch_time": self.head_watch(joined).squeeze(-1),
            }

    return ActionConditionedTransformer()


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return float(step) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _ranknet_pairwise_loss(pred, target, eps: float = 1e-6):
    """RankNet pairwise BCE: directly optimizes pairwise concordance (Spearman-friendly).

    Sub-samples up to ``max_pairs`` pairs per batch when batch is large, to keep memory
    O(B) instead of O(B^2). For the default training batch=512 here we keep all pairs
    where target_i != target_j (typically dense — watch_time is rarely tied).

    Returns 0 if no informative pairs exist (all targets tied within batch).
    """
    import torch
    import torch.nn.functional as F
    diff_pred = pred.unsqueeze(0) - pred.unsqueeze(1)        # [B, B]
    diff_t = target.unsqueeze(0) - target.unsqueeze(1)       # [B, B]
    sign = (diff_t > eps).float() - (diff_t < -eps).float()  # +1 / -1 / 0
    informative = sign.abs() > 0
    if not informative.any():
        return pred.new_tensor(0.0)
    diff_pred_sel = diff_pred[informative]
    sign_sel = sign[informative]
    target_label = (sign_sel > 0).float()  # 1 if pred_i > pred_j should hold
    return F.binary_cross_entropy_with_logits(diff_pred_sel, target_label)


def run_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    device,
    train: bool,
    log_path: Path | None,
    epoch: int,
    log_every: int,
    pos_weights=None,  # dict[head -> tensor] or None
    watch_time_loss: str = "mse",
    huber_delta: float = 1.0,
    rank_alpha: float = 0.5,
):
    import torch
    import torch.nn.functional as F

    model.train() if train else model.eval()
    bce = F.binary_cross_entropy_with_logits
    if watch_time_loss == "huber":
        def wt_loss(pred, target):
            return F.huber_loss(pred, target, delta=huber_delta)
    elif watch_time_loss == "ranknet":
        def wt_loss(pred, target):
            return _ranknet_pairwise_loss(pred, target)
    elif watch_time_loss == "mixed":
        def wt_loss(pred, target):
            h = F.huber_loss(pred, target, delta=huber_delta)
            r = _ranknet_pairwise_loss(pred, target)
            return rank_alpha * r + (1 - rank_alpha) * h
    else:
        wt_loss = F.mse_loss
    pw = pos_weights or {}
    totals = {"loss": 0.0, "click_loss": 0.0, "like_loss": 0.0, "lv_loss": 0.0, "wt_loss": 0.0, "n": 0}

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for step, batch in enumerate(loader):
            tensors = [b.to(device) for b in batch]
            if len(tensors) == 9:
                hist, mask, actions, rc, rl, rlv, rwt, user_feat, video_feat = tensors
                out = model(hist, mask, actions, user_feat=user_feat, video_feat=video_feat)
            else:
                hist, mask, actions, rc, rl, rlv, rwt = tensors
                out = model(hist, mask, actions)
            l_c = bce(out["click"], rc, pos_weight=pw.get("click"))
            l_l = bce(out["like"], rl, pos_weight=pw.get("like"))
            l_v = bce(out["long_view"], rlv, pos_weight=pw.get("long_view"))
            l_w = wt_loss(out["watch_time"], rwt)
            loss = l_c + l_l + l_v + 0.5 * l_w
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            bsz = rc.size(0)
            totals["n"] += bsz
            totals["loss"] += loss.item() * bsz
            totals["click_loss"] += l_c.item() * bsz
            totals["like_loss"] += l_l.item() * bsz
            totals["lv_loss"] += l_v.item() * bsz
            totals["wt_loss"] += l_w.item() * bsz

            if train and log_path is not None and (step + 1) % log_every == 0:
                lr_now = scheduler.get_last_lr()[0] if scheduler else 0.0
                with log_path.open("a") as f:
                    json.dump(
                        {
                            "phase": "train",
                            "epoch": epoch,
                            "step": step + 1,
                            "loss": loss.item(),
                            "click_loss": l_c.item(),
                            "like_loss": l_l.item(),
                            "lv_loss": l_v.item(),
                            "wt_loss": l_w.item(),
                            "lr": lr_now,
                        },
                        f,
                    )
                    f.write("\n")

    n = max(1, totals["n"])
    summary = {k: v / n if k != "n" else v for k, v in totals.items()}
    summary["phase"] = "train" if train else "val"
    summary["epoch"] = epoch
    if log_path is not None:
        with log_path.open("a") as f:
            json.dump(summary, f)
            f.write("\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-history", type=int, default=50)
    parser.add_argument("--d-model", type=int, default=512,
                        help="v0.2 default per D-023 (was 128 in v0.1).")
    parser.add_argument("--layers", type=int, default=6,
                        help="v0.2 default per D-023 (was 2 in v0.1).")
    parser.add_argument("--heads", type=int, default=8,
                        help="v0.2 default per D-023 (was 4 in v0.1).")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20,
                        help="v0.2 default per D-023 (was 2 in v0.1).")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="v0.2 attempt-2 default per D-023 follow-up (was 1e-3 in attempt-1; "
                             "lr=1e-3 plateaued on 23M params with click_loss stuck at 1.1).")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=4000,
                        help="v0.2 attempt-2 default per D-023 follow-up (was 1000 in attempt-1; "
                             "1000 was <0.5 epoch on full data, lr ramped before model warmed).")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--early-stop-patience", type=int, default=3,
                        help="v0.2 default per D-023 (was 1 in v0.1).")
    parser.add_argument("--seed", type=int, choices=(42, 137, 256), default=42,
                        help="Fixed seed list per ct-wm-training-plan.md §6 / paper protocol.")
    parser.add_argument("--max-users", type=int, default=0,
                        help="Optional smoke-test cap on distinct users. 0 = all.")
    parser.add_argument("--max-items", type=int, default=0,
                        help="Optional smoke-test cap on action-id vocab. 0 = all.")
    parser.add_argument("--max-rows-per-user", type=int, default=0,
                        help="v0.5 Stage 2 RAM control: cap each user's most recent "
                             "N click rows. 0 = all. KuaiRand-27K full has ~12K rows/user; "
                             "cap=500 produces ~14M samples (≈14GB list) vs uncapped 323M.")
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--device", type=str, default="auto",
                        help="'auto' picks cuda when available else cpu.")
    parser.add_argument("--pos-weight-balance", action="store_true",
                        help="Apply BCE pos_weight to click/like/long_view heads. "
                             "Treats v0.1 like-AUC ≈ 0.51 random-floor on sparse positives. Per D-023.")
    parser.add_argument("--pos-weight-mode", choices=("linear", "sqrt", "cap"), default="sqrt",
                        help="How to derive pos_weight from class ratio. "
                             "linear=N_neg/N_pos (full balance, can be >100 on rare classes); "
                             "sqrt=sqrt(N_neg/N_pos) (smoothed, default per D-023 attempt-2); "
                             "cap=min(N_neg/N_pos, --pos-weight-cap).")
    parser.add_argument("--pos-weight-cap", type=float, default=10.0,
                        help="Upper cap when --pos-weight-mode=cap.")
    parser.add_argument("--watch-time-loss", choices=("mse", "huber", "ranknet", "mixed"), default="huber",
                        help="Watch-time head loss. v0.2 default Huber (delta=1.0). "
                             "ranknet=RankNet pairwise BCE (directly optimizes pairwise concordance, "
                             "Spearman-friendly; introduced v0.3 to attack Spearman ceiling). "
                             "mixed=rank_alpha*RankNet + (1-rank_alpha)*Huber. Per D-023.")
    parser.add_argument("--huber-delta", type=float, default=1.0,
                        help="Huber loss delta when --watch-time-loss=huber.")
    parser.add_argument("--rank-alpha", type=float, default=0.5,
                        help="Weight on RankNet term when --watch-time-loss=mixed.")
    parser.add_argument("--user-features", type=str, default="",
                        help="Path to user_features.parquet (v0.4+ meta tower). "
                             "Empty = no user meta tower (v0.1-v0.3 behavior).")
    parser.add_argument("--video-features", type=str, default="",
                        help="Path to video_features.parquet (v0.4+ meta tower). "
                             "Empty = no video meta tower (v0.1-v0.3 behavior).")
    args = parser.parse_args()

    cfg = TrainConfig(
        data=str(args.data),
        out=str(args.out),
        max_history=args.max_history,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        val_frac=args.val_frac,
        early_stop_patience=args.early_stop_patience,
        seed=args.seed,
        max_users=args.max_users,
        max_items=args.max_items,
        max_rows_per_user=args.max_rows_per_user,
        log_every=args.log_every,
        device=args.device,
        pos_weight_balance=args.pos_weight_balance,
        pos_weight_mode=args.pos_weight_mode,
        pos_weight_cap=args.pos_weight_cap,
        watch_time_loss=args.watch_time_loss,
        huber_delta=args.huber_delta,
        rank_alpha=args.rank_alpha,
        user_features=args.user_features,
        video_features=args.video_features,
    )

    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required to train. On the dev workstation (no GPU) this "
            "script is staged but not executed; install on autodl per the module "
            f"docstring autodl quickstart. Original error: {exc}"
        )

    device = torch.device("cuda" if (cfg.device == "auto" and torch.cuda.is_available()) else (cfg.device if cfg.device != "auto" else "cpu"))
    print(f"[ct-wm-train] device={device} seed={cfg.seed}")
    set_seed(cfg.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "training_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    print(f"[ct-wm-train] loading {cfg.data}")
    df = pd.read_parquet(cfg.data)
    required = {"user_id", "item_id", "action_id", "timestamp",
                "reward_click", "reward_like", "reward_long_view", "reward_watch_time"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")
    # Drop rows with any unmapped reward.
    df = df.dropna(subset=["reward_click", "reward_like", "reward_long_view", "reward_watch_time"]).reset_index(drop=True)
    df["reward_click"] = df["reward_click"].astype(int)
    df["reward_like"] = df["reward_like"].astype(int)
    df["reward_long_view"] = df["reward_long_view"].astype(int)
    # KuaiRand-27K has negative watch_time sentinels in dirty rows; clamp to 0
    # so log1p stays finite. Pure data has no such rows so this is a no-op there.
    rwt = df["reward_watch_time"].astype(float).clip(lower=0)
    df["log1p_watch_time"] = np.log1p(rwt)

    if cfg.max_users:
        keep_users = df["user_id"].drop_duplicates().head(cfg.max_users)
        df = df[df["user_id"].isin(keep_users)].reset_index(drop=True)
    # Action id integer encoding (reserve 0 for PAD; reserve 1 for OOV when capped).
    if cfg.max_items:
        # v0.5 long-tail vocab control: keep top-K most frequent action_ids; map
        # the long tail to a single "<OOV>" token rather than dropping rows.
        # KuaiRand-27K has ~10M unique videos in the click logs — embedding all
        # of them at d_model=256 produces a 7+ GB action embedding whose Adam
        # state alone exceeds 24 GB GPU memory. The OOV bucket preserves every
        # row's reward/meta signal at the cost of a shared embedding for cold
        # videos (video-side meta features still differentiate them).
        top = df["action_id"].value_counts().head(cfg.max_items).index
        df = df.copy()
        df.loc[~df["action_id"].isin(top), "action_id"] = "<OOV>"
        action_to_int = {a: i + 1 for i, a in enumerate(top.tolist())}
        action_to_int["<OOV>"] = len(action_to_int) + 1
    else:
        uniq = df["action_id"].astype(str).unique().tolist()
        action_to_int = {a: i + 1 for i, a in enumerate(uniq)}  # +1 to skip PAD
    df["action_id_int"] = df["action_id"].astype(str).map(action_to_int).astype(np.int64)
    n_actions = len(action_to_int)
    print(f"[ct-wm-train] rows={len(df):,} users={df['user_id'].nunique():,} actions={n_actions:,}")

    meta_loaded = load_meta_features(cfg.user_features, cfg.video_features)
    if meta_loaded:
        user_feat_dict, video_feat_dict, meta_stats = meta_loaded
    else:
        user_feat_dict = video_feat_dict = meta_stats = None

    print("[ct-wm-train] building sequences...")
    samples = build_user_sequences(
        df, cfg.max_history,
        max_users=cfg.max_users,
        max_rows_per_user=cfg.max_rows_per_user,
        user_features_dict=user_feat_dict,
        video_features_dict=video_feat_dict,
    )
    print(f"[ct-wm-train] samples={len(samples):,}{' (with meta features)' if meta_loaded else ''}")

    train_samples, val_samples = temporal_split(samples, cfg.val_frac, df)
    print(f"[ct-wm-train] split train={len(train_samples):,} val={len(val_samples):,}")

    def make_loader(items, shuffle: bool):
        gen = torch.Generator().manual_seed(cfg.seed)
        return DataLoader(
            items,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=2,
            collate_fn=lambda b: collate(b, cfg.max_history),
            generator=gen,
            drop_last=False,
        )

    train_loader = make_loader(train_samples, shuffle=True)
    val_loader = make_loader(val_samples, shuffle=False)

    pos_weights: dict = {}
    if cfg.pos_weight_balance:
        import math
        for head, key in (("click", "rc"), ("like", "rl"), ("long_view", "rlv")):
            n_pos = sum(1 for s in train_samples if s[key] > 0.5)
            n_neg = len(train_samples) - n_pos
            if n_pos > 0:
                ratio = n_neg / n_pos
                if cfg.pos_weight_mode == "linear":
                    pw = ratio
                elif cfg.pos_weight_mode == "sqrt":
                    pw = math.sqrt(ratio)
                else:  # cap
                    pw = min(ratio, cfg.pos_weight_cap)
                pw = max(1.0, pw)
                pos_weights[head] = torch.tensor(pw, device=device, dtype=torch.float32)
                print(f"[ct-wm-train] pos_weight {head}={pw:.3f} mode={cfg.pos_weight_mode} "
                      f"(raw ratio {ratio:.3f}, n_pos={n_pos:,}, n_neg={n_neg:,})")
            else:
                print(f"[ct-wm-train] pos_weight {head}=skipped (no positives in train)")

    model = build_model(n_actions=n_actions, cfg=cfg).to(device)
    n_params = count_params(model)
    print(f"[ct-wm-train] params={n_params:,} (D-023 4090D VRAM rail; v0.2 default ≈ 23M @ dim=512 layers=6)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: lr_lambda(s, cfg.warmup_steps, total_steps)
    )

    best_val = float("inf")
    bad_epochs = 0
    best_epoch = -1
    ckpt_path = args.out / "checkpoint.pt"
    summaries = []
    t0 = time.time()
    for epoch in range(cfg.epochs):
        train_summary = run_epoch(
            model, train_loader, optimizer, scheduler, device,
            train=True, log_path=log_path, epoch=epoch, log_every=cfg.log_every,
            pos_weights=pos_weights, watch_time_loss=cfg.watch_time_loss,
            huber_delta=cfg.huber_delta, rank_alpha=cfg.rank_alpha,
        )
        val_summary = run_epoch(
            model, val_loader, optimizer, scheduler, device,
            train=False, log_path=log_path, epoch=epoch, log_every=cfg.log_every,
            pos_weights=pos_weights, watch_time_loss=cfg.watch_time_loss,
            huber_delta=cfg.huber_delta, rank_alpha=cfg.rank_alpha,
        )
        summaries.append({"train": train_summary, "val": val_summary})
        print(f"[ct-wm-train] epoch={epoch} train_loss={train_summary['loss']:.4f} val_loss={val_summary['loss']:.4f}")
        if val_summary["loss"] < best_val - 1e-4:
            best_val = val_summary["loss"]
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg.to_json(),
                    "n_actions": n_actions,
                    "action_to_int": action_to_int,
                    "epoch": epoch,
                    "val_loss": best_val,
                },
                ckpt_path,
            )
            print(f"[ct-wm-train] saved checkpoint epoch={epoch} val_loss={best_val:.4f}")
        else:
            bad_epochs += 1
            if bad_epochs > cfg.early_stop_patience:
                print(f"[ct-wm-train] early stop at epoch={epoch}")
                break
    wall = time.time() - t0

    run_config = {
        "cli": cfg.to_json(),
        "n_params": n_params,
        "n_rows": int(len(df)),
        "n_users": int(df["user_id"].nunique()),
        "n_actions": n_actions,
        "n_train": len(train_samples),
        "n_val": len(val_samples),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "wall_seconds": wall,
        "epoch_summaries": summaries,
        "device": str(device),
    }
    (args.out / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    print(f"[ct-wm-train] done. wall={wall:.1f}s best_epoch={best_epoch} val_loss={best_val:.4f}")
    print(f"[ct-wm-train] artifacts: {ckpt_path}, {log_path}, {args.out / 'run_config.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
