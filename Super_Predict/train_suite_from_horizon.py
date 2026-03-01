"""Train 4-model suite for one (fusion_strategy, horizon_days) pair and snapshot artifacts to S3."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Data.common.s3_artifact_store import S3ArtifactStore  # noqa: E402

STRATEGIES = ("concat", "sum_pool", "max_pool")
HORIZONS = (7, 30)
MODEL_FAMILIES = ("all", "concat_mlp", "gated_fusion_mlp", "ridge", "gbdt")
LOW_CARD_CATEGORICAL_CANDIDATES = ["channel_country", "default_language", "default_audio_language", "query"]
HIGH_CARD_EXCLUDE = {"channel_id", "channel_title"}
LEAKAGE_COLUMNS = {
    "horizon_view_count",
    "horizon_days",
    "horizon_label_type",
    "view_count",
    "like_count",
    "comment_count",
    "views_per_day",
    "likes_per_day",
    "comments_per_day",
    "likes_per_view",
    "comments_per_view",
    "views_per_hour",
    "age_days",
    "virality_score",
}


@dataclass
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


@dataclass
class NNMetaBundle:
    numeric_cols: List[str]
    categorical_cols: List[str]
    cat_mappings: Dict[str, Dict[str, int]]
    num_median: Dict[str, float]
    num_mean: Dict[str, float]
    num_std: Dict[str, float]


class ConcatMLP(nn.Module):
    def __init__(
        self,
        fused_dim: int,
        numeric_dim: int,
        cat_cardinalities: List[int],
        hidden_dims: List[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.cat_embeddings = nn.ModuleList()
        emb_dims: List[int] = []
        for card in cat_cardinalities:
            emb_dim = int(min(32, max(2, round(math.sqrt(card)))))
            self.cat_embeddings.append(nn.Embedding(card, emb_dim))
            emb_dims.append(emb_dim)

        input_dim = int(fused_dim + numeric_dim + sum(emb_dims))
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, fused: torch.Tensor, numeric: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        parts = [fused, numeric]
        if self.cat_embeddings:
            cat_parts = []
            for i, emb in enumerate(self.cat_embeddings):
                cat_parts.append(emb(cat[:, i]))
            parts.extend(cat_parts)
        x = torch.cat(parts, dim=1)
        return self.net(x).squeeze(-1)


class GatedFusionMLP(nn.Module):
    def __init__(
        self,
        video_dim: int,
        audio_dim: int,
        text_dim: int,
        numeric_dim: int,
        cat_cardinalities: List[int],
        tower_dim: int = 256,
        gate_hidden: int = 128,
        head_hidden: List[int] | None = None,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if head_hidden is None:
            head_hidden = [256, 128]

        self.video_tower = nn.Sequential(nn.Linear(video_dim, tower_dim), nn.GELU(), nn.LayerNorm(tower_dim))
        self.audio_tower = nn.Sequential(nn.Linear(audio_dim, tower_dim), nn.GELU(), nn.LayerNorm(tower_dim))
        self.text_tower = nn.Sequential(nn.Linear(text_dim, tower_dim), nn.GELU(), nn.LayerNorm(tower_dim))

        self.cat_embeddings = nn.ModuleList()
        emb_dims: List[int] = []
        for card in cat_cardinalities:
            emb_dim = int(min(32, max(2, round(math.sqrt(card)))))
            self.cat_embeddings.append(nn.Embedding(card, emb_dim))
            emb_dims.append(emb_dim)

        gate_input_dim = int(numeric_dim + sum(emb_dims) + 1)
        self.gate_net = nn.Sequential(
            nn.Linear(gate_input_dim, gate_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, 3),
        )

        head_input_dim = int(tower_dim + numeric_dim + sum(emb_dims) + 1)
        layers: List[nn.Module] = []
        prev = head_input_dim
        for h in head_hidden:
            layers.extend([nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*layers)

    def _cat_emb(self, cat: torch.Tensor) -> List[torch.Tensor]:
        if not self.cat_embeddings:
            return []
        return [emb(cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]

    def forward(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        text: torch.Tensor,
        numeric: torch.Tensor,
        cat: torch.Tensor,
        text_present: torch.Tensor,
    ) -> torch.Tensor:
        cat_parts = self._cat_emb(cat)
        gate_features = [numeric, text_present]
        if cat_parts:
            gate_features.extend(cat_parts)
        gate_input = torch.cat(gate_features, dim=1)
        gate_logits = self.gate_net(gate_input)
        gate_weights = torch.softmax(gate_logits, dim=1)

        stacked = torch.stack(
            [
                self.video_tower(video),
                self.audio_tower(audio),
                self.text_tower(text),
            ],
            dim=1,
        )
        fused = (stacked * gate_weights.unsqueeze(-1)).sum(dim=1)

        head_features = [fused, numeric, text_present]
        if cat_parts:
            head_features.extend(cat_parts)
        head_input = torch.cat(head_features, dim=1)
        return self.head(head_input).squeeze(-1)


class ProjectorRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, projector_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projector_dim),
        )
        self.head = nn.Linear(projector_dim, 1)

    def forward(self, fused: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        proj = self.projector(fused)
        pred = self.head(proj).squeeze(-1)
        return pred, proj


class ArrayDataset(torch.utils.data.Dataset):
    def __init__(self, payload: Dict[str, np.ndarray], keys: Iterable[str], y: np.ndarray) -> None:
        self.payload = payload
        self.keys = list(keys)
        self.y = y.astype(np.float32)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        out = [self.payload[k][idx] for k in self.keys]
        out.append(self.y[idx])
        return tuple(out)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 4-model suite for one fusion strategy and horizon")
    parser.add_argument("--metadata_csv", default="Data/raw/Metadata/shorts_metadata_horizon.csv")
    parser.add_argument("--s3_bucket", default=os.getenv("S3_BUCKET", ""))
    parser.add_argument("--s3_region", default=os.getenv("AWS_REGION", ""))
    parser.add_argument("--fused_manifest_s3_key", required=True)
    parser.add_argument("--fusion_strategy", choices=STRATEGIES, required=True)
    parser.add_argument("--target_horizon_days", type=int, choices=HORIZONS, required=True)
    parser.add_argument("--model_family", choices=MODEL_FAMILIES, default="all")
    parser.add_argument("--snapshot_prefix", default="clipfarm/models/snapshots")
    parser.add_argument("--run_id", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_train", type=float, default=0.70)
    parser.add_argument("--split_val", type=float, default=0.15)
    parser.add_argument("--split_test", type=float, default=0.15)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--rank_metric", default="rmse_log")
    parser.add_argument("--max_epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--projector_dim", type=int, default=128)
    parser.add_argument("--text_dim", type=int, default=768)
    return parser.parse_args()


def check_split_ratios(args: argparse.Namespace) -> None:
    total = args.split_train + args.split_val + args.split_test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {total}")


def selected_models(model_family: str) -> List[str]:
    if model_family == "all":
        return ["concat_mlp", "gated_fusion_mlp", "ridge", "gbdt"]
    return [model_family]


def pick_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_numeric_bool(series: pd.Series) -> pd.Series:
    return series.map(
        lambda x: 1
        if str(x).strip().lower() in {"1", "true", "t", "yes", "y"}
        else (0 if str(x).strip().lower() in {"0", "false", "f", "no", "n"} else np.nan)
    )


def build_metadata_frame(metadata_csv: Path, horizon_days: int) -> pd.DataFrame:
    df = pd.read_csv(metadata_csv)
    if "horizon_days" not in df.columns or "horizon_view_count" not in df.columns:
        raise ValueError("metadata CSV must include horizon_days and horizon_view_count")

    df = df[df["horizon_days"] == int(horizon_days)].copy()
    if df.empty:
        raise ValueError(f"No rows found for horizon_days={horizon_days}")

    if "captured_at" in df.columns:
        df["_captured_at_dt"] = pd.to_datetime(df["captured_at"], errors="coerce", utc=True)
        df = df.sort_values("_captured_at_dt")
    df = df.drop_duplicates(subset=["video_id"], keep="last")

    if "published_at" in df.columns:
        published = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
        df["published_hour"] = published.dt.hour.astype("Float64")
        df["published_dayofweek"] = published.dt.dayofweek.astype("Float64")

    for col in ["has_hashtags", "has_shorts_hashtag", "has_clickbait_words", "caption_available", "is_vertical_thumb"]:
        if col in df.columns:
            df[col] = to_numeric_bool(df[col])

    y_raw = pd.to_numeric(df["horizon_view_count"], errors="coerce").fillna(0).clip(lower=0)
    df["target_log"] = y_raw.map(lambda x: float(math.log1p(x)))
    df["target_raw"] = y_raw.astype(float)

    for col in LEAKAGE_COLUMNS:
        if col in df.columns:
            df[f"_forbidden_{col}"] = df[col]

    return df


def get_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    protected_cols = {
        "video_id",
        "target_log",
        "target_raw",
        "captured_at",
        "published_at",
        "_captured_at_dt",
        "horizon_days",
        "horizon_label_type",
        "horizon_view_count",
        "horizon_label_type",
    }
    protected_cols.update({f"_forbidden_{c}" for c in LEAKAGE_COLUMNS})

    candidate_cols = [c for c in df.columns if c not in protected_cols and not c.startswith("_forbidden_")]
    dropped_cols = sorted([c for c in candidate_cols if c in HIGH_CARD_EXCLUDE])
    candidate_cols = [c for c in candidate_cols if c not in HIGH_CARD_EXCLUDE]

    categorical_cols = [c for c in LOW_CARD_CATEGORICAL_CANDIDATES if c in candidate_cols]

    numeric_cols: List[str] = []
    passthrough_cols: List[str] = []
    for c in candidate_cols:
        if c in categorical_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
        else:
            passthrough_cols.append(c)

    dropped_cols.extend(passthrough_cols)
    dropped_cols = sorted(set(dropped_cols))
    return numeric_cols, categorical_cols, dropped_cols


def assert_no_feature_leakage(feature_columns: List[str]) -> None:
    leaked = sorted(set(feature_columns) & LEAKAGE_COLUMNS)
    if leaked:
        raise ValueError(f"Leakage columns included in feature set: {leaked}")


def load_sharded_vectors(
    s3: S3ArtifactStore,
    manifest: pd.DataFrame,
    text_dim_default: int,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required_cols = {"video_id", "fused_key", "shard_idx"}
    missing = sorted(required_cols - set(manifest.columns))
    if missing:
        raise ValueError(f"manifest missing required cols: {missing}")

    mf = manifest.copy()
    if "captured_at" in mf.columns:
        mf = mf.sort_values("captured_at")
    mf = mf.drop_duplicates(subset=["video_id"], keep="last").reset_index(drop=True)

    shard_cache: Dict[str, np.lib.npyio.NpzFile] = {}
    emb_cache: Dict[str, np.ndarray] = {}

    def fetch_shard_vec(key: str, idx: int, tmp_dir: Path) -> np.ndarray:
        if key not in shard_cache:
            local = tmp_dir / Path(key).name
            s3.download_file(key, local)
            shard_cache[key] = np.load(local, allow_pickle=True)
        shard = shard_cache[key]
        arr = np.asarray(shard["vectors"][idx], dtype=np.float32).reshape(-1)
        return arr

    def fetch_emb_vec(key: str, tmp_dir: Path) -> np.ndarray:
        if key in emb_cache:
            return emb_cache[key]
        local = tmp_dir / (Path(key).name + ".npy")
        s3.download_file(key, local)
        vec = np.asarray(np.load(local, allow_pickle=True), dtype=np.float32).reshape(-1)
        emb_cache[key] = vec
        return vec

    rows: List[Dict[str, Any]] = []
    fused_list: List[np.ndarray] = []
    video_list: List[np.ndarray] = []
    audio_list: List[np.ndarray] = []
    text_list: List[np.ndarray] = []
    text_present_list: List[np.ndarray] = []

    inferred_text_dim: int | None = None

    with tempfile.TemporaryDirectory(prefix="train_vectors_") as tmp_dir_raw:
        tmp_dir = Path(tmp_dir_raw)
        for row in mf.itertuples(index=False):
            video_id = str(row.video_id)
            fused_key = str(row.fused_key)
            shard_idx = int(row.shard_idx)
            try:
                fused = fetch_shard_vec(fused_key, shard_idx, tmp_dir)
            except Exception:
                continue

            # For gated model we need modality vectors; skip rows where required modality keys are absent.
            if not all(hasattr(row, col) for col in ["video_emb_key", "audio_emb_key", "text_emb_key"]):
                continue

            try:
                video_vec = fetch_emb_vec(str(row.video_emb_key), tmp_dir)
                audio_vec = fetch_emb_vec(str(row.audio_emb_key), tmp_dir)
            except Exception:
                continue

            text_present = int(getattr(row, "text_present", 1) or 0)
            text_key = str(getattr(row, "text_emb_key", "") or "")
            text_vec: np.ndarray
            if text_present == 1 and text_key:
                try:
                    text_vec = fetch_emb_vec(text_key, tmp_dir)
                except Exception:
                    text_present = 0
                    text_vec = np.zeros((inferred_text_dim or text_dim_default,), dtype=np.float32)
            else:
                text_vec = np.zeros((inferred_text_dim or text_dim_default,), dtype=np.float32)

            if inferred_text_dim is None:
                inferred_text_dim = int(text_vec.shape[0])
            if text_vec.shape[0] != inferred_text_dim:
                if text_vec.shape[0] > inferred_text_dim:
                    text_vec = text_vec[:inferred_text_dim]
                else:
                    padded = np.zeros((inferred_text_dim,), dtype=np.float32)
                    padded[: text_vec.shape[0]] = text_vec
                    text_vec = padded

            rows.append(
                {
                    "video_id": video_id,
                    "text_present": int(text_present),
                    "captured_at": str(getattr(row, "captured_at", "")),
                }
            )
            fused_list.append(fused)
            video_list.append(video_vec)
            audio_list.append(audio_vec)
            text_list.append(text_vec)
            text_present_list.append(np.array([float(text_present)], dtype=np.float32))

    if not rows:
        raise ValueError("No usable rows after loading fused/modality vectors")

    out_df = pd.DataFrame(rows)
    return (
        out_df,
        np.stack(fused_list).astype(np.float32),
        np.stack(video_list).astype(np.float32),
        np.stack(audio_list).astype(np.float32),
        np.stack(text_list).astype(np.float32),
        np.stack(text_present_list).astype(np.float32),
    )


def make_splits(n_rows: int, split_train: float, split_val: float, split_test: float, seed: int) -> SplitIndices:
    if n_rows < 12:
        raise ValueError(f"Need at least 12 rows to split robustly, got {n_rows}")

    idx = np.arange(n_rows)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    n_train = int(round(n_rows * split_train))
    n_val = int(round(n_rows * split_val))
    n_test = n_rows - n_train - n_val

    if n_train < 1 or n_val < 1 or n_test < 1:
        raise ValueError(f"Invalid split counts train={n_train} val={n_val} test={n_test} for n={n_rows}")

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return SplitIndices(train=train_idx, val=val_idx, test=test_idx)


def fit_nn_metadata_bundle(
    frame: pd.DataFrame,
    train_idx: np.ndarray,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> NNMetaBundle:
    train_df = frame.iloc[train_idx]

    num_median: Dict[str, float] = {}
    num_mean: Dict[str, float] = {}
    num_std: Dict[str, float] = {}
    for c in numeric_cols:
        vals = pd.to_numeric(train_df[c], errors="coerce")
        med = float(vals.median()) if vals.notna().any() else 0.0
        filled = vals.fillna(med)
        mean = float(filled.mean())
        std = float(filled.std(ddof=0))
        if std <= 1e-8:
            std = 1.0
        num_median[c] = med
        num_mean[c] = mean
        num_std[c] = std

    cat_mappings: Dict[str, Dict[str, int]] = {}
    for c in categorical_cols:
        raw = train_df[c].fillna("<NA>").astype(str)
        cats = sorted(set(raw.tolist()))
        cat_mappings[c] = {cat: i + 1 for i, cat in enumerate(cats)}

    return NNMetaBundle(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        cat_mappings=cat_mappings,
        num_median=num_median,
        num_mean=num_mean,
        num_std=num_std,
    )


def transform_nn_metadata(frame: pd.DataFrame, bundle: NNMetaBundle) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    if bundle.numeric_cols:
        num_arr = np.zeros((len(frame), len(bundle.numeric_cols)), dtype=np.float32)
        for i, c in enumerate(bundle.numeric_cols):
            vals = pd.to_numeric(frame[c], errors="coerce").fillna(bundle.num_median[c]).astype(float)
            vals = (vals - bundle.num_mean[c]) / bundle.num_std[c]
            num_arr[:, i] = vals.to_numpy(dtype=np.float32)
    else:
        num_arr = np.zeros((len(frame), 0), dtype=np.float32)

    cat_cardinalities: List[int] = []
    if bundle.categorical_cols:
        cat_arr = np.zeros((len(frame), len(bundle.categorical_cols)), dtype=np.int64)
        for i, c in enumerate(bundle.categorical_cols):
            mapping = bundle.cat_mappings[c]
            cat_cardinalities.append(len(mapping) + 1)
            raw = frame[c].fillna("<NA>").astype(str)
            cat_arr[:, i] = raw.map(lambda x: mapping.get(x, 0)).to_numpy(dtype=np.int64)
    else:
        cat_arr = np.zeros((len(frame), 0), dtype=np.int64)

    return num_arr, cat_arr, cat_cardinalities


def regression_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> Dict[str, float]:
    y_true_log = np.asarray(y_true_log, dtype=np.float64)
    y_pred_log = np.asarray(y_pred_log, dtype=np.float64)
    rmse_log = float(np.sqrt(np.mean((y_true_log - y_pred_log) ** 2)))
    mae_log = float(np.mean(np.abs(y_true_log - y_pred_log)))
    r2_log = float(r2_score(y_true_log, y_pred_log)) if y_true_log.size > 1 else float("nan")

    y_true_raw = np.expm1(np.clip(y_true_log, -20, 30))
    y_pred_raw = np.expm1(np.clip(y_pred_log, -20, 30))
    y_pred_raw = np.clip(y_pred_raw, 0.0, None)

    rmse_raw = float(np.sqrt(np.mean((y_true_raw - y_pred_raw) ** 2)))
    mae_raw = float(np.mean(np.abs(y_true_raw - y_pred_raw)))

    return {
        "rmse_log": rmse_log,
        "mae_log": mae_log,
        "r2_log": r2_log,
        "rmse_raw": rmse_raw,
        "mae_raw": mae_raw,
    }


def compute_slice_metrics(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    text_present: np.ndarray,
) -> Dict[str, Dict[str, float | int]]:
    out: Dict[str, Dict[str, float | int]] = {}
    mask_1 = text_present.reshape(-1) >= 0.5
    mask_0 = ~mask_1
    for name, mask in (("text_present_1", mask_1), ("text_present_0", mask_0)):
        count = int(mask.sum())
        if count == 0:
            out[name] = {"n": 0}
            continue
        m = regression_metrics(y_true_log[mask], y_pred_log[mask])
        m["n"] = count
        out[name] = m
    return out


def fit_torch_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    batch_forward,
) -> Tuple[Dict[str, Any], Dict[str, List[float]]]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_rmse = float("inf")
    best_state: Dict[str, Any] = {}
    epochs_no_improve = 0
    curves = {"train_loss": [], "val_loss": [], "val_rmse_log": []}

    model.to(device)

    for _epoch in range(max_epochs):
        model.train()
        train_losses: List[float] = []
        for batch in train_loader:
            batch = [b.to(device) for b in batch]
            *features, y = batch
            pred = batch_forward(model, features)
            loss = criterion(pred, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        val_losses: List[float] = []
        val_true: List[np.ndarray] = []
        val_pred: List[np.ndarray] = []
        with torch.no_grad():
            for batch in val_loader:
                batch = [b.to(device) for b in batch]
                *features, y = batch
                pred = batch_forward(model, features)
                loss = criterion(pred, y)
                val_losses.append(float(loss.detach().cpu().item()))
                val_true.append(y.detach().cpu().numpy())
                val_pred.append(pred.detach().cpu().numpy())

        val_true_arr = np.concatenate(val_true, axis=0)
        val_pred_arr = np.concatenate(val_pred, axis=0)
        val_rmse = float(np.sqrt(np.mean((val_true_arr - val_pred_arr) ** 2)))

        curves["train_loss"].append(float(np.mean(train_losses) if train_losses else 0.0))
        curves["val_loss"].append(float(np.mean(val_losses) if val_losses else 0.0))
        curves["val_rmse_log"].append(val_rmse)

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return best_state, curves


def predict_torch_model(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    batch_forward,
) -> np.ndarray:
    model.eval()
    model.to(device)
    preds: List[np.ndarray] = []
    with torch.no_grad():
        for batch in data_loader:
            batch = [b.to(device) for b in batch]
            *features, _y = batch
            pred = batch_forward(model, features)
            preds.append(pred.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def build_loader(payload: Dict[str, np.ndarray], keys: List[str], y: np.ndarray, batch_size: int, shuffle: bool) -> torch.utils.data.DataLoader:
    ds = ArrayDataset(payload=payload, keys=keys, y=y)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def fit_ridge_model(
    metadata: pd.DataFrame,
    fused: np.ndarray,
    y: np.ndarray,
    split: SplitIndices,
    categorical_cols: List[str],
    numeric_cols: List[str],
) -> Tuple[Pipeline, Dict[str, np.ndarray]]:
    fused_cols = [f"fused_{i}" for i in range(fused.shape[1])]
    fused_df = pd.DataFrame(fused, columns=fused_cols)
    X = pd.concat([metadata.reset_index(drop=True), fused_df], axis=1)

    cat_cols = [c for c in categorical_cols if c in X.columns]
    num_cols = [c for c in numeric_cols if c in X.columns] + fused_cols

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                num_cols,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    model = RidgeCV(alphas=np.array([0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]))
    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X.iloc[split.train], y[split.train])

    preds = {
        "val": pipe.predict(X.iloc[split.val]).astype(np.float32),
        "test": pipe.predict(X.iloc[split.test]).astype(np.float32),
    }
    return pipe, preds


def fit_projector(
    fused: np.ndarray,
    y: np.ndarray,
    split: SplitIndices,
    device: torch.device,
    max_epochs: int,
    patience: int,
    projector_dim: int,
) -> Tuple[ProjectorRegressor, Dict[str, List[float]], np.ndarray]:
    model = ProjectorRegressor(input_dim=fused.shape[1], hidden_dim=256, projector_dim=projector_dim, dropout=0.1)

    payload = {"fused": fused.astype(np.float32)}
    train_loader = build_loader(payload, ["fused"], y[split.train], batch_size=256, shuffle=True)
    val_loader = build_loader({"fused": fused[split.val].astype(np.float32)}, ["fused"], y[split.val], batch_size=512, shuffle=False)

    def fwd(m: ProjectorRegressor, features: List[torch.Tensor]) -> torch.Tensor:
        pred, _proj = m(features[0])
        return pred

    _best_state, curves = fit_torch_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        max_epochs=max_epochs,
        patience=patience,
        learning_rate=1e-3,
        weight_decay=1e-4,
        batch_forward=fwd,
    )

    model.eval()
    model.to(device)
    all_proj: List[np.ndarray] = []
    with torch.no_grad():
        data = torch.from_numpy(fused.astype(np.float32)).to(device)
        _, proj = model(data)
        all_proj.append(proj.detach().cpu().numpy().astype(np.float32))
    projected = np.concatenate(all_proj, axis=0)
    return model, curves, projected


def fit_gbdt_model(
    metadata: pd.DataFrame,
    projected_fused: np.ndarray,
    y: np.ndarray,
    split: SplitIndices,
    categorical_cols: List[str],
    numeric_cols: List[str],
) -> Tuple[Pipeline, Dict[str, np.ndarray]]:
    proj_cols = [f"proj_{i}" for i in range(projected_fused.shape[1])]
    proj_df = pd.DataFrame(projected_fused, columns=proj_cols)
    X = pd.concat([metadata.reset_index(drop=True), proj_df], axis=1)

    cat_cols = [c for c in categorical_cols if c in X.columns]
    num_cols = [c for c in numeric_cols if c in X.columns] + proj_cols

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                ]),
                num_cols,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=6,
        max_iter=500,
        min_samples_leaf=20,
        l2_regularization=1e-4,
        early_stopping=True,
        random_state=42,
    )
    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X.iloc[split.train], y[split.train])

    preds = {
        "val": pipe.predict(X.iloc[split.val]).astype(np.float32),
        "test": pipe.predict(X.iloc[split.test]).astype(np.float32),
    }
    return pipe, preds


def upload_tree_to_s3(s3: S3ArtifactStore, root: Path, s3_prefix: str) -> None:
    for p in root.rglob("*"):
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            key = f"{s3_prefix.rstrip('/')}/{rel}"
            s3.upload_file(p, key)


def make_run_id(cli_run_id: str) -> str:
    value = (cli_run_id or "").strip()
    if value:
        return value
    gh_run = os.getenv("GITHUB_RUN_ID", "").strip()
    if gh_run:
        return f"gh-{gh_run}"
    return "local-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def get_git_sha() -> str:
    if os.getenv("GITHUB_SHA"):
        return str(os.getenv("GITHUB_SHA"))
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def main() -> None:
    args = parse_args()
    check_split_ratios(args)

    set_global_seed(args.seed)
    device = pick_device(args.device)
    run_id = make_run_id(args.run_id)

    s3 = S3ArtifactStore(bucket=args.s3_bucket, region=args.s3_region)

    metadata_df = build_metadata_frame(Path(args.metadata_csv), args.target_horizon_days)

    if args.fusion_strategy not in STRATEGIES:
        raise ValueError(f"Unsupported fusion_strategy: {args.fusion_strategy}")

    with tempfile.TemporaryDirectory(prefix="train_suite_") as tmp_dir_raw:
        tmp_dir = Path(tmp_dir_raw)
        local_manifest = tmp_dir / "fused_manifest.parquet"
        if not s3.exists(args.fused_manifest_s3_key):
            raise FileNotFoundError(f"Missing fused manifest in S3: {args.fused_manifest_s3_key}")
        s3.download_file(args.fused_manifest_s3_key, local_manifest)
        fused_manifest = pd.read_parquet(local_manifest)

        if "fusion_strategy" in fused_manifest.columns:
            fused_manifest = fused_manifest[fused_manifest["fusion_strategy"].astype(str) == args.fusion_strategy].copy()
        if fused_manifest.empty:
            raise ValueError("No rows in fused manifest after strategy filter")

        (
            emb_meta_df,
            fused_mat,
            video_mat,
            audio_mat,
            text_mat,
            text_present_arr,
        ) = load_sharded_vectors(s3=s3, manifest=fused_manifest, text_dim_default=args.text_dim)

        joined = emb_meta_df.merge(metadata_df, on="video_id", how="inner")
        if joined.empty:
            raise ValueError("No rows after joining fused rows with metadata")

        id_to_idx = {vid: i for i, vid in enumerate(emb_meta_df["video_id"].tolist())}
        emb_idx = joined["video_id"].map(id_to_idx).to_numpy(dtype=np.int64)

        fused_use = fused_mat[emb_idx]
        video_use = video_mat[emb_idx]
        audio_use = audio_mat[emb_idx]
        text_use = text_mat[emb_idx]
        text_present_use = text_present_arr[emb_idx]

        y_log = joined["target_log"].to_numpy(dtype=np.float32)
        y_raw = joined["target_raw"].to_numpy(dtype=np.float32)

        numeric_cols, categorical_cols, dropped_metadata_cols = get_feature_columns(joined)
        selected_feature_cols = numeric_cols + categorical_cols
        assert_no_feature_leakage(selected_feature_cols)

        split = make_splits(
            n_rows=len(joined),
            split_train=args.split_train,
            split_val=args.split_val,
            split_test=args.split_test,
            seed=args.seed,
        )

        nn_bundle = fit_nn_metadata_bundle(joined, split.train, numeric_cols=numeric_cols, categorical_cols=categorical_cols)
        nn_num, nn_cat, cat_cardinalities = transform_nn_metadata(joined, nn_bundle)

        requested_models = selected_models(args.model_family)

        payload_concat = {
            "fused": fused_use.astype(np.float32),
            "num": nn_num.astype(np.float32),
            "cat": nn_cat.astype(np.int64),
        }
        payload_gated = {
            "video": video_use.astype(np.float32),
            "audio": audio_use.astype(np.float32),
            "text": text_use.astype(np.float32),
            "num": nn_num.astype(np.float32),
            "cat": nn_cat.astype(np.int64),
            "text_present": text_present_use.astype(np.float32),
        }

        def sub_payload(payload: Dict[str, np.ndarray], idx: np.ndarray) -> Dict[str, np.ndarray]:
            return {k: v[idx] for k, v in payload.items()}

        pred_val_map: Dict[str, np.ndarray] = {}
        pred_test_map: Dict[str, np.ndarray] = {}
        model_curves: Dict[str, Dict[str, List[float]]] = {}
        concat_model: ConcatMLP | None = None
        gated_model: GatedFusionMLP | None = None
        projector_model: ProjectorRegressor | None = None
        ridge_pipe: Pipeline | None = None
        gbdt_pipe: Pipeline | None = None

        if "concat_mlp" in requested_models:
            train_loader_concat = build_loader(
                sub_payload(payload_concat, split.train),
                ["fused", "num", "cat"],
                y_log[split.train],
                256,
                True,
            )
            val_loader_concat = build_loader(
                sub_payload(payload_concat, split.val),
                ["fused", "num", "cat"],
                y_log[split.val],
                512,
                False,
            )
            test_loader_concat = build_loader(
                sub_payload(payload_concat, split.test),
                ["fused", "num", "cat"],
                y_log[split.test],
                512,
                False,
            )

            concat_model = ConcatMLP(
                fused_dim=fused_use.shape[1],
                numeric_dim=nn_num.shape[1],
                cat_cardinalities=cat_cardinalities,
                hidden_dims=[1024, 512, 256],
                dropout=0.20,
            )

            def concat_forward(model: ConcatMLP, features: List[torch.Tensor]) -> torch.Tensor:
                return model(features[0], features[1], features[2].long())

            _best_concat_state, concat_curves = fit_torch_model(
                model=concat_model,
                train_loader=train_loader_concat,
                val_loader=val_loader_concat,
                device=device,
                max_epochs=args.max_epochs,
                patience=args.patience,
                learning_rate=1e-3,
                weight_decay=1e-4,
                batch_forward=concat_forward,
            )
            model_curves["concat_mlp"] = concat_curves
            pred_val_map["concat_mlp"] = predict_torch_model(concat_model, val_loader_concat, device, concat_forward)
            pred_test_map["concat_mlp"] = predict_torch_model(concat_model, test_loader_concat, device, concat_forward)

        if "gated_fusion_mlp" in requested_models:
            train_loader_gated = build_loader(
                sub_payload(payload_gated, split.train),
                ["video", "audio", "text", "num", "cat", "text_present"],
                y_log[split.train],
                256,
                True,
            )
            val_loader_gated = build_loader(
                sub_payload(payload_gated, split.val),
                ["video", "audio", "text", "num", "cat", "text_present"],
                y_log[split.val],
                512,
                False,
            )
            test_loader_gated = build_loader(
                sub_payload(payload_gated, split.test),
                ["video", "audio", "text", "num", "cat", "text_present"],
                y_log[split.test],
                512,
                False,
            )

            gated_model = GatedFusionMLP(
                video_dim=video_use.shape[1],
                audio_dim=audio_use.shape[1],
                text_dim=text_use.shape[1],
                numeric_dim=nn_num.shape[1],
                cat_cardinalities=cat_cardinalities,
                tower_dim=256,
                gate_hidden=128,
                head_hidden=[256, 128],
                dropout=0.15,
            )

            def gated_forward(model: GatedFusionMLP, features: List[torch.Tensor]) -> torch.Tensor:
                return model(
                    features[0],
                    features[1],
                    features[2],
                    features[3],
                    features[4].long(),
                    features[5],
                )

            _best_gated_state, gated_curves = fit_torch_model(
                model=gated_model,
                train_loader=train_loader_gated,
                val_loader=val_loader_gated,
                device=device,
                max_epochs=args.max_epochs,
                patience=args.patience,
                learning_rate=7e-4,
                weight_decay=1e-4,
                batch_forward=gated_forward,
            )
            model_curves["gated_fusion_mlp"] = gated_curves
            pred_val_map["gated_fusion_mlp"] = predict_torch_model(gated_model, val_loader_gated, device, gated_forward)
            pred_test_map["gated_fusion_mlp"] = predict_torch_model(gated_model, test_loader_gated, device, gated_forward)

        if "ridge" in requested_models:
            ridge_pipe, ridge_preds = fit_ridge_model(
                metadata=joined[numeric_cols + categorical_cols].copy(),
                fused=fused_use,
                y=y_log,
                split=split,
                categorical_cols=categorical_cols,
                numeric_cols=numeric_cols,
            )
            pred_val_map["ridge"] = ridge_preds["val"]
            pred_test_map["ridge"] = ridge_preds["test"]

        if "gbdt" in requested_models:
            projector_model, projector_curves = None, {}
            projector_model, projector_curves, projected = fit_projector(
                fused=fused_use,
                y=y_log,
                split=split,
                device=device,
                max_epochs=max(10, min(args.max_epochs, 20)),
                patience=max(4, min(args.patience, 8)),
                projector_dim=args.projector_dim,
            )
            model_curves["gbdt_projector"] = projector_curves
            gbdt_pipe, gbdt_preds = fit_gbdt_model(
                metadata=joined[numeric_cols + categorical_cols].copy(),
                projected_fused=projected,
                y=y_log,
                split=split,
                categorical_cols=categorical_cols,
                numeric_cols=numeric_cols,
            )
            pred_val_map["gbdt"] = gbdt_preds["val"]
            pred_test_map["gbdt"] = gbdt_preds["test"]

        model_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
        model_slice_metrics: Dict[str, Dict[str, Dict[str, Dict[str, float | int]]]] = {}

        y_val = y_log[split.val]
        y_test = y_log[split.test]
        tp_val = text_present_use[split.val]
        tp_test = text_present_use[split.test]

        for model_name in requested_models:
            if model_name not in pred_val_map:
                continue
            val_metrics = regression_metrics(y_val, pred_val_map[model_name])
            test_metrics = regression_metrics(y_test, pred_test_map[model_name])
            model_metrics[model_name] = {"val": val_metrics, "test": test_metrics}
            model_slice_metrics[model_name] = {
                "val": compute_slice_metrics(y_val, pred_val_map[model_name], tp_val),
                "test": compute_slice_metrics(y_test, pred_test_map[model_name], tp_test),
            }

        if not model_metrics:
            raise ValueError(f"No model metrics produced for model_family={args.model_family}")

        rank_col = args.rank_metric if str(args.rank_metric).startswith("val_") else f"val_{args.rank_metric}"
        leaderboard_rows: List[Dict[str, Any]] = []
        for model_name, payload in model_metrics.items():
            row = {
                "model": model_name,
                "val_rmse_log": payload["val"]["rmse_log"],
                "val_mae_log": payload["val"]["mae_log"],
                "val_r2_log": payload["val"]["r2_log"],
                "val_rmse_raw": payload["val"]["rmse_raw"],
                "val_mae_raw": payload["val"]["mae_raw"],
                "test_rmse_log": payload["test"]["rmse_log"],
                "test_mae_log": payload["test"]["mae_log"],
                "test_r2_log": payload["test"]["r2_log"],
                "test_rmse_raw": payload["test"]["rmse_raw"],
                "test_mae_raw": payload["test"]["mae_raw"],
            }
            row[rank_col] = row.get(rank_col, row["val_rmse_log"])
            leaderboard_rows.append(row)

        leaderboard_df = pd.DataFrame(leaderboard_rows)
        if rank_col not in leaderboard_df.columns:
            rank_col = "val_rmse_log"
        leaderboard_df = leaderboard_df.sort_values(rank_col, ascending=True).reset_index(drop=True)
        leaderboard_df.insert(0, "rank", np.arange(1, len(leaderboard_df) + 1))

        best_model = str(leaderboard_df.iloc[0]["model"])

        if args.model_family == "all":
            snapshot_root = (
                f"{args.snapshot_prefix.strip('/')}/run_id={run_id}/strategy={args.fusion_strategy}/horizon={args.target_horizon_days}"
            )
        else:
            snapshot_root = (
                f"{args.snapshot_prefix.strip('/')}/run_id={run_id}/model={args.model_family}/"
                f"strategy={args.fusion_strategy}/horizon={args.target_horizon_days}"
            )

        artifact_dir = tmp_dir / "snapshot_artifacts"
        (artifact_dir / "models").mkdir(parents=True, exist_ok=True)
        (artifact_dir / "curves").mkdir(parents=True, exist_ok=True)
        (artifact_dir / "predictions").mkdir(parents=True, exist_ok=True)

        split_manifest = pd.DataFrame(
            {
                "video_id": joined["video_id"].astype(str),
                "split": "",
                "text_present": text_present_use.reshape(-1),
                "target_log": y_log,
                "target_raw": y_raw,
            }
        )
        split_manifest.loc[split.train, "split"] = "train"
        split_manifest.loc[split.val, "split"] = "val"
        split_manifest.loc[split.test, "split"] = "test"
        split_manifest.to_parquet(artifact_dir / "split_manifest.parquet", index=False)

        all_hyperparams = {
            "concat_mlp": {
                "hidden_dims": [1024, 512, 256],
                "dropout": 0.2,
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "max_epochs": int(args.max_epochs),
                "patience": int(args.patience),
            },
            "gated_fusion_mlp": {
                "tower_dim": 256,
                "gate_hidden": 128,
                "head_hidden": [256, 128],
                "dropout": 0.15,
                "lr": 7e-4,
                "weight_decay": 1e-4,
                "max_epochs": int(args.max_epochs),
                "patience": int(args.patience),
            },
            "ridge": {
                "alphas": [0.1, 0.3, 1, 3, 10, 30, 100],
            },
            "gbdt": {
                "learning_rate": 0.05,
                "max_depth": 6,
                "max_iter": 500,
                "min_samples_leaf": 20,
                "l2_regularization": 1e-4,
                "projector_dim": int(args.projector_dim),
            },
        }

        config_used = {
            "generated_at": utc_now_iso(),
            "run_id": run_id,
            "git_sha": get_git_sha(),
            "fusion_strategy": args.fusion_strategy,
            "target_horizon_days": int(args.target_horizon_days),
            "model_family": args.model_family,
            "trained_models": requested_models,
            "target_transform": "log1p(horizon_view_count)",
            "seed": int(args.seed),
            "splits": {
                "train": float(args.split_train),
                "val": float(args.split_val),
                "test": float(args.split_test),
            },
            "device": str(device),
            "rank_metric": rank_col,
            "feature_set_version": "v1_training_suite",
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "dropped_metadata_cols": dropped_metadata_cols,
            "excluded_high_card": sorted(HIGH_CARD_EXCLUDE),
            "excluded_leakage": sorted(LEAKAGE_COLUMNS),
            "fused_manifest_s3_key": args.fused_manifest_s3_key,
            "hyperparams": {k: all_hyperparams[k] for k in requested_models},
            "versions": {
                "python": sys.version,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
                "torch": torch.__version__,
            },
        }

        numeric_feature_stats = {}
        for c in numeric_cols:
            vals = pd.to_numeric(joined[c], errors="coerce")
            if vals.notna().sum() == 0:
                numeric_feature_stats[c] = {"n_non_null": 0}
            else:
                numeric_feature_stats[c] = {
                    "n_non_null": int(vals.notna().sum()),
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=0)),
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                }

        data_summary = {
            "generated_at": utc_now_iso(),
            "n_total": int(len(joined)),
            "n_train": int(len(split.train)),
            "n_val": int(len(split.val)),
            "n_test": int(len(split.test)),
            "split_ratios": {
                "train": float(args.split_train),
                "val": float(args.split_val),
                "test": float(args.split_test),
            },
            "text_present_rate": float(text_present_use.mean()),
            "target_log_mean": float(np.mean(y_log)),
            "target_log_std": float(np.std(y_log)),
            "target_raw_mean": float(np.mean(y_raw)),
            "target_raw_p50": float(np.percentile(y_raw, 50)),
            "target_raw_p90": float(np.percentile(y_raw, 90)),
            "metadata_missing_rate": {
                c: float(joined[c].isna().mean()) for c in (numeric_cols + categorical_cols)
            },
            "numeric_feature_stats": numeric_feature_stats,
            "horizon_days": int(args.target_horizon_days),
            "horizon_definition": f"horizon_days == {int(args.target_horizon_days)}",
            "leakage_columns_checked": sorted(LEAKAGE_COLUMNS),
        }

        metrics_summary = {
            "generated_at": utc_now_iso(),
            "run_id": run_id,
            "fusion_strategy": args.fusion_strategy,
            "target_horizon_days": int(args.target_horizon_days),
            "model_family": args.model_family,
            "trained_models": requested_models,
            "target_transform": "log1p(horizon_view_count)",
            "rank_metric": rank_col,
            "best_model": best_model,
            "models": model_metrics,
        }

        (artifact_dir / "config_used.json").write_text(json.dumps(config_used, indent=2), encoding="utf-8")
        (artifact_dir / "data_summary.json").write_text(json.dumps(data_summary, indent=2), encoding="utf-8")
        (artifact_dir / "metrics_summary.json").write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")
        (artifact_dir / "slice_metrics_text_present.json").write_text(
            json.dumps(model_slice_metrics, indent=2),
            encoding="utf-8",
        )
        leaderboard_df.to_csv(artifact_dir / "leaderboard.csv", index=False)

        preprocess_bundle = {
            "numeric_cols": nn_bundle.numeric_cols,
            "categorical_cols": nn_bundle.categorical_cols,
            "num_median": nn_bundle.num_median,
            "num_mean": nn_bundle.num_mean,
            "num_std": nn_bundle.num_std,
            "cat_mappings": nn_bundle.cat_mappings,
        }

        if concat_model is not None:
            concat_ckpt = {
                "model_state_dict": concat_model.state_dict(),
                "model_config": {
                    "fused_dim": int(fused_use.shape[1]),
                    "numeric_dim": int(nn_num.shape[1]),
                    "cat_cardinalities": cat_cardinalities,
                },
                "preprocess": preprocess_bundle,
            }
            torch.save(concat_ckpt, artifact_dir / "models" / "concat_mlp.pt")

        if gated_model is not None:
            gated_ckpt = {
                "model_state_dict": gated_model.state_dict(),
                "model_config": {
                    "video_dim": int(video_use.shape[1]),
                    "audio_dim": int(audio_use.shape[1]),
                    "text_dim": int(text_use.shape[1]),
                    "numeric_dim": int(nn_num.shape[1]),
                    "cat_cardinalities": cat_cardinalities,
                },
                "preprocess": preprocess_bundle,
            }
            torch.save(gated_ckpt, artifact_dir / "models" / "gated_fusion_mlp.pt")

        if ridge_pipe is not None:
            joblib.dump(ridge_pipe, artifact_dir / "models" / "ridge.joblib")
        if gbdt_pipe is not None:
            joblib.dump(gbdt_pipe, artifact_dir / "models" / "gbdt.joblib")
        if projector_model is not None:
            torch.save(
                {
                    "state_dict": projector_model.state_dict(),
                    "input_dim": int(fused_use.shape[1]),
                    "projector_dim": int(args.projector_dim),
                },
                artifact_dir / "models" / "gbdt_projector.pt",
            )

        for curve_name, curve_payload in model_curves.items():
            (artifact_dir / "curves" / f"{curve_name}_curve.json").write_text(
                json.dumps(curve_payload, indent=2),
                encoding="utf-8",
            )

        val_pred_df = pd.DataFrame(
            {
                "video_id": joined.iloc[split.val]["video_id"].astype(str).to_numpy(),
                "text_present": text_present_use[split.val].reshape(-1),
                "y_true_log": y_val,
                "y_true_raw": np.expm1(np.clip(y_val, -20, 30)),
            }
        )
        for model_name in requested_models:
            if model_name not in pred_val_map:
                continue
            col = f"pred_{model_name}_log"
            val_pred_df[col] = pred_val_map[model_name]
            val_pred_df[col.replace("_log", "_raw")] = np.clip(np.expm1(np.clip(val_pred_df[col], -20, 30)), 0.0, None)
        val_pred_df.to_parquet(artifact_dir / "predictions" / "val.parquet", index=False)

        test_pred_df = pd.DataFrame(
            {
                "video_id": joined.iloc[split.test]["video_id"].astype(str).to_numpy(),
                "text_present": text_present_use[split.test].reshape(-1),
                "y_true_log": y_test,
                "y_true_raw": np.expm1(np.clip(y_test, -20, 30)),
            }
        )
        for model_name in requested_models:
            if model_name not in pred_test_map:
                continue
            col = f"pred_{model_name}_log"
            test_pred_df[col] = pred_test_map[model_name]
            test_pred_df[col.replace("_log", "_raw")] = np.clip(np.expm1(np.clip(test_pred_df[col], -20, 30)), 0.0, None)
        test_pred_df.to_parquet(artifact_dir / "predictions" / "test.parquet", index=False)

        upload_tree_to_s3(s3=s3, root=artifact_dir, s3_prefix=snapshot_root)

        print("Training suite summary")
        print(f"run_id: {run_id}")
        print(f"model_family: {args.model_family}")
        print(f"fusion_strategy: {args.fusion_strategy}")
        print(f"target_horizon_days: {args.target_horizon_days}")
        print(f"rows_total: {len(joined)}")
        print(f"rows_train: {len(split.train)}")
        print(f"rows_val: {len(split.val)}")
        print(f"rows_test: {len(split.test)}")
        print(f"best_model: {best_model}")
        print(f"trained_models: {','.join(requested_models)}")
        print(f"rank_metric: {rank_col}")
        print(f"snapshot_root: {snapshot_root}")


if __name__ == "__main__":
    main()
