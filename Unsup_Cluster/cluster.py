"""Cluster fused embeddings from S3 manifests using canonical video_id keys.

This script implements Dev1/2 outputs:
- Loads fused vectors from S3 shard manifests for one or more fusion strategies.
- Builds reduced spaces for clustering and visualization.
- Auto-selects strategy + K based on silhouette, stability, and tiny-cluster penalties.
- Saves deterministic cluster assignments and diagnostics.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except Exception:
    matplotlib = None
    plt = None
    MATPLOTLIB_AVAILABLE = False

try:
    import umap  # type: ignore

    UMAP_AVAILABLE = True
except Exception:
    umap = None
    UMAP_AVAILABLE = False

REPO_ROOT = Path(__file__).resolve().parents[1]

# Add repo root so local package imports resolve when executed directly.
import sys

sys.path.insert(0, str(REPO_ROOT))

from Data.common.horizon_delta import DEFAULT_METADATA_CSV, load_latest_horizon_rows  # noqa: E402
from Data.common.s3_artifact_store import S3ArtifactStore  # noqa: E402

DEFAULT_FUSED_PREFIX_BASE = "clipfarm/fused"
DEFAULT_STRATEGIES = ("concat", "sum_pool", "max_pool")
DEFAULT_TERMINAL_FUSION_STATUSES = ("success_full", "success_text_placeholder")
DEFAULT_RANDOM_SEEDS = (13, 23, 37, 53, 71)

DEFAULT_OUTPUT_CSV = REPO_ROOT / "Unsup_Cluster" / "cluster_results.csv"
DEFAULT_DIAGNOSTICS_JSON = REPO_ROOT / "Unsup_Cluster" / "cluster_diagnostics.json"
DEFAULT_EMBEDDING_DIR = REPO_ROOT / "Latence" / "latent_space_outputs" / "embeddings"
DEFAULT_PLOTS_DIR = REPO_ROOT / "Latence" / "latent_space_outputs" / "plots"


@dataclass
class StrategyDataset:
    strategy: str
    video_ids: list[str]
    vectors: np.ndarray
    manifest_rows_raw: int
    manifest_rows_filtered: int


@dataclass
class ReducedSpace:
    strategy: str
    video_ids: list[str]
    vectors: np.ndarray
    cluster_matrix: np.ndarray
    cluster_space_name: str
    pca_cluster: np.ndarray
    pca_viz2d: np.ndarray
    umap_viz2d: Optional[np.ndarray]
    scaler: StandardScaler


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_csv_set(raw: str) -> set[str]:
    return {x.strip() for x in (raw or "").split(",") if x.strip()}


def _parse_csv_list(raw: str) -> list[str]:
    return [x.strip() for x in (raw or "").split(",") if x.strip()]


def _parse_int_csv(raw: str) -> list[int]:
    values: list[int] = []
    for tok in _parse_csv_list(raw):
        try:
            values.append(int(tok))
        except ValueError as exc:
            raise ValueError(f"Invalid integer in CSV list: {tok!r}") from exc
    if not values:
        raise ValueError("At least one integer seed is required")
    return values


def _resolve_manifest_key(fused_prefix_base: str, strategy: str) -> str:
    return f"{fused_prefix_base.strip('/')}/{strategy}/fused_manifest.parquet"


def _resolve_strategy_prefix(fused_prefix_base: str, strategy: str) -> str:
    return f"{fused_prefix_base.strip('/')}/{strategy}"


def _validate_manifest_columns(df: pd.DataFrame, key: str) -> None:
    required = {"video_id", "source_hash", "captured_at", "fused_key", "shard_idx", "fusion_status"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Manifest {key} missing required columns: {missing}")


def _dedupe_latest_by_video_id(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for col in ("video_id", "source_hash", "captured_at", "fused_key"):
        if col not in work.columns:
            work[col] = ""
        work[col] = work[col].fillna("").astype(str)

    work = work.sort_values(["video_id", "captured_at", "source_hash", "fused_key", "shard_idx"])
    work = work.drop_duplicates(subset=["video_id"], keep="last")
    work = work.sort_values("video_id").reset_index(drop=True)
    return work


def _load_vectors_from_manifest(
    s3: S3ArtifactStore,
    manifest_df: pd.DataFrame,
    tmp_dir: Path,
) -> tuple[list[str], np.ndarray]:
    if manifest_df.empty:
        raise ValueError("Manifest frame is empty")

    # Preserve canonical manifest order for output alignment, while processing
    # one shard at a time to keep memory bounded.
    ordered = manifest_df.reset_index(drop=True).copy()
    ordered["row_idx"] = np.arange(len(ordered), dtype=np.int64)
    scan = ordered.sort_values(["fused_key", "shard_idx", "row_idx"]).reset_index(drop=True)

    video_ids = ordered["video_id"].astype(str).tolist()
    vectors: list[Optional[np.ndarray]] = [None] * len(ordered)

    for fused_key, group in scan.groupby("fused_key", sort=False):
        key = str(fused_key)
        local = tmp_dir / Path(key).name
        s3.download_file(key, local)

        try:
            with np.load(local, allow_pickle=True) as payload:
                if "vectors" not in payload:
                    raise ValueError(f"Shard missing 'vectors' array: {key}")
                arr = np.asarray(payload["vectors"], dtype=np.float32)
        finally:
            if local.exists():
                local.unlink()

        if arr.ndim != 2:
            raise ValueError(f"Shard vectors must be 2D, got {arr.shape} for {key}")

        for row in group.itertuples(index=False):
            out_idx = int(getattr(row, "row_idx"))
            shard_idx = int(getattr(row, "shard_idx"))
            video_id = str(getattr(row, "video_id"))

            if shard_idx < 0 or shard_idx >= arr.shape[0]:
                raise IndexError(
                    f"shard_idx out of range for {video_id}: idx={shard_idx} rows={arr.shape[0]} key={key}"
                )
            vectors[out_idx] = np.asarray(arr[shard_idx], dtype=np.float32).copy()

        del arr
        gc.collect()

    if any(v is None for v in vectors):
        raise ValueError("Failed to reconstruct some vectors from manifest shards")

    stacked = np.stack([v for v in vectors if v is not None]).astype(np.float32, copy=False)
    return video_ids, stacked


def _load_strategy_dataset(
    s3: S3ArtifactStore,
    strategy: str,
    fused_prefix_base: str,
    statuses: set[str],
    metadata_video_ids: Optional[set[str]],
    tmp_dir: Path,
) -> StrategyDataset:
    manifest_key = _resolve_manifest_key(fused_prefix_base=fused_prefix_base, strategy=strategy)
    if not s3.exists(manifest_key):
        raise FileNotFoundError(f"Manifest not found for strategy={strategy}: {manifest_key}")

    local_manifest = tmp_dir / f"{strategy}_manifest.parquet"
    s3.download_file(manifest_key, local_manifest)
    manifest = pd.read_parquet(local_manifest)

    _validate_manifest_columns(manifest, manifest_key)
    raw_rows = int(len(manifest))

    filtered = manifest[manifest["fusion_status"].astype(str).isin(statuses)].copy()
    if metadata_video_ids is not None:
        filtered = filtered[filtered["video_id"].astype(str).isin(metadata_video_ids)].copy()
    filtered_rows = int(len(filtered))

    filtered = _dedupe_latest_by_video_id(filtered)
    if filtered.empty:
        raise ValueError(f"No candidate rows after filters for strategy={strategy}")

    video_ids, vectors = _load_vectors_from_manifest(s3=s3, manifest_df=filtered, tmp_dir=tmp_dir)

    if local_manifest.exists():
        local_manifest.unlink()

    return StrategyDataset(
        strategy=strategy,
        video_ids=video_ids,
        vectors=vectors,
        manifest_rows_raw=raw_rows,
        manifest_rows_filtered=filtered_rows,
    )


def _build_reduced_space(
    dataset: StrategyDataset,
    random_seed: int,
    enable_umap_cluster: bool,
    umap_cluster_dim: int,
    umap_viz_neighbors: int,
) -> ReducedSpace:
    vectors = np.asarray(dataset.vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D vectors for {dataset.strategy}, got {vectors.shape}")
    if vectors.shape[0] < 3:
        raise ValueError(f"Need at least 3 samples for clustering. strategy={dataset.strategy}")

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(vectors)

    pca_dim = min(50, x_scaled.shape[1], max(2, x_scaled.shape[0] - 1))
    if pca_dim < 2:
        raise ValueError(f"Insufficient rank for PCA reduction. strategy={dataset.strategy}")

    pca_cluster = PCA(n_components=pca_dim, random_state=random_seed).fit_transform(x_scaled)

    cluster_matrix = pca_cluster
    cluster_space_name = "pca"

    if enable_umap_cluster and UMAP_AVAILABLE and pca_cluster.shape[0] >= 30 and pca_cluster.shape[1] >= 4:
        n_neighbors = min(30, max(10, pca_cluster.shape[0] // 30))
        n_components = min(max(2, umap_cluster_dim), pca_cluster.shape[1])
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=0.0,
            metric="euclidean",
            random_state=random_seed,
        )
        cluster_matrix = reducer.fit_transform(pca_cluster).astype(np.float32)
        cluster_space_name = f"umap{n_components}"

    pca_viz2d = PCA(n_components=2, random_state=random_seed).fit_transform(x_scaled)

    umap_viz2d: Optional[np.ndarray] = None
    if UMAP_AVAILABLE and pca_cluster.shape[0] >= 15:
        viz_neighbors = max(5, min(int(umap_viz_neighbors), pca_cluster.shape[0] - 1))
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=viz_neighbors,
            min_dist=0.1,
            metric="euclidean",
            random_state=random_seed,
        )
        umap_viz2d = reducer.fit_transform(pca_cluster).astype(np.float32)

    return ReducedSpace(
        strategy=dataset.strategy,
        video_ids=list(dataset.video_ids),
        vectors=vectors,
        cluster_matrix=np.asarray(cluster_matrix, dtype=np.float32),
        cluster_space_name=cluster_space_name,
        pca_cluster=np.asarray(pca_cluster, dtype=np.float32),
        pca_viz2d=np.asarray(pca_viz2d, dtype=np.float32),
        umap_viz2d=umap_viz2d,
        scaler=scaler,
    )


def _tiny_cluster_penalty(labels_collection: Sequence[np.ndarray], k: int, min_cluster_size: int) -> float:
    if k <= 0:
        return 1.0
    penalties: list[float] = []
    for labels in labels_collection:
        counts = np.bincount(labels, minlength=k)
        penalties.append(float(np.mean(counts < min_cluster_size)))
    return float(np.mean(penalties)) if penalties else 1.0


def _stability_score(labels_collection: Sequence[np.ndarray]) -> float:
    if len(labels_collection) <= 1:
        return 1.0

    aris: list[float] = []
    for i in range(len(labels_collection)):
        for j in range(i + 1, len(labels_collection)):
            aris.append(float(adjusted_rand_score(labels_collection[i], labels_collection[j])))
    return float(np.mean(aris)) if aris else 1.0


def _evaluate_strategy_candidates(
    reduced: ReducedSpace,
    k_values: Iterable[int],
    random_seeds: Sequence[int],
    min_cluster_fraction: float,
    max_silhouette_samples: int,
) -> list[dict]:
    x = reduced.cluster_matrix
    n = x.shape[0]
    candidates: list[dict] = []
    sample_size_cap = int(max_silhouette_samples)

    for k in k_values:
        if k < 2 or k >= n:
            continue

        labels_collection: list[np.ndarray] = []
        silhouettes: list[float] = []

        for seed in random_seeds:
            model = KMeans(n_clusters=int(k), random_state=int(seed), n_init=20)
            labels = model.fit_predict(x)
            labels_collection.append(labels)

            # silhouette_score requires at least 2 non-empty clusters.
            if len(np.unique(labels)) < 2:
                silhouettes.append(-1.0)
            else:
                if sample_size_cap > 0 and n > sample_size_cap:
                    silhouettes.append(
                        float(
                            silhouette_score(
                                x,
                                labels,
                                metric="euclidean",
                                sample_size=sample_size_cap,
                                random_state=int(seed),
                            )
                        )
                    )
                else:
                    silhouettes.append(float(silhouette_score(x, labels, metric="euclidean")))

        silhouette_mean = float(np.mean(silhouettes)) if silhouettes else -1.0
        stability = _stability_score(labels_collection)
        min_cluster_size = max(2, int(np.ceil(min_cluster_fraction * n)))
        tiny_penalty = _tiny_cluster_penalty(labels_collection, int(k), min_cluster_size)

        composite = (0.55 * silhouette_mean) + (0.35 * stability) - (0.10 * tiny_penalty)

        candidates.append(
            {
                "strategy": reduced.strategy,
                "k": int(k),
                "n_samples": int(n),
                "feature_dim": int(x.shape[1]),
                "cluster_space": reduced.cluster_space_name,
                "silhouette": silhouette_mean,
                "stability": float(stability),
                "tiny_cluster_penalty": float(tiny_penalty),
                "composite": float(composite),
            }
        )

    return candidates


def _candidate_sort_key(candidate: dict) -> tuple:
    return (
        float(candidate["composite"]),
        float(candidate["silhouette"]),
        float(candidate["stability"]),
        -float(candidate["tiny_cluster_penalty"]),
        -int(candidate["k"]),
    )


def _canonicalize_labels(labels: np.ndarray, features: np.ndarray) -> np.ndarray:
    unique = sorted(int(x) for x in np.unique(labels))
    centroids: dict[int, np.ndarray] = {}

    for label in unique:
        centroids[label] = np.mean(features[labels == label], axis=0)

    ordered_labels = sorted(
        unique,
        key=lambda old: (tuple(np.round(centroids[old], 12).tolist()), int(old)),
    )
    mapping = {old: new for new, old in enumerate(ordered_labels)}
    return np.asarray([mapping[int(x)] for x in labels], dtype=np.int64)


def _render_scatter(points: np.ndarray, labels: np.ndarray, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not MATPLOTLIB_AVAILABLE:
        # Keep artifact paths stable even if matplotlib is unavailable.
        path.write_text("plot skipped: matplotlib unavailable\n", encoding="utf-8")
        return

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(points[:, 0], points[:, 1], c=labels, cmap="tab20", s=10, alpha=0.75)
    plt.colorbar(scatter, label="Cluster")
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _save_latent_outputs(
    reduced: ReducedSpace,
    labels: np.ndarray,
    strategy: str,
    embeddings_dir: Path,
    plots_dir: Path,
) -> dict:
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Backward-compatible filenames + explicit keyed outputs.
    pca20 = reduced.pca_cluster[:, : min(20, reduced.pca_cluster.shape[1])]
    np.save(embeddings_dir / "cluster_pca20.npy", pca20)

    if reduced.cluster_space_name.startswith("umap"):
        np.save(embeddings_dir / "cluster_pca50_umap15.npy", reduced.cluster_matrix)
    else:
        fallback = reduced.pca_cluster[:, : min(15, reduced.pca_cluster.shape[1])]
        np.save(embeddings_dir / "cluster_pca50_umap15.npy", fallback)

    np.savez(
        embeddings_dir / "cluster_reduction.npz",
        video_ids=np.asarray(reduced.video_ids),
        embeddings=np.asarray(reduced.cluster_matrix, dtype=np.float32),
        strategy=np.asarray([strategy]),
        cluster_space=np.asarray([reduced.cluster_space_name]),
    )
    np.savez(
        embeddings_dir / "viz_pca2d.npz",
        video_ids=np.asarray(reduced.video_ids),
        embeddings=np.asarray(reduced.pca_viz2d, dtype=np.float32),
    )

    if reduced.umap_viz2d is not None:
        np.savez(
            embeddings_dir / "viz_umap2d.npz",
            video_ids=np.asarray(reduced.video_ids),
            embeddings=np.asarray(reduced.umap_viz2d, dtype=np.float32),
        )

    pca_plot = plots_dir / "pca_2d.png"
    _render_scatter(
        points=reduced.pca_viz2d,
        labels=labels,
        path=pca_plot,
        title=f"Cluster Visualization (PCA 2D) | strategy={strategy}",
    )

    umap_plot = plots_dir / "umap_2d.png"
    if reduced.umap_viz2d is not None:
        _render_scatter(
            points=reduced.umap_viz2d,
            labels=labels,
            path=umap_plot,
            title=f"Cluster Visualization (UMAP 2D) | strategy={strategy}",
        )
    else:
        # Keep expected artifact path stable even when UMAP is unavailable.
        _render_scatter(
            points=reduced.pca_viz2d,
            labels=labels,
            path=umap_plot,
            title=f"Cluster Visualization (PCA fallback for UMAP) | strategy={strategy}",
        )

    return {
        "pca_plot": str(pca_plot),
        "umap_plot": str(umap_plot),
        "cluster_space": reduced.cluster_space_name,
        "pca_cluster_shape": [int(x) for x in reduced.pca_cluster.shape],
        "cluster_matrix_shape": [int(x) for x in reduced.cluster_matrix.shape],
    }


def _load_metadata_video_ids(metadata_csv: Path) -> set[str]:
    items = load_latest_horizon_rows(csv_path=metadata_csv)
    return {item.video_id for item in items}


def run_clustering(
    *,
    s3: S3ArtifactStore,
    metadata_video_ids: set[str],
    fused_prefix_base: str,
    strategies: Sequence[str],
    statuses: set[str],
    k_values: Sequence[int],
    random_seeds: Sequence[int],
    enable_umap_cluster: bool,
    umap_cluster_dim: int,
    umap_viz_neighbors: int,
    min_cluster_fraction: float,
    max_silhouette_samples: int,
    output_csv: Path,
    diagnostics_json: Path,
    embeddings_dir: Path,
    plots_dir: Path,
) -> dict:
    strategy_summaries: list[dict] = []
    best: Optional[dict] = None
    best_reduced: Optional[ReducedSpace] = None
    candidates: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="cluster_refactor_") as tmp:
        tmp_dir = Path(tmp)

        for strategy in strategies:
            print(f"[cluster] loading strategy={strategy}", flush=True)
            ds = _load_strategy_dataset(
                s3=s3,
                strategy=strategy,
                fused_prefix_base=fused_prefix_base,
                statuses=statuses,
                metadata_video_ids=metadata_video_ids,
                tmp_dir=tmp_dir,
            )
            print(
                f"[cluster] loaded strategy={strategy} rows={len(ds.video_ids)} dim={ds.vectors.shape[1]}",
                flush=True,
            )
            strategy_summaries.append(
                {
                    "strategy": strategy,
                    "manifest_rows_raw": int(ds.manifest_rows_raw),
                    "manifest_rows_filtered": int(ds.manifest_rows_filtered),
                    "cluster_rows": int(len(ds.video_ids)),
                    "vector_dim": int(ds.vectors.shape[1]),
                }
            )

            reduced = _build_reduced_space(
                dataset=ds,
                random_seed=int(random_seeds[0]),
                enable_umap_cluster=enable_umap_cluster,
                umap_cluster_dim=umap_cluster_dim,
                umap_viz_neighbors=umap_viz_neighbors,
            )
            print(
                f"[cluster] reduced strategy={strategy} space={reduced.cluster_space_name} "
                f"shape={list(reduced.cluster_matrix.shape)}",
                flush=True,
            )

            strategy_candidates = _evaluate_strategy_candidates(
                reduced=reduced,
                k_values=k_values,
                random_seeds=random_seeds,
                min_cluster_fraction=min_cluster_fraction,
                max_silhouette_samples=max_silhouette_samples,
            )
            print(
                f"[cluster] evaluated strategy={strategy} candidates={len(strategy_candidates)}",
                flush=True,
            )
            if not strategy_candidates:
                print(f"[cluster] no valid candidates for strategy={strategy}", flush=True)
                del ds
                del reduced
                gc.collect()
                continue

            candidates.extend(strategy_candidates)

            strategy_best = max(strategy_candidates, key=_candidate_sort_key)
            if best is None or _candidate_sort_key(strategy_best) > _candidate_sort_key(best):
                best = strategy_best
                best_reduced = reduced
                print(
                    f"[cluster] current best strategy={strategy} k={int(strategy_best['k'])} "
                    f"composite={float(strategy_best['composite']):.4f}",
                    flush=True,
                )
            else:
                del reduced

            del ds
            gc.collect()

    if not candidates:
        raise ValueError("No valid clustering candidates were produced")
    if best is None or best_reduced is None:
        raise ValueError("No best strategy could be selected from candidates")

    best_strategy = str(best["strategy"])
    best_k = int(best["k"])
    reduced = best_reduced

    final_model = KMeans(n_clusters=best_k, random_state=int(random_seeds[0]), n_init=50)
    final_labels = final_model.fit_predict(reduced.cluster_matrix)
    canonical_labels = _canonicalize_labels(final_labels, reduced.cluster_matrix)

    out_df = pd.DataFrame(
        {
            "video_id": reduced.video_ids,
            "cluster": canonical_labels.astype(int),
            "cluster_id": canonical_labels.astype(int),
            "fusion_strategy": best_strategy,
            "k_selected": int(best_k),
        }
    )
    out_df = out_df.sort_values(["cluster", "video_id"]).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    latent_meta = _save_latent_outputs(
        reduced=reduced,
        labels=canonical_labels,
        strategy=best_strategy,
        embeddings_dir=embeddings_dir,
        plots_dir=plots_dir,
    )

    diagnostics_payload = {
        "generated_at": _utc_now_iso(),
        "selected": {
            "fusion_strategy": best_strategy,
            "k": best_k,
            "cluster_space": reduced.cluster_space_name,
            "candidate_metrics": best,
        },
        "strategies": strategy_summaries,
        "candidates": candidates,
        "scoring": {
            "max_silhouette_samples": int(max_silhouette_samples),
        },
        "artifacts": {
            "cluster_results_csv": str(output_csv),
            "latent_embeddings_dir": str(embeddings_dir),
            "latent_plots_dir": str(plots_dir),
            **latent_meta,
        },
        "row_counts": {
            "clustered_rows": int(len(out_df)),
        },
    }

    diagnostics_json.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_json.write_text(json.dumps(diagnostics_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Clustering summary")
    print(f"selected_strategy: {best_strategy}")
    print(f"selected_k: {best_k}")
    print(f"cluster_space: {reduced.cluster_space_name}")
    print(f"clustered_rows: {len(out_df)}")
    print(f"cluster_results_csv: {output_csv}")
    print(f"diagnostics_json: {diagnostics_json}")

    return diagnostics_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster fused embeddings using video_id-native manifests")
    parser.add_argument("--metadata_csv", default=str(DEFAULT_METADATA_CSV))
    parser.add_argument("--s3_bucket", default=os.getenv("S3_BUCKET", ""))
    parser.add_argument("--s3_region", default=os.getenv("AWS_REGION", ""))

    parser.add_argument("--fused_prefix_base", default=DEFAULT_FUSED_PREFIX_BASE)
    parser.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES))
    parser.add_argument("--fusion_statuses", default=",".join(DEFAULT_TERMINAL_FUSION_STATUSES))

    parser.add_argument("--strategy_selection", choices=("auto",), default="auto")
    parser.add_argument("--k_min", type=int, default=6)
    parser.add_argument("--k_max", type=int, default=16)
    parser.add_argument("--random_seeds", default=",".join(str(x) for x in DEFAULT_RANDOM_SEEDS))
    parser.add_argument("--min_cluster_fraction", type=float, default=0.01)
    parser.add_argument(
        "--max_silhouette_samples",
        type=int,
        default=4000,
        help="Max sample size for silhouette scoring per candidate (<=0 means full dataset).",
    )

    parser.add_argument("--enable_umap_cluster", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--umap_cluster_dim", type=int, default=15)
    parser.add_argument("--umap_viz_neighbors", type=int, default=15)

    parser.add_argument("--output_csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--diagnostics_json", default=str(DEFAULT_DIAGNOSTICS_JSON))
    parser.add_argument("--embeddings_dir", default=str(DEFAULT_EMBEDDING_DIR))
    parser.add_argument("--plots_dir", default=str(DEFAULT_PLOTS_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.s3_bucket:
        raise ValueError("--s3_bucket is required (or set S3_BUCKET env var)")

    if args.k_min < 2:
        raise ValueError("--k_min must be >= 2")
    if args.k_max < args.k_min:
        raise ValueError("--k_max must be >= k_min")

    strategies = _parse_csv_list(args.strategies)
    if not strategies:
        raise ValueError("At least one strategy is required")

    statuses = _parse_csv_set(args.fusion_statuses)
    if not statuses:
        raise ValueError("At least one fusion status is required")

    random_seeds = _parse_int_csv(args.random_seeds)
    k_values = list(range(int(args.k_min), int(args.k_max) + 1))

    s3 = S3ArtifactStore(bucket=args.s3_bucket, region=args.s3_region)
    metadata_video_ids = _load_metadata_video_ids(Path(args.metadata_csv))

    payload = run_clustering(
        s3=s3,
        metadata_video_ids=metadata_video_ids,
        fused_prefix_base=args.fused_prefix_base,
        strategies=strategies,
        statuses=statuses,
        k_values=k_values,
        random_seeds=random_seeds,
        enable_umap_cluster=bool(args.enable_umap_cluster),
        umap_cluster_dim=int(args.umap_cluster_dim),
        umap_viz_neighbors=int(args.umap_viz_neighbors),
        min_cluster_fraction=float(args.min_cluster_fraction),
        max_silhouette_samples=int(args.max_silhouette_samples),
        output_csv=Path(args.output_csv),
        diagnostics_json=Path(args.diagnostics_json),
        embeddings_dir=Path(args.embeddings_dir),
        plots_dir=Path(args.plots_dir),
    )

    # Keep CLI output concise but informative for scripting logs.
    print(
        json.dumps(
            {
                "selected_strategy": payload["selected"]["fusion_strategy"],
                "selected_k": payload["selected"]["k"],
                "clustered_rows": payload["row_counts"]["clustered_rows"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
