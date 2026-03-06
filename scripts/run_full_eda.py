"""Generate a full exploratory data analysis report for the virality dataset.

Outputs:
- Markdown report
- Structured CSV summaries
- Core plots
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except Exception:
    matplotlib = None
    plt = None
    MATPLOTLIB_AVAILABLE = False

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_METADATA_CSV = REPO_ROOT / "Data" / "raw" / "Metadata" / "shorts_metadata_horizon.csv"
DEFAULT_CLUSTER_CSV = REPO_ROOT / "Unsup_Cluster" / "cluster_results.csv"
DEFAULT_INTERPRETATION_CSV = REPO_ROOT / "Interpretation" / "interpretation.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "EDA" / "output"
DEFAULT_FUSION_STATE_DBS = [
    REPO_ROOT / "state" / "fusion_concat.sqlite",
    REPO_ROOT / "state" / "fusion_sum_pool.sqlite",
    REPO_ROOT / "state" / "fusion_max_pool.sqlite",
]
DEFAULT_FUSION_SUCCESS_STATUSES = ["success_full", "success_text_placeholder"]

NUMERIC_CANDIDATES = [
    "channel_description_length",
    "channel_subscriber_count",
    "channel_video_count",
    "channel_view_count",
    "channel_age_days",
    "title_length",
    "description_length",
    "emoji_count",
    "hashtag_count",
    "duration_seconds",
    "thumb_width",
    "thumb_height",
    "thumb_aspect_ratio",
    "view_count",
    "like_count",
    "comment_count",
    "age_days",
    "views_per_day",
    "likes_per_day",
    "comments_per_day",
    "likes_per_view",
    "comments_per_view",
    "views_per_hour",
    "virality_score",
    "horizon_days",
    "horizon_view_count",
    "channel_median_shorts_view_count",
]

CATEGORICAL_CANDIDATES = [
    "query",
    "category_type",
    "default_language",
    "default_audio_language",
    "dimension",
    "definition",
    "projection",
    "privacy_status",
    "upload_status",
    "license",
    "horizon_label_type",
]

BOOLEANISH_CANDIDATES = [
    "channel_hidden_subscriber_count",
    "has_hashtags",
    "has_shorts_hashtag",
    "has_clickbait_words",
    "caption_available",
    "licensed_content",
    "is_vertical_thumb",
    "shorts_by_duration",
    "shorts_by_hashtag",
    "is_short",
    "embeddable",
    "madeForKids",
    "publicStatsViewable",
    "likes_hidden",
    "comments_disabled",
    "stats_hidden",
    "too_new_for_rates",
]

DATETIME_CANDIDATES = ["captured_at", "published_at", "channel_created_at"]


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _normalize_boolish(series: pd.Series) -> pd.Series:
    as_str = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0,
        "yes": 1,
        "no": 0,
    }
    out = as_str.map(mapping)
    return pd.to_numeric(out, errors="coerce")


def _ensure_dirs(base: Path) -> dict[str, Path]:
    paths = {
        "base": base,
        "plots": base / "plots",
        "tables": base / "tables",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def _save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _parse_csv_items(raw: str) -> list[str]:
    return [x.strip() for x in (raw or "").split(",") if x.strip()]


def _optional_existing_path(raw: Optional[str]) -> Optional[Path]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    p = Path(s)
    if p.exists() and p.is_file():
        return p
    return None


def _write_plot_placeholder(path: Path, reason: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"plot skipped: {reason}\n", encoding="utf-8")


def _plot_missingness(missing_df: pd.DataFrame, out_path: Path, top_n: int = 20) -> None:
    if not MATPLOTLIB_AVAILABLE:
        _write_plot_placeholder(out_path, "matplotlib unavailable")
        return
    top = missing_df.head(top_n).copy()
    plt.figure(figsize=(11, 7))
    plt.barh(top["column"][::-1], top["missing_pct"][::-1])
    plt.xlabel("Missing %")
    plt.title(f"Top {top_n} Missing Columns")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_hist(series: pd.Series, out_path: Path, title: str, bins: int = 50, log1p: bool = False) -> None:
    if not MATPLOTLIB_AVAILABLE:
        _write_plot_placeholder(out_path, "matplotlib unavailable")
        return
    clean = series.dropna()
    if clean.empty:
        _write_plot_placeholder(out_path, "no numeric data")
        return
    data = np.log1p(clean) if log1p else clean
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel("log1p(value)" if log1p else "value")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_top_categories(series: pd.Series, out_path: Path, title: str, top_n: int = 15) -> None:
    if not MATPLOTLIB_AVAILABLE:
        _write_plot_placeholder(out_path, "matplotlib unavailable")
        return
    counts = series.fillna("<NA>").astype(str).value_counts().head(top_n)
    if counts.empty:
        _write_plot_placeholder(out_path, "no categorical data")
        return
    plt.figure(figsize=(11, 6))
    plt.bar(range(len(counts)), counts.values)
    plt.xticks(range(len(counts)), counts.index, rotation=45, ha="right")
    plt.title(title)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_daily_counts(df: pd.DataFrame, dt_col: str, out_path: Path, title: str) -> None:
    if not MATPLOTLIB_AVAILABLE:
        _write_plot_placeholder(out_path, "matplotlib unavailable")
        return
    if dt_col not in df.columns:
        _write_plot_placeholder(out_path, f"missing datetime column: {dt_col}")
        return
    dt = _safe_datetime(df[dt_col])
    daily = dt.dropna().dt.date.value_counts().sort_index()
    if daily.empty:
        _write_plot_placeholder(out_path, "no valid datetime values")
        return
    plt.figure(figsize=(12, 6))
    plt.plot(daily.index.astype(str), daily.values)
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.ylabel("rows")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_corr_heatmap(corr: pd.DataFrame, out_path: Path) -> None:
    if not MATPLOTLIB_AVAILABLE:
        _write_plot_placeholder(out_path, "matplotlib unavailable")
        return
    if corr.empty:
        _write_plot_placeholder(out_path, "no correlation matrix")
        return
    fig, ax = plt.subplots(figsize=(12, 10))
    mat = ax.imshow(corr.values, interpolation="nearest", aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    ax.set_title("Correlation Heatmap")
    fig.colorbar(mat, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _latest_by_video_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "video_id" not in out.columns:
        return out
    cap = _safe_datetime(out["captured_at"]) if "captured_at" in out.columns else pd.Series(pd.NaT, index=out.index)
    out["_captured_at_parsed"] = cap
    out = out.sort_values(["video_id", "_captured_at_parsed"], na_position="last")
    out = out.drop_duplicates(subset=["video_id"], keep="last")
    return out.drop(columns=["_captured_at_parsed"], errors="ignore")


def _load_fusion_state_tables(sqlite_paths: Iterable[Path]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for p in sqlite_paths:
        if not p.exists():
            continue
        strategy = p.stem.replace("fusion_", "")
        try:
            with sqlite3.connect(str(p)) as conn:
                frame = pd.read_sql_query("SELECT video_id, status FROM processed_items", conn)
            if "video_id" not in frame.columns or "status" not in frame.columns:
                continue
            frame["video_id"] = frame["video_id"].astype(str)
            frame["status"] = frame["status"].astype(str)
            out[strategy] = frame
        except Exception:
            continue
    return out


def _build_training_coverage(
    metadata_df: pd.DataFrame,
    fusion_state: dict[str, pd.DataFrame],
    success_statuses: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    fusion_status_rows: list[dict] = []
    funnel_rows: list[dict] = []
    horizon_status_rows: list[dict] = []

    if "video_id" not in metadata_df.columns:
        return (
            pd.DataFrame(columns=["strategy", "status", "count"]),
            pd.DataFrame(
                columns=[
                    "strategy",
                    "horizon_days",
                    "rows_after_horizon_filter",
                    "rows_after_latest_dedupe",
                    "rows_after_fusion_success",
                    "dropped_by_fusion_join",
                    "success_rate_after_dedupe_pct",
                ]
            ),
            pd.DataFrame(columns=["strategy", "horizon_days", "status", "count"]),
            {"available": False, "reason": "video_id_missing"},
        )

    horizon_values = []
    if "horizon_days" in metadata_df.columns:
        horizon_values = [h for h in metadata_df["horizon_days"].dropna().unique().tolist()]
    horizon_values = sorted(horizon_values)

    if not fusion_state:
        return (
            pd.DataFrame(columns=["strategy", "status", "count"]),
            pd.DataFrame(
                columns=[
                    "strategy",
                    "horizon_days",
                    "rows_after_horizon_filter",
                    "rows_after_latest_dedupe",
                    "rows_after_fusion_success",
                    "dropped_by_fusion_join",
                    "success_rate_after_dedupe_pct",
                ]
            ),
            pd.DataFrame(columns=["strategy", "horizon_days", "status", "count"]),
            {
                "available": False,
                "reason": "no_fusion_state",
                "detected_strategies": [],
                "success_statuses": sorted(success_statuses),
            },
        )

    for strategy, state_df in fusion_state.items():
        status_counts = state_df["status"].value_counts(dropna=False)
        for status, count in status_counts.items():
            fusion_status_rows.append({"strategy": strategy, "status": str(status), "count": int(count)})

        for horizon in horizon_values:
            filtered = metadata_df[metadata_df["horizon_days"] == horizon].copy()
            dedup = _latest_by_video_id(filtered)

            joined = dedup[["video_id"]].astype(str).merge(state_df, on="video_id", how="left")
            joined_status = joined["status"].fillna("missing_in_fusion_state")
            success_count = int(joined_status.isin(success_statuses).sum())

            funnel_rows.append(
                {
                    "strategy": strategy,
                    "horizon_days": horizon,
                    "rows_after_horizon_filter": int(len(filtered)),
                    "rows_after_latest_dedupe": int(len(dedup)),
                    "rows_after_fusion_success": success_count,
                    "dropped_by_fusion_join": int(len(dedup) - success_count),
                    "success_rate_after_dedupe_pct": float((success_count / len(dedup)) * 100.0) if len(dedup) > 0 else 0.0,
                }
            )

            hs = joined_status.value_counts(dropna=False)
            for status, count in hs.items():
                horizon_status_rows.append(
                    {
                        "strategy": strategy,
                        "horizon_days": horizon,
                        "status": str(status),
                        "count": int(count),
                    }
                )

    fusion_status_df = pd.DataFrame(fusion_status_rows).sort_values(["strategy", "count"], ascending=[True, False])
    funnel_df = pd.DataFrame(funnel_rows).sort_values(["strategy", "horizon_days"]).reset_index(drop=True)
    horizon_status_df = pd.DataFrame(horizon_status_rows).sort_values(["strategy", "horizon_days", "count"], ascending=[True, True, False]).reset_index(drop=True)

    summary = {
        "available": True,
        "detected_strategies": sorted(fusion_state.keys()),
        "success_statuses": sorted(success_statuses),
        "n_fusion_state_rows_total": int(sum(len(v) for v in fusion_state.values())),
    }
    return fusion_status_df, funnel_df, horizon_status_df, summary


def run_eda(
    *,
    metadata_csv: Path,
    out_dir: Path,
    cluster_csv: Optional[Path],
    interpretation_csv: Optional[Path],
    top_n: int,
    fusion_state_dbs: list[Path],
    fusion_success_statuses: set[str],
) -> dict:
    dirs = _ensure_dirs(out_dir)

    df = pd.read_csv(metadata_csv, low_memory=False)
    latest = _latest_by_video_id(df)

    overview = {
        "metadata_csv": str(metadata_csv),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "unique_video_id": int(df["video_id"].nunique()) if "video_id" in df.columns else 0,
        "duplicate_video_id_rows": int(len(df) - df["video_id"].nunique()) if "video_id" in df.columns else 0,
        "rows_latest_per_video": int(len(latest)),
        "horizon_days_distribution": df["horizon_days"].value_counts(dropna=False).to_dict()
        if "horizon_days" in df.columns
        else {},
    }

    # Missingness table
    missing_df = (
        pd.DataFrame(
            {
                "column": df.columns,
                "missing_count": [int(df[c].isna().sum()) for c in df.columns],
                "missing_pct": [float(df[c].isna().mean() * 100.0) for c in df.columns],
            }
        )
        .sort_values("missing_pct", ascending=False)
        .reset_index(drop=True)
    )
    _save_table(missing_df, dirs["tables"] / "missingness.csv")

    # Numeric profile
    numeric_data: dict[str, pd.Series] = {}
    for col in NUMERIC_CANDIDATES:
        if col in df.columns:
            numeric_data[col] = _safe_numeric(df[col])
    for col in BOOLEANISH_CANDIDATES:
        if col in df.columns:
            numeric_data[col] = _normalize_boolish(df[col])

    numeric_df = pd.DataFrame(numeric_data)
    if numeric_df.shape[1] > 0:
        numeric_summary = numeric_df.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).transpose()
        numeric_summary = numeric_summary.reset_index().rename(columns={"index": "column"})
    else:
        numeric_summary = pd.DataFrame(
            columns=[
                "column",
                "count",
                "mean",
                "std",
                "min",
                "1%",
                "5%",
                "25%",
                "50%",
                "75%",
                "95%",
                "99%",
                "max",
            ]
        )
    _save_table(numeric_summary, dirs["tables"] / "numeric_summary.csv")

    # Correlations on stable subset
    corr_cols = [
        c
        for c in [
            "view_count",
            "like_count",
            "comment_count",
            "horizon_view_count",
            "virality_score",
            "duration_seconds",
            "views_per_day",
            "likes_per_view",
            "comments_per_view",
        ]
        if c in numeric_df.columns
    ]
    corr_df = numeric_df[corr_cols].corr() if corr_cols else pd.DataFrame()
    corr_out = corr_df.reset_index().rename(columns={"index": "column"}) if not corr_df.empty else pd.DataFrame(columns=["column"])
    _save_table(corr_out, dirs["tables"] / "correlations.csv")

    # Categorical profile
    cat_rows: list[dict] = []
    for col in CATEGORICAL_CANDIDATES:
        if col not in df.columns:
            continue
        s = df[col].astype(str)
        vc = s.fillna("<NA>").value_counts(dropna=False)
        top_values = " | ".join(f"{k}:{v}" for k, v in vc.head(5).items())
        cat_rows.append(
            {
                "column": col,
                "unique_values": int(s.nunique(dropna=True)),
                "missing_pct": float(df[col].isna().mean() * 100.0),
                "top_5": top_values,
            }
        )
    categorical_summary = pd.DataFrame(cat_rows)
    if not categorical_summary.empty:
        categorical_summary = categorical_summary.sort_values("unique_values", ascending=False)
    else:
        categorical_summary = pd.DataFrame(columns=["column", "unique_values", "missing_pct", "top_5"])
    _save_table(categorical_summary, dirs["tables"] / "categorical_summary.csv")

    # Horizon summary
    horizon_summary = pd.DataFrame(columns=["horizon_days", "rows", "unique_videos", "mean_horizon_view_count", "mean_virality_score", "mean_view_count", "mean_likes_per_view", "mean_comments_per_view"])
    if "horizon_days" in df.columns:
        rows = []
        for horizon, grp in df.groupby("horizon_days", dropna=False):
            row = {
                "horizon_days": horizon,
                "rows": int(len(grp)),
                "unique_videos": int(grp["video_id"].astype(str).nunique()) if "video_id" in grp.columns else None,
            }
            for col in ["horizon_view_count", "virality_score", "view_count", "likes_per_view", "comments_per_view"]:
                key = f"mean_{col}"
                row[key] = float(_safe_numeric(grp[col]).mean()) if col in grp.columns else None
            rows.append(row)
        horizon_summary = pd.DataFrame(rows).sort_values("horizon_days")
        _save_table(horizon_summary, dirs["tables"] / "horizon_summary.csv")

    # Optional training-coverage analysis from local fusion state DBs.
    training_coverage_summary = {}
    fusion_status_df = pd.DataFrame(columns=["strategy", "status", "count"])
    training_funnel_df = pd.DataFrame(
        columns=[
            "strategy",
            "horizon_days",
            "rows_after_horizon_filter",
            "rows_after_latest_dedupe",
            "rows_after_fusion_success",
            "dropped_by_fusion_join",
            "success_rate_after_dedupe_pct",
        ]
    )
    horizon_fusion_status_df = pd.DataFrame(columns=["strategy", "horizon_days", "status", "count"])
    if fusion_state_dbs:
        fusion_state = _load_fusion_state_tables(fusion_state_dbs)
        (
            fusion_status_df,
            training_funnel_df,
            horizon_fusion_status_df,
            training_coverage_summary,
        ) = _build_training_coverage(
            metadata_df=df,
            fusion_state=fusion_state,
            success_statuses=fusion_success_statuses,
        )
        _save_table(fusion_status_df, dirs["tables"] / "fusion_status_counts.csv")
        _save_table(training_funnel_df, dirs["tables"] / "training_row_funnel.csv")
        _save_table(horizon_fusion_status_df, dirs["tables"] / "horizon_fusion_status_counts.csv")
    else:
        training_coverage_summary = {
            "available": False,
            "reason": "fusion_state_dbs_not_provided",
            "detected_strategies": [],
            "success_statuses": sorted(fusion_success_statuses),
        }

    # Optional cluster join
    cluster_join_summary = {}
    cluster_summary_df = pd.DataFrame()
    if cluster_csv and cluster_csv.exists() and "video_id" in latest.columns:
        clusters = pd.read_csv(cluster_csv)
        merged = latest.merge(clusters, on="video_id", how="left")
        coverage = float(merged["cluster"].notna().mean() * 100.0) if "cluster" in merged.columns else 0.0
        cluster_join_summary = {
            "cluster_csv": str(cluster_csv),
            "rows_in_cluster_csv": int(len(clusters)),
            "coverage_pct_on_latest_metadata": coverage,
        }
        if "cluster" in merged.columns:
            cluster_summary_df = (
                merged.groupby("cluster", dropna=True)
                .agg(
                    n_videos=("video_id", "nunique"),
                    mean_horizon_view_count=("horizon_view_count", "mean") if "horizon_view_count" in merged.columns else ("video_id", "size"),
                    mean_virality_score=("virality_score", "mean") if "virality_score" in merged.columns else ("video_id", "size"),
                )
                .reset_index()
                .sort_values("n_videos", ascending=False)
            )
            _save_table(cluster_summary_df, dirs["tables"] / "cluster_summary.csv")

    # Optional interpretation profile
    interpretation_summary = {}
    interpretation_df = pd.DataFrame()
    if interpretation_csv and interpretation_csv.exists():
        interpretation_df = pd.read_csv(interpretation_csv)
        interpretation_summary = {
            "interpretation_csv": str(interpretation_csv),
            "rows": int(len(interpretation_df)),
            "columns": int(len(interpretation_df.columns)),
            "video_available_pct": float(interpretation_df["video_available"].mean() * 100.0)
            if "video_available" in interpretation_df.columns and len(interpretation_df) > 0
            else None,
            "audio_available_pct": float(interpretation_df["audio_available"].mean() * 100.0)
            if "audio_available" in interpretation_df.columns and len(interpretation_df) > 0
            else None,
        }
        if "cluster" in interpretation_df.columns:
            num_interp_cols = [
                c
                for c in ["motion_mean", "cut_rate_per_min", "audio_rms_mean", "audio_rms_std", "visual_density"]
                if c in interpretation_df.columns
            ]
            if num_interp_cols:
                interp_cluster = interpretation_df.groupby("cluster", dropna=False)[num_interp_cols].mean().reset_index()
                _save_table(interp_cluster, dirs["tables"] / "interpretation_cluster_means.csv")

    # Plots
    _plot_missingness(missing_df=missing_df, out_path=dirs["plots"] / "missingness_top20.png", top_n=min(20, len(missing_df)))

    if "horizon_days" in df.columns:
        _plot_hist(_safe_numeric(df["horizon_days"]), dirs["plots"] / "horizon_days_distribution.png", "Horizon Days Distribution", bins=20)

    if "horizon_view_count" in numeric_df.columns:
        _plot_hist(numeric_df["horizon_view_count"], dirs["plots"] / "horizon_view_count_log_hist.png", "Horizon View Count (log1p)", bins=60, log1p=True)

    if "virality_score" in numeric_df.columns:
        _plot_hist(numeric_df["virality_score"], dirs["plots"] / "virality_score_hist.png", "Virality Score Distribution", bins=60)

    if "query" in df.columns:
        _plot_top_categories(df["query"], dirs["plots"] / "top_queries.png", "Top Queries", top_n=top_n)

    _plot_daily_counts(df, "captured_at", dirs["plots"] / "daily_captured_rows.png", "Rows Collected per Captured Day")

    if not corr_df.empty:
        _plot_corr_heatmap(corr_df, dirs["plots"] / "correlation_heatmap.png")
    else:
        _plot_corr_heatmap(corr_df, dirs["plots"] / "correlation_heatmap.png")

    cluster_sizes_plot_written = False
    if not cluster_summary_df.empty and "cluster" in cluster_summary_df.columns:
        if not MATPLOTLIB_AVAILABLE:
            _write_plot_placeholder(dirs["plots"] / "cluster_sizes.png", "matplotlib unavailable")
            cluster_sizes_plot_written = True
        else:
            plt.figure(figsize=(10, 6))
            plt.bar(cluster_summary_df["cluster"].astype(str), cluster_summary_df["n_videos"])
            plt.title("Cluster Size Distribution")
            plt.xlabel("cluster")
            plt.ylabel("n_videos")
            plt.tight_layout()
            plt.savefig(dirs["plots"] / "cluster_sizes.png")
            plt.close()
            cluster_sizes_plot_written = True

    # Markdown report
    report_path = dirs["base"] / "eda_report.md"
    report_lines = [
        "# Full Dataset EDA",
        "",
        f"Generated at: `{pd.Timestamp.utcnow().isoformat()}`",
        "",
        "## Dataset Snapshot",
        f"- Metadata file: `{metadata_csv}`",
        f"- Rows: `{overview['rows']}`",
        f"- Columns: `{overview['columns']}`",
        f"- Unique video_id: `{overview['unique_video_id']}`",
        f"- Duplicate video_id rows: `{overview['duplicate_video_id_rows']}`",
        f"- Rows in latest-per-video view: `{overview['rows_latest_per_video']}`",
        "",
        "## Data Quality",
        "- Missingness table: `tables/missingness.csv`",
        "- Numeric summary: `tables/numeric_summary.csv`",
        "- Categorical summary: `tables/categorical_summary.csv`",
        "",
        "## Target / Outcome Signals",
        "- Horizon summary: `tables/horizon_summary.csv`",
        "- Correlations: `tables/correlations.csv`",
        "",
        "## Plots",
        "- `plots/missingness_top20.png`",
        "- `plots/horizon_days_distribution.png`",
        "- `plots/horizon_view_count_log_hist.png`",
        "- `plots/virality_score_hist.png`",
        "- `plots/top_queries.png`",
        "- `plots/daily_captured_rows.png`",
        "- `plots/correlation_heatmap.png`",
    ]

    if cluster_join_summary:
        report_lines.extend(
            [
                "",
                "## Cluster Join Coverage",
                f"- Cluster file: `{cluster_join_summary['cluster_csv']}`",
                f"- Rows in cluster file: `{cluster_join_summary['rows_in_cluster_csv']}`",
                f"- Coverage on latest metadata: `{cluster_join_summary['coverage_pct_on_latest_metadata']:.2f}%`",
                "- Cluster summary table: `tables/cluster_summary.csv`",
            ]
        )
        if cluster_sizes_plot_written:
            report_lines.append("- Cluster size plot: `plots/cluster_sizes.png`")
        else:
            report_lines.append("- Cluster size plot: not generated (no matched cluster rows).")

    if interpretation_summary:
        report_lines.extend(
            [
                "",
                "## Interpretation Coverage",
                f"- Interpretation file: `{interpretation_summary['interpretation_csv']}`",
                f"- Rows: `{interpretation_summary['rows']}`",
                f"- Columns: `{interpretation_summary['columns']}`",
                f"- video_available %: `{interpretation_summary['video_available_pct']}`",
                f"- audio_available %: `{interpretation_summary['audio_available_pct']}`",
                "- Cluster means (interpretation): `tables/interpretation_cluster_means.csv`",
            ]
        )

    if training_coverage_summary:
        report_lines.extend(
            [
                "",
                "## Training Readiness Coverage",
                f"- Success statuses used: `{', '.join(training_coverage_summary.get('success_statuses', []))}`",
                f"- Fusion strategies detected: `{', '.join(training_coverage_summary.get('detected_strategies', []))}`",
                f"- Coverage table: `tables/training_row_funnel.csv`",
                f"- Fusion statuses table: `tables/fusion_status_counts.csv`",
                f"- Horizon x fusion status table: `tables/horizon_fusion_status_counts.csv`",
            ]
        )

    report_lines.extend(
        [
            "",
            "## Suggested Follow-ups",
            "- Investigate high-missingness columns before model usage.",
            "- Re-check cluster join coverage after re-running video_id-native clustering outputs.",
            "- For skewed count features, use log1p transforms in downstream modeling.",
        ]
    )

    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    overview_payload = {
        "overview": overview,
        "cluster_join_summary": cluster_join_summary,
        "interpretation_summary": interpretation_summary,
        "training_coverage_summary": training_coverage_summary,
        "artifacts": {
            "report": str(report_path),
            "tables_dir": str(dirs["tables"]),
            "plots_dir": str(dirs["plots"]),
        },
    }

    (dirs["base"] / "overview.json").write_text(json.dumps(overview_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("EDA summary")
    print(f"metadata_rows: {overview['rows']}")
    print(f"metadata_columns: {overview['columns']}")
    print(f"unique_video_id: {overview['unique_video_id']}")
    print(f"rows_latest_per_video: {overview['rows_latest_per_video']}")
    if cluster_join_summary:
        print(f"cluster_coverage_pct: {cluster_join_summary['coverage_pct_on_latest_metadata']:.2f}")
    if interpretation_summary:
        print(f"interpretation_rows: {interpretation_summary['rows']}")
    if training_coverage_summary.get("available"):
        print(f"fusion_strategies_detected: {','.join(training_coverage_summary.get('detected_strategies', []))}")
    print(f"report: {report_path}")

    return overview_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full EDA and emit report/tables/plots")
    parser.add_argument("--metadata_csv", default=str(DEFAULT_METADATA_CSV))
    parser.add_argument("--cluster_csv", default=str(DEFAULT_CLUSTER_CSV))
    parser.add_argument("--interpretation_csv", default=str(DEFAULT_INTERPRETATION_CSV))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--top_n_categories", type=int, default=15)
    parser.add_argument(
        "--fusion_state_dbs",
        default=",".join(str(p) for p in DEFAULT_FUSION_STATE_DBS),
        help="Comma-separated sqlite paths (processed_items with video_id/status) for training coverage analysis.",
    )
    parser.add_argument(
        "--fusion_success_statuses",
        default=",".join(DEFAULT_FUSION_SUCCESS_STATUSES),
        help="Comma-separated statuses treated as train-eligible in fusion coverage.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    metadata_csv = Path(args.metadata_csv)
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")

    cluster_csv = _optional_existing_path(args.cluster_csv)
    interpretation_csv = _optional_existing_path(args.interpretation_csv)
    raw_fusion_paths = [Path(p) for p in _parse_csv_items(args.fusion_state_dbs)]
    fusion_state_dbs = [p for p in raw_fusion_paths if p.exists() and p.is_file()]
    fusion_success_statuses = set(_parse_csv_items(args.fusion_success_statuses)) or set(DEFAULT_FUSION_SUCCESS_STATUSES)

    run_eda(
        metadata_csv=metadata_csv,
        out_dir=Path(args.out_dir),
        cluster_csv=cluster_csv,
        interpretation_csv=interpretation_csv,
        top_n=max(5, int(args.top_n_categories)),
        fusion_state_dbs=fusion_state_dbs,
        fusion_success_statuses=fusion_success_statuses,
    )


if __name__ == "__main__":
    main()
