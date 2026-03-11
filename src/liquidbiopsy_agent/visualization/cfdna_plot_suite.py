from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from liquidbiopsy_agent.utils.storage import resolve_data_path


sns.set_style("whitegrid")


def _to_data_path(path_value: str | Path, *, path_kind: str, must_exist: bool) -> Path:
    return resolve_data_path(path_value, path_kind=path_kind, must_exist=must_exist)


def _read_table(path: Path) -> pd.DataFrame:
    suffixes = [s.lower() for s in path.suffixes]
    if ".parquet" in suffixes or path.suffix.lower() in {".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _load_label_map(
    labels_table: Path | None,
    *,
    sample_id_col: str,
    label_col: str,
) -> tuple[pd.DataFrame | None, dict[str, str]]:
    if labels_table is None:
        return None, {}
    labels_df = _read_table(labels_table)
    if sample_id_col not in labels_df.columns:
        raise ValueError(f"labels_table missing sample id column: {sample_id_col}")
    if label_col not in labels_df.columns:
        raise ValueError(f"labels_table missing label column: {label_col}")
    labels_df = labels_df.dropna(subset=[sample_id_col, label_col]).copy()
    labels_df[sample_id_col] = labels_df[sample_id_col].astype(str)
    labels_df[label_col] = labels_df[label_col].astype(str)
    label_map = dict(zip(labels_df[sample_id_col].tolist(), labels_df[label_col].tolist()))
    return labels_df, label_map


def _load_cfdna_feature_map(cfdna_features_pt: Path) -> dict[str, np.ndarray]:
    payload = torch.load(cfdna_features_pt, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"cfDNA feature PT must be a dict(sample_id -> feature), got {type(payload)}")
    out: dict[str, np.ndarray] = {}
    for sid, value in payload.items():
        arr = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            continue
        out[str(sid)] = arr
    if not out:
        raise ValueError(f"No usable sample vectors in: {cfdna_features_pt}")
    dims = sorted({v.shape[0] for v in out.values()})
    if len(dims) != 1:
        raise ValueError(f"Inconsistent feature dimensions in cfDNA PT: {dims}")
    return out


def _zscore_columns(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (x - mean) / std


def _pca_2d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x.ndim != 2:
        raise ValueError(f"PCA input must be 2D, got {x.shape}")
    centered = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    n_comp = min(2, vt.shape[0])
    basis = vt[:n_comp].T
    proj = centered @ basis
    if n_comp < 2:
        proj = np.concatenate([proj, np.zeros((proj.shape[0], 2 - n_comp), dtype=proj.dtype)], axis=1)
    return proj.astype(np.float32), basis.astype(np.float32)


def plot_feature_space_with_hyperplane(
    *,
    cfdna_features_pt: str | Path,
    output_dir: str | Path,
    labels_table: str | Path | None = None,
    sample_id_col: str = "sample_id",
    label_col: str = "her2_status",
    random_seed: int = 42,
) -> dict[str, Any]:
    out_dir = _to_data_path(output_dir, path_kind="cfDNA viz output dir", must_exist=False)
    out_dir.mkdir(parents=True, exist_ok=True)

    features_pt = _to_data_path(cfdna_features_pt, path_kind="cfDNA feature PT", must_exist=True)
    labels_path = (
        _to_data_path(labels_table, path_kind="cfDNA labels table", must_exist=True)
        if labels_table is not None
        else None
    )
    _, label_map = _load_label_map(
        labels_path,
        sample_id_col=sample_id_col,
        label_col=label_col,
    )
    feature_map = _load_cfdna_feature_map(features_pt)
    sample_ids = sorted(feature_map.keys())
    x = np.stack([feature_map[sid] for sid in sample_ids], axis=0)
    x = _zscore_columns(x)
    proj, _ = _pca_2d(x)

    labels = [label_map.get(sid, "Unknown") for sid in sample_ids]
    uniq_labels = sorted(set(labels))
    label_ids = np.array([uniq_labels.index(v) for v in labels], dtype=np.int64)

    projected_df = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "pc1": proj[:, 0],
            "pc2": proj[:, 1],
            "label": labels,
            "label_id": label_ids,
        }
    )

    clf_summary: dict[str, Any] = {
        "hyperplane_drawn": False,
        "reason": None,
        "acc_2d": None,
        "auc_2d": None,
    }
    decision_score = np.full((len(sample_ids),), np.nan, dtype=np.float32)

    fig_path = out_dir / "cfdna_feature_space__pca2d__linear_hyperplane.png"
    plt.figure(figsize=(8, 6))
    palette = sns.color_palette("tab10", n_colors=max(1, len(uniq_labels)))
    for i, label in enumerate(uniq_labels):
        mask = projected_df["label"] == label
        plt.scatter(
            projected_df.loc[mask, "pc1"],
            projected_df.loc[mask, "pc2"],
            s=40,
            alpha=0.8,
            label=label,
            color=palette[i],
        )

    if labels_path is not None and len(uniq_labels) == 2:
        try:
            from sklearn.metrics import roc_auc_score
            from sklearn.svm import LinearSVC

            y = np.array([0 if v == uniq_labels[0] else 1 for v in labels], dtype=np.int64)
            clf = LinearSVC(random_state=random_seed, dual="auto")
            clf.fit(proj, y)
            pred = clf.predict(proj)
            acc = float((pred == y).mean())
            score = clf.decision_function(proj).astype(np.float32)
            decision_score = score
            auc = float(roc_auc_score(y, score))
            clf_summary["hyperplane_drawn"] = True
            clf_summary["acc_2d"] = acc
            clf_summary["auc_2d"] = auc

            w = clf.coef_[0]
            b = float(clf.intercept_[0])
            x_min, x_max = float(projected_df["pc1"].min()), float(projected_df["pc1"].max())
            xs = np.linspace(x_min - 0.2, x_max + 0.2, 300)
            if abs(float(w[1])) > 1e-8:
                ys = -(w[0] * xs + b) / w[1]
                ys_m_pos = -(w[0] * xs + b - 1.0) / w[1]
                ys_m_neg = -(w[0] * xs + b + 1.0) / w[1]
                plt.plot(xs, ys, color="black", linewidth=2.0, label="Linear hyperplane")
                plt.plot(xs, ys_m_pos, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
                plt.plot(xs, ys_m_neg, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
            else:
                x0 = -(b / w[0]) if abs(float(w[0])) > 1e-8 else 0.0
                plt.axvline(x=x0, color="black", linewidth=2.0, label="Linear hyperplane")

            plt.title(f"cfDNA Feature Space (PCA 2D) | ACC={acc:.3f} AUC={auc:.3f}")
        except Exception as exc:
            clf_summary["reason"] = f"Hyperplane skipped: {type(exc).__name__}: {exc}"
            warnings.warn(clf_summary["reason"], stacklevel=2)
            plt.title("cfDNA Feature Space (PCA 2D)")
    else:
        if labels_path is None:
            clf_summary["reason"] = "No labels_table provided."
        else:
            clf_summary["reason"] = f"Need binary labels for hyperplane; got {len(uniq_labels)} classes."
        plt.title("cfDNA Feature Space (PCA 2D)")

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    projected_df["decision_score"] = decision_score
    projected_csv = out_dir / "cfdna_feature_space__projected_points.csv"
    projected_df.to_csv(projected_csv, index=False)

    return {
        "figure": str(fig_path),
        "projected_csv": str(projected_csv),
        "samples": int(len(sample_ids)),
        "feature_dim": int(x.shape[1]),
        "labels_provided": labels_path is not None,
        "label_classes": uniq_labels,
        "hyperplane": clf_summary,
    }


def visualize_fragmentomics(
    *,
    frag_dir: str | Path,
    output_dir: str | Path,
    labels_table: str | Path | None = None,
    sample_id_col: str = "sample_id",
    label_col: str = "her2_status",
    max_samples_for_overlay: int = 40,
) -> dict[str, Any]:
    out_dir = _to_data_path(output_dir, path_kind="cfDNA viz output dir", must_exist=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    frag_path = _to_data_path(frag_dir, path_kind="fragmentomics dir", must_exist=True)
    if not frag_path.is_dir():
        raise NotADirectoryError(f"fragmentomics dir is not a folder: {frag_path}")

    labels_path = (
        _to_data_path(labels_table, path_kind="cfDNA labels table", must_exist=True)
        if labels_table is not None
        else None
    )
    _, label_map = _load_label_map(
        labels_path,
        sample_id_col=sample_id_col,
        label_col=label_col,
    )

    hist_files = sorted(p for p in frag_path.glob("*_length_hist.parquet") if p.is_file())
    if not hist_files:
        raise ValueError(f"No *_length_hist.parquet found in: {frag_path}")

    sampled_files = hist_files[: max(1, int(max_samples_for_overlay))]
    rows: list[pd.DataFrame] = []
    for f in sampled_files:
        sid = f.name.replace("_length_hist.parquet", "")
        df = pd.read_parquet(f)
        if df.empty:
            continue
        part = df.copy()
        part["sample_id"] = sid
        part["label"] = label_map.get(sid, "Unknown")
        part["bin_mid"] = (part["bin_start"] + part["bin_end"]) / 2.0
        rows.append(part)

    if not rows:
        raise ValueError(f"No valid histogram rows loaded from: {frag_path}")

    hist_df = pd.concat(rows, ignore_index=True)
    agg = (
        hist_df.groupby(["label", "bin_mid"], as_index=False)["count"]
        .sum()
        .sort_values(["label", "bin_mid"])
        .reset_index(drop=True)
    )
    agg["density"] = agg.groupby("label")["count"].transform(lambda s: s / (s.sum() + 1e-9))

    fig_dist = out_dir / "cfdna_fragmentomics__length_distribution_by_label.png"
    plt.figure(figsize=(9, 5))
    for label, part in agg.groupby("label"):
        plt.plot(part["bin_mid"], part["density"], linewidth=2, label=str(label))
    plt.title("Fragment Length Distribution (Grouped)")
    plt.xlabel("Fragment length")
    plt.ylabel("Density")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dist)
    plt.close()

    short_ratio_fig = None
    short_ratio_csv = None
    frag_summary_path = frag_path / "frag_summary.parquet"
    if frag_summary_path.exists():
        summary_df = pd.read_parquet(frag_summary_path)
        if "sample_id" in summary_df.columns and "short_100_150_ratio" in summary_df.columns:
            summary_df = summary_df.copy()
            summary_df["sample_id"] = summary_df["sample_id"].astype(str)
            summary_df["label"] = summary_df["sample_id"].map(label_map).fillna("Unknown")
            short_ratio_csv = out_dir / "cfdna_fragmentomics__short_fragment_ratio_table.csv"
            summary_df.to_csv(short_ratio_csv, index=False)

            short_ratio_fig = out_dir / "cfdna_fragmentomics__short_fragment_ratio_boxplot.png"
            plt.figure(figsize=(8, 5))
            sns.boxplot(data=summary_df, x="label", y="short_100_150_ratio")
            sns.stripplot(
                data=summary_df,
                x="label",
                y="short_100_150_ratio",
                color="black",
                size=3,
                alpha=0.5,
                jitter=0.2,
            )
            plt.title("Short Fragment Ratio (100-150bp)")
            plt.xlabel("Label")
            plt.ylabel("short_100_150_ratio")
            plt.tight_layout()
            plt.savefig(short_ratio_fig)
            plt.close()

    density_csv = out_dir / "cfdna_fragmentomics__length_density_by_label.csv"
    agg.to_csv(density_csv, index=False)

    return {
        "hist_files_used": int(len(sampled_files)),
        "density_csv": str(density_csv),
        "distribution_figure": str(fig_dist),
        "short_ratio_csv": str(short_ratio_csv) if short_ratio_csv else None,
        "short_ratio_figure": str(short_ratio_fig) if short_ratio_fig else None,
    }


def visualize_methylation_proxy(
    *,
    meth_summary_path: str | Path,
    output_dir: str | Path,
    labels_table: str | Path | None = None,
    sample_id_col: str = "sample_id",
    label_col: str = "her2_status",
    max_features: int = 50,
) -> dict[str, Any]:
    out_dir = _to_data_path(output_dir, path_kind="cfDNA viz output dir", must_exist=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = _to_data_path(meth_summary_path, path_kind="meth summary table", must_exist=True)
    df = _read_table(summary_path)
    if "sample_id" not in df.columns:
        raise ValueError("meth summary table must contain column: sample_id")

    labels_path = (
        _to_data_path(labels_table, path_kind="cfDNA labels table", must_exist=True)
        if labels_table is not None
        else None
    )
    _, label_map = _load_label_map(
        labels_path,
        sample_id_col=sample_id_col,
        label_col=label_col,
    )

    value_cols = [c for c in df.columns if c.endswith("_mean_count")]
    if not value_cols:
        value_cols = [c for c in df.columns if c != "sample_id" and pd.api.types.is_numeric_dtype(df[c])]
    if not value_cols:
        raise ValueError("No numeric methylation-proxy columns found.")

    work = df[["sample_id"] + value_cols].copy()
    work["sample_id"] = work["sample_id"].astype(str)
    if len(value_cols) > int(max_features):
        variances = work[value_cols].var(axis=0).sort_values(ascending=False)
        keep_cols = variances.head(int(max_features)).index.tolist()
    else:
        keep_cols = value_cols

    matrix = work.set_index("sample_id")[keep_cols].astype(float)
    matrix = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0).replace(0.0, 1.0)
    matrix = matrix.fillna(0.0)

    row_labels = pd.Series(matrix.index, index=matrix.index).map(label_map).fillna("Unknown")
    order = row_labels.sort_values().index
    matrix = matrix.loc[order]
    row_labels = row_labels.loc[order]

    fig_path = out_dir / "cfdna_methylation_proxy__heatmap.png"
    plt.figure(figsize=(max(8, 0.25 * matrix.shape[1]), max(6, 0.25 * matrix.shape[0])))
    sns.heatmap(matrix, cmap="vlag", center=0.0, cbar_kws={"label": "z-score"})
    plt.title("Methylation Proxy Heatmap")
    plt.xlabel("Features")
    plt.ylabel("Samples")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    out_matrix = matrix.copy()
    out_matrix.insert(0, "label", row_labels.values)
    matrix_csv = out_dir / "cfdna_methylation_proxy__matrix.csv"
    out_matrix.to_csv(matrix_csv)

    return {
        "samples": int(matrix.shape[0]),
        "features": int(matrix.shape[1]),
        "matrix_csv": str(matrix_csv),
        "figure": str(fig_path),
    }


def visualize_cnv_heatmap(
    *,
    cnv_dir: str | Path,
    output_dir: str | Path,
    labels_table: str | Path | None = None,
    sample_id_col: str = "sample_id",
    label_col: str = "her2_status",
    max_bins: int = 1200,
    max_samples: int | None = None,
) -> dict[str, Any]:
    out_dir = _to_data_path(output_dir, path_kind="cfDNA viz output dir", must_exist=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    cnv_path = _to_data_path(cnv_dir, path_kind="cnv dir", must_exist=True)
    if not cnv_path.is_dir():
        raise NotADirectoryError(f"cnv dir is not a folder: {cnv_path}")

    labels_path = (
        _to_data_path(labels_table, path_kind="cfDNA labels table", must_exist=True)
        if labels_table is not None
        else None
    )
    _, label_map = _load_label_map(
        labels_path,
        sample_id_col=sample_id_col,
        label_col=label_col,
    )

    bin_files = sorted(p for p in cnv_path.glob("*_bin_counts.parquet") if p.is_file())
    if not bin_files:
        raise ValueError(f"No *_bin_counts.parquet found in: {cnv_path}")
    if max_samples is not None:
        bin_files = bin_files[: max(1, int(max_samples))]

    sample_series: dict[str, pd.Series] = {}
    for f in bin_files:
        sid = f.name.replace("_bin_counts.parquet", "")
        df = pd.read_parquet(f)
        if df.empty:
            continue
        needed = {"chrom", "start", "end", "count"}
        if not needed.issubset(set(df.columns)):
            continue
        key = df["chrom"].astype(str) + ":" + df["start"].astype(str) + "-" + df["end"].astype(str)
        counts = df["count"].astype(float).to_numpy()
        norm = np.log2((counts + 1.0) / (counts.mean() + 1.0))
        sample_series[sid] = pd.Series(norm, index=key)

    if not sample_series:
        raise ValueError(f"No valid CNV bin matrices parsed from: {cnv_path}")

    matrix = pd.DataFrame.from_dict(sample_series, orient="index").fillna(0.0)
    if matrix.shape[1] > int(max_bins):
        top_bins = matrix.var(axis=0).sort_values(ascending=False).head(int(max_bins)).index
        matrix = matrix.loc[:, top_bins]

    row_labels = pd.Series(matrix.index, index=matrix.index).map(label_map).fillna("Unknown")
    order = row_labels.sort_values().index
    matrix = matrix.loc[order]
    row_labels = row_labels.loc[order]

    fig_path = out_dir / "cfdna_cnv__heatmap.png"
    plt.figure(figsize=(max(8, 0.008 * matrix.shape[1]), max(6, 0.25 * matrix.shape[0])))
    sns.heatmap(matrix, cmap="coolwarm", center=0.0, cbar_kws={"label": "log2 normalized count"})
    plt.title("CNV Heatmap")
    plt.xlabel("Genomic bins")
    plt.ylabel("Samples")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    out_matrix = matrix.copy()
    out_matrix.insert(0, "label", row_labels.values)
    matrix_csv = out_dir / "cfdna_cnv__matrix.csv"
    out_matrix.to_csv(matrix_csv)

    return {
        "samples": int(matrix.shape[0]),
        "bins": int(matrix.shape[1]),
        "matrix_csv": str(matrix_csv),
        "figure": str(fig_path),
    }


def run_cfdna_plot_suite(
    *,
    output_dir: str | Path,
    cfdna_features_pt: str | Path | None = None,
    labels_table: str | Path | None = None,
    labels_sample_col: str = "sample_id",
    labels_label_col: str = "her2_status",
    frag_dir: str | Path | None = None,
    meth_summary_path: str | Path | None = None,
    cnv_dir: str | Path | None = None,
    random_seed: int = 42,
) -> dict[str, Any]:
    out_dir = _to_data_path(output_dir, path_kind="cfDNA viz output dir", must_exist=False)
    out_dir.mkdir(parents=True, exist_ok=True)

    # TODO(superhy): after data disk is mounted, validate these visualizations on full cohort
    # and tune default plotting scales for publication-quality output.
    summary: dict[str, Any] = {
        "output_dir": str(out_dir),
        "modules": {},
    }

    if cfdna_features_pt is not None:
        summary["modules"]["feature_space"] = plot_feature_space_with_hyperplane(
            cfdna_features_pt=cfdna_features_pt,
            output_dir=out_dir / "feature_space",
            labels_table=labels_table,
            sample_id_col=labels_sample_col,
            label_col=labels_label_col,
            random_seed=random_seed,
        )

    if frag_dir is not None:
        summary["modules"]["fragmentomics"] = visualize_fragmentomics(
            frag_dir=frag_dir,
            output_dir=out_dir / "fragmentomics",
            labels_table=labels_table,
            sample_id_col=labels_sample_col,
            label_col=labels_label_col,
        )

    if meth_summary_path is not None:
        summary["modules"]["methylation_proxy"] = visualize_methylation_proxy(
            meth_summary_path=meth_summary_path,
            output_dir=out_dir / "methylation_proxy",
            labels_table=labels_table,
            sample_id_col=labels_sample_col,
            label_col=labels_label_col,
        )

    if cnv_dir is not None:
        summary["modules"]["cnv"] = visualize_cnv_heatmap(
            cnv_dir=cnv_dir,
            output_dir=out_dir / "cnv",
            labels_table=labels_table,
            sample_id_col=labels_sample_col,
            label_col=labels_label_col,
        )

    summary_path = out_dir / "cfdna_visualisation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    return summary


def run_cfdna_visualization_suite(
    *,
    output_dir: str | Path,
    cfdna_features_pt: str | Path | None = None,
    labels_table: str | Path | None = None,
    labels_sample_col: str = "sample_id",
    labels_label_col: str = "her2_status",
    frag_dir: str | Path | None = None,
    meth_summary_path: str | Path | None = None,
    cnv_dir: str | Path | None = None,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Backward-compatible alias; prefer run_cfdna_plot_suite()."""
    return run_cfdna_plot_suite(
        output_dir=output_dir,
        cfdna_features_pt=cfdna_features_pt,
        labels_table=labels_table,
        labels_sample_col=labels_sample_col,
        labels_label_col=labels_label_col,
        frag_dir=frag_dir,
        meth_summary_path=meth_summary_path,
        cnv_dir=cnv_dir,
        random_seed=random_seed,
    )
