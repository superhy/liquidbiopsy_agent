from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")


def save_histogram(series: pd.Series, title: str, xlabel: str, path: Path, bins: int = 50) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    sns.histplot(series, bins=bins, kde=False)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_barplot(data: pd.Series, title: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    data.plot(kind="bar")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_boxplot(df: pd.DataFrame, value_col: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    sns.boxplot(y=df[value_col])
    plt.title(value_col)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def embed_image_base64(path: Path) -> str:
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{encoded}"


def save_scatter(x: pd.Series, y: pd.Series, title: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    plt.scatter(x, y, alpha=0.6)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def pca_plot(df: pd.DataFrame, sample_col: str, path: Path) -> None:
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        return
    numeric = df.select_dtypes(include=["number"])
    if numeric.empty or len(numeric) < 2:
        return
    pca = PCA(n_components=2)
    comps = pca.fit_transform(numeric)
    plt.figure(figsize=(5, 4))
    plt.scatter(comps[:, 0], comps[:, 1], alpha=0.7)
    for i, sid in enumerate(df[sample_col]):
        plt.text(comps[i, 0], comps[i, 1], str(sid), fontsize=6)
    plt.title("PCA (features)")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def save_lineplot(x: pd.Series, y: pd.Series, title: str, xlabel: str, ylabel: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 3))
    plt.plot(x, y, linewidth=1.0)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_heatmap(df: pd.DataFrame, title: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    sns.heatmap(df, cmap="viridis")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_multi_hist(data: List[pd.Series], labels: List[str], title: str, xlabel: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    for series, label in zip(data, labels):
        sns.histplot(series, bins=60, kde=False, stat="density", label=label, element="step", fill=False)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_violinplot(df: pd.DataFrame, x_col: str, y_col: str, title: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 4))
    sns.violinplot(data=df, x=x_col, y=y_col, inner="quartile", cut=0)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_cnv_chrom_plot(df: pd.DataFrame, path: Path, title: str) -> None:
    if df.empty:
        return
    df = df.sort_values(["chrom", "start"]).reset_index(drop=True)
    df["norm"] = df["count"] / (df["count"].mean() + 1e-6)
    chrom_changes = df["chrom"].ne(df["chrom"].shift()).to_numpy()
    boundaries = df.index[chrom_changes].tolist()
    labels = df.loc[boundaries, "chrom"].tolist()
    x = df.index.to_numpy()
    y = df["norm"].to_numpy()

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 3))
    plt.plot(x, y, linewidth=0.8)
    for b in boundaries:
        plt.axvline(b, color="grey", linewidth=0.5, alpha=0.5)
    if boundaries:
        plt.xticks(boundaries, labels, rotation=90, fontsize=6)
    plt.title(title)
    plt.ylabel("Normalised count")
    plt.xlabel("Genomic bins (by chromosome)")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
