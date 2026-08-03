"""
Reusable exploratory data analysis and visualization helpers.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List


def set_plot_style() -> None:
    """Apply a clean, professional plotting style."""
    sns.set_theme(style="whitegrid", context="notebook", palette="muted")
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 120,
    })


def plot_price_distribution(
    df: pd.DataFrame,
    price_col: str = "Price (INR)",
    save_path: Optional[str | Path] = None,
) -> None:
    """Histogram + KDE of dish prices (capped for readability)."""
    fig, ax = plt.subplots(figsize=(12, 5))
    data = df[df[price_col] < 1500][price_col]
    sns.histplot(data, bins=60, kde=True, ax=ax, color="#e74c3c", alpha=0.7)
    ax.axvline(data.median(), color="black", linestyle="--", label=f"Median: ₹{data.median():.0f}")
    ax.set_title("Distribution of Dish Prices (INR) — Capped at ₹1500")
    ax.set_xlabel("Price (INR)")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_rating_distribution(
    df: pd.DataFrame,
    rating_col: str = "Rating",
    save_path: Optional[str | Path] = None,
) -> None:
    """Distribution of ratings."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df[rating_col], bins=30, kde=True, ax=ax, color="#3498db")
    ax.set_title("Distribution of Dish Ratings")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_top_cities(
    df: pd.DataFrame,
    n: int = 15,
    save_path: Optional[str | Path] = None,
) -> None:
    """Bar chart of top cities by number of listings."""
    top = df["City"].value_counts().head(n)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=top.values, y=top.index, ax=ax, palette="viridis")
    ax.set_title(f"Top {n} Cities by Number of Dish Listings")
    ax.set_xlabel("Number of Listings")
    ax.set_ylabel("City")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_price_vs_rating(
    df: pd.DataFrame,
    sample_size: int = 5000,
    save_path: Optional[str | Path] = None,
) -> None:
    """Scatter of Price vs Rating (sampled for performance)."""
    sample = df.sample(min(sample_size, len(df)), random_state=42)
    sample = sample[sample["Price (INR)"] < 1500]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=sample,
        x="Price (INR)",
        y="Rating",
        alpha=0.4,
        ax=ax,
        color="#9b59b6",
    )
    ax.set_title("Price vs Rating (Sampled)")
    ax.set_xlabel("Price (INR)")
    ax.set_ylabel("Rating")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def correlation_heatmap(
    df: pd.DataFrame,
    cols: List[str],
    save_path: Optional[str | Path] = None,
) -> None:
    """Correlation heatmap for selected numeric columns."""
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".3f", ax=ax)
    ax.set_title("Correlation Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()