"""
Machine learning utilities: clustering and regression.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns


def prepare_clustering_data(
    df: pd.DataFrame,
    features: list[str] = ["Price (INR)", "Rating", "Rating Count"],
    price_cap: float = 1500.0,
) -> Tuple[pd.DataFrame, np.ndarray, StandardScaler]:
    """
    Prepare and scale features for K-Means.
    Returns: filtered dataframe, scaled feature matrix, fitted scaler.
    """
    data = df[features].dropna().copy()
    data = data[data["Price (INR)"] < price_cap]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    return data, X_scaled, scaler


def find_optimal_k(
    X_scaled: np.ndarray,
    k_range: range = range(2, 11),
    random_state: int = 42,
) -> Dict[str, list]:
    """Compute inertia (elbow) for a range of k."""
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
    return {"k": list(k_range), "inertia": inertias}


def run_kmeans(
    X_scaled: np.ndarray,
    n_clusters: int = 3,
    random_state: int = 42,
) -> KMeans:
    """Fit K-Means and return the model."""
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    model.fit(X_scaled)
    return model


def plot_elbow(elbow_data: Dict[str, list], save_path: str | None = None) -> None:
    """Elbow plot for choosing k."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(elbow_data["k"], elbow_data["inertia"], marker="o", color="#e74c3c")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method for Optimal k")
    ax.set_xticks(elbow_data["k"])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_clusters(
    data: pd.DataFrame,
    labels: np.ndarray,
    x_col: str = "Price (INR)",
    y_col: str = "Rating",
    save_path: str | None = None,
) -> None:
    """2D scatter of clusters."""
    plot_df = data.copy()
    plot_df["Cluster"] = labels
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.scatterplot(
        data=plot_df,
        x=x_col,
        y=y_col,
        hue="Cluster",
        palette="Set1",
        alpha=0.6,
        ax=ax,
    )
    ax.set_title("Dish Segmentation (K-Means) — Price vs Rating")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def train_price_regression(
    df: pd.DataFrame,
    features: list[str] = ["Rating", "Rating Count"],
    target: str = "Price (INR)",
    price_cap: float = 1500.0,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train a simple Linear Regression model to predict price.
    Returns metrics and the fitted model.
    """
    model_df = df[[target] + features].dropna()
    model_df = model_df[model_df[target] < price_cap]

    X = model_df[features]
    y = model_df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred),
        "coefficients": dict(zip(features, model.coef_)),
        "intercept": model.intercept_,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    return {"model": model, "metrics": metrics, "y_test": y_test, "y_pred": y_pred}
