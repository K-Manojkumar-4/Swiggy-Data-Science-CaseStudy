"""
Data loading, cleaning, and feature engineering utilities.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


def load_raw_data(filepath: str | Path) -> pd.DataFrame:
    """Load the raw Swiggy dataset from CSV."""
    df = pd.read_csv(filepath)
    return df


def clean_and_engineer(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """
    Perform standard cleaning and feature engineering.

    Steps:
    - Parse Order Date to datetime
    - Extract temporal features (Month, Month_num, Day, Year)
    - Optionally remove exact duplicates
    - Strip whitespace from object columns
    """
    df = df.copy()

    # Date parsing
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")

    # Temporal features
    df["Month"] = df["Order Date"].dt.month_name()
    df["Month_num"] = df["Order Date"].dt.month
    df["Day"] = df["Order Date"].dt.day_name()
    df["Year"] = df["Order Date"].dt.year

    # Clean string columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()

    if drop_duplicates:
        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)
        if removed > 0:
            print(f"Removed {removed} duplicate rows.")

    return df


def get_basic_info(df: pd.DataFrame) -> None:
    """Print a concise summary of the dataframe."""
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape          : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory usage   : {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    print(f"Date range     : {df['Order Date'].min().date()} → {df['Order Date'].max().date()}")
    print(f"Cities         : {df['City'].nunique()}")
    print(f"States         : {df['State'].nunique()}")
    print(f"Restaurants    : {df['Restaurant Name'].nunique():,}")
    print(f"Categories     : {df['Category'].nunique()}")
    print(f"Missing values : {df.isnull().sum().sum()}")
    print("=" * 60)


def filter_price_outliers(
    df: pd.DataFrame,
    price_col: str = "Price (INR)",
    upper_quantile: float = 0.99,
) -> pd.DataFrame:
    """Optionally filter extreme price outliers for modeling."""
    threshold = df[price_col].quantile(upper_quantile)
    return df[df[price_col] <= threshold].copy()