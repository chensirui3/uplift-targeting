"""Download, verify, load and split the Hillstrom e-mail experiment."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config

FEATURES = [
    "recency",  # months since last purchase (1-12)
    "history",  # dollars spent in the past year
    "mens",  # bought men's merchandise in the past year
    "womens",  # bought women's merchandise in the past year
    "newbie",  # new customer in the past twelve months
    "zip_urban",  # zip code type (suburban is the baseline)
    "zip_rural",
    "channel_web",  # purchase channels used in the past year (phone is the baseline)
    "channel_multichannel",
]
OUTCOMES = ["visit", "conversion", "spend"]


def download(path: Path = config.DATA_FILE) -> Path:
    """Fetch the raw CSV once and refuse to continue if its checksum has changed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"Downloading {config.DATA_URL}")
        urllib.request.urlretrieve(config.DATA_URL, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != config.DATA_SHA256:
        raise ValueError(
            f"{path} has SHA-256 {digest}, expected {config.DATA_SHA256}. "
            "Delete the file and rerun to download a fresh copy."
        )
    return path


def load(path: Path = config.DATA_FILE) -> pd.DataFrame:
    """Return one row per customer: model features, arm code and outcomes."""
    raw = pd.read_csv(path)
    arm = raw["segment"].map({name: code for code, name in enumerate(config.ARMS)})
    if arm.isna().any():
        raise ValueError(f"Unexpected segment values: {sorted(raw['segment'].unique())}")
    df = pd.DataFrame(
        {
            "recency": raw["recency"],
            "history": raw["history"],
            "mens": raw["mens"],
            "womens": raw["womens"],
            "newbie": raw["newbie"],
            # The source file spells the suburban category "Surburban".
            "zip_urban": (raw["zip_code"] == "Urban").astype(int),
            "zip_rural": (raw["zip_code"] == "Rural").astype(int),
            "channel_web": (raw["channel"] == "Web").astype(int),
            "channel_multichannel": (raw["channel"] == "Multichannel").astype(int),
            "arm": arm.astype(int),
            "visit": raw["visit"],
            "conversion": raw["conversion"],
            "spend": raw["spend"],
        }
    )
    expected_rows = 64_000
    if len(df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(df)}")
    return df


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train / test split, stratified so every arm keeps its share of rare purchases."""
    strata = df["arm"] * 2 + df["conversion"]
    train, test = train_test_split(
        df, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=strata
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Feature matrix, arm codes and the outcomes table, as the learners expect them."""
    return df[FEATURES].to_numpy(dtype=float), df["arm"].to_numpy(), df[OUTCOMES]
