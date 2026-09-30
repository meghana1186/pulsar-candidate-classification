import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "HTRU_2.csv"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/372/htru2.zip"

# Column order verified in check_moment_order below
FEATURES = [
    "ip_mean", "ip_std", "ip_skew", "ip_kurt",
    "dm_mean", "dm_std", "dm_skew", "dm_kurt",
]
LABEL = "label"


def download(path=DATA_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(UCI_ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            archive = zipfile.ZipFile(io.BytesIO(resp.read()))
    except Exception as err:
        raise SystemExit(
            f"Could not download HTRU2 ({err}). Download {UCI_ZIP_URL} manually "
            f"and put HTRU_2.csv in {path.parent}"
        )
    member = next(n for n in archive.namelist() if n.endswith("HTRU_2.csv"))
    path.write_bytes(archive.read(member))


def check_moment_order(df, tol=1e-6):
    for prefix in ("ip", "dm"):
        skew, kurt = df[f"{prefix}_skew"], df[f"{prefix}_kurt"]
        ok = (kurt >= skew ** 2 - 2 - tol).mean()
        if ok < 1.0:
            raise ValueError(
                f"{prefix}: excess kurtosis >= skewness^2 - 2 fails for "
                f"{(1 - ok):.2%} of rows, so the skew/kurt columns are swapped"
            )


def load_htru2(path=DATA_PATH):
    path = Path(path)
    if not path.exists():
        download(path)
    df = pd.read_csv(path, header=None)
    if df.shape[1] != len(FEATURES) + 1:
        raise ValueError(f"expected 9 columns in {path.name}, found {df.shape[1]}")
    df = df.apply(pd.to_numeric, errors="coerce")
    if df.iloc[0].isna().all():
        df = df.iloc[1:].reset_index(drop=True)
    df.columns = FEATURES + [LABEL]
    if df.isna().any().any():
        raise ValueError("HTRU2 file has missing values, re-download it")
    if not set(df[LABEL].unique()) <= {0, 1}:
        raise ValueError("unexpected class labels in HTRU2 file")
    df[LABEL] = df[LABEL].astype(int)
    check_moment_order(df)
    return df


def pulsars_only(df):
    return df.loc[df[LABEL] == 1, FEATURES].reset_index(drop=True)
