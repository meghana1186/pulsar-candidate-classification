
"""
Loads the HTRU2 pulsar dataset.

The raw CSV has NO header row, so we assign column names manually.
Column order (verified from UCI HTRU2 documentation):
    1-4  : statistics of the integrated pulse profile
    5-8  : statistics of the DM-SNR curve
    9    : class label (0 = non-pulsar, 1 = pulsar)
"""

import os
import pandas as pd

COLUMN_NAMES = [
    "mean_ip",
    "std_ip",
    "skewness_ip",
    "kurtosis_ip",
    "mean_dmsnr",
    "std_dmsnr",
    "skewness_dmsnr",
    "kurtosis_dmsnr",
    "class",
]

# Path to the dataset, relative to the project root
DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "HTRU_2.csv",
)


def load_data(path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load HTRU2 dataset and return a DataFrame with named columns."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download HTRU_2.csv from "
            "https://archive.ics.uci.edu/dataset/372/htru2 "
            "and place it in the data/ folder."
        )

    df = pd.read_csv(path, header=None, names=COLUMN_NAMES)
    return df


if __name__ == "__main__":
    # Quick sanity check when run directly
    df = load_data()
    print("Shape:", df.shape)
    print("\nClass counts:")
    print(df["class"].value_counts())
    print("\nFirst 5 rows:")
    print(df.head())