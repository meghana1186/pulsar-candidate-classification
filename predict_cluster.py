import argparse
from pathlib import Path

import joblib
import numpy as np

from src.data import FEATURES, load_htru2, pulsars_only
from src.unsupervised import som_cells

MODEL_PATH = Path(__file__).resolve().parent / "models" / "unsupervised.joblib"


def assign(values, bundle):
    z = bundle["scaler"].transform(np.asarray(values, dtype=float).reshape(1, -1))
    km_cluster = int(bundle["kmeans_map"][bundle["kmeans"].predict(z)[0]])
    cell = int(som_cells(bundle["som"], z)[0])
    som_cluster = int(bundle["som_proto_labels"][cell])
    n_cols = bundle["som"].get_weights().shape[1]
    return km_cluster, (cell // n_cols, cell % n_cols), som_cluster


def main():
    parser = argparse.ArgumentParser(description="Assign a pulsar to a cluster")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--row", type=int, help="index into the 1,639 pulsar rows")
    group.add_argument("--values", type=float, nargs=8, metavar="X",
                       help=" ".join(FEATURES))
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit("Model not found. Run: python src/unsupervised.py")
    bundle = joblib.load(MODEL_PATH)

    if args.row is not None:
        pulsars = pulsars_only(load_htru2())
        if not 0 <= args.row < len(pulsars):
            raise SystemExit(f"--row must be between 0 and {len(pulsars) - 1}")
        values = pulsars.iloc[args.row].to_numpy()
    else:
        values = np.array(args.values)

    km_cluster, cell, som_cluster = assign(values, bundle)
    prof = bundle["profiles"]

    print("Input:")
    for name, v in zip(FEATURES, values):
        print(f"  {name:8s} {v:10.3f}   cluster {km_cluster} median {prof.loc[km_cluster, name]:10.3f}")
    print(f"K-means cluster: {km_cluster} (cluster size {prof.loc[km_cluster, 'count']})")
    print(f"SOM best matching cell: row {cell[0]}, col {cell[1]} -> cluster {som_cluster}")
    print("Note: clustering was fit on confirmed pulsars only, so run the classifier first.")


if __name__ == "__main__":
    main()
