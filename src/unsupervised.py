import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from minisom import MiniSom
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from src.data import FEATURES, load_htru2, pulsars_only

SEED = 42
SKEW_CUTOFF = 1.0


def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


class PulsarScaler:
    """Signed log on heavy-tailed columns, then z-scores. Fit on pulsars only."""

    def __init__(self, skew_cutoff=SKEW_CUTOFF):
        self.skew_cutoff = skew_cutoff

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.skewness_ = pd.DataFrame(X).skew().to_numpy()
        self.log_cols_ = np.abs(self.skewness_) > self.skew_cutoff
        self.scaler_ = StandardScaler().fit(self._log(X))
        return self

    def _log(self, X):
        X = np.array(X, dtype=float, copy=True)
        X[:, self.log_cols_] = signed_log(X[:, self.log_cols_])
        return X

    def transform(self, X):
        return self.scaler_.transform(self._log(np.atleast_2d(X)))

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def fit_kmeans(Z, k, seed=SEED):
    return KMeans(n_clusters=k, n_init=20, random_state=seed).fit(Z)


def bootstrap_stability(Z, k, n_boot=30, seed=SEED):
    ref = fit_kmeans(Z, k, seed).labels_
    rng = np.random.default_rng(seed)
    scores = []
    for b in range(n_boot):
        idx = rng.choice(len(Z), size=len(Z), replace=True)
        km = KMeans(n_clusters=k, n_init=10, random_state=seed + b + 1).fit(Z[idx])
        scores.append(adjusted_rand_score(ref, km.predict(Z)))
    return float(np.mean(scores)), float(np.min(scores))


def k_selection(Z, ks=range(2, 7), seed=SEED):
    rows = []
    for k in ks:
        km = fit_kmeans(Z, k, seed)
        stab_mean, stab_min = bootstrap_stability(Z, k, seed=seed)
        rows.append({
            "k": k,
            "silhouette": silhouette_score(Z, km.labels_),
            "davies_bouldin": davies_bouldin_score(Z, km.labels_),
            "calinski_harabasz": calinski_harabasz_score(Z, km.labels_),
            "inertia": km.inertia_,
            "bootstrap_ari_mean": stab_mean,
            "bootstrap_ari_min": stab_min,
            "smallest_cluster": int(np.bincount(km.labels_).min()),
        })
    return pd.DataFrame(rows)


def order_by(labels, values):
    """Relabel clusters so cluster 0 has the lowest median of `values`."""
    k = labels.max() + 1
    medians = [np.median(values[labels == c]) for c in range(k)]
    mapping = np.empty(k, dtype=int)
    mapping[np.argsort(medians)] = np.arange(k)
    return mapping


def match_labels(reference, labels):
    k = max(reference.max(), labels.max()) + 1
    overlap = np.zeros((k, k), dtype=int)
    for r, l in zip(reference, labels):
        overlap[l, r] += 1
    rows, cols = linear_sum_assignment(-overlap)
    mapping = np.empty(k, dtype=int)
    mapping[rows] = cols
    return mapping


def train_som(Z, rows=5, cols=5, sigma=1.5, learning_rate=0.5, epochs=100, seed=SEED):
    som = MiniSom(rows, cols, Z.shape[1], sigma=sigma,
                  learning_rate=learning_rate, random_seed=seed)
    som.pca_weights_init(Z)
    som.train(Z, epochs, random_order=True, use_epochs=True)
    return som


def som_cells(som, Z):
    n_cols = som.get_weights().shape[1]
    return np.array([r * n_cols + c for r, c in (som.winner(z) for z in Z)])


def som_prototype_clusters(som, k, seed=SEED):
    codebook = som.get_weights().reshape(-1, som.get_weights().shape[2])
    return fit_kmeans(codebook, k, seed).labels_


def pca_2d(Z):
    pca = PCA(n_components=2, random_state=SEED).fit(Z)
    return pca, pca.transform(Z)


def feature_pc_correlations(Z, scores, names):
    return pd.DataFrame(
        [[np.corrcoef(Z[:, j], scores[:, i])[0, 1] for i in range(scores.shape[1])]
         for j in range(Z.shape[1])],
        index=names, columns=[f"PC{i + 1}" for i in range(scores.shape[1])],
    )


def cluster_profile(X_raw, labels, names):
    df = pd.DataFrame(X_raw, columns=names)
    prof = df.groupby(labels).median()
    prof.insert(0, "count", pd.Series(labels).value_counts().sort_index().to_numpy())
    prof.index.name = "cluster"
    return prof


OUT = ROOT / "results" / "unsupervised"
MODELS = ROOT / "models"
K_RANGE = range(2, 7)
SOM_SIGMAS = (1.0, 1.5, 2.0)
SOM_SIGMA = 1.5
COLORS = plt.cm.tab10.colors


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)


def scatter_clusters(ax, pts, labels, title):
    for c in np.unique(labels):
        m = labels == c
        ax.scatter(pts[m, 0], pts[m, 1], s=8, alpha=0.7, color=COLORS[c],
                   label=f"cluster {c} (n={m.sum()})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend(fontsize=8, markerscale=2)


def main():
    plt.switch_backend("Agg")
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    df = load_htru2()
    X = pulsars_only(df).to_numpy()
    print(f"Pulsars used for clustering: {len(X)} of {len(df)} candidates")

    # 1. Reproduce the paper setting: K-means (k=3) on raw, unscaled features
    km_raw = KMeans(n_clusters=3, n_init=20, random_state=SEED).fit(X)
    raw_sil = silhouette_score(X, km_raw.labels_)
    raw_counts = sorted(np.bincount(km_raw.labels_).tolist(), reverse=True)
    var_share = pd.Series(X.var(axis=0) / X.var(axis=0).sum(), index=FEATURES)
    var_share.sort_values(ascending=False).rename("share_of_total_variance").to_csv(
        OUT / "raw_variance_share.csv")
    top3 = var_share.sort_values(ascending=False).head(3)
    print(f"Paper replication (raw k=3): silhouette={raw_sil:.3f}, sizes={raw_counts}")
    print(f"In raw units {', '.join(top3.index)} hold {top3.sum():.1%} of total variance")

    raw_pts = PCA(n_components=2, random_state=SEED).fit_transform(X)
    raw_labels = order_by(km_raw.labels_, X[:, FEATURES.index("dm_mean")])[km_raw.labels_]

    # 2. Preprocessing: signed log on heavy-tailed columns, then standardize
    scaler = PulsarScaler().fit(X)
    Z = scaler.transform(X)
    log_cols = [f for f, flag in zip(FEATURES, scaler.log_cols_) if flag]
    pd.DataFrame({"skewness_raw": scaler.skewness_,
                  "skewness_after": pd.DataFrame(Z).skew().to_numpy(),
                  "log_applied": scaler.log_cols_}, index=FEATURES).to_csv(
        OUT / "feature_skewness.csv")
    print(f"Signed log applied to: {log_cols}")

    fig, axes = plt.subplots(2, len(log_cols), figsize=(4 * len(log_cols), 6))
    for j, name in enumerate(log_cols):
        i = FEATURES.index(name)
        axes[0, j].hist(X[:, i], bins=50, color=COLORS[0])
        axes[0, j].set_title(f"{name} raw (skew {scaler.skewness_[i]:.2f})")
        axes[1, j].hist(Z[:, i], bins=50, color=COLORS[1])
        axes[1, j].set_title(f"{name} after log + z-score")
    savefig(fig, "feature_transform.png")

    spaces = {
        "raw": X,
        "standardized": StandardScaler().fit_transform(X),
        "log_standardized": Z,
    }
    comp = pd.DataFrame(
        {name: [silhouette_score(S, fit_kmeans(S, k).labels_) for k in K_RANGE]
         for name, S in spaces.items()}, index=pd.Index(list(K_RANGE), name="k"))
    comp.to_csv(OUT / "silhouette_by_preprocessing.csv")

    # 3. Choose k on the transformed data
    ksel = k_selection(Z, K_RANGE)
    ksel.to_csv(OUT / "k_selection.csv", index=False)
    best_k = int(ksel.loc[ksel["silhouette"].idxmax(), "k"])
    print(ksel.round(3).to_string(index=False))
    print(f"Chosen k = {best_k} (highest silhouette)")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, col, title in zip(axes,
                              ["silhouette", "davies_bouldin", "bootstrap_ari_mean"],
                              ["Silhouette (higher is better)",
                               "Davies-Bouldin (lower is better)",
                               "Bootstrap stability, ARI (higher is better)"]):
        ax.plot(ksel["k"], ksel[col], marker="o", color=COLORS[0])
        ax.axvline(best_k, ls="--", color="grey")
        ax.set_xlabel("k")
        ax.set_title(title)
    savefig(fig, "k_selection.png")

    # 4. K-means with the chosen k
    km = fit_kmeans(Z, best_k)
    km_map = order_by(km.labels_, X[:, FEATURES.index("dm_mean")])
    km_labels = km_map[km.labels_]
    km_sil = silhouette_score(Z, km_labels)

    # 5. Self-Organizing Map, 5x5
    sweep = []
    for s in SOM_SIGMAS:
        som_s = train_som(Z, sigma=s)
        sweep.append({"sigma": s,
                      "quantization_error": som_s.quantization_error(Z),
                      "topographic_error": som_s.topographic_error(Z),
                      "non_empty_cells": len(np.unique(som_cells(som_s, Z)))})
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(OUT / "som_sigma_sweep.csv", index=False)

    som = train_som(Z, sigma=SOM_SIGMA)
    cells = som_cells(som, Z)
    proto_labels = som_prototype_clusters(som, best_k)
    som_raw_labels = proto_labels[cells]
    som_map = match_labels(km_labels, som_raw_labels)
    proto_labels = som_map[proto_labels]
    som_labels = proto_labels[cells]
    som_sil = silhouette_score(Z, som_labels)
    cell_sil = silhouette_score(Z, cells)
    ari = adjusted_rand_score(km_labels, som_labels)
    qe, te = som.quantization_error(Z), som.topographic_error(Z)
    print(f"SOM: {len(np.unique(cells))}/25 cells used, QE={qe:.3f}, TE={te:.3f}")
    print(f"SOM prototypes grouped into k={best_k}: silhouette={som_sil:.3f}, "
          f"ARI vs K-means={ari:.3f}")

    rows, cols = som.get_weights().shape[:2]
    hits = np.bincount(cells, minlength=rows * cols).reshape(rows, cols)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    im = axes[0].imshow(som.distance_map(scaling="mean"), cmap="bone_r")
    axes[0].set_title("U-matrix (darker = farther from neighbours)")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    im = axes[1].imshow(hits, cmap="Blues")
    for r in range(rows):
        for c in range(cols):
            axes[1].text(c, r, f"{hits[r, c]}\nC{proto_labels[r * cols + c]}",
                         ha="center", va="center", fontsize=8,
                         color="white" if hits[r, c] > hits.max() * 0.6 else "black")
    axes[1].set_title("Hits per cell and prototype cluster")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    for ax in axes:
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
    savefig(fig, "som_maps.png")

    # 6. PCA, used only for plotting
    pca, pts = pca_2d(Z)
    evr = pca.explained_variance_ratio_
    pd.DataFrame(pca.components_.T, index=FEATURES, columns=["PC1", "PC2"]).to_csv(
        OUT / "pca_loadings.csv")
    corr = feature_pc_correlations(Z, pts, FEATURES)
    corr.to_csv(OUT / "pca_feature_correlations.csv")
    print(f"PCA on transformed data: PC1 {evr[0]:.1%}, PC2 {evr[1]:.1%} of variance")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    scatter_clusters(axes[0], raw_pts, raw_labels,
                     f"Paper setting: raw features, k=3 (silhouette {raw_sil:.2f})")
    scatter_clusters(axes[1], pts, km_labels,
                     f"Ours: log + z-score, k={best_k} (silhouette {km_sil:.2f})")
    savefig(fig, "paper_vs_ours_pca.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
    scatter_clusters(axes[0], pts, km_labels, f"K-means, k={best_k}")
    scatter_clusters(axes[1], pts, som_labels, f"SOM 5x5, prototypes grouped into {best_k}")
    savefig(fig, "pca_kmeans_vs_som.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    ypos = np.arange(len(FEATURES))[::-1]
    ax.barh(ypos + 0.2, corr["PC1"], height=0.4, label="PC1", color=COLORS[0])
    ax.barh(ypos - 0.2, corr["PC2"], height=0.4, label="PC2", color=COLORS[1])
    ax.set_yticks(ypos)
    ax.set_yticklabels(FEATURES)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Correlation with component")
    ax.legend()
    savefig(fig, "pca_feature_correlations.png")

    # 7. Cluster interpretation
    km_prof = cluster_profile(X, km_labels, FEATURES)
    som_prof = cluster_profile(X, som_labels, FEATURES)
    km_prof.to_csv(OUT / "kmeans_cluster_medians.csv")
    som_prof.to_csv(OUT / "som_cluster_medians.csv")
    zmeans = pd.DataFrame(Z, columns=FEATURES).groupby(km_labels).mean()
    print("K-means cluster medians (original units):")
    print(km_prof.round(2).to_string())

    fig, ax = plt.subplots(figsize=(8, 1.2 + 0.6 * best_k))
    lim = np.abs(zmeans.to_numpy()).max()
    im = ax.imshow(zmeans.to_numpy(), cmap="coolwarm", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(FEATURES)))
    ax.set_xticklabels(FEATURES, rotation=45, ha="right")
    ax.set_yticks(range(best_k))
    ax.set_yticklabels([f"cluster {c} (n={n})" for c, n in km_prof["count"].items()])
    for (r, c), v in np.ndenumerate(zmeans.to_numpy()):
        ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="mean z-score")
    ax.set_title("K-means cluster profiles")
    savefig(fig, "kmeans_cluster_profiles.png")

    summary = {
        "n_pulsars": int(len(X)),
        "paper_replication_raw_k3": {"silhouette": round(raw_sil, 4), "sizes": raw_counts},
        "raw_top3_variance_share": {k: round(v, 4) for k, v in top3.items()},
        "log_transformed_features": log_cols,
        "chosen_k": best_k,
        "kmeans": {"silhouette": round(km_sil, 4),
                   "sizes": km_prof["count"].tolist()},
        "som": {"grid": [rows, cols], "sigma": SOM_SIGMA, "epochs": 100,
                "non_empty_cells": int(len(np.unique(cells))),
                "quantization_error": round(qe, 4), "topographic_error": round(te, 4),
                "per_cell_silhouette": round(cell_sil, 4),
                "grouped_silhouette": round(som_sil, 4),
                "sizes": som_prof["count"].tolist(),
                "ari_vs_kmeans": round(ari, 4)},
        "pca_explained_variance": [round(v, 4) for v in evr],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    joblib.dump({"features": FEATURES, "scaler": scaler, "kmeans": km,
                 "kmeans_map": km_map, "som": som, "som_proto_labels": proto_labels,
                 "profiles": km_prof}, MODELS / "unsupervised.joblib")
    print(f"Saved figures and tables to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    # re-import so the saved model points to src.unsupervised, not __main__
    from src.unsupervised import main as run
    run()