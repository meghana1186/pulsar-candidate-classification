# Pulsar Candidate Classification and Clustering (HTRU2)

Mini-project for UE24CS352A Machine Learning, PES University.

Radio telescope surveys pick up far more pulsar candidates than anyone can check by hand, and most of them are noise or radio interference. This project does two things with the HTRU2 dataset:

1. **Identify** real pulsars among all candidates using supervised learning (GDA, Random Forest, SMOTE, XGBoost).
2. **Categorize** the confirmed pulsars using unsupervised learning (K-means and a Self-Organizing Map, with PCA for plots).

It reproduces and extends *Application of Machine Learning Methods to Identify and Categorize Radio Pulsar Signal Candidates* (Debesai, Gutierrez and Koyluoglu, CS229, Stanford, 2020).

## Team

| Member | Part | Files |
|---|---|---|
| Shriya | EDA and supervised learning | `notebooks/01_eda.ipynb`, `src/data_loader.py`, `src/supervised.py` |
| Meghana K | Data checks and unsupervised learning | `src/data.py`, `src/unsupervised.py`, `predict_cluster.py` |

## Project structure

```
Pulsar-Classification-Mini-Project/
|-- notebooks/
|   `-- 01_eda.ipynb          exploratory data analysis
|-- src/
|   |-- data_loader.py        loads the data for the supervised part and the notebook
|   |-- data.py               loads the data for the unsupervised part, downloads it if missing,
|   |                         and checks the skewness/kurtosis column order
|   |-- supervised.py         GDA, Random Forest, RF + SMOTE, XGBoost + SMOTE
|   `-- unsupervised.py       K-means, Self-Organizing Map, PCA plots
|-- predict_cluster.py        demo: assigns one pulsar to its cluster
|-- results/
|   |-- supervised/           classifier metrics and curves
|   `-- unsupervised/         clustering tables and figures
|-- requirements.txt
`-- README.md
```

The folders `data/` and `models/` are created when you run the code. They are not stored in the repository.

## Setup

You need Python and Git installed. We developed and tested everything on Python 3.12.

**Step 1. Clone the repository**

```bash
git clone https://github.com/shriya5752/Pulsar-Classification-Mini-Project.git
cd Pulsar-Classification-Mini-Project
```

**Step 2. Create and activate a virtual environment**

Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\activate
```

macOS or Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Step 3. Install the packages**

```bash
pip install -r requirements.txt
```

**Step 4. Get the dataset**

```bash
python -c "from src.data import load_htru2; df = load_htru2(); print(df.shape)"
```

This downloads HTRU2 from the UCI Machine Learning Repository into `data/HTRU_2.csv` and prints `(17898, 9)` when it is done.

If the download is blocked on your network, do it by hand instead:

1. Download `htru2.zip` from https://archive.ics.uci.edu/dataset/372/htru2
2. Unzip it.
3. Create a folder called `data` in the project folder and put `HTRU_2.csv` inside it.

Run every command in this README from the main project folder, the one that contains `README.md`.

## Running the project

### 1. Unsupervised learning (about 1 minute)

```bash
python src/unsupervised.py
```

This clusters the 1,639 confirmed pulsars. It first reproduces the paper's K-means result on raw features, then applies a signed log transform and standardization, picks the number of clusters, fits K-means and a 5x5 Self-Organizing Map, and draws PCA plots. The console output should include these lines:

```
Paper replication (raw k=3): silhouette=0.441, sizes=[974, 621, 44]
Chosen k = 2 (highest silhouette)
SOM: 25/25 cells used, QE=0.850, TE=0.024
SOM prototypes grouped into k=2: silhouette=0.382, ARI vs K-means=0.850
```

It writes 17 files to `results/unsupervised/` and saves the trained models to `models/unsupervised.joblib`, which the demo script needs.

### 2. Supervised learning (about 2 to 5 minutes)

```bash
python src/supervised.py
```

This holds out a stratified 20% test set, runs 5-fold cross-validation on the rest, and trains GDA (written from scratch), Random Forest, Random Forest with SMOTE, and XGBoost with SMOTE. Each model is tested at thresholds 0.5 and 0.2. The last lines of the output should be:

```
Saved test results to results/supervised/test_results.csv
Saved CV summary to results/supervised/cv_summary.csv
```

### 3. Demo: assign a pulsar to a cluster

Run the unsupervised step first, since the demo loads the models it saves.

Pick any of the 1,639 confirmed pulsars by its row number (0 to 1638):

```bash
python predict_cluster.py --row 10
```

Or type in the 8 feature values yourself, in this order: `ip_mean ip_std ip_skew ip_kurt dm_mean dm_std dm_skew dm_kurt`

```bash
python predict_cluster.py --values 33.9 33.7 4.47 23.5 74.6 68.7 0.85 -0.05
```

The script prints the input next to the median values of its cluster, then the K-means cluster and the Self-Organizing Map cell it falls into. For `--row 10` the result is K-means cluster 1 and SOM cell (row 2, col 2).

The clusters were fit on confirmed pulsars only, so the demo assumes the candidate is already known to be a pulsar.

### 4. EDA notebook (optional)

Jupyter is not in `requirements.txt`, so install it first:

```bash
pip install notebook
jupyter notebook notebooks/01_eda.ipynb
```

Run the cells from top to bottom. The notebook saves its figures to `results/`.

## Output files

### `results/supervised/`

| File | Contents |
|---|---|
| `test_results.csv` | Accuracy, recall, precision, F1, MCC, ROC-AUC and PR-AUC on the test set, for each model at thresholds 0.5 and 0.2 |
| `cv_summary.csv` | Mean 5-fold cross-validation accuracy, recall, F1 and AUC for each model |
| `roc_curves.png` | ROC curves for all four models |
| `pr_curves.png` | Precision-recall curves for all four models |

### `results/unsupervised/`

| File | Contents |
|---|---|
| `summary.json` | All headline numbers in one place |
| `k_selection.csv`, `k_selection.png` | Silhouette, Davies-Bouldin, Calinski-Harabasz and bootstrap stability for k = 2 to 6 |
| `silhouette_by_preprocessing.csv` | Silhouette per k on raw, standardized, and log plus standardized features |
| `raw_variance_share.csv` | How much of the total variance each feature holds in raw units |
| `feature_skewness.csv`, `feature_transform.png` | Skewness before and after the log transform |
| `paper_vs_ours_pca.png` | The paper's clustering next to ours, on PCA axes |
| `pca_kmeans_vs_som.png` | K-means and SOM clusters on the same PCA axes |
| `pca_loadings.csv`, `pca_feature_correlations.csv`, `pca_feature_correlations.png` | What the two principal components represent |
| `som_maps.png`, `som_sigma_sweep.csv` | SOM distance map, hits per cell, and the sigma comparison |
| `kmeans_cluster_medians.csv`, `som_cluster_medians.csv`, `kmeans_cluster_profiles.png` | What each cluster looks like in the original units |

The EDA figures (`feature_distributions.png`, `correlation_matrix.png`, `pairplot.png`) are in `results/`.

## Key results

**Supervised, test set (3,580 candidates, 328 pulsars)**

| Model | Threshold | Accuracy | Recall | Precision | F1 |
|---|---|---|---|---|---|
| GDA | 0.5 | 97.82% | 89.63% | 86.98% | 88.29% |
| Random Forest | 0.5 | 97.85% | 87.80% | 88.62% | 88.21% |
| Random Forest | 0.2 | 96.87% | 90.55% | 78.57% | 84.14% |
| RF + SMOTE | 0.5 | 97.09% | 89.33% | 80.94% | 84.93% |
| XGBoost + SMOTE | 0.5 | 97.04% | 89.94% | 80.16% | 84.77% |

**Unsupervised, 1,639 confirmed pulsars**

| Method | Setting | Silhouette | Cluster sizes |
|---|---|---|---|
| Paper replication | Raw features, k = 3 | 0.441 | 974 / 621 / 44 |
| K-means | Signed log + z-score, k = 2 | 0.391 | 764 / 875 |
| SOM, prototypes grouped | 5x5 grid, sigma 1.5, k = 2 | 0.382 | 776 / 863 |

The paper's three clusters come from unscaled features: in raw units, three features hold 89.3% of the variance and decide all the distances. After scaling, the pulsars form two stable groups (bootstrap ARI 0.968) that K-means and the SOM agree on (ARI 0.850).

## A note on the data

The UCI page and the reference paper disagree on whether skewness or kurtosis comes first in each group of four features. The file itself has mean, standard deviation, **skewness**, then **excess kurtosis**. We confirmed this mathematically: for any distribution, excess kurtosis is at least skewness squared minus 2. With our column order that holds for every row, and with the paper's order it fails for 16.8% of rows (pulse profile) and 93.1% (DM-SNR curve). `src/data.py` repeats this check every time it loads the data.

## Reproducibility

All random seeds are fixed (`random_state = 42`), so running the scripts again gives the same numbers on the same package versions. XGBoost results can shift slightly with a different `xgboost` version.

## References

1. Lyon, R. J., Stappers, B. W., Cooper, S., Brooke, J. M., and Knowles, J. D. (2016). Fifty years of pulsar candidate selection: from simple filters to a new principled real-time classification approach. *Monthly Notices of the Royal Astronomical Society*, 459(1), 1104-1123.
2. Debesai, S., Gutierrez, C., and Koyluoglu, N. U. (2020). Application of machine learning methods to identify and categorize radio pulsar signal candidates. CS229, Stanford University.
3. Lyon, R. (2015). HTRU2 [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DK6R
