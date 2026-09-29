# src/supervised.py
"""
Supervised learning for pulsar candidate classification.

Implements:
  1. Gaussian Discriminant Analysis (GDA) - baseline
  2. Random Forest - main model (added next)
"""
import os
import sys
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold 
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    matthews_corrcoef,              # ← ADD THIS
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_data

# ---------------- GDA (from scratch) ----------------

def fit_gda(X, y):
    """
    Fit Gaussian Discriminant Analysis with shared covariance.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,) with values in {0, 1}

    Returns
    -------
    dict with phi, mu0, mu1, Sigma, Sigma_inv, Sigma_det
    """
    n, d = X.shape

    phi = np.mean(y)                              # P(y=1)
    mu0 = X[y == 0].mean(axis=0)                  # mean of class 0
    mu1 = X[y == 1].mean(axis=0)                  # mean of class 1

    # Shared covariance
    X_centered = X.copy()
    X_centered[y == 0] -= mu0
    X_centered[y == 1] -= mu1
    Sigma = (X_centered.T @ X_centered) / n

    # Add small ridge for numerical stability (covariance may be near-singular)
    Sigma += np.eye(d) * 1e-6

    Sigma_inv = np.linalg.inv(Sigma)
    Sigma_det = np.linalg.det(Sigma)

    return {
        "phi": phi,
        "mu0": mu0,
        "mu1": mu1,
        "Sigma": Sigma,
        "Sigma_inv": Sigma_inv,
        "Sigma_det": Sigma_det,
    }


def _gaussian_log_pdf(X, mu, Sigma_inv, Sigma_det):
    """Log N(X | mu, Sigma) for each row of X."""
    d = X.shape[1]
    diff = X - mu
    # (x-mu)^T Sigma^-1 (x-mu) for each row
    mahalanobis = np.einsum("ij,jk,ik->i", diff, Sigma_inv, diff)
    log_norm = -0.5 * (d * np.log(2 * np.pi) + np.log(Sigma_det))
    return log_norm - 0.5 * mahalanobis


def predict_proba_gda(params, X):
    """Return P(y=1 | x) for each row of X."""
    log_p0 = np.log(1 - params["phi"]) + _gaussian_log_pdf(
        X, params["mu0"], params["Sigma_inv"], params["Sigma_det"]
    )
    log_p1 = np.log(params["phi"]) + _gaussian_log_pdf(
        X, params["mu1"], params["Sigma_inv"], params["Sigma_det"]
    )

    # Normalize in log-space to avoid overflow
    log_sum = np.logaddexp(log_p0, log_p1)
    return np.exp(log_p1 - log_sum)


def predict_gda(params, X, threshold=0.5):
    return (predict_proba_gda(params, X) >= threshold).astype(int)


# ---------------- Evaluation helpers ----------------

def evaluate(y_true, y_pred, y_proba=None, name="model"):
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        metrics["pr_auc"] = average_precision_score(y_true, y_proba)
    return metrics



# ---------------- Random Forest ----------------

def fit_random_forest(X_train, y_train, tune=True):
    """
    Fit Random Forest. If tune=True, use RandomizedSearchCV
    over hyperparameters (matching paper's approach).
    """
    if not tune:
        rf = RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        )
        rf.fit(X_train, y_train)
        return rf

    param_dist = {
        "n_estimators": randint(100, 500),
        "max_depth": [None, 10, 20, 30, 50, 100, 110],
        "min_samples_split": randint(2, 20),
        "max_features": ["sqrt", "log2", 2, 4, 6],
    }

    base = RandomForestClassifier(random_state=42, n_jobs=-1)

    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=30,
        cv=3,
        scoring="recall",   # maximize recall, matching paper's objective
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    print(f"\n  Best RF params: {search.best_params_}")
    print(f"  Best CV recall: {search.best_score_:.4f}")
    return search.best_estimator_

# ---------------- SMOTE ----------------

def apply_smote(X_train, y_train, random_state=42):
    """Apply SMOTE to the training set only."""
    smote = SMOTE(random_state=random_state)
    return smote.fit_resample(X_train, y_train)

# ---------------- Plotting ----------------

def plot_roc_pr_curves(results_dict, out_dir="results"):
    """
    Plot ROC and Precision-Recall curves for multiple models.

    Parameters
    ----------
    results_dict : dict
        {model_name: (y_true, y_proba)}
    out_dir : str
        Directory to save the plots.
    """
    os.makedirs(out_dir, exist_ok=True)

    # --- ROC curve ---
    plt.figure(figsize=(7, 6))
    for name, (y_true, y_proba) in results_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_curves.png"), dpi=120)
    plt.close()

    # --- PR curve ---
    plt.figure(figsize=(7, 6))
    for name, (y_true, y_proba) in results_dict.items():
        prec, rec, _ = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        plt.plot(rec, prec, label=f"{name} (AP = {ap:.4f})", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pr_curves.png"), dpi=120)
    plt.close()

    print(f"\nSaved ROC and PR curves to {out_dir}/")
    
# ---------------- Main: GDA baseline ----------------

def main():
    # Seed everything for reproducibility
    np.random.seed(42)
    print("Loading data...")
    df = load_data()
    features = [c for c in df.columns if c != "class"]
    X = df[features].values.astype(float)
    y = df["class"].values.astype(int)
    print(f"  X shape: {X.shape}, y shape: {y.shape}")
    print(f"  Positive rate: {y.mean():.4%}")

    # Hold out 20% test set, stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"\nTrain size: {len(y_train)} | Test size: {len(y_test)}")

    # --- GDA with 5-fold CV on the training set ---
    print("\n=== GDA: 5-fold cross-validation on training set ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_metrics = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
        X_va, y_va = X_train[va_idx], y_train[va_idx]

        # Upsample positives in TRAINING fold only (matches paper)
        pos_idx = np.where(y_tr == 1)[0]
        neg_idx = np.where(y_tr == 0)[0]
        # Bring positives up to match negatives
        upsampled_pos = np.random.choice(
            pos_idx, size=len(neg_idx), replace=True
        )
        idx = np.concatenate([neg_idx, upsampled_pos])
        X_tr_up, y_tr_up = X_tr[idx], y_tr[idx]

        params = fit_gda(X_tr_up, y_tr_up)
        y_va_pred = predict_gda(params, X_va)
        y_va_proba = predict_proba_gda(params, X_va)

        m = evaluate(y_va, y_va_pred, y_va_proba, name=f"GDA fold {fold}")
        fold_metrics.append(m)
        print(f"  Fold {fold}: acc={m['accuracy']:.4f}  "
              f"recall={m['recall']:.4f}  "
              f"f1={m['f1']:.4f}  auc={m['roc_auc']:.4f}")

    # Average across folds
    avg = {}
    for k in ["accuracy", "recall", "precision", "f1", "roc_auc"]:
        avg[k] = np.mean([m[k] for m in fold_metrics])
    print(f"\n  CV average: acc={avg['accuracy']:.4f}  "
          f"recall={avg['recall']:.4f}  "
          f"f1={avg['f1']:.4f}  auc={avg['roc_auc']:.4f}")

    # --- Final GDA trained on all training data, evaluated on test set ---
    print("\n=== GDA: trained on full train set, evaluated on test set ===")
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    upsampled_pos = np.random.choice(pos_idx, size=len(neg_idx), replace=True)
    idx = np.concatenate([neg_idx, upsampled_pos])
    X_train_up, y_train_up = X_train[idx], y_train[idx]

    params = fit_gda(X_train_up, y_train_up)
    y_test_pred = predict_gda(params, X_test)
    y_test_proba = predict_proba_gda(params, X_test)

    test_metrics = evaluate(y_test, y_test_pred, y_test_proba, name="GDA test")
    print(f"  Test: acc={test_metrics['accuracy']:.4f}  "
          f"recall={test_metrics['recall']:.4f}  "
          f"precision={test_metrics['precision']:.4f}  "
          f"f1={test_metrics['f1']:.4f}  "
          f"auc={test_metrics['roc_auc']:.4f}")
    print(f"  Confusion matrix:\n{confusion_matrix(y_test, y_test_pred)}")


    # --- Random Forest ---
    print("\n=== Random Forest: 5-fold CV on training set ===")

    fold_metrics = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
        X_va, y_va = X_train[va_idx], y_train[va_idx]

        # Upsample positives in training fold only
        pos_idx = np.where(y_tr == 1)[0]
        neg_idx = np.where(y_tr == 0)[0]
        upsampled_pos = np.random.choice(
            pos_idx, size=len(neg_idx), replace=True
        )
        idx = np.concatenate([neg_idx, upsampled_pos])
        X_tr_up, y_tr_up = X_tr[idx], y_tr[idx]

        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=110,
            max_features=4,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_tr_up, y_tr_up)

        y_va_proba = rf.predict_proba(X_va)[:, 1]

        # Test two thresholds: 0.5 and 0.2 (paper found 0.2 optimal)
        for thr in [0.5, 0.2]:
            y_va_pred = (y_va_proba >= thr).astype(int)
            m = evaluate(y_va, y_va_pred, y_va_proba,
                         name=f"RF fold {fold} thr={thr}")
            m["threshold"] = thr
            if thr == 0.5:
                fold_metrics.append(m)
            print(f"  Fold {fold} thr={thr}: "
                  f"acc={m['accuracy']:.4f}  "
                  f"recall={m['recall']:.4f}  "
                  f"f1={m['f1']:.4f}")

    # Average across folds at threshold 0.5
    avg_rf = {}
    for k in ["accuracy", "recall", "precision", "f1", "roc_auc"]:
        avg_rf[k] = np.mean([m[k] for m in fold_metrics])
    print(f"\n  RF CV average (thr=0.5): acc={avg_rf['accuracy']:.4f}  "
          f"recall={avg_rf['recall']:.4f}  f1={avg_rf['f1']:.4f}  "
          f"auc={avg_rf['roc_auc']:.4f}")

    # --- Final RF on full training set ---
    print("\n=== Random Forest: full train set, test set ===")

    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    upsampled_pos = np.random.choice(pos_idx, size=len(neg_idx), replace=True)
    idx = np.concatenate([neg_idx, upsampled_pos])
    X_train_up, y_train_up = X_train[idx], y_train[idx]

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=110,
        max_features=4,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_up, y_train_up)
    y_test_proba = rf.predict_proba(X_test)[:, 1]

    for thr in [0.5, 0.2]:
        y_test_pred = (y_test_proba >= thr).astype(int)
        m = evaluate(y_test, y_test_pred, y_test_proba,
                     name=f"RF test thr={thr}")
        print(f"\n  RF test (thr={thr}): "
              f"acc={m['accuracy']:.4f}  recall={m['recall']:.4f}  "
              f"precision={m['precision']:.4f}  f1={m['f1']:.4f}  "
              f"auc={m['roc_auc']:.4f}")
        print(f"  Confusion matrix:\n{confusion_matrix(y_test, y_test_pred)}")

    # --- Feature importances ---
    print("\n=== RF Feature Importances ===")
    for name, imp in sorted(
        zip(features, rf.feature_importances_),
        key=lambda x: -x[1],
    ):
        print(f"  {name:>18}: {imp:.4f}")

        # --- SMOTE + Random Forest ---
    print("\n=== Random Forest with SMOTE: 5-fold CV ===")

    fold_metrics_smote = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
        X_va, y_va = X_train[va_idx], y_train[va_idx]

        X_tr_sm, y_tr_sm = apply_smote(X_tr, y_tr)

        rf_sm = RandomForestClassifier(
            n_estimators=200, max_depth=110, max_features=4,
            min_samples_split=5, random_state=42, n_jobs=-1,
        )
        rf_sm.fit(X_tr_sm, y_tr_sm)
        y_va_proba_sm = rf_sm.predict_proba(X_va)[:, 1]
        y_va_pred_sm = (y_va_proba_sm >= 0.5).astype(int)

        m = evaluate(y_va, y_va_pred_sm, y_va_proba_sm,
                     name=f"RF-SMOTE fold {fold}")
        fold_metrics_smote.append(m)
        print(f"  Fold {fold}: acc={m['accuracy']:.4f}  "
              f"recall={m['recall']:.4f}  f1={m['f1']:.4f}")

    avg_sm = {k: np.mean([m[k] for m in fold_metrics_smote])
              for k in ["accuracy", "recall", "precision", "f1", "roc_auc"]}
    print(f"\n  RF-SMOTE CV average: acc={avg_sm['accuracy']:.4f}  "
          f"recall={avg_sm['recall']:.4f}  f1={avg_sm['f1']:.4f}  "
          f"auc={avg_sm['roc_auc']:.4f}")

    # --- Final RF-SMOTE on full train set ---
    print("\n=== RF-SMOTE: test set ===")
    X_train_sm, y_train_sm = apply_smote(X_train, y_train)
    rf_sm = RandomForestClassifier(
        n_estimators=200, max_depth=110, max_features=4,
        min_samples_split=5, random_state=42, n_jobs=-1,
    )
    rf_sm.fit(X_train_sm, y_train_sm)
    y_test_proba_sm = rf_sm.predict_proba(X_test)[:, 1]

    for thr in [0.5, 0.2]:
        y_test_pred_sm = (y_test_proba_sm >= thr).astype(int)
        m = evaluate(y_test, y_test_pred_sm, y_test_proba_sm,
                     name=f"RF-SMOTE test thr={thr}")
        print(f"\n  RF-SMOTE test (thr={thr}): "
              f"acc={m['accuracy']:.4f}  recall={m['recall']:.4f}  "
              f"precision={m['precision']:.4f}  f1={m['f1']:.4f}  "
              f"auc={m['roc_auc']:.4f}")
        print(f"  Confusion matrix:\n{confusion_matrix(y_test, y_test_pred_sm)}")

    # --- XGBoost with SMOTE ---
    print("\n=== XGBoost with SMOTE: 5-fold CV ===")

    fold_metrics_xgb = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
        X_va, y_va = X_train[va_idx], y_train[va_idx]

        X_tr_sm, y_tr_sm = apply_smote(X_tr, y_tr)

        xgb = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42, n_jobs=-1,
            tree_method="hist",
        )
        xgb.fit(X_tr_sm, y_tr_sm)
        y_va_proba_xgb = xgb.predict_proba(X_va)[:, 1]
        y_va_pred_xgb = (y_va_proba_xgb >= 0.5).astype(int)

        m = evaluate(y_va, y_va_pred_xgb, y_va_proba_xgb,
                     name=f"XGBoost fold {fold}")
        fold_metrics_xgb.append(m)
        print(f"  Fold {fold}: acc={m['accuracy']:.4f}  "
              f"recall={m['recall']:.4f}  f1={m['f1']:.4f}")

    avg_xgb = {k: np.mean([m[k] for m in fold_metrics_xgb])
               for k in ["accuracy", "recall", "precision", "f1", "roc_auc"]}
    print(f"\n  XGBoost CV average: acc={avg_xgb['accuracy']:.4f}  "
          f"recall={avg_xgb['recall']:.4f}  f1={avg_xgb['f1']:.4f}  "
          f"auc={avg_xgb['roc_auc']:.4f}")

    # --- Final XGBoost on full train set ---
    print("\n=== XGBoost: test set ===")
    X_train_sm, y_train_sm = apply_smote(X_train, y_train)
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42, n_jobs=-1,
        tree_method="hist",
    )
    xgb.fit(X_train_sm, y_train_sm)
    y_test_proba_xgb = xgb.predict_proba(X_test)[:, 1]

    for thr in [0.5, 0.2]:
        y_test_pred_xgb = (y_test_proba_xgb >= thr).astype(int)
        m = evaluate(y_test, y_test_pred_xgb, y_test_proba_xgb,
                     name=f"XGBoost test thr={thr}")
        print(f"\n  XGBoost test (thr={thr}): "
              f"acc={m['accuracy']:.4f}  recall={m['recall']:.4f}  "
              f"precision={m['precision']:.4f}  f1={m['f1']:.4f}  "
              f"auc={m['roc_auc']:.4f}")
        print(f"  Confusion matrix:\n{confusion_matrix(y_test, y_test_pred_xgb)}")

    # --- Final ROC/PR plots with all four models ---
    # Recompute GDA probas on test set for the plot
    gda_test_proba = predict_proba_gda(
        fit_gda(X_train_up, y_train_up), X_test
    )
    plot_roc_pr_curves({
        "GDA": (y_test, gda_test_proba),
        "RF (upsampled)": (y_test, y_test_proba),
        "RF + SMOTE": (y_test, y_test_proba_sm),
        "XGBoost + SMOTE": (y_test, y_test_proba_xgb),
    })


    # --- Save all test results to CSV ---
    import csv
    test_results = [
        {"model": "GDA", "threshold": 0.5,
         **{k: v for k, v in test_metrics.items() if k != "model"}},
        {"model": "RF (upsampled)", "threshold": 0.5,
         **{k: v for k, v in
            evaluate(y_test, (y_test_proba >= 0.5).astype(int),
                     y_test_proba, name="rf").items() if k != "model"}},
        {"model": "RF (upsampled)", "threshold": 0.2,
         **{k: v for k, v in
            evaluate(y_test, (y_test_proba >= 0.2).astype(int),
                     y_test_proba, name="rf").items() if k != "model"}},
        {"model": "RF + SMOTE", "threshold": 0.5,
         **{k: v for k, v in
            evaluate(y_test, (y_test_proba_sm >= 0.5).astype(int),
                     y_test_proba_sm, name="rf_sm").items() if k != "model"}},
        {"model": "RF + SMOTE", "threshold": 0.2,
         **{k: v for k, v in
            evaluate(y_test, (y_test_proba_sm >= 0.2).astype(int),
                     y_test_proba_sm, name="rf_sm").items() if k != "model"}},
        {"model": "XGBoost + SMOTE", "threshold": 0.5,
         **{k: v for k, v in
            evaluate(y_test, (y_test_proba_xgb >= 0.5).astype(int),
                     y_test_proba_xgb, name="xgb").items() if k != "model"}},
        {"model": "XGBoost + SMOTE", "threshold": 0.2,
         **{k: v for k, v in
            evaluate(y_test, (y_test_proba_xgb >= 0.2).astype(int),
                     y_test_proba_xgb, name="xgb").items() if k != "model"}},
    ]

    os.makedirs("results", exist_ok=True)
    with open("results/test_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=test_results[0].keys())
        writer.writeheader()
        writer.writerows(test_results)
    print(f"\nSaved test results to results/test_results.csv")

    # --- Save CV summary ---
    cv_summary = [
        {"model": "GDA", "cv_accuracy": avg["accuracy"],
         "cv_recall": avg["recall"], "cv_f1": avg["f1"],
         "cv_auc": avg["roc_auc"]},
        {"model": "RF (upsampled)", "cv_accuracy": avg_rf["accuracy"],
         "cv_recall": avg_rf["recall"], "cv_f1": avg_rf["f1"],
         "cv_auc": avg_rf["roc_auc"]},
        {"model": "RF + SMOTE", "cv_accuracy": avg_sm["accuracy"],
         "cv_recall": avg_sm["recall"], "cv_f1": avg_sm["f1"],
         "cv_auc": avg_sm["roc_auc"]},
        {"model": "XGBoost + SMOTE", "cv_accuracy": avg_xgb["accuracy"],
         "cv_recall": avg_xgb["recall"], "cv_f1": avg_xgb["f1"],
         "cv_auc": avg_xgb["roc_auc"]},
    ]

    with open("results/cv_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cv_summary[0].keys())
        writer.writeheader()
        writer.writerows(cv_summary)
    print("Saved CV summary to results/cv_summary.csv")

if __name__ == "__main__":
    main()