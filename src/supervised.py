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

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, confusion_matrix, roc_auc_score,
)

# Allow imports from src/
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
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    return metrics


def print_metrics(m):
    print(f"--- {m['model']} ---")
    for k in ["accuracy", "recall", "precision", "f1"]:
        print(f"  {k:>10}: {m[k]:.4f}")
    if "roc_auc" in m:
        print(f"  {'roc_auc':>10}: {m['roc_auc']:.4f}")
    print(f"  confusion matrix:\n{confusion_matrix}"
          if False else "")


# ---------------- Main: GDA baseline ----------------

def main():
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


if __name__ == "__main__":
    main()