#!/usr/bin/env python3



#======================================================================================
#======================================================================================
#lopo_trainer.py
#======================================================================================
#======================================================================================

"""
LOPO Trainer (production-ready)

Usage:
    python lopo_trainer.py --features path/to/features.csv --output results/lopo_out \
        --aggregate patient --model rf

Notes:
- Requires feature CSV with columns: patient_id, label, <feature columns...>
- If label is missing per-window but you have an annotations.csv mapping patient->label,
  you can join externally and provide a features CSV that includes patient-level label.

Outputs (in output_dir):
- fold_models/model_patient_<PATIENT>.joblib
- fold_predictions/predictions_patient_<PATIENT>.csv
- results_summary.csv   (per-fold metrics + aggregated mean/std)
- roc_curve.png, pr_curve.png
- preprocessing/scaler.joblib (last fold's scaler)
- training_log.json
"""

import argparse
import os
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

RND = 42

# -----------------------
# Utilities
# -----------------------
def ensure_dir(d):
    Path(d).mkdir(parents=True, exist_ok=True)

def patient_majority_label(df, patient_col="patient_id", label_col="label"):
    """Return dict patient->majority label."""
    out = {}
    for pid, g in df.groupby(patient_col):
        labels = g[label_col].dropna().astype(int).values
        if len(labels) == 0:
            out[pid] = None
        else:
            out[pid] = int(Counter(labels).most_common(1)[0][0])
    return out

def aggregate_patient_features(df, patient_col="patient_id", agg="mean"):
    """
    Aggregate windows per patient.
    agg in ["mean","median"]
    Returns X (patients x features), y (patients), patient_ids (list)
    """
    # Only numeric columns
    feature_cols = df.select_dtypes(include=['float64','int64']).columns.tolist()
    # Remove patient_col and label if accidentally included
    feature_cols = [c for c in feature_cols if c not in [patient_col, "label"]]

    if agg == "mean":
        agg_df = df.groupby(patient_col)[feature_cols].mean()
    elif agg == "median":
        agg_df = df.groupby(patient_col)[feature_cols].median()
    else:
        raise ValueError("Unsupported agg")
    return agg_df.reset_index(), feature_cols


def compute_metrics(y_true, y_pred, y_score=None):
    m = {}
    try:
        m["auc"] = roc_auc_score(y_true, y_score) if (y_score is not None and len(np.unique(y_true))>1) else np.nan
    except Exception:
        m["auc"] = np.nan
    try:
        m["auprc"] = average_precision_score(y_true, y_score) if (y_score is not None and len(np.unique(y_true))>1) else np.nan
    except Exception:
        m["auprc"] = np.nan
    m["accuracy"] = accuracy_score(y_true, y_pred)
    m["precision"] = precision_score(y_true, y_pred, zero_division=0)
    m["recall"] = recall_score(y_true, y_pred, zero_division=0)
    m["f1"] = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    m["tn"], m["fp"], m["fn"], m["tp"] = int(tn), int(fp), int(fn), int(tp)
    return m

# -----------------------
# Trainer
# -----------------------
def run_lopo(features_csv, output_dir, aggregate_mode="patient", model_type="rf",
             patient_col="patient_id", label_col="label", scaler=True):
    df = pd.read_csv(features_csv)
    assert patient_col in df.columns, f"{patient_col} not found in features CSV"
    assert label_col in df.columns, f"{label_col} not found in features CSV"

    # Optional: drop rows with NaN label (we can't use them)
    df = df[~df[label_col].isna()].copy()
    df[label_col] = df[label_col].astype(int)

    ensure_dir(output_dir)
    models_dir = Path(output_dir) / "fold_models"
    preds_dir = Path(output_dir) / "fold_predictions"
    plots_dir = Path(output_dir) / "plots"
    ensure_dir(models_dir); ensure_dir(preds_dir); ensure_dir(plots_dir)

    # If aggregation requested -> patient-level table
    if aggregate_mode == "patient":
        agg_df, feat_cols = aggregate_patient_features(df, patient_col=patient_col, agg="mean")
        # bring labels (majority) back
        labels_map = patient_majority_label(df, patient_col, label_col)
        agg_df[label_col] = agg_df[patient_col].map(labels_map)
        agg_df = agg_df.dropna(subset=[label_col]).reset_index(drop=True)
        X_all = agg_df[feat_cols].values
        y_all = agg_df[label_col].values
        patient_ids = agg_df[patient_col].astype(str).tolist()
        is_patient_level = True
    elif aggregate_mode == "none":
        feat_cols = [c for c in df.columns if c not in [patient_col, label_col, "window_id", "file"]]
        X_all = df[feat_cols].values
        y_all = df[label_col].values
        patient_ids = df[patient_col].astype(str).tolist()
        is_patient_level = False
    else:
        raise ValueError("Unsupported aggregate_mode")

    # Build mapping patient -> indices (for LOPO)
    unique_patients = list(pd.Series(patient_ids).unique())
    print(f"Found {len(unique_patients)} unique patients for LOPO")

    per_fold_metrics = []
    all_fold_scores = []
    y_true_all = []
    y_score_all = []
    y_pred_all = []
    y_pid_all = []

    for test_pid in unique_patients:
        # select indices: test indices are those with patient == test_pid
        test_mask = np.array([str(p) == str(test_pid) for p in patient_ids])
        train_mask = ~test_mask

        X_train, X_test = X_all[train_mask], X_all[test_mask]
        y_train, y_test = y_all[train_mask], y_all[test_mask]

        # If training labels are single-class, skip fold (can't train)
        if len(np.unique(y_train)) < 2:
            print(f"Skipping LOPO fold for patient {test_pid} — training labels single-class")
            continue

        # Scaling
        scaler_obj = None
        if scaler:
            scaler_obj = StandardScaler().fit(X_train)
            X_train = scaler_obj.transform(X_train)
            X_test = scaler_obj.transform(X_test)

        # Model selection
        if model_type == "rf":
            clf = RandomForestClassifier(n_estimators=200, random_state=RND, class_weight="balanced")
        elif model_type == "lr":
            clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RND)
        else:
            raise ValueError("Unsupported model type")

        # Train
        clf.fit(X_train, y_train)

        # Predict (for patient-level, test set may have 1 row)
        y_score = None
        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(X_test)[:, 1]
        else:
            y_score = clf.decision_function(X_test)
        y_pred = (y_score >= 0.5).astype(int)

        # If fold has multiple rows (window-level), aggregate predictions to patient-level by mean score
        if len(y_score) > 1 and not is_patient_level:
            # aggregate
            score_mean = np.mean(y_score)
            pred_patient = int(score_mean >= 0.5)
            y_score_fold = np.array([score_mean])
            y_pred_fold = np.array([pred_patient])
            y_test_fold = np.array([Counter(y_test).most_common(1)[0][0]])  # majority true label
        else:
            y_score_fold = y_score
            y_pred_fold = y_pred
            y_test_fold = y_test

        # Metrics
        metrics = compute_metrics(y_test_fold, y_pred_fold, y_score_fold)
        metrics["test_patient"] = test_pid
        per_fold_metrics.append(metrics)

        # Save model + scaler
        joblib.dump(clf, str(models_dir / f"model_patient_{test_pid}.joblib"))
        if scaler and scaler_obj is not None:
            joblib.dump(scaler_obj, str(models_dir / f"scaler_patient_{test_pid}.joblib"))

        # Save predictions details
        pred_df = pd.DataFrame({
            "patient_id": [test_pid]*len(y_score),
            "y_true": y_test,
            "y_score": y_score,
            "y_pred": y_pred
        })
        pred_df.to_csv(str(preds_dir / f"predictions_patient_{test_pid}.csv"), index=False)

        # Accumulate for global curves
        y_true_all.extend(y_test_fold.tolist())
        y_score_all.extend(y_score_fold.tolist())
        y_pred_all.extend(y_pred_fold.tolist())
        y_pid_all.extend([test_pid]*len(y_score_fold))

    # Summarize
    per_fold_df = pd.DataFrame(per_fold_metrics)
    if per_fold_df.shape[0] == 0:
        raise RuntimeError("No LOPO folds ran successfully (check labels and patient distribution).")
    # overall aggregated stats (mean, std)
    summary = per_fold_df.describe().loc[["mean","std"],:].T
    summary.columns = ["mean", "std"]
    per_fold_df.to_csv(Path(output_dir)/"results_per_fold.csv", index=False)
    summary.to_csv(Path(output_dir)/"results_summary.csv")

    # Save combined predictions
    all_pred_df = pd.DataFrame({
        "patient_id": y_pid_all,
        "y_true": y_true_all,
        "y_score": y_score_all,
        "y_pred": y_pred_all
    })
    all_pred_df.to_csv(Path(output_dir)/"results_all_predictions.csv", index=False)

    # Plot ROC & PR if possible
    try:
        import sklearn.metrics as skm
        if len(np.unique(y_true_all)) > 1:
            fpr, tpr, _ = skm.roc_curve(y_true_all, y_score_all)
            pr_recall, pr_prec, _ = skm.precision_recall_curve(y_true_all, y_score_all)
            plt.figure(figsize=(6,5))
            plt.plot(fpr, tpr, label=f'ROC (AUC={roc_auc_score(y_true_all, y_score_all):.3f})')
            plt.plot([0,1],[0,1],"--", color="grey")
            plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve")
            plt.legend(); plt.grid(True); plt.savefig(Path(output_dir)/"plots/roc_curve.png"); plt.close()

            plt.figure(figsize=(6,5))
            plt.plot(pr_recall, pr_prec, label=f'PR (AUPRC={average_precision_score(y_true_all, y_score_all):.3f})')
            plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve")
            plt.legend(); plt.grid(True); plt.savefig(Path(output_dir)/"plots/pr_curve.png"); plt.close()
    except Exception as e:
        print("Could not plot ROC/PR:", e)

    # Save run log
    run_log = {
        "n_patients": len(unique_patients),
        "n_folds_run": len(per_fold_metrics),
        "model": model_type,
        "aggregate_mode": aggregate_mode,
        "feature_count": len(feat_cols),
        "timestamp": pd.Timestamp.now().isoformat()
    }
    with open(Path(output_dir)/"training_log.json", "w") as f:
        json.dump(run_log, f, indent=2)

    print("LOPO complete. Summary saved at:", Path(output_dir)/"results_summary.csv")
    return Path(output_dir)

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LOPO Trainer")
    parser.add_argument("--features", required=True, help="Features CSV (one row per window)")
    parser.add_argument("--output", required=True, help="Output folder")
    parser.add_argument("--aggregate", choices=["patient","none"], default="patient",
                        help="Aggregation mode. patient -> aggregate windows to patient (mean). none -> window-level LOPO")
    parser.add_argument("--model", choices=["rf","lr"], default="rf", help="Model type")
    parser.add_argument("--patient_col", default="patient_id", help="Patient id column name")
    parser.add_argument("--label_col", default="label", help="Label column name")
    args = parser.parse_args()

    run_lopo(args.features, args.output, aggregate_mode=args.aggregate,
             model_type=args.model, patient_col=args.patient_col, label_col=args.label_col)


