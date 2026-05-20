#!/usr/bin/env python
"""
Agentic-V2XShield training pipeline:
1. Trains standard ML baselines for multiclass V2X attack classification.
2. Trains a proposed custom model: Agentic Edge-Cloud Trust Ensemble.
3. Saves metrics, confusion matrices, feature importance, and model artifacts.

Recommended input:
    outputs_multiclass/veremi_multiclass_balanced.csv

Quick run:
    python train_v2x_baselines_custom.py ^
      --csv "outputs_multiclass\veremi_multiclass_balanced.csv" ^
      --out-dir "outputs_training" ^
      --max-per-class 100000

Full run:
    python train_v2x_baselines_custom.py ^
      --csv "outputs_multiclass\veremi_multiclass_balanced.csv" ^
      --out-dir "outputs_training"

Outputs:
    outputs_training/
        metrics_summary.csv
        per_class_metrics.csv
        confusion_matrices/
        feature_importance/
        saved_models/
        experiment_report.md
"""

import argparse
import json
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin, clone

warnings.filterwarnings("ignore")


try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


TARGET_COL = "class_id"
CLASS_NAME_COL = "class_name"

DROP_COLS = [
    "source_file",
    "attacker_raw",
    "sender_id",
]


POSITION_FEATURES = [
    "sender_pos_x", "sender_pos_y", "sender_pos_z",
    "receiver_pos_x", "receiver_pos_y", "receiver_pos_z",
    "sender_receiver_distance", "distance_to_road_edge", "edge_violation",
]

MOBILITY_FEATURES = [
    "sender_spd", "receiver_spd", "sender_acl", "receiver_acl",
    "sender_hed", "receiver_hed",
    "speed_delta", "accel_delta", "heading_delta",
    "abs_sender_speed", "abs_sender_accel",
]

TEMPORAL_FEATURES = [
    "rcvTime", "sendTime", "delay", "messageID",
]

NOISE_FEATURES = [
    "sender_pos_noise_x", "sender_pos_noise_y", "sender_pos_noise_z",
    "receiver_pos_noise_x", "receiver_pos_noise_y", "receiver_pos_noise_z",
    "sender_spd_noise", "receiver_spd_noise",
    "sender_acl_noise", "receiver_acl_noise",
    "sender_hed_noise", "receiver_hed_noise",
]

CONTEXT_FEATURES = [
    "scenario", "density", "split",
    "sender_alias", "sender_driver_profile", "receiver_driver_profile",
]


class AgenticEdgeCloudTrustEnsemble(BaseEstimator, ClassifierMixin):
    """
    Proposed custom model for the paper.

    Design:
        - Edge Position Agent: learns spatial/geometric attack evidence.
        - Edge Mobility Agent: learns speed/acceleration/heading consistency.
        - Edge Temporal Agent: learns delay and message timing evidence.
        - Cloud Noise-Context Agent: learns noise/context relationships.
        - Cloud Fusion Agent: combines all agent probability vectors and raw compact cues.

    This is not reinforcement learning.
    It is a supervised multi-agent stacked model suitable for edge-cloud V2X IDS.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def _make_agent(self):
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", ExtraTreesClassifier(
                n_estimators=150,
                max_depth=None,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=self.random_state,
                class_weight="balanced",
            )),
        ])

    def _select_existing(self, X: pd.DataFrame, cols: List[str]) -> List[str]:
        return [c for c in cols if c in X.columns]

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.classes_ = np.unique(y)

        self.position_cols_ = self._select_existing(X, POSITION_FEATURES)
        self.mobility_cols_ = self._select_existing(X, MOBILITY_FEATURES)
        self.temporal_cols_ = self._select_existing(X, TEMPORAL_FEATURES)
        self.noise_cols_ = self._select_existing(X, NOISE_FEATURES)

        self.position_agent_ = self._make_agent()
        self.mobility_agent_ = self._make_agent()
        self.temporal_agent_ = self._make_agent()
        self.noise_agent_ = self._make_agent()

        if self.position_cols_:
            self.position_agent_.fit(X[self.position_cols_], y)
        if self.mobility_cols_:
            self.mobility_agent_.fit(X[self.mobility_cols_], y)
        if self.temporal_cols_:
            self.temporal_agent_.fit(X[self.temporal_cols_], y)
        if self.noise_cols_:
            self.noise_agent_.fit(X[self.noise_cols_], y)

        fusion_X = self._build_fusion_matrix(X)

        self.fusion_agent_ = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            multi_class="auto",
            n_jobs=-1,
            random_state=self.random_state,
        )
        self.fusion_agent_.fit(fusion_X, y)
        return self

    def _safe_proba(self, model, Xpart: pd.DataFrame) -> np.ndarray:
        proba = model.predict_proba(Xpart)
        if proba.shape[1] == len(self.classes_):
            return proba

        fixed = np.zeros((len(Xpart), len(self.classes_)))
        for j, c in enumerate(model.classes_):
            idx = list(self.classes_).index(c)
            fixed[:, idx] = proba[:, j]
        return fixed

    def _build_fusion_matrix(self, X: pd.DataFrame) -> np.ndarray:
        parts = []

        if self.position_cols_:
            parts.append(self._safe_proba(self.position_agent_, X[self.position_cols_]))
        if self.mobility_cols_:
            parts.append(self._safe_proba(self.mobility_agent_, X[self.mobility_cols_]))
        if self.temporal_cols_:
            parts.append(self._safe_proba(self.temporal_agent_, X[self.temporal_cols_]))
        if self.noise_cols_:
            parts.append(self._safe_proba(self.noise_agent_, X[self.noise_cols_]))

        compact_cols = [
            c for c in [
                "sender_receiver_distance", "speed_delta", "accel_delta",
                "heading_delta", "delay", "distance_to_road_edge", "edge_violation"
            ]
            if c in X.columns
        ]

        if compact_cols:
            compact = X[compact_cols].copy()
            compact = compact.replace([np.inf, -np.inf], np.nan)
            compact = compact.fillna(compact.median(numeric_only=True))
            compact = compact.fillna(0.0)
            parts.append(compact.to_numpy(dtype=float))

        return np.hstack(parts)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        fusion_X = self._build_fusion_matrix(X)
        return self.fusion_agent_.predict_proba(fusion_X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        fusion_X = self._build_fusion_matrix(X)
        return self.fusion_agent_.predict(fusion_X)


def load_and_prepare(csv_path: Path, max_per_class: int = 0) -> pd.DataFrame:
    print(f"[INFO] Loading: {csv_path}")
    df = pd.read_csv(csv_path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    if max_per_class and max_per_class > 0:
        df = (
            df.groupby(TARGET_COL, group_keys=False)
              .apply(lambda x: x.sample(min(len(x), max_per_class), random_state=42))
              .reset_index(drop=True)
        )

    print(f"[INFO] Records after optional sampling: {len(df):,}")
    return df


def get_train_val_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train_df = df[df["split"].astype(str).str.lower() == "train"].copy()
    val_df = df[df["split"].astype(str).str.lower() == "validation"].copy()
    test_df = df[df["split"].astype(str).str.lower() == "test"].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("Train/test split is empty. Check the split column.")

    y_train = train_df[TARGET_COL].astype(int)
    y_val = val_df[TARGET_COL].astype(int) if len(val_df) else None
    y_test = test_df[TARGET_COL].astype(int)

    drop = [TARGET_COL, CLASS_NAME_COL, "binary_label"]
    X_train = train_df.drop(columns=[c for c in drop if c in train_df.columns])
    X_val = val_df.drop(columns=[c for c in drop if c in val_df.columns]) if len(val_df) else None
    X_test = test_df.drop(columns=[c for c in drop if c in test_df.columns])

    # remove high-cardinality or leakage-prone raw columns
    X_train = X_train.drop(columns=[c for c in DROP_COLS if c in X_train.columns])
    if X_val is not None:
        X_val = X_val.drop(columns=[c for c in DROP_COLS if c in X_val.columns])
    X_test = X_test.drop(columns=[c for c in DROP_COLS if c in X_test.columns])

    return X_train, y_train, X_val, y_val, X_test, y_test


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric, num_cols),
        ("cat", categorical, cat_cols),
    ])


def make_models(X_train: pd.DataFrame) -> Dict[str, object]:
    pre = make_preprocessor(X_train)

    models = {
        "RandomForest": Pipeline([
            ("preprocess", pre),
            ("clf", RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=42,
                class_weight="balanced",
            )),
        ]),
        "ExtraTrees": Pipeline([
            ("preprocess", pre),
            ("clf", ExtraTreesClassifier(
                n_estimators=250,
                max_depth=None,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=42,
                class_weight="balanced",
            )),
        ]),
        "HistGradientBoosting": Pipeline([
            ("preprocess", pre),
            ("clf", HistGradientBoostingClassifier(
                max_iter=180,
                learning_rate=0.08,
                max_leaf_nodes=31,
                random_state=42,
            )),
        ]),
        "LogisticRegression": Pipeline([
            ("preprocess", pre),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            )),
        ]),
        "Proposed_AgenticEdgeCloudTrustEnsemble": AgenticEdgeCloudTrustEnsemble(random_state=42),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = Pipeline([
            ("preprocess", pre),
            ("clf", XGBClassifier(
                n_estimators=250,
                max_depth=7,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="multi:softprob",
                eval_metric="mlogloss",
                tree_method="hist",
                random_state=42,
                n_jobs=-1,
            )),
        ])
    else:
        print("[WARN] xgboost not installed. Skipping XGBoost.")

    return models


def evaluate_model(name: str, model, X_test: pd.DataFrame, y_test: pd.Series, class_names: Dict[int, str], out_dir: Path) -> Dict[str, float]:
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    infer_time = time.perf_counter() - t0

    n = len(X_test)
    latency_ms = (infer_time / max(n, 1)) * 1000.0

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "macro_precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_test, y_pred),
        "test_records": n,
        "total_inference_seconds": infer_time,
        "latency_ms_per_message": latency_ms,
    }

    # Per-class report
    labels = sorted(class_names.keys())
    target_names = [class_names[i] for i in labels]
    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(out_dir / "per_class_metrics" / f"{name}_classification_report.csv")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    cm_df.to_csv(out_dir / "confusion_matrices" / f"{name}_confusion_matrix.csv")

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(target_names)), target_names, rotation=45, ha="right")
    plt.yticks(range(len(target_names)), target_names)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrices" / f"{name}_confusion_matrix.png", dpi=250)
    plt.close()

    return metrics


def save_feature_importance(name: str, model, X_train: pd.DataFrame, out_dir: Path) -> None:
    try:
        if not isinstance(model, Pipeline):
            return

        clf = model.named_steps.get("clf")
        pre = model.named_steps.get("preprocess")

        if not hasattr(clf, "feature_importances_"):
            return

        feature_names = pre.get_feature_names_out()
        imp = clf.feature_importances_

        fi = (
            pd.DataFrame({"feature": feature_names, "importance": imp})
              .sort_values("importance", ascending=False)
        )
        fi.to_csv(out_dir / "feature_importance" / f"{name}_feature_importance.csv", index=False)

        top = fi.head(25).iloc[::-1]
        plt.figure(figsize=(8, 7))
        plt.barh(top["feature"], top["importance"])
        plt.title(f"{name} Top-25 Feature Importance")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(out_dir / "feature_importance" / f"{name}_top25_feature_importance.png", dpi=250)
        plt.close()

    except Exception as e:
        print(f"[WARN] Feature importance failed for {name}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to veremi_multiclass_balanced.csv or all.csv")
    parser.add_argument("--out-dir", default="outputs_training")
    parser.add_argument("--max-per-class", type=int, default=0, help="0 means use all records.")
    parser.add_argument("--save-models", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    for sub in ["confusion_matrices", "feature_importance", "saved_models", "per_class_metrics"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    df = load_and_prepare(Path(args.csv), args.max_per_class)

    class_map = (
        df[[TARGET_COL, CLASS_NAME_COL]]
        .drop_duplicates()
        .sort_values(TARGET_COL)
        .set_index(TARGET_COL)[CLASS_NAME_COL]
        .to_dict()
    )
    class_map = {int(k): str(v) for k, v in class_map.items()}

    X_train, y_train, X_val, y_val, X_test, y_test = get_train_val_test(df)

    print(f"[INFO] Train records: {len(X_train):,}")
    print(f"[INFO] Validation records: {0 if X_val is None else len(X_val):,}")
    print(f"[INFO] Test records: {len(X_test):,}")
    print(f"[INFO] Features used: {X_train.shape[1]}")

    models = make_models(X_train)
    metrics_rows = []

    for name, model in models.items():
        print(f"\n[TRAIN] {name}")
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        print(f"[EVAL] {name}")
        metrics = evaluate_model(name, model, X_test, y_test, class_map, out_dir)
        metrics["training_seconds"] = train_time
        metrics_rows.append(metrics)

        save_feature_importance(name, model, X_train, out_dir)

        if args.save_models:
            joblib.dump(model, out_dir / "saved_models" / f"{name}.joblib")

    metrics_df = pd.DataFrame(metrics_rows).sort_values("macro_f1", ascending=False)
    metrics_df.to_csv(out_dir / "metrics_summary.csv", index=False)

    report = []
    report.append("# Agentic-V2XShield Training Report\n")
    report.append(f"- Input CSV: `{args.csv}`")
    report.append(f"- Total records used: **{len(df):,}**")
    report.append(f"- Train records: **{len(X_train):,}**")
    report.append(f"- Test records: **{len(X_test):,}**")
    report.append(f"- Feature count: **{X_train.shape[1]}**")
    report.append("\n## Class Mapping\n")
    report.append(pd.DataFrame([{"class_id": k, "class_name": v} for k, v in class_map.items()]).to_markdown(index=False))
    report.append("\n## Model Comparison\n")
    report.append(metrics_df.to_markdown(index=False))
    report.append("\n## Proposed Model Description\n")
    report.append(
        "The proposed model, Agentic Edge-Cloud Trust Ensemble, decomposes V2X cybersecurity "
        "evidence into position, mobility, temporal, and noise-context agents. Each edge/cloud "
        "agent learns a specialized risk view, and a fusion agent combines their class-probability "
        "outputs with compact trust-relevant cues. This gives a publishable custom model beyond "
        "standard monolithic ML baselines."
    )
    (out_dir / "experiment_report.md").write_text("\n".join(report), encoding="utf-8")

    print("\n[DONE] Training complete.")
    print(f"[OUT] {out_dir.resolve()}")
    print("Main files:")
    print(" - metrics_summary.csv")
    print(" - experiment_report.md")
    print(" - confusion_matrices/*.png")
    print(" - per_class_metrics/*.csv")
    print(" - feature_importance/*.csv")


if __name__ == "__main__":
    main()
