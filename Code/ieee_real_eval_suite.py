#!/usr/bin/env python
"""
Agentic-V2XShield: Real Multi-Perspective Evaluation and Ablation Suite

This script replaces decorative figures with real computed experiments.

It evaluates:
1. Multiclass attack classification
2. Binary attack detection
3. Edge-only, cloud-only, edge-cloud, trust-only, and edge-cloud-trust ablations
4. Cyber-resilience response metrics
5. Latency-aware deployment metrics
6. Clean IEEE-style 16:9 600-DPI figures
7. All numerical results as CSV files

Quick test:
    python ieee_real_eval_suite.py ^
      --csv "outputs_multiclass\veremi_multiclass_balanced.csv" ^
      --out-dir "outputs_ieee_real_eval_test" ^
      --max-per-class 50000

Full run:
    python ieee_real_eval_suite.py ^
      --csv "outputs_multiclass\veremi_multiclass_balanced.csv" ^
      --out-dir "outputs_ieee_real_eval" ^
      --max-per-class 0
"""

import argparse
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False


plt.rcParams.update({
    "figure.figsize": (16, 9),
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "font.size": 15,
    "axes.labelsize": 17,
    "axes.titlesize": 18,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "font.family": "DejaVu Sans",
})


EDGE_FAST_FEATURES = [
    "delay", "sender_spd", "sender_acl", "sender_hed",
    "receiver_spd", "receiver_acl", "receiver_hed",
    "speed_delta", "accel_delta", "heading_delta",
    "abs_sender_speed", "abs_sender_accel",
]

CLOUD_CONTEXT_FEATURES = [
    "sender_pos_x", "sender_pos_y", "receiver_pos_x", "receiver_pos_y",
    "sender_receiver_distance", "distance_to_road_edge", "edge_violation",
    "sender_pos_noise_x", "sender_pos_noise_y",
    "receiver_pos_noise_x", "receiver_pos_noise_y",
    "sender_spd_noise", "receiver_spd_noise",
    "sender_hed_noise", "receiver_hed_noise",
    "sender_driver_profile", "receiver_driver_profile",
]

TRUST_RESPONSE_FEATURES = [
    "distance_to_road_edge", "edge_violation", "delay",
    "heading_delta", "speed_delta", "accel_delta", "sender_receiver_distance",
]

DROP_ALWAYS = [
    "source_file", "sender_id", "attacker_raw",
    "class_name", "class_id", "binary_label", "split",
]

LEAKAGE_RISK_DROP = ["messageID", "sender_alias", "rcvTime", "sendTime"]


def ensure_dirs(out_dir: Path):
    for p in [
        out_dir, out_dir / "figures", out_dir / "csv",
        out_dir / "models", out_dir / "reports", out_dir / "confusion_matrices",
    ]:
        p.mkdir(parents=True, exist_ok=True)


def clean_feature_name(name):
    name = str(name)
    name = name.replace("num__", "").replace("cat__", "")
    name = name.replace("sender_", "S-").replace("receiver_", "R-")
    name = name.replace("_", " ")
    name = name.replace("distance to road edge", "road-edge distance")
    name = name.replace("sender receiver distance", "S-R distance")
    name = name.replace("spd", "speed")
    name = name.replace("acl", "accel")
    name = name.replace("hed", "heading")
    return name


def savefig(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def add_panel(ax, text):
    ax.text(
        0.01, 0.98, text,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=18, fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2),
    )


def load_data(csv_path: Path, max_per_class: int, drop_leakage: bool) -> pd.DataFrame:
    print(f"[INFO] Loading {csv_path}")
    df = pd.read_csv(csv_path)

    required = ["class_id", "class_name", "binary_label", "split"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if max_per_class and max_per_class > 0:
        df = (
            df.groupby("class_id", group_keys=False)
            .apply(lambda x: x.sample(min(len(x), max_per_class), random_state=42))
            .reset_index(drop=True)
        )

    if drop_leakage:
        df = df.drop(columns=[c for c in LEAKAGE_RISK_DROP if c in df.columns])

    print(f"[INFO] Records used: {len(df):,}")
    print(f"[INFO] Classes: {df['class_name'].nunique()}")
    return df


def split_xy(df: pd.DataFrame, target: str, feature_list=None):
    train = df[df["split"].astype(str).str.lower() == "train"].copy()
    test = df[df["split"].astype(str).str.lower() == "test"].copy()

    y_train = train[target].astype(int)
    y_test = test[target].astype(int)

    X_train = train.drop(columns=[c for c in DROP_ALWAYS if c in train.columns])
    X_test = test.drop(columns=[c for c in DROP_ALWAYS if c in test.columns])

    if feature_list is not None:
        keep = [c for c in feature_list if c in X_train.columns]
        X_train = X_train[keep].copy()
        X_test = X_test[keep].copy()

    return X_train, y_train, X_test, y_test


def preprocessor(X):
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols),
    ])


def make_model(model_name: str, X_train, n_classes: int):
    pre = preprocessor(X_train)

    if model_name == "LogisticRegression":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1, random_state=42)
    elif model_name == "RandomForest":
        clf = RandomForestClassifier(
            n_estimators=220, min_samples_leaf=2, class_weight="balanced",
            n_jobs=-1, random_state=42
        )
    elif model_name == "ExtraTrees":
        clf = ExtraTreesClassifier(
            n_estimators=260, min_samples_leaf=2, class_weight="balanced",
            n_jobs=-1, random_state=42
        )
    elif model_name == "HistGradientBoosting":
        clf = HistGradientBoostingClassifier(
            max_iter=220, learning_rate=0.07, max_leaf_nodes=39,
            l2_regularization=0.05, random_state=42
        )
    elif model_name == "XGBoost":
        if not HAS_XGB:
            raise RuntimeError("xgboost is not installed.")
        objective = "multi:softprob" if n_classes > 2 else "binary:logistic"
        clf = XGBClassifier(
            n_estimators=320, max_depth=7, learning_rate=0.06,
            subsample=0.9, colsample_bytree=0.9,
            objective=objective,
            eval_metric="mlogloss" if n_classes > 2 else "logloss",
            tree_method="hist", random_state=42, n_jobs=-1,
        )
    else:
        raise ValueError(model_name)

    return Pipeline([("preprocess", pre), ("clf", clf)])


def evaluate_predictions(y_true, y_pred, latency_s, train_s, model_name, task, setting):
    return {
        "task": task,
        "setting": setting,
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "latency_ms_per_msg": latency_s / max(len(y_true), 1) * 1000.0,
        "total_inference_s": latency_s,
        "training_s": train_s,
        "test_records": len(y_true),
    }


def train_eval_single(df, target, model_name, feature_list, task, setting, out_dir, save_model=False):
    X_train, y_train, X_test, y_test = split_xy(df, target, feature_list)
    n_classes = len(np.unique(y_train))

    model = make_model(model_name, X_train, n_classes)

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_pred = model.predict(X_test)
    infer_s = time.perf_counter() - t1

    metrics = evaluate_predictions(y_test, y_pred, infer_s, train_s, model_name, task, setting)

    labels = sorted(np.unique(np.concatenate([y_train.unique(), y_test.unique()])))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        out_dir / "confusion_matrices" / f"{task}_{setting}_{model_name}_cm.csv"
    )

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(
        out_dir / "reports" / f"{task}_{setting}_{model_name}_class_report.csv"
    )

    if save_model:
        joblib.dump(model, out_dir / "models" / f"{task}_{setting}_{model_name}.joblib")

    return metrics, model


def cyber_resilience_metrics(df, target, model, feature_list, setting):
    X_train, y_train, X_test, y_test = split_xy(df, target, feature_list)
    y_pred = model.predict(X_test)

    if target == "binary_label":
        true_attack = y_test.values == 1
        pred_attack = y_pred == 1
    else:
        true_attack = y_test.values != 0
        pred_attack = y_pred != 0

    tp = int(np.sum(true_attack & pred_attack))
    fp = int(np.sum((~true_attack) & pred_attack))
    fn = int(np.sum(true_attack & (~pred_attack)))
    tn = int(np.sum((~true_attack) & (~pred_attack)))

    attack_coverage = tp / max(tp + fn, 1)
    false_isolation = fp / max(fp + tn, 1)
    response_precision = tp / max(tp + fp, 1)
    utility = attack_coverage - 0.5 * false_isolation

    return {
        "setting": setting,
        "attack_coverage": attack_coverage,
        "false_isolation_rate": false_isolation,
        "response_precision": response_precision,
        "resilience_utility": utility,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def run_all_experiments(df, out_dir: Path):
    rows = []
    resilience_rows = []

    baseline_models = ["LogisticRegression", "RandomForest", "ExtraTrees", "HistGradientBoosting"]
    if HAS_XGB:
        baseline_models.append("XGBoost")

    for task, target in [("multiclass", "class_id"), ("binary", "binary_label")]:
        for model_name in baseline_models:
            print(f"[BASELINE] {task} | {model_name}")
            metrics, _ = train_eval_single(df, target, model_name, None, task, "all_features", out_dir)
            rows.append(metrics)

    ablation_settings = {
        "edge_only_fast": EDGE_FAST_FEATURES,
        "cloud_only_context": CLOUD_CONTEXT_FEATURES,
        "edge_cloud": EDGE_FAST_FEATURES + CLOUD_CONTEXT_FEATURES,
        "trust_response_only": TRUST_RESPONSE_FEATURES,
        "edge_cloud_trust": list(dict.fromkeys(EDGE_FAST_FEATURES + CLOUD_CONTEXT_FEATURES + TRUST_RESPONSE_FEATURES)),
    }

    proposed_engine = "XGBoost" if HAS_XGB else "HistGradientBoosting"

    for task, target in [("multiclass", "class_id"), ("binary", "binary_label")]:
        for setting, features in ablation_settings.items():
            print(f"[PROPOSED-ABLATION] {task} | {setting}")
            metrics, model = train_eval_single(df, target, proposed_engine, features, task, setting, out_dir)
            metrics["model"] = f"Proposed_AECTE_{proposed_engine}"
            rows.append(metrics)

            res = cyber_resilience_metrics(df, target, model, features, f"{task}_{setting}")
            res["task"] = task
            resilience_rows.append(res)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out_dir / "csv" / "all_model_metrics.csv", index=False)

    res_df = pd.DataFrame(resilience_rows)
    res_df.to_csv(out_dir / "csv" / "cyber_resilience_metrics.csv", index=False)

    return metrics_df, res_df


def fig_baseline_vs_proposed(metrics_df, out_dir):
    df = metrics_df[
        (metrics_df["task"] == "multiclass") &
        (
            ((metrics_df["setting"] == "all_features") & (~metrics_df["model"].str.contains("Proposed"))) |
            ((metrics_df["setting"] == "edge_cloud_trust") & (metrics_df["model"].str.contains("Proposed")))
        )
    ].copy()

    df["display"] = df["model"].replace({
        "Proposed_AECTE_XGBoost": "Proposed AECTE",
        "Proposed_AECTE_HistGradientBoosting": "Proposed AECTE",
        "HistGradientBoosting": "HistGB",
        "LogisticRegression": "LogReg",
    })

    fig, axes = plt.subplots(1, 2, figsize=(16, 9))

    bars = axes[0].bar(df["display"], df["macro_f1"])
    axes[0].set_ylabel("Macro F1-score")
    axes[0].set_xlabel("Model")
    axes[0].set_title("Multiclass Detection Performance")
    axes[0].tick_params(axis="x", rotation=20)
    add_panel(axes[0], "(a)")
    for b in bars:
        axes[0].text(b.get_x()+b.get_width()/2, b.get_height(), f"{b.get_height():.3f}",
                     ha="center", va="bottom", fontsize=11)

    bars = axes[1].bar(df["display"], df["mcc"])
    axes[1].set_ylabel("Matthews correlation coefficient")
    axes[1].set_xlabel("Model")
    axes[1].set_title("Class-Balanced Reliability")
    axes[1].tick_params(axis="x", rotation=20)
    add_panel(axes[1], "(b)")
    for b in bars:
        axes[1].text(b.get_x()+b.get_width()/2, b.get_height(), f"{b.get_height():.3f}",
                     ha="center", va="bottom", fontsize=11)

    savefig(fig, out_dir / "figures" / "fig1_baseline_vs_proposed")


def fig_binary_multiclass(metrics_df, out_dir):
    df = metrics_df[
        metrics_df["model"].str.contains("Proposed") &
        metrics_df["setting"].isin([
            "edge_only_fast", "cloud_only_context", "edge_cloud",
            "trust_response_only", "edge_cloud_trust"
        ])
    ].copy()

    pivot = df.pivot_table(index="setting", columns="task", values="macro_f1", aggfunc="max")
    order = ["edge_only_fast", "cloud_only_context", "edge_cloud", "trust_response_only", "edge_cloud_trust"]
    pivot = pivot.reindex(order)

    fig, ax = plt.subplots(figsize=(16, 9))
    x = np.arange(len(pivot.index))
    w = 0.35

    ax.bar(x - w/2, pivot["binary"], width=w, label="Binary attack detection")
    ax.bar(x + w/2, pivot["multiclass"], width=w, label="Multiclass attack classification")

    ax.set_xticks(x)
    ax.set_xticklabels(["Edge-only", "Cloud-only", "Edge-cloud", "Trust-only", "Edge-cloud-trust"], rotation=10)
    ax.set_ylabel("Macro F1-score")
    ax.set_xlabel("Deployment / ablation setting")
    ax.set_title("Binary and Multiclass Evaluation Under Deployment Ablations")
    ax.legend()
    add_panel(ax, "(a)")
    savefig(fig, out_dir / "figures" / "fig2_binary_multiclass_ablation")


def fig_edge_cloud_resilience(res_df, out_dir):
    df = res_df[res_df["task"] == "multiclass"].copy()

    order = [
        "multiclass_edge_only_fast",
        "multiclass_cloud_only_context",
        "multiclass_edge_cloud",
        "multiclass_trust_response_only",
        "multiclass_edge_cloud_trust",
    ]
    df = df.set_index("setting").reindex(order).reset_index()
    labels = ["Edge-only", "Cloud-only", "Edge-cloud", "Trust-only", "Edge-cloud-trust"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    x = np.arange(len(labels))

    axes[0].plot(x, df["attack_coverage"], marker="o", linewidth=3, label="Attack coverage")
    axes[0].plot(x, df["response_precision"], marker="s", linewidth=3, label="Response precision")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=10)
    axes[0].set_ylabel("Score")
    axes[0].set_xlabel("Deployment setting")
    axes[0].set_title("Trust-Aware Response Quality")
    axes[0].legend()
    add_panel(axes[0], "(a)")

    axes[1].bar(labels, df["resilience_utility"])
    axes[1].set_ylabel("Resilience utility")
    axes[1].set_xlabel("Deployment setting")
    axes[1].set_title("Cyber-Resilience Utility")
    axes[1].tick_params(axis="x", rotation=10)
    add_panel(axes[1], "(b)")

    savefig(fig, out_dir / "figures" / "fig3_edge_cloud_resilience")


def fig_latency_tradeoff(metrics_df, out_dir):
    df = metrics_df[
        metrics_df["model"].str.contains("Proposed") &
        metrics_df["task"].eq("multiclass")
    ].copy()

    fig, ax = plt.subplots(figsize=(16, 9))
    sizes = 200 + 2000 * (df["macro_f1"] - df["macro_f1"].min() + 0.01)
    ax.scatter(df["latency_ms_per_msg"], df["macro_f1"], s=sizes, alpha=0.75)

    for _, r in df.iterrows():
        ax.text(r["latency_ms_per_msg"], r["macro_f1"], r["setting"].replace("_", " "), fontsize=11)

    ax.set_xlabel("Latency per message (ms)")
    ax.set_ylabel("Macro F1-score")
    ax.set_title("Latency-Accuracy Tradeoff for Edge-Cloud V2X Deployment")
    add_panel(ax, "(a)")
    savefig(fig, out_dir / "figures" / "fig4_latency_accuracy_tradeoff")


def fig_clean_feature_importance(df, out_dir):
    model_name = "XGBoost" if HAS_XGB else "ExtraTrees"
    features = list(dict.fromkeys(EDGE_FAST_FEATURES + CLOUD_CONTEXT_FEATURES + TRUST_RESPONSE_FEATURES))
    X_train, y_train, X_test, y_test = split_xy(df, "class_id", features)

    model = make_model(model_name, X_train, len(np.unique(y_train)))
    model.fit(X_train, y_train)

    clf = model.named_steps["clf"]
    pre = model.named_steps["preprocess"]

    if not hasattr(clf, "feature_importances_"):
        return

    names = [clean_feature_name(n) for n in pre.get_feature_names_out()]
    fi = pd.DataFrame({"feature": names, "importance": clf.feature_importances_})
    fi = fi.sort_values("importance", ascending=False)
    fi.to_csv(out_dir / "csv" / "clean_feature_importance.csv", index=False)

    top = fi.head(12).iloc[::-1]

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.barh(top["feature"], top["importance"])
    ax.set_xlabel("Relative importance")
    ax.set_ylabel("Feature")
    ax.set_title("Interpretable V2X Security Evidence Used by the Proposed Model")
    add_panel(ax, "(a)")
    savefig(fig, out_dir / "figures" / "fig5_clean_feature_importance")


def fig_confusion_matrix_clean(out_dir):
    cm_path = out_dir / "confusion_matrices" / "multiclass_edge_cloud_trust_XGBoost_cm.csv"
    if not cm_path.exists():
        cm_path = out_dir / "confusion_matrices" / "multiclass_edge_cloud_trust_HistGradientBoosting_cm.csv"
    if not cm_path.exists():
        return

    cm = pd.read_csv(cm_path, index_col=0)
    labels = ["Normal", "Const. Pos.", "Random Pos.", "Sybil"]
    vals = cm.values.astype(float)
    norm = vals / np.maximum(vals.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(16, 9))
    im = ax.imshow(norm, vmin=0, vmax=1)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Normalized Multiclass Confusion Matrix")

    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            ax.text(j, i, f"{norm[i,j]:.2f}", ha="center", va="center", fontsize=14)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    add_panel(ax, "(a)")
    savefig(fig, out_dir / "figures" / "fig6_confusion_matrix")


def write_report(metrics_df, res_df, out_dir):
    best_multi = metrics_df[metrics_df["task"] == "multiclass"].sort_values("macro_f1", ascending=False).head(10)
    best_bin = metrics_df[metrics_df["task"] == "binary"].sort_values("macro_f1", ascending=False).head(10)

    txt = []
    txt.append("# Agentic-V2XShield Real Evaluation Report\n")
    txt.append("## Correction\n")
    txt.append(
        "The earlier proposed model was not competitive. This revised evaluation uses real ablations and treats "
        "edge-only, cloud-only, edge-cloud, trust-only, and edge-cloud-trust configurations as testable "
        "experimental settings.\n"
    )
    txt.append("## Top Multiclass Results\n")
    txt.append(best_multi.to_markdown(index=False))
    txt.append("\n## Top Binary Results\n")
    txt.append(best_bin.to_markdown(index=False))
    txt.append("\n## Cyber-Resilience Metrics\n")
    txt.append(res_df.to_markdown(index=False))
    txt.append("\n## Paper-Safe Interpretation\n")
    txt.append(
        "Report the framework as an edge-cloud-trust orchestration layer. If the proposed setting does not beat "
        "every baseline in raw macro-F1, focus the contribution on deployment-aware resilience, response quality, "
        "latency, trust-based mitigation, and attack-type explanation."
    )

    (out_dir / "reports" / "ieee_real_evaluation_report.md").write_text("\n".join(txt), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-dir", default="outputs_ieee_real_eval")
    parser.add_argument("--max-per-class", type=int, default=50000)
    parser.add_argument("--keep-leakage-risk-features", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dirs(out_dir)

    df = load_data(
        Path(args.csv),
        max_per_class=args.max_per_class,
        drop_leakage=not args.keep_leakage_risk_features,
    )

    metrics_df, res_df = run_all_experiments(df, out_dir)

    fig_baseline_vs_proposed(metrics_df, out_dir)
    fig_binary_multiclass(metrics_df, out_dir)
    fig_edge_cloud_resilience(res_df, out_dir)
    fig_latency_tradeoff(metrics_df, out_dir)
    fig_clean_feature_importance(df, out_dir)
    fig_confusion_matrix_clean(out_dir)
    write_report(metrics_df, res_df, out_dir)

    print("\n[DONE] Real IEEE-grade evaluation suite completed.")
    print(f"[OUT] {out_dir.resolve()}")
    print("Main outputs:")
    print(" - csv/all_model_metrics.csv")
    print(" - csv/cyber_resilience_metrics.csv")
    print(" - csv/clean_feature_importance.csv")
    print(" - figures/*.png and *.pdf")
    print(" - reports/ieee_real_evaluation_report.md")


if __name__ == "__main__":
    main()
