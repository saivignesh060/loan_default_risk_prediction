"""Ablation of lender-set features. Does not change the saved model or existing outputs.

Uses the Stage 5 XGBoost parameters and class weights. The threshold for each
variant is chosen from training out-of-fold scores only. The test set is scored once
per variant.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train

RESULTS_NAME = "19_ablation_results.csv"
DECILE_NAME = "20_ablation_decile_table.csv"
BEST_PARAMS = {
    "n_estimators": 400,
    "max_depth": 5,
    "learning_rate": 0.1,
}


def make_preprocessor(drop_columns: set[str]) -> ColumnTransformer:
    """Same transformers as train.py, with the named columns left out."""
    numeric = [column for column in train.CONFIG["numeric_columns"] if column not in drop_columns]
    transformers = [
        (
            "numeric",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric,
        )
    ]
    if train.CONFIG["ordinal_column"] not in drop_columns:
        transformers.append(
            (
                "ordinal",
                OrdinalEncoder(
                    categories=[train.CONFIG["ordinal_categories"]],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                [train.CONFIG["ordinal_column"]],
            )
        )
    transformers.append(
        (
            "binary",
            OrdinalEncoder(
                categories=[train.CONFIG["binary_categories"]],
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
            [train.CONFIG["binary_column"]],
        )
    )
    transformers.append(
        (
            "nominal",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            list(train.CONFIG["nominal_columns"]),
        )
    )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_model(y_train: pd.Series) -> XGBClassifier:
    n_positive = int((y_train == 1).sum())
    n_negative = int((y_train == 0).sum())
    scale_pos_weight = n_negative / n_positive
    print(
        "Class weights kept from the main model: "
        f"scale_pos_weight = {n_negative} / {n_positive} = {scale_pos_weight:.4f}. "
        "SMOTE is not in this pipeline."
    )
    return XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=train.CONFIG["random_state"],
        **BEST_PARAMS,
    )


def make_pipeline(drop_columns: set[str], y_train: pd.Series) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(drop_columns)),
            ("model", make_model(y_train)),
        ]
    )


def decile_table(y_true: pd.Series, probabilities: np.ndarray) -> pd.DataFrame:
    """Decile 10 is the highest risk."""
    ranks = pd.Series(probabilities).rank(method="first", ascending=True)
    deciles = pd.qcut(ranks, 10, labels=list(range(1, 11))).astype(int)
    frame = pd.DataFrame({"decile": deciles.to_numpy(), "actual": np.asarray(y_true)})
    table = frame.groupby("decile", as_index=False).agg(
        applicants=("actual", "size"),
        defaults=("actual", "sum"),
    )
    table["default_rate"] = table["defaults"] / table["applicants"]
    return table.sort_values("decile")


def score_once(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict:
    predicted = (probabilities >= threshold).astype(int)
    y_array = np.asarray(y_true)
    tn, fp, fn, tp = confusion_matrix(y_array, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_array, predicted)),
        "precision": float(precision_score(y_array, predicted, zero_division=0)),
        "recall": float(recall_score(y_array, predicted, zero_division=0)),
        "f1": float(f1_score(y_array, predicted, zero_division=0)),
        "f2": float(fbeta_score(y_array, predicted, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_array, probabilities)),
        "pr_auc": float(average_precision_score(y_array, probabilities)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def run_variant(
    name: str,
    drop_columns: set[str],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[dict, pd.DataFrame]:
    print()
    print("=" * 72)
    print(f"Variant: {name}. Dropped: {sorted(drop_columns) or 'none'}")
    print("=" * 72)
    pipeline = make_pipeline(drop_columns, y_train)
    print("Out-of-fold probabilities, 10-fold cross_val_predict, training set only.")
    oof = cross_val_predict(
        pipeline,
        X_train,
        y_train,
        cv=StratifiedKFold(
            n_splits=train.CONFIG["cv_splits"],
            shuffle=True,
            random_state=train.CONFIG["random_state"],
        ),
        method="predict_proba",
        n_jobs=-1,
    )
    # Binary classes are ordered [0, 1], so column 1 is the default probability.
    oof_probabilities = np.asarray(oof[:, 1], dtype=np.float64)
    _scan, chosen = train.scan_thresholds(y_train, oof_probabilities)
    point = train.operating_point(_scan, chosen)
    print(
        f"Chosen threshold: {chosen:.2f} "
        f"(OOF precision={point['precision']:.4f}, recall={point['recall']:.4f}, F2={point['f2']:.4f})"
    )
    print("Fitting once on the full training set, then scoring the test set once.")
    pipeline.fit(X_train, y_train)
    fitted_classes = [int(label) for label in pipeline.named_steps["model"].classes_]
    if fitted_classes != [0, 1]:
        raise RuntimeError(f"Expected classes [0, 1], found {fitted_classes}.")
    test_probabilities = np.asarray(pipeline.predict_proba(X_test)[:, 1], dtype=np.float64)
    metrics = score_once(y_test, test_probabilities, chosen)
    metrics["variant"] = name
    lift = decile_table(y_test, test_probabilities)
    print(
        "Test: "
        f"recall={metrics['recall']:.4f}, precision={metrics['precision']:.4f}, "
        f"f1={metrics['f1']:.4f}, f2={metrics['f2']:.4f}, "
        f"accuracy={metrics['accuracy']:.4f}, roc_auc={metrics['roc_auc']:.4f}, "
        f"pr_auc={metrics['pr_auc']:.4f}"
    )
    print(
        f"Confusion matrix: tn={metrics['tn']}, fp={metrics['fp']}, "
        f"fn={metrics['fn']}, tp={metrics['tp']}"
    )
    print("Decile lift (10 = highest risk):")
    print(lift.to_string(index=False))
    return metrics, lift


def full_model_row() -> dict:
    metrics = json.loads((train.ROOT / "outputs" / "final_metrics.json").read_text(encoding="utf-8"))
    matrix = metrics["confusion_matrix"]
    lift = pd.read_csv(train.ROOT / "outputs" / "16_decile_lift_table.csv")
    top = lift.loc[lift["decile"] == 10].iloc[0]
    return {
        "variant": "full",
        "threshold": float(metrics["threshold"]),
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "f2": float(metrics["f2"]),
        "roc_auc": float(metrics["roc_auc"]),
        "pr_auc": float(metrics["pr_auc"]),
        "tn": int(matrix["tn"]),
        "fp": int(matrix["fp"]),
        "fn": int(matrix["fn"]),
        "tp": int(matrix["tp"]),
        "top_decile_applicants": int(top["applicants"]),
        "top_decile_defaults": int(top["defaults"]),
        "top_decile_default_rate": float(top["default_rate"]),
    }


def verify_split(X_test: pd.DataFrame, y_test: pd.Series) -> None:
    saved = pd.read_csv(train.ROOT / "outputs" / "16_test_risk_scores.csv")
    saved_ids = saved["row_id"].astype(int).sort_values().to_numpy()
    test_ids = pd.Index(X_test.index).astype(int).sort_values().to_numpy()
    index_match = np.array_equal(saved_ids, test_ids)
    saved_actual = saved.set_index("row_id")["actual"].loc[X_test.index].to_numpy()
    label_match = np.array_equal(saved_actual, np.asarray(y_test))
    print(f"Test index match with 16_test_risk_scores.csv: {index_match}")
    print(f"Test label match with 16_test_risk_scores.csv: {label_match}")
    if not index_match or not label_match:
        raise SystemExit(1)


def save_new(name: str, frame: pd.DataFrame) -> Path:
    path = train.ROOT / "outputs" / name
    if path.exists():
        raise SystemExit(f"Refusing to overwrite {path}")
    frame.to_csv(path, index=False, float_format="%.6f")
    print(f"Saved outputs/{name}")
    return path


def main() -> None:
    started = time.perf_counter()
    cleaned = train.load_and_clean()
    X_train, X_test, y_train, y_test = train.stratified_split(cleaned)
    print(f"Train rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    verify_split(X_test, y_test)

    rows = [full_model_row()]
    neither_lift = None
    for name, dropped in (
        ("no_grade", {"loan_grade"}),
        ("no_int_rate", {"loan_int_rate"}),
        ("neither", {"loan_grade", "loan_int_rate"}),
    ):
        metrics, lift = run_variant(name, dropped, X_train, X_test, y_train, y_test)
        top = lift.loc[lift["decile"] == 10].iloc[0]
        metrics["top_decile_applicants"] = int(top["applicants"])
        metrics["top_decile_defaults"] = int(top["defaults"])
        metrics["top_decile_default_rate"] = float(top["default_rate"])
        rows.append(metrics)
        if name == "neither":
            neither_lift = lift

    results = pd.DataFrame(rows)
    column_order = [
        "variant",
        "threshold",
        "recall",
        "precision",
        "f1",
        "f2",
        "accuracy",
        "roc_auc",
        "pr_auc",
        "tn",
        "fp",
        "fn",
        "tp",
        "top_decile_applicants",
        "top_decile_defaults",
        "top_decile_default_rate",
    ]
    results = results[column_order]
    save_new(RESULTS_NAME, results)
    save_new(DECILE_NAME, neither_lift)

    full_f2 = float(results.loc[results["variant"] == "full", "f2"].iloc[0])
    full_top = float(results.loc[results["variant"] == "full", "top_decile_default_rate"].iloc[0])
    print()
    print("Comparison with the saved full model (not rerun):")
    print(
        results[
            ["variant", "threshold", "recall", "precision", "f2", "roc_auc", "pr_auc", "top_decile_default_rate"]
        ].to_string(index=False)
    )
    print()
    for _, row in results.iterrows():
        if row["variant"] == "full":
            continue
        f2_delta = float(row["f2"]) - full_f2
        top_delta = float(row["top_decile_default_rate"]) - full_top
        print(
            f"{row['variant']}: test F2 {float(row['f2']):.4f} "
            f"(change {f2_delta:+.4f} from {full_f2:.4f}); "
            f"top-decile default rate {float(row['top_decile_default_rate']):.4f} "
            f"(change {top_delta:+.4f} from {full_top:.4f})."
        )
    elapsed = time.perf_counter() - started
    print(f"Part B runtime seconds: {elapsed:.1f}")


if __name__ == "__main__":
    main()
