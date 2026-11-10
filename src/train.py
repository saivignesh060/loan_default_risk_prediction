"""Loan default risk training script.

Stage 5 implements:
  A. Load and clean
  B. EDA figures
  C. Stratified 80/20 split
  D. Preprocessing pipeline and VIF diagnostic
  E. Seven-model bake-off
  F. Imbalance experiment on the winning model
  G. Tuning, threshold, test evaluation, and the saved artifact
"""

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from xgboost import XGBClassifier

# =============================================================================
# CONFIG
# Change these if the CSV columns or cleaning limits differ from the PRD.
# =============================================================================
CONFIG = {
    "data_path": Path("data") / "credit_risk_dataset.csv",
    "output_dir": Path("outputs"),
    "target_column": "loan_status",
    # Names from PRD Section 4. Order here is the PRD table order, not file order.
    "expected_columns": [
        "person_age",
        "person_income",
        "person_home_ownership",
        "person_emp_length",
        "loan_intent",
        "loan_grade",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_default_on_file",
        "cb_person_cred_hist_length",
        "loan_status",
    ],
    "numeric_columns": [
        "person_age",
        "person_income",
        "person_emp_length",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length",
    ],
    "ordinal_column": "loan_grade",
    "ordinal_categories": ["A", "B", "C", "D", "E", "F", "G"],
    "binary_column": "cb_person_default_on_file",
    # N maps to 0 and Y maps to 1.
    "binary_categories": ["N", "Y"],
    "nominal_columns": ["person_home_ownership", "loan_intent"],
    "test_size": 0.2,
    "random_state": 42,
    "max_person_age": 100,
    "max_person_emp_length": 60,
    "vif_threshold": 5.0,
    "figure_01": "01_target_distribution.png",
    "figure_02": "02_numeric_histograms.png",
    "figure_03": "03_default_rate_by_loan_grade.png",
    "figure_04": "04_default_rate_by_home_ownership.png",
    "figure_05": "05_default_rate_by_loan_intent.png",
    "figure_06": "06_correlation_heatmap.png",
    "figure_07": "07_vif_before_after.png",
    "table_07": "07_vif_before_after.csv",
    "cv_splits": 10,
    "f2_tie_tolerance": 0.01,
    "logistic_max_iter": 1000,
    "knn_neighbors": 15,
    "tree_max_depth": 8,
    "rf_n_estimators": 200,
    "figure_09": "09_cv_boxplots_f2_recall.png",
    "table_08": "08_model_comparison_results.csv",
    "model_comparison_csv": "model_comparison_results.csv",
    "smote_f2_margin": 0.01,
    "figure_10": "10_imbalance_experiment.png",
    "table_10": "10_imbalance_experiment.csv",
    "tuning_cv_splits": 5,
    "threshold_start": 0.05,
    "threshold_stop": 0.95,
    "threshold_step": 0.01,
    "permutation_repeats": 5,
    "permutation_top_n": 15,
    "figure_11": "11_threshold_metrics.png",
    "table_11": "11_threshold_metrics.csv",
    "figure_12": "12_confusion_matrix.png",
    "figure_13": "13_roc_curve.png",
    "figure_14": "14_precision_recall_curve.png",
    "figure_15": "15_permutation_importance.png",
    "table_15": "15_permutation_importance.csv",
    "figure_16": "16_decile_default_rate.png",
    "table_16_decile": "16_decile_lift_table.csv",
    "table_16_risk": "16_test_risk_scores.csv",
    "figure_18": "18_logistic_regression_coefficients.png",
    "table_18": "18_logistic_regression_coefficients.csv",
    "final_metrics_json": "final_metrics.json",
    "model_path": Path("models") / "loan_default_model.joblib",
    # Grids from PRD Section 5G. Keys are pipeline parameter names.
    "param_grids": {
        "Logistic Regression": {
            "model__C": [0.01, 0.1, 1, 10],
            "model__penalty": ["l1", "l2"],
        },
        "Random Forest": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [8, 12, None],
            "model__min_samples_leaf": [1, 5],
        },
        "XGBoost": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [3, 5],
            "model__learning_rate": [0.05, 0.1],
        },
        "Gradient Boosting": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [2, 3],
            "model__learning_rate": [0.05, 0.1],
        },
        "Decision Tree": {
            "model__max_depth": [4, 6, 8, 10],
            "model__min_samples_leaf": [1, 5, 20],
        },
        "KNN": {"model__n_neighbors": [5, 15, 25]},
        "Gaussian NB": {"model__var_smoothing": [1e-9, 1e-8, 1e-7]},
    },
}

# ROC AUC and PR AUC use predict_proba, not hard 0/1 labels.
SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f2": make_scorer(fbeta_score, beta=2, zero_division=0),
    "roc_auc": make_scorer(roc_auc_score, response_method="predict_proba"),
    "pr_auc": make_scorer(average_precision_score, response_method="predict_proba"),
}
METRIC_ORDER = ["f2", "recall", "precision", "roc_auc", "pr_auc", "accuracy"]

# Project root is the parent of src/. Paths stay relative to that root.
ROOT = Path(__file__).resolve().parent.parent


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def class_ratio_text(y: pd.Series) -> str:
    """Return counts, default rate, and non-default:default ratio."""
    counts = y.value_counts(dropna=False)
    n = int(y.shape[0])
    n0 = int(counts.get(0, 0))
    n1 = int(counts.get(1, 0))
    default_rate = (n1 / n) if n else float("nan")
    if n1 == 0:
        balance = "non-default:default=undefined (no defaults)"
    else:
        balance = f"non-default:default={n0 / n1:.4f}:1"
    return (
        f"n={n}, non-default (0)={n0}, default (1)={n1}, "
        f"default rate={default_rate:.4f}, {balance}"
    )


def print_frame_profile(df: pd.DataFrame, label: str) -> None:
    target = CONFIG["target_column"]
    print(f"\n{label} shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"\n{label} dtypes:")
    print(df.dtypes.to_string())
    print(f"\n{label} missing-value counts:")
    print(df.isna().sum().to_string())
    print(f"\n{label} target distribution ({target}):")
    print(df[target].value_counts(dropna=False).sort_index().to_string())
    print(f"{label} class ratio: {class_ratio_text(df[target])}")


def output_path(name: str) -> Path:
    directory = ROOT / CONFIG["output_dir"]
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name


def save_figure(fig: plt.Figure, name: str) -> None:
    path = output_path(name)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {CONFIG['output_dir'].as_posix()}/{name}")


def load_and_clean() -> pd.DataFrame:
    """Section A. Read the CSV and apply the PRD cleaning rules."""
    banner("A. LOAD AND CLEAN")
    path = ROOT / CONFIG["data_path"]
    print(f"Reading {CONFIG['data_path'].as_posix()}")
    df = pd.read_csv(path)
    print(f"Loaded shape: {df.shape[0]} rows x {df.shape[1]} columns")

    expected = CONFIG["expected_columns"]
    actual = list(df.columns)
    missing = [col for col in expected if col not in actual]
    extra = [col for col in actual if col not in expected]
    if missing or extra:
        print("CSV columns do not match the PRD.")
        print(f"Missing from CSV: {missing if missing else 'none'}")
        print(f"Extra in CSV: {extra if extra else 'none'}")
        print(f"CSV columns: {actual}")
        raise SystemExit(1)
    if actual == expected:
        print("CSV column names and order match the PRD.")
    else:
        print("CSV column names match the PRD. Column order in the file differs.")
        print(f"CSV order: {actual}")

    print_frame_profile(df, "Raw")

    # Step 1: exact duplicate rows.
    n_before = len(df)
    n_dup = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"\nExact duplicate rows removed: {n_dup}")
    print(f"Rows remaining after duplicate removal: {len(df)} (started at {n_before})")

    # Step 2: impossible values. Age and employment length are counted
    # separately, then removed together so a row that fails both is dropped once.
    # NaN employment length is kept; imputation happens later inside the pipeline.
    age_limit = CONFIG["max_person_age"]
    emp_limit = CONFIG["max_person_emp_length"]
    age_mask = df["person_age"] > age_limit
    emp_mask = df["person_emp_length"] > emp_limit
    n_age = int(age_mask.sum())
    n_emp = int(emp_mask.sum())
    n_both = int((age_mask & emp_mask).sum())
    n_impossible = int((age_mask | emp_mask).sum())
    df = df.loc[~(age_mask | emp_mask)].reset_index(drop=True)
    print(f"Rows with person_age > {age_limit} removed: {n_age}")
    print(f"Rows with person_emp_length > {emp_limit} removed: {n_emp}")
    print(f"Rows counted in both impossible-value rules: {n_both}")
    print(f"Impossible-value rows removed: {n_impossible}")
    print(f"Rows remaining after cleaning: {len(df)}")

    print_frame_profile(df, "Cleaned")
    return df


def plot_target_distribution(df: pd.DataFrame) -> None:
    target = CONFIG["target_column"]
    counts = df[target].value_counts().sort_index()
    total = int(counts.sum())
    labels = ["Non-default (0)", "Default (1)"]
    values = [int(counts.get(0, 0)), int(counts.get(1, 0))]
    default_rate = values[1] / total if total else float("nan")

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=["#4C78A8", "#E45756"])
    ax.set_ylabel("Applicants")
    ax.set_title(f"Target distribution (default rate = {default_rate:.1%})")
    for bar, value in zip(bars, values):
        share = value / total if total else 0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value}\n({share:.1%})",
            ha="center",
            va="bottom",
        )
    ax.set_ylim(0, max(values) * 1.18 if values else 1)
    save_figure(fig, CONFIG["figure_01"])


def plot_numeric_histograms(df: pd.DataFrame) -> None:
    columns = CONFIG["numeric_columns"]
    fig, axes = plt.subplots(2, 4, figsize=(12, 7))
    flat_axes = axes.ravel()
    for ax, column in zip(flat_axes, columns):
        values = df[column].dropna()
        ax.hist(values, bins=30, color="#4C78A8", edgecolor="white")
        ax.set_title(column)
        ax.set_ylabel("Count")
    for ax in flat_axes[len(columns) :]:
        ax.axis("off")
    fig.suptitle("Numeric feature histograms")
    fig.tight_layout()
    save_figure(fig, CONFIG["figure_02"])


def plot_default_rate(df: pd.DataFrame, column: str, filename: str, category_order=None) -> None:
    target = CONFIG["target_column"]
    grouped = (
        df.groupby(column, dropna=False)[target]
        .agg(default_rate="mean", applicants="count")
        .reset_index()
    )
    if category_order is None:
        grouped = grouped.sort_values(
            ["default_rate", column], ascending=[False, True]
        )
    else:
        present = [category for category in category_order if category in set(grouped[column])]
        grouped["_order"] = grouped[column].map({category: i for i, category in enumerate(present)})
        grouped = grouped.dropna(subset=["_order"]).sort_values("_order")

    print(f"\nDefault rate by {column}:")
    print(grouped[[column, "default_rate", "applicants"]].to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 4))
    positions = range(len(grouped))
    rates = grouped["default_rate"].tolist()
    ax.bar(positions, rates, color="#E45756")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(grouped[column].astype(str).tolist(), rotation=30, ha="right")
    ax.set_ylabel("Default rate")
    ax.set_xlabel(column)
    ax.set_title(f"Default rate by {column}")
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _pos: f"{value:.0%}"))
    for pos, rate, count in zip(positions, rates, grouped["applicants"].tolist()):
        ax.text(pos, rate + 0.03, f"{rate:.1%}\n(n={int(count)})", ha="center", va="bottom", fontsize=8)
    save_figure(fig, filename)


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    target = CONFIG["target_column"]
    columns = CONFIG["numeric_columns"] + [target]
    # Pairwise-complete Pearson correlation. EDA does not impute.
    corr = df[columns].corr(method="pearson")
    target_corr = corr[target].drop(labels=[target]).sort_values()
    negative_name = str(target_corr.index[0])
    positive_name = str(target_corr.index[-1])
    print(f"\nPearson correlations with {target} (pairwise complete observations):")
    print(target_corr.sort_values(ascending=False).to_string())
    print(
        f"Top negative correlation with {target}: "
        f"{negative_name} = {float(target_corr.loc[negative_name]):.4f}"
    )
    print(
        f"Top positive correlation with {target}: "
        f"{positive_name} = {float(target_corr.loc[positive_name]):.4f}"
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title("Correlation of numeric features")
    save_figure(fig, CONFIG["figure_06"])


def run_eda(df: pd.DataFrame) -> None:
    """Section B. Descriptive figures on the cleaned data, before the split."""
    banner("B. EDA")
    print("EDA uses the cleaned dataset. No imputer is fitted in this section.")
    plot_target_distribution(df)
    plot_numeric_histograms(df)
    plot_default_rate(
        df,
        CONFIG["ordinal_column"],
        CONFIG["figure_03"],
        category_order=CONFIG["ordinal_categories"],
    )
    plot_default_rate(df, "person_home_ownership", CONFIG["figure_04"])
    plot_default_rate(df, "loan_intent", CONFIG["figure_05"])
    plot_correlation_heatmap(df)


def stratified_split(df: pd.DataFrame):
    """Section C. 80/20 stratified split before any fitting."""
    banner("C. STRATIFIED 80/20 SPLIT")
    target = CONFIG["target_column"]
    feature_columns = [col for col in df.columns if col != target]
    X = df[feature_columns]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"],
        stratify=y,
    )
    print(f"Feature columns ({len(feature_columns)}): {feature_columns}")
    print(f"Train rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Train class ratio: {class_ratio_text(y_train)}")
    print(f"Test class ratio: {class_ratio_text(y_test)}")
    print("Split is done before fitting. No model has been fitted.")
    return X_train, X_test, y_train, y_test


def compute_vif(frame: pd.DataFrame) -> pd.DataFrame:
    """VIF for each column. An intercept is included and is not a feature."""
    design = add_constant(frame, has_constant="add")
    rows = []
    for index, feature in enumerate(frame.columns):
        vif_value = float(variance_inflation_factor(design.to_numpy(dtype=float), index + 1))
        rows.append({"feature": feature, "vif": vif_value})
    return (
        pd.DataFrame(rows)
        .sort_values(["vif", "feature"], ascending=[False, True])
        .reset_index(drop=True)
    )


def build_preprocessor(numeric_columns: list[str]) -> ColumnTransformer:
    """Unfitted preprocessing. Callers fit it only inside a training pipeline."""
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    ordinal_pipe = OrdinalEncoder(
        categories=[CONFIG["ordinal_categories"]],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    # Categories [N, Y] map N to 0 and Y to 1.
    binary_pipe = OrdinalEncoder(
        categories=[CONFIG["binary_categories"]],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    nominal_pipe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, list(numeric_columns)),
            ("ordinal", ordinal_pipe, [CONFIG["ordinal_column"]]),
            ("binary", binary_pipe, [CONFIG["binary_column"]]),
            ("nominal", nominal_pipe, list(CONFIG["nominal_columns"])),
        ],
        remainder="drop",
    )


def run_preprocessing(X_train: pd.DataFrame) -> dict:
    """Section D. VIF on imputed training numerics, then an unfitted pipeline."""
    banner("D. PREPROCESSING PIPELINE AND VIF")
    numeric_columns = list(CONFIG["numeric_columns"])
    print(
        "Fitting a median imputer on the training numeric columns only, "
        "for the VIF diagnostic."
    )
    print(f"Training rows used for VIF: {len(X_train)}")
    vif_imputer = SimpleImputer(strategy="median")
    imputed_values = vif_imputer.fit_transform(X_train[numeric_columns])
    imputed = pd.DataFrame(imputed_values, columns=numeric_columns)

    threshold = float(CONFIG["vif_threshold"])
    before = compute_vif(imputed)
    print("\nVIF before dropping features:")
    print(before.to_string(index=False))

    kept = list(numeric_columns)
    current = before
    dropped_rows = []
    step = 0
    while len(kept) > 1 and float(current["vif"].max()) > threshold:
        worst = current.iloc[0]
        feature = str(worst["feature"])
        vif_value = float(worst["vif"])
        step += 1
        reason = (
            f"Highest VIF on pass {step} was {vif_value:.4f}, above {threshold:.0f}."
        )
        print(f"Dropping {feature}: {reason}")
        dropped_rows.append(
            {
                "feature": feature,
                "vif_at_drop": vif_value,
                "drop_step": step,
                "reason": reason,
            }
        )
        kept.remove(feature)
        current = compute_vif(imputed[kept])
        print("VIF after this drop:")
        print(current.to_string(index=False))

    after = current
    if not dropped_rows:
        print(f"No numeric feature had VIF above {threshold:.0f}. None were dropped.")
    print(f"Numeric features kept for the model pipeline: {kept}")

    drop_lookup = {row["feature"]: row for row in dropped_rows}
    before_map = dict(zip(before["feature"], before["vif"]))
    after_map = dict(zip(after["feature"], after["vif"]))
    table_rows = []
    for feature in before["feature"]:
        dropped = drop_lookup.get(feature)
        table_rows.append(
            {
                "feature": feature,
                "vif_before": round(float(before_map[feature]), 4),
                "vif_after": None if dropped else round(float(after_map[feature]), 4),
                "decision": "dropped" if dropped else "kept",
                "drop_step": None if dropped is None else int(dropped["drop_step"]),
                "reason": (
                    dropped["reason"]
                    if dropped
                    else f"VIF is at or below {threshold:.0f} after iterative removal."
                ),
            }
        )
    vif_table = pd.DataFrame(table_rows)
    table_file = output_path(CONFIG["table_07"])
    vif_table.to_csv(table_file, index=False, float_format="%.4f")
    print(f"Saved {CONFIG['output_dir'].as_posix()}/{CONFIG['table_07']}")
    print("\nVIF table before and after:")
    print(vif_table.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 5))
    positions = list(range(len(vif_table)))
    ax.barh(
        [pos - 0.18 for pos in positions],
        vif_table["vif_before"],
        height=0.36,
        color="#4C78A8",
        label="Before",
    )
    after_positions = []
    after_values = []
    for pos, vif_after in zip(positions, vif_table["vif_after"]):
        if pd.notna(vif_after):
            after_positions.append(pos + 0.18)
            after_values.append(float(vif_after))
    ax.barh(after_positions, after_values, height=0.36, color="#F58518", label="After")
    ax.axvline(threshold, color="black", linestyle="--", label=f"VIF = {threshold:.0f}")
    ax.set_yticks(positions)
    ax.set_yticklabels(vif_table["feature"].tolist())
    ax.invert_yaxis()
    ax.set_xlabel("VIF")
    ax.set_title("VIF before and after iterative removal")
    ax.legend()
    save_figure(fig, CONFIG["figure_07"])

    preprocessor = build_preprocessor(kept)
    print("\nPreprocessing ColumnTransformer (not fitted):")
    print(preprocessor)
    print(
        "The modeling preprocessor is not fitted here. "
        "Later sections fit it only inside training pipelines."
    )
    print(f"Numeric columns excluded by VIF: {[row['feature'] for row in dropped_rows] or 'none'}")
    return {
        "preprocessor": preprocessor,
        "numeric_columns": kept,
        "dropped_columns": [row["feature"] for row in dropped_rows],
        "vif_table": vif_table,
    }


def build_bakeoff_models(y_train: pd.Series) -> list[tuple[str, object]]:
    """Seven models with the settings in PRD Section 5E."""
    seed = CONFIG["random_state"]
    n_positive = int((y_train == 1).sum())
    n_negative = int((y_train == 0).sum())
    scale_pos_weight = n_negative / n_positive
    print(
        "XGBoost scale_pos_weight = "
        f"negatives / positives = {n_negative} / {n_positive} = {scale_pos_weight:.4f}"
    )
    return [
        (
            "Logistic Regression",
            LogisticRegression(
                class_weight="balanced",
                max_iter=CONFIG["logistic_max_iter"],
                random_state=seed,
            ),
        ),
        ("KNN", KNeighborsClassifier(n_neighbors=CONFIG["knn_neighbors"])),
        ("Gaussian NB", GaussianNB()),
        (
            "Decision Tree",
            DecisionTreeClassifier(
                max_depth=CONFIG["tree_max_depth"],
                class_weight="balanced",
                random_state=seed,
            ),
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=CONFIG["rf_n_estimators"],
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=seed,
            ),
        ),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=seed)),
        (
            "XGBoost",
            XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=seed,
            ),
        ),
    ]


def select_winner(summary: pd.DataFrame) -> str:
    """Highest mean CV F2. Within 0.01 F2, pick the higher mean ROC AUC."""
    tolerance = float(CONFIG["f2_tie_tolerance"])
    best_f2 = float(summary["f2_mean"].max())
    contenders = summary.loc[summary["f2_mean"] >= best_f2 - tolerance].copy()
    contenders = contenders.sort_values(
        ["roc_auc_mean", "f2_mean", "model"],
        ascending=[False, False, True],
    )
    winner = str(contenders.iloc[0]["model"])
    print(f"\nHighest mean CV F2: {best_f2:.4f}")
    print(f"Models within {tolerance:.2f} mean F2 of the best:")
    print(contenders[["model", "f2_mean", "roc_auc_mean"]].to_string(index=False))
    if len(contenders) == 1:
        print(f"Winner: {winner}. It leads on mean CV F2 by more than {tolerance:.2f}.")
    else:
        print(
            f"Winner: {winner}. More than one model is within {tolerance:.2f} mean F2, "
            "so the higher mean ROC AUC is selected."
        )
    return winner


def plot_cv_boxplots(fold_scores: pd.DataFrame) -> None:
    model_order = list(dict.fromkeys(fold_scores["model"].tolist()))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, metric in zip(axes, ["F2", "Recall"]):
        subset = fold_scores.loc[fold_scores["metric"] == metric]
        sns.boxplot(
            data=subset,
            x="model",
            y="score",
            order=model_order,
            color="#4C78A8",
            ax=ax,
        )
        ax.set_title(f"Per-fold {metric}")
        ax.set_xlabel("")
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=30)
        for label in ax.get_xticklabels():
            label.set_ha("right")
    fig.suptitle("Stratified 10-fold CV")
    fig.tight_layout()
    save_figure(fig, CONFIG["figure_09"])


def run_model_comparison(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
) -> dict:
    """Section E. Stratified 10-fold CV on the training set only."""
    banner("E. SEVEN-MODEL BAKE-OFF")
    print("Each model is wrapped in a pipeline with a clone of the unfitted preprocessor.")
    print("Imputers, encoders, and scalers are fit inside each training fold only.")
    cv = StratifiedKFold(
        n_splits=CONFIG["cv_splits"],
        shuffle=True,
        random_state=CONFIG["random_state"],
    )
    models = build_bakeoff_models(y_train)
    summary_rows = []
    fold_rows = []
    for name, model in models:
        print(f"\nCross-validating {name}...")
        pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", model),
            ]
        )
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=SCORING,
            n_jobs=-1,
            error_score="raise",
        )
        row = {"model": name}
        for metric in METRIC_ORDER:
            fold_values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = float(fold_values.mean())
            row[f"{metric}_std"] = float(fold_values.std(ddof=1))
            metric_label = "F2" if metric == "f2" else "Recall" if metric == "recall" else None
            if metric_label is not None:
                for fold_id, value in enumerate(fold_values, start=1):
                    fold_rows.append(
                        {
                            "model": name,
                            "metric": metric_label,
                            "fold": fold_id,
                            "score": float(value),
                        }
                    )
            print(
                f"  {metric}: {row[f'{metric}_mean']:.4f} "
                f"+/- {row[f'{metric}_std']:.4f}"
            )
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    for name in (CONFIG["table_08"], CONFIG["model_comparison_csv"]):
        path = output_path(name)
        summary.to_csv(path, index=False, float_format="%.6f")
        print(f"Saved {CONFIG['output_dir'].as_posix()}/{name}")
    print("\nModel comparison (mean and sample std across 10 folds):")
    print(summary.to_string(index=False))

    plot_cv_boxplots(pd.DataFrame(fold_rows))
    winner = select_winner(summary)
    return {"summary": summary, "winner": winner}


def estimator_for(model_name: str, use_class_weights: bool, y_train: pd.Series):
    """Build one unfitted estimator. Class weights follow the PRD model settings."""
    seed = CONFIG["random_state"]
    if model_name == "Logistic Regression":
        return LogisticRegression(
            class_weight="balanced" if use_class_weights else None,
            max_iter=CONFIG["logistic_max_iter"],
            random_state=seed,
        )
    if model_name == "KNN":
        return KNeighborsClassifier(n_neighbors=CONFIG["knn_neighbors"])
    if model_name == "Gaussian NB":
        return GaussianNB()
    if model_name == "Decision Tree":
        return DecisionTreeClassifier(
            max_depth=CONFIG["tree_max_depth"],
            class_weight="balanced" if use_class_weights else None,
            random_state=seed,
        )
    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=CONFIG["rf_n_estimators"],
            class_weight="balanced_subsample" if use_class_weights else None,
            n_jobs=-1,
            random_state=seed,
        )
    if model_name == "Gradient Boosting":
        return GradientBoostingClassifier(random_state=seed)
    if model_name == "XGBoost":
        n_positive = int((y_train == 1).sum())
        n_negative = int((y_train == 0).sum())
        scale_pos_weight = (n_negative / n_positive) if use_class_weights else 1.0
        return XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=seed,
        )
    raise ValueError(f"Unknown model: {model_name}")


def plot_imbalance_chart(table: pd.DataFrame, model_name: str) -> None:
    labels = table["label"].tolist()
    means = table["f2_mean"].tolist()
    stds = table["f2_std"].tolist()
    positions = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        positions,
        means,
        yerr=stds,
        color=["#4C78A8", "#F58518", "#54A24B"],
        capsize=4,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean CV F2")
    ax.set_ylim(0, 1)
    ax.set_title(f"Imbalance experiment ({model_name})")
    for pos, mean in zip(positions, means):
        ax.text(pos, min(mean + 0.04, 0.97), f"{mean:.3f}", ha="center", va="bottom")
    save_figure(fig, CONFIG["figure_10"])


def run_imbalance_experiment(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    winner: str,
) -> dict:
    """Section F. None vs class weights vs SMOTE on the winning model only."""
    banner("F. IMBALANCE EXPERIMENT")
    print(f"Winning model: {winner}")
    print("SMOTE is inside an imblearn Pipeline, so it fits on each training fold only.")
    if winner == "XGBoost":
        print("For XGBoost, class weights means scale_pos_weight = negatives / positives.")
        print("The none and SMOTE arms use scale_pos_weight = 1.")
    elif winner in {"KNN", "Gaussian NB", "Gradient Boosting"}:
        print(
            f"{winner} has no class_weight parameter in the PRD settings. "
            "The class-weight arm uses the same estimator as no handling."
        )
    cv = StratifiedKFold(
        n_splits=CONFIG["cv_splits"],
        shuffle=True,
        random_state=CONFIG["random_state"],
    )
    strategies = [
        ("none", "None", False, False),
        ("class_weights", "Class weights", True, False),
        ("smote", "SMOTE", False, True),
    ]
    rows = []
    for strategy, label, use_weights, use_smote in strategies:
        print(f"\nCross-validating {winner} with {label}...")
        estimator = estimator_for(winner, use_weights, y_train)
        if use_smote:
            pipeline = ImbPipeline(
                steps=[
                    ("preprocessor", clone(preprocessor)),
                    ("smote", SMOTE(random_state=CONFIG["random_state"])),
                    ("model", estimator),
                ]
            )
        else:
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", clone(preprocessor)),
                    ("model", estimator),
                ]
            )
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=SCORING,
            n_jobs=-1,
            error_score="raise",
        )
        row = {"strategy": strategy, "label": label}
        for metric in METRIC_ORDER:
            fold_values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = float(fold_values.mean())
            row[f"{metric}_std"] = float(fold_values.std(ddof=1))
            print(
                f"  {metric}: {row[f'{metric}_mean']:.4f} "
                f"+/- {row[f'{metric}_std']:.4f}"
            )
        rows.append(row)

    table = pd.DataFrame(rows)
    class_f2 = float(table.loc[table["strategy"] == "class_weights", "f2_mean"].iloc[0])
    smote_f2 = float(table.loc[table["strategy"] == "smote", "f2_mean"].iloc[0])
    margin = float(CONFIG["smote_f2_margin"])
    f2_gap = smote_f2 - class_f2
    if f2_gap > margin:
        chosen = "smote"
        print(
            f"\nSMOTE mean F2 is {f2_gap:.4f} above class weights, "
            f"more than {margin:.2f}. The final model will use SMOTE."
        )
    else:
        chosen = "class_weights"
        print(
            f"\nSMOTE mean F2 minus class weights is {f2_gap:.4f}, "
            f"which is not more than {margin:.2f}. The final model keeps class weights."
        )
    table["selected"] = table["strategy"].map(lambda name: "yes" if name == chosen else "no")
    path = output_path(CONFIG["table_10"])
    table.to_csv(path, index=False, float_format="%.6f")
    print(f"Saved {CONFIG['output_dir'].as_posix()}/{CONFIG['table_10']}")
    print("\nImbalance experiment:")
    print(table.to_string(index=False))
    plot_imbalance_chart(table, winner)
    return {"table": table, "strategy": chosen, "model": winner}


def as_builtin(value):
    """Convert numpy values so the metrics file is plain JSON."""
    if isinstance(value, dict):
        return {str(key): as_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_builtin(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_tuned_pipeline(preprocessor, model_name: str, strategy: str, y_train: pd.Series):
    """Unfitted pipeline using the imbalance strategy chosen in section F."""
    estimator = estimator_for(model_name, strategy == "class_weights", y_train)
    if model_name == "Logistic Regression":
        estimator.set_params(solver="liblinear")
    if strategy == "smote":
        return ImbPipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("smote", SMOTE(random_state=CONFIG["random_state"])),
                ("model", estimator),
            ]
        )
    return Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("model", estimator),
        ]
    )


def scan_thresholds(y_true: pd.Series, probabilities: np.ndarray) -> tuple[pd.DataFrame, float]:
    """Highest F2 on the 0.05-0.95 grid. Ties keep the lowest threshold."""
    step = float(CONFIG["threshold_step"])
    thresholds = np.round(
        np.arange(
            float(CONFIG["threshold_start"]),
            float(CONFIG["threshold_stop"]) + step / 2,
            step,
        ),
        2,
    )
    y_array = np.asarray(y_true)
    rows = []
    for threshold in thresholds:
        predicted = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(y_array, predicted, zero_division=0)),
                "recall": float(recall_score(y_array, predicted, zero_division=0)),
                "f2": float(fbeta_score(y_array, predicted, beta=2, zero_division=0)),
            }
        )
    scan = pd.DataFrame(rows)
    chosen = float(scan.loc[scan["f2"].idxmax(), "threshold"])
    n_tied = int(np.isclose(scan["f2"], scan["f2"].max()).sum())
    scan["selected"] = np.where(np.isclose(scan["threshold"], chosen), "yes", "no")
    if n_tied > 1:
        print(
            f"{n_tied} thresholds tie for the highest F2. "
            "The lowest of those thresholds is selected."
        )
    return scan, chosen


def operating_point(scan: pd.DataFrame, threshold: float) -> pd.Series:
    matched = scan.loc[np.isclose(scan["threshold"], threshold)]
    if matched.empty:
        raise ValueError(f"Threshold {threshold} is not on the scan grid.")
    return matched.iloc[0]


def plot_threshold_scan(scan: pd.DataFrame, chosen: float) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(scan["threshold"], scan["precision"], label="Precision")
    ax.plot(scan["threshold"], scan["recall"], label="Recall")
    ax.plot(scan["threshold"], scan["f2"], label="F2")
    ax.axvline(chosen, color="black", linestyle="--", label=f"Chosen {chosen:.2f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Out-of-fold threshold scan")
    ax.legend()
    save_figure(fig, CONFIG["figure_11"])


def plot_confusion_matrix(counts: dict) -> None:
    matrix = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-default", "Default"],
        yticklabels=["Non-default", "Default"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Test confusion matrix")
    save_figure(fig, CONFIG["figure_12"])


def plot_roc_curve(y_true: pd.Series, probabilities: np.ndarray, roc_auc: float) -> None:
    fpr, tpr, _thresholds = roc_curve(y_true, probabilities)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="#4C78A8", label=f"ROC AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Test ROC curve")
    ax.legend(loc="lower right")
    save_figure(fig, CONFIG["figure_13"])


def plot_pr_curve(y_true: pd.Series, probabilities: np.ndarray, pr_auc: float) -> None:
    precision, recall, _thresholds = precision_recall_curve(y_true, probabilities)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, color="#E45756", label=f"PR AUC = {pr_auc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Test precision-recall curve")
    ax.legend(loc="lower left")
    save_figure(fig, CONFIG["figure_14"])


def assign_band(score: float, threshold: float) -> str:
    """High if score is at or above the operating threshold (probability x 100)."""
    if score >= float(threshold) * 100:
        return "High"
    return "Low"


def run_final_evaluation(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
    winner: str,
    strategy: str,
) -> dict:
    """Section G. Tune, choose a threshold from OOF scores, then test once."""
    banner("G. TUNING, THRESHOLD, AND TEST EVALUATION")
    print(f"Model: {winner}. Imbalance handling: {strategy}.")
    print("GridSearchCV uses the training set only. The test set is not used here.")
    pipeline = build_tuned_pipeline(preprocessor, winner, strategy, y_train)
    param_grid = CONFIG["param_grids"][winner]
    n_candidates = 1
    for values in param_grid.values():
        n_candidates *= len(values)
    n_folds = int(CONFIG["tuning_cv_splits"])
    print(
        f"Grid size: {n_candidates} candidates x {n_folds} folds "
        f"= {n_candidates * n_folds} fits. Scoring is F2 at the model's default predict() threshold."
    )
    search = GridSearchCV(
        pipeline,
        param_grid,
        scoring=SCORING["f2"],
        cv=StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=CONFIG["random_state"],
        ),
        n_jobs=-1,
        refit=True,
        verbose=1,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    best_params = as_builtin(search.best_params_)
    best_cv_f2 = float(search.best_score_)
    print(f"Best parameters: {best_params}")
    print(f"Best mean CV F2: {best_cv_f2:.4f}")

    print("\nOut-of-fold probabilities from 10-fold cross_val_predict on the tuned pipeline.")
    fitted = search.best_estimator_
    class_labels = [int(label) for label in fitted.named_steps["model"].classes_]
    if class_labels != [0, 1]:
        raise RuntimeError(f"Expected classes [0, 1], found {class_labels}.")
    oof_probabilities = cross_val_predict(
        fitted,
        X_train,
        y_train,
        cv=StratifiedKFold(
            n_splits=CONFIG["cv_splits"],
            shuffle=True,
            random_state=CONFIG["random_state"],
        ),
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    scan, chosen = scan_thresholds(y_train, oof_probabilities)
    at_chosen = operating_point(scan, chosen)
    at_half = operating_point(scan, 0.5)
    print(
        "OOF at default 0.50: "
        f"precision={at_half['precision']:.4f}, "
        f"recall={at_half['recall']:.4f}, "
        f"F2={at_half['f2']:.4f}"
    )
    print(
        f"OOF at chosen {chosen:.2f}: "
        f"precision={at_chosen['precision']:.4f}, "
        f"recall={at_chosen['recall']:.4f}, "
        f"F2={at_chosen['f2']:.4f}"
    )
    scan_path = output_path(CONFIG["table_11"])
    scan.to_csv(scan_path, index=False, float_format="%.6f")
    print(f"Saved {CONFIG['output_dir'].as_posix()}/{CONFIG['table_11']}")
    plot_threshold_scan(scan, chosen)

    oof_scores = oof_probabilities * 100
    score_cutoffs = {
        "p50": float(np.percentile(oof_scores, 50)),
        "p80": float(np.percentile(oof_scores, 80)),
    }
    print(
        "Score cutoffs from out-of-fold risk scores (probability x 100), "
        "not from in-sample training predictions."
    )
    print(f"50th percentile cutoff: {score_cutoffs['p50']:.4f}")
    print(f"80th percentile cutoff: {score_cutoffs['p80']:.4f}")
    print(
        "Bands are binary at the operating threshold: "
        f"High if score >= {chosen * 100:.2f}, else Low. "
        "Percentile cutoffs are kept in the artifact for compatibility."
    )

    print("\nTest evaluation, run once with the tuned model and chosen threshold.")
    test_probabilities = fitted.predict_proba(X_test)[:, 1]
    test_predicted = (test_probabilities >= chosen).astype(int)
    y_test_array = np.asarray(y_test)
    tn, fp, fn, tp = confusion_matrix(y_test_array, test_predicted, labels=[0, 1]).ravel()
    test_metrics = {
        "accuracy": float(accuracy_score(y_test_array, test_predicted)),
        "precision": float(precision_score(y_test_array, test_predicted, zero_division=0)),
        "recall": float(recall_score(y_test_array, test_predicted, zero_division=0)),
        "f1": float(f1_score(y_test_array, test_predicted, zero_division=0)),
        "f2": float(fbeta_score(y_test_array, test_predicted, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test_array, test_probabilities)),
        "pr_auc": float(average_precision_score(y_test_array, test_probabilities)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "threshold": chosen,
    }
    print(
        "Test metrics: "
        f"accuracy={test_metrics['accuracy']:.4f}, "
        f"precision={test_metrics['precision']:.4f}, "
        f"recall={test_metrics['recall']:.4f}, "
        f"f1={test_metrics['f1']:.4f}, "
        f"f2={test_metrics['f2']:.4f}, "
        f"roc_auc={test_metrics['roc_auc']:.4f}, "
        f"pr_auc={test_metrics['pr_auc']:.4f}"
    )
    print(
        "Confusion matrix counts: "
        f"tn={tn}, fp={fp}, fn={fn}, tp={tp}"
    )
    plot_confusion_matrix(test_metrics["confusion_matrix"])
    plot_roc_curve(y_test_array, test_probabilities, test_metrics["roc_auc"])
    plot_pr_curve(y_test_array, test_probabilities, test_metrics["pr_auc"])

    test_scores = test_probabilities * 100
    ranks = pd.Series(test_probabilities).rank(method="first", ascending=True)
    deciles = pd.qcut(ranks, 10, labels=list(range(1, 11))).astype(int)
    risk = pd.DataFrame(
        {
            "row_id": X_test.index.to_numpy(),
            "actual": y_test_array,
            "predicted": test_predicted,
            "risk_score": np.round(test_scores, 2),
            "decile": deciles.to_numpy(),
            "band": [
                assign_band(float(score), chosen)
                for score in test_scores
            ],
        }
    ).sort_values("row_id")
    risk_path = output_path(CONFIG["table_16_risk"])
    risk.to_csv(risk_path, index=False)
    print(f"Saved {CONFIG['output_dir'].as_posix()}/{CONFIG['table_16_risk']}")

    lift = (
        risk.groupby("decile", as_index=False)
        .agg(applicants=("actual", "size"), defaults=("actual", "sum"))
        .sort_values("decile")
    )
    lift["default_rate"] = lift["defaults"] / lift["applicants"]
    lift_path = output_path(CONFIG["table_16_decile"])
    lift.to_csv(lift_path, index=False, float_format="%.6f")
    print(f"Saved {CONFIG['output_dir'].as_posix()}/{CONFIG['table_16_decile']}")
    print("\nDecile lift table (1 = lowest risk, 10 = highest risk):")
    print(lift.to_string(index=False))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(lift["decile"], lift["default_rate"], color="#E45756")
    ax.set_xticks(list(lift["decile"]))
    ax.set_xlabel("Decile (1 = lowest risk, 10 = highest risk)")
    ax.set_ylabel("Actual default rate")
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _pos: f"{value:.0%}"))
    ax.set_title("Test default rate by risk decile")
    for decile, rate in zip(lift["decile"], lift["default_rate"]):
        ax.text(decile, rate + 0.02, f"{rate:.1%}", ha="center", va="bottom", fontsize=8)
    save_figure(fig, CONFIG["figure_16"])

    print("\nPermutation importance on the transformed test set (ROC AUC, 5 repeats).")
    transformed = fitted.named_steps["preprocessor"].transform(X_test)
    feature_names = [
        str(name) for name in fitted.named_steps["preprocessor"].get_feature_names_out()
    ]
    importance_result = permutation_importance(
        fitted.named_steps["model"],
        transformed,
        y_test_array,
        n_repeats=CONFIG["permutation_repeats"],
        random_state=CONFIG["random_state"],
        scoring="roc_auc",
        n_jobs=-1,
    )
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": importance_result.importances_mean,
            "importance_std": importance_result.importances_std,
        }
    ).sort_values(["importance_mean", "feature"], ascending=[False, True])
    importance_path = output_path(CONFIG["table_15"])
    importance.to_csv(importance_path, index=False, float_format="%.6f")
    print(f"Saved {CONFIG['output_dir'].as_posix()}/{CONFIG['table_15']}")
    top = importance.head(CONFIG["permutation_top_n"])
    print(f"Top {len(top)} features:")
    print(top.to_string(index=False))
    top_plot = top.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        top_plot["feature"],
        top_plot["importance_mean"],
        xerr=top_plot["importance_std"],
        color="#4C78A8",
    )
    ax.set_xlabel("Mean ROC AUC decrease")
    ax.set_title("Permutation importance on the test set")
    save_figure(fig, CONFIG["figure_15"])

    print("\nPlain logistic regression fit on the training set for coefficient signs.")
    logistic = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=CONFIG["logistic_max_iter"],
                    random_state=CONFIG["random_state"],
                ),
            ),
        ]
    )
    logistic.fit(X_train, y_train)
    coefficients = pd.DataFrame(
        {
            "feature": [
                str(name)
                for name in logistic.named_steps["preprocessor"].get_feature_names_out()
            ],
            "coefficient": logistic.named_steps["model"].coef_.ravel(),
        }
    ).sort_values(["coefficient", "feature"], ascending=[False, True])
    coefficient_path = output_path(CONFIG["table_18"])
    coefficients.to_csv(coefficient_path, index=False, float_format="%.6f")
    print(f"Saved {CONFIG['output_dir'].as_posix()}/{CONFIG['table_18']}")
    print("Largest positive coefficient:", coefficients.iloc[0].to_dict())
    print("Largest negative coefficient:", coefficients.iloc[-1].to_dict())
    coefficient_plot = coefficients.sort_values("coefficient", ascending=True)
    colors = [
        "#E45756" if value >= 0 else "#4C78A8"
        for value in coefficient_plot["coefficient"]
    ]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(coefficient_plot["feature"], coefficient_plot["coefficient"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient")
    ax.set_title("Logistic regression coefficients (positive = higher default risk)")
    save_figure(fig, CONFIG["figure_18"])

    final_metrics = {
        "model": winner,
        "imbalance_strategy": strategy,
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "f2": test_metrics["f2"],
        "roc_auc": test_metrics["roc_auc"],
        "pr_auc": test_metrics["pr_auc"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "threshold": chosen,
        "best_params": best_params,
        "best_cv_f2": best_cv_f2,
    }
    metrics_path = output_path(CONFIG["final_metrics_json"])
    metrics_path.write_text(json.dumps(as_builtin(final_metrics), indent=2), encoding="utf-8")
    print(f"Saved {CONFIG['output_dir'].as_posix()}/{CONFIG['final_metrics_json']}")

    artifact = {
        "pipeline": fitted,
        "threshold": chosen,
        "score_cutoffs": score_cutoffs,
        "feature_columns": list(X_train.columns),
        "test_metrics": test_metrics,
    }
    model_path = ROOT / CONFIG["model_path"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    print(f"Saved {CONFIG['model_path'].as_posix()}")
    return {
        "best_params": best_params,
        "best_cv_f2": best_cv_f2,
        "threshold": chosen,
        "test_metrics": test_metrics,
        "score_cutoffs": score_cutoffs,
    }


def main() -> None:
    df = load_and_clean()
    run_eda(df)
    X_train, X_test, y_train, y_test = stratified_split(df)
    preparation = run_preprocessing(X_train)
    comparison = run_model_comparison(X_train, y_train, preparation["preprocessor"])
    imbalance = run_imbalance_experiment(
        X_train,
        y_train,
        preparation["preprocessor"],
        comparison["winner"],
    )
    final = run_final_evaluation(
        X_train,
        X_test,
        y_train,
        y_test,
        preparation["preprocessor"],
        comparison["winner"],
        imbalance["strategy"],
    )
    banner("STAGE 5 COMPLETE")
    print(f"Winning model: {comparison['winner']}")
    print(f"Imbalance handling: {imbalance['strategy']}")
    print(f"Best parameters: {final['best_params']}")
    print(f"Best mean CV F2: {final['best_cv_f2']:.4f}")
    print(f"Chosen threshold: {final['threshold']:.2f}")
    print(f"Score cutoffs: {final['score_cutoffs']}")
    print(f"Test metrics: {final['test_metrics']}")


if __name__ == "__main__":
    main()
