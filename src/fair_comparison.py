"""10-fold comparison with SMOTE applied the same way to every model.

Default hyperparameters. SMOTE is inside each training fold. This does not replace
outputs/08_model_comparison_results.csv or the saved model.
"""

import sys
import time
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train

OUTPUT_NAME = "21_fair_model_comparison.csv"
METRICS = ["recall", "f2", "roc_auc", "pr_auc"]


def default_models() -> list[tuple[str, object]]:
    """Library defaults, plus random_state 42 where the estimator accepts it."""
    seed = train.CONFIG["random_state"]
    return [
        ("Logistic Regression", LogisticRegression(random_state=seed)),
        ("KNN", KNeighborsClassifier()),
        ("Gaussian NB", GaussianNB()),
        ("Decision Tree", DecisionTreeClassifier(random_state=seed)),
        ("Random Forest", RandomForestClassifier(random_state=seed)),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=seed)),
        ("XGBoost", XGBClassifier(random_state=seed)),
    ]


def main() -> None:
    started = time.perf_counter()
    cleaned = train.load_and_clean()
    X_train, _X_test, y_train, _y_test = train.stratified_split(cleaned)
    preprocessor = train.build_preprocessor(list(train.CONFIG["numeric_columns"]))
    cv = StratifiedKFold(
        n_splits=train.CONFIG["cv_splits"],
        shuffle=True,
        random_state=train.CONFIG["random_state"],
    )
    scoring = {name: train.SCORING[name] for name in METRICS}
    rows = []
    for name, model in default_models():
        print(f"Cross-validating {name} with SMOTE inside each training fold...")
        pipeline = ImbPipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("smote", SMOTE(random_state=train.CONFIG["random_state"])),
                ("model", model),
            ]
        )
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
        )
        row = {"model": name, "imbalance": "SMOTE"}
        for metric in METRICS:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
            print(f"  {metric}: {row[f'{metric}_mean']:.4f} +/- {row[f'{metric}_std']:.4f}")
        rows.append(row)

    table = pd.DataFrame(rows)
    path = train.ROOT / "outputs" / OUTPUT_NAME
    if path.exists():
        raise SystemExit(f"Refusing to overwrite {path}")
    table.to_csv(path, index=False, float_format="%.6f")
    print(f"Saved outputs/{OUTPUT_NAME}")
    print(table.to_string(index=False))
    print(f"Part C runtime seconds: {time.perf_counter() - started:.1f}")


if __name__ == "__main__":
    main()
