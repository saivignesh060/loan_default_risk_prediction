"""Streamlit demo for loan default risk scores.

Loads models/loan_default_model.joblib. It does not retrain the model.
Run the app with: streamlit run app.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
ARTIFACT_PATH = ROOT / "models" / "loan_default_model.joblib"
IMPORTANCE_PATH = ROOT / "outputs" / "15_permutation_importance.csv"

# Bounds follow the training cleaning rules and the observed cleaned data,
# with a little room beyond the sample minimums and maximums.
NUMERIC_RANGES = {
    "person_age": (18, 100),
    "person_income": (1000, 2_500_000),
    "person_emp_length": (0, 60),
    "loan_amnt": (500, 50_000),
    "loan_int_rate": (0, 40),
    "loan_percent_income": (0, 1),
    "cb_person_cred_hist_length": (0, 40),
}
# These two columns were missing in the training data and are imputed in the pipeline.
NULLABLE_NUMERIC = {"person_emp_length", "loan_int_rate"}
CATEGORICAL_VALUES = {
    "person_home_ownership": ["RENT", "MORTGAGE", "OWN", "OTHER"],
    "loan_intent": [
        "DEBTCONSOLIDATION",
        "EDUCATION",
        "HOMEIMPROVEMENT",
        "MEDICAL",
        "PERSONAL",
        "VENTURE",
    ],
    "loan_grade": ["A", "B", "C", "D", "E", "F", "G"],
    "cb_person_default_on_file": ["N", "Y"],
}
BAND_COLORS = {"Low": "#2e7d32", "Medium": "#ef6c00", "High": "#c62828"}
REVIEW_MESSAGE = "Likely default: flag for review"


def load_artifact(path: Path | None = None) -> dict:
    artifact_path = ARTIFACT_PATH if path is None else Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Saved model not found: {artifact_path}")
    artifact = joblib.load(artifact_path)
    required = {"pipeline", "threshold", "score_cutoffs", "feature_columns", "test_metrics"}
    missing = required.difference(artifact)
    if missing:
        raise ValueError(f"Model file is missing keys: {sorted(missing)}")
    return artifact


def risk_band(score: float, cutoffs: dict) -> str:
    """High is at or above the 80th percentile. Low is below the 50th."""
    if score >= float(cutoffs["p80"]):
        return "High"
    if score >= float(cutoffs["p50"]):
        return "Medium"
    return "Low"


def assign_deciles(probabilities: np.ndarray) -> np.ndarray:
    """Decile 10 is the highest risk in this batch.

    With at least 10 rows this matches training: rank, then pd.qcut into 10 bins.
    Fewer than 10 rows cannot fill 10 equal groups, so ranks are scaled onto 1-10
    and the highest score in the file is still decile 10.
    """
    ranks = pd.Series(probabilities).rank(method="first", ascending=True)
    if len(ranks) >= 10:
        deciles = pd.qcut(ranks, 10, labels=list(range(1, 11)))
        return deciles.astype(int).to_numpy()
    scaled = np.ceil(ranks.to_numpy() / len(ranks) * 10).astype(int)
    return np.clip(scaled, 1, 10)


def _problem_rows(index) -> str:
    labels = [str(int(row) + 2) if str(row).isdigit() else str(row) for row in list(index)[:8]]
    suffix = "" if len(index) <= 8 else f" and {len(index) - 8} more"
    return ", ".join(labels) + suffix


def validate_applicants(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Return a model-ready frame or raise ValueError with a readable message.

    Columns other than the model features, including loan_status, are ignored.
    """
    if frame is None or frame.empty:
        raise ValueError("The file has no applicant rows.")
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))

    data = frame.loc[:, feature_columns].copy()
    problems = []
    for column, (low, high) in NUMERIC_RANGES.items():
        numeric = pd.to_numeric(data[column], errors="coerce")
        blank = data[column].isna() | data[column].astype(str).str.strip().isin(["", "nan", "None"])
        non_numeric = numeric.isna() & ~blank
        if non_numeric.any():
            problems.append(f"{column} is not numeric at CSV row {_problem_rows(data.index[non_numeric])}.")
        if column not in NULLABLE_NUMERIC and numeric.isna().any():
            problems.append(f"{column} is missing at CSV row {_problem_rows(data.index[numeric.isna()])}.")
        out_of_range = numeric.notna() & ((numeric < low) | (numeric > high))
        if out_of_range.any():
            problems.append(
                f"{column} must be between {low} and {high}. "
                f"Out of range at CSV row {_problem_rows(data.index[out_of_range])}."
            )
        data[column] = numeric

    for column, allowed in CATEGORICAL_VALUES.items():
        text = data[column].astype(str).str.strip()
        blank = data[column].isna() | text.isin(["", "nan", "None"])
        if blank.any():
            problems.append(f"{column} is missing at CSV row {_problem_rows(data.index[blank])}.")
        unknown = ~blank & ~text.isin(allowed)
        if unknown.any():
            found = sorted(text.loc[unknown].unique())
            problems.append(
                f"{column} has unrecognised values {found}. Allowed values: {', '.join(allowed)}."
            )
        data[column] = text

    comparable = data["person_age"].notna() & data["person_emp_length"].notna()
    too_long_job = comparable & (data["person_emp_length"] > data["person_age"])
    if too_long_job.any():
        problems.append(
            "person_emp_length cannot be greater than person_age. "
            f"Check CSV row {_problem_rows(data.index[too_long_job])}."
        )
    comparable_hist = data["person_age"].notna() & data["cb_person_cred_hist_length"].notna()
    too_long_hist = comparable_hist & (data["cb_person_cred_hist_length"] > data["person_age"])
    if too_long_hist.any():
        problems.append(
            "cb_person_cred_hist_length cannot be greater than person_age. "
            f"Check CSV row {_problem_rows(data.index[too_long_hist])}."
        )

    if problems:
        raise ValueError("Invalid applicant data:\n" + "\n".join(problems))
    return data


def score_applicants(frame: pd.DataFrame, artifact: dict | None = None) -> pd.DataFrame:
    """Score applicants. Decile 10 is the highest risk in this batch."""
    if artifact is None:
        artifact = load_artifact()
    features = validate_applicants(frame, list(artifact["feature_columns"]))
    model = artifact["pipeline"]
    classes = [int(label) for label in model.named_steps["model"].classes_]
    if classes != [0, 1]:
        raise RuntimeError(f"Expected classes [0, 1], found {classes}.")
    probabilities = np.asarray(model.predict_proba(features)[:, 1], dtype=np.float64)
    scores = np.round(probabilities * 100, 2)
    threshold = float(artifact["threshold"])
    # Same rule as the test-set labels: flag when probability is at or above the threshold.
    flagged = probabilities >= threshold
    scored = features.copy()
    scored["risk_score"] = scores
    scored["band"] = [risk_band(float(score), artifact["score_cutoffs"]) for score in scores]
    scored["decile"] = assign_deciles(probabilities)
    scored["review_flag"] = np.where(flagged, REVIEW_MESSAGE, "")
    return scored.sort_values(["risk_score", "decile"], ascending=[False, False]).reset_index(drop=True)


def test_band_default_rates() -> dict[str, float]:
    """Observed test-set default rate by band, read from the saved risk table."""
    table = pd.read_csv(ROOT / "outputs" / "16_test_risk_scores.csv")
    rates = {}
    for band in ("Low", "Medium", "High"):
        part = table.loc[table["band"] == band, "actual"]
        rates[band] = float(part.mean()) if len(part) else float("nan")
    return rates


def ranking_caption() -> str:
    rates = test_band_default_rates()
    return (
        "The score is a ranking, not a probability of default. "
        "On the held-out test set the observed default rate was "
        f"Low {rates['Low']:.1%}, Medium {rates['Medium']:.1%}, High {rates['High']:.1%}."
    )


def driver_note() -> str:
    fallback = (
        "Top global risk drivers from the test-set check: "
        "person income, loan percent income, and loan grade."
    )
    if not IMPORTANCE_PATH.exists():
        return fallback
    table = pd.read_csv(IMPORTANCE_PATH)
    if "feature" not in table.columns or table.empty:
        return fallback
    names = []
    for raw_name in table["feature"].head(3):
        names.append(str(raw_name).split("__", 1)[-1].replace("_", " "))
    return "Top global risk drivers from the test-set check: " + ", ".join(names) + "."


def _show_score(row: pd.Series, cutoffs: dict) -> None:
    band = str(row["band"])
    color = BAND_COLORS[band]
    st.metric("Risk score", f"{float(row['risk_score']):.2f}")
    st.markdown(
        f"<p style='font-size:1.6rem;margin-top:0'>Band: "
        f"<strong style='color:{color}'>{band}</strong></p>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Low is below {float(cutoffs['p50']):.2f}. "
        f"Medium is from {float(cutoffs['p50']):.2f} up to {float(cutoffs['p80']):.2f}. "
        f"High is {float(cutoffs['p80']):.2f} or higher."
    )
    if row["review_flag"]:
        st.error(REVIEW_MESSAGE)
    st.caption(ranking_caption())
    st.caption(
        "Band comes from the applicant's rank (Low = bottom 50%, Medium = next 30%, "
        "High = top 20% of scores). The review flag uses a separate threshold optimised "
        "for catching defaulters, so some Medium-band applicants are also flagged."
    )
    st.info(driver_note())


def _single_applicant_tab(artifact: dict) -> None:
    st.write("Enter one applicant. The score runs from 0 to 100.")
    with st.form("single_applicant"):
        left, right = st.columns(2)
        with left:
            person_age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
            person_income = st.number_input(
                "Annual income", min_value=1000, max_value=2_500_000, value=50_000, step=1000
            )
            person_home_ownership = st.selectbox(
                "Home ownership", CATEGORICAL_VALUES["person_home_ownership"]
            )
            person_emp_length = st.number_input(
                "Employment length (years)", min_value=0.0, max_value=60.0, value=5.0, step=0.5
            )
            loan_intent = st.selectbox("Loan intent", CATEGORICAL_VALUES["loan_intent"])
            loan_grade = st.selectbox("Loan grade", CATEGORICAL_VALUES["loan_grade"], index=1)
        with right:
            loan_amnt = st.number_input(
                "Loan amount", min_value=500, max_value=50_000, value=10_000, step=500
            )
            loan_int_rate = st.number_input(
                "Interest rate (%)", min_value=0.0, max_value=40.0, value=11.0, step=0.1
            )
            loan_percent_income = st.number_input(
                "Loan as a fraction of income",
                min_value=0.0,
                max_value=1.0,
                value=0.20,
                step=0.01,
                format="%.2f",
            )
            cb_person_default_on_file = st.selectbox(
                "Previous default on file", CATEGORICAL_VALUES["cb_person_default_on_file"]
            )
            cb_person_cred_hist_length = st.number_input(
                "Credit history length (years)", min_value=0, max_value=40, value=5, step=1
            )
        submitted = st.form_submit_button("Score applicant")

    if not submitted:
        return
    applicant = pd.DataFrame(
        [
            {
                "person_age": person_age,
                "person_income": person_income,
                "person_home_ownership": person_home_ownership,
                "person_emp_length": person_emp_length,
                "loan_intent": loan_intent,
                "loan_grade": loan_grade,
                "loan_amnt": loan_amnt,
                "loan_int_rate": loan_int_rate,
                "loan_percent_income": loan_percent_income,
                "cb_person_default_on_file": cb_person_default_on_file,
                "cb_person_cred_hist_length": cb_person_cred_hist_length,
            }
        ]
    )
    try:
        scored = score_applicants(applicant, artifact)
    except ValueError as exc:
        st.error(str(exc))
        return
    _show_score(scored.iloc[0], artifact["score_cutoffs"])


def _batch_tab(artifact: dict) -> None:
    st.write("Upload a CSV with the applicant columns.")
    st.caption("Extra columns, such as loan_status, are ignored.")
    st.caption("Decile 10 is the highest risk in the uploaded file.")
    st.caption(ranking_caption())
    upload = st.file_uploader("Applicant CSV", type=["csv"])
    if upload is None:
        return
    try:
        frame = pd.read_csv(upload)
        scored = score_applicants(frame, artifact)
    except ValueError as exc:
        st.error(str(exc))
        return
    except pd.errors.EmptyDataError:
        st.error("The CSV is empty.")
        return
    if len(scored) < 10:
        st.caption(
            "This file has fewer than 10 rows, so deciles are scaled from rank. "
            "Decile 10 is still the highest risk in the file."
        )
    display = scored[
        ["risk_score", "band", "decile", "review_flag", *artifact["feature_columns"]]
    ]
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.download_button(
        "Download scored CSV",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name="loan_risk_scores.csv",
        mime="text/csv",
    )


def main() -> None:
    st.set_page_config(page_title="Loan default risk", layout="centered")
    st.title("Loan default risk")
    st.write("The score ranks applicants. It is not a calibrated probability.")
    try:
        artifact = load_artifact()
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        return
    single_tab, batch_tab = st.tabs(["Single applicant", "Batch CSV"])
    with single_tab:
        _single_applicant_tab(artifact)
    with batch_tab:
        _batch_tab(artifact)


if __name__ == "__main__":
    main()
