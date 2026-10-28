# Predictive analytics for loan default risk

This project scores loan applicants by the risk that they default. The score is a rank from 0 to 100, with bands Low, Medium, and High. Decile 10 is the highest risk. A separate review flag marks applicants whose default probability is at or above a threshold chosen to catch defaulters.

The training script compares seven classifiers, keeps class weights rather than SMOTE, tunes the winner, and saves one model file. A Streamlit app scores one applicant or a CSV batch from that file. It does not retrain.

## Setup

The machine used for the reported numbers was Python 3.13.7. The pinned NumPy 2.3.1 build needs Python 3.11 or newer.

```text
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with `source .venv/bin/activate` instead.

The dataset is the Kaggle [Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset), published by Lao Tse. Put the file at `data/credit_risk_dataset.csv`. The copy in this folder is that dataset (32,581 rows, 12 columns). It was saved from the public GitHub mirror [Jv89/Credit-Risk-SQL-Analysis](https://github.com/Jv89/Credit-Risk-SQL-Analysis) so the script can run without a Kaggle login. Column names match the project spec. In the file, `loan_status` is not the last column.

## Run order

From the project folder:

```text
python src/train.py
streamlit run app.py
```

`src/train.py` loads the data, writes the figures and tables in `outputs/`, and saves `models/loan_default_model.joblib`. Then start the app. The app only loads that file.

A full run of `python src/train.py` on this machine took 4 minutes 5 seconds (245 seconds) and exited 0. A second run, with no setting changes, rewrote `outputs/final_metrics.json` with the same values: accuracy 0.8744, precision 0.6603, recall 0.8773, F1 0.7535, F2 0.8232, ROC AUC 0.9548, PR AUC 0.9149, threshold 0.35, and best 5-fold CV F2 0.8017.

## Output index

Numbering follows the project spec. A few tables sit beside the numbered figure.

| # | File | What it is |
| --- | --- | --- |
| 01 | `outputs/01_target_distribution.png` | Class balance |
| 02 | `outputs/02_numeric_histograms.png` | Numeric histograms |
| 03 | `outputs/03_default_rate_by_loan_grade.png` | Default rate by loan grade |
| 04 | `outputs/04_default_rate_by_home_ownership.png` | Default rate by home ownership |
| 05 | `outputs/05_default_rate_by_loan_intent.png` | Default rate by loan intent |
| 06 | `outputs/06_correlation_heatmap.png` | Numeric correlations, including the target |
| 07 | `outputs/07_vif_before_after.csv`, `outputs/07_vif_before_after.png` | VIF before and after |
| 08 | `outputs/08_model_comparison_results.csv` | Seven-model means and standard deviations |
| 09 | `outputs/09_cv_boxplots_f2_recall.png` | Per-fold F2 and recall |
| 10 | `outputs/10_imbalance_experiment.csv`, `outputs/10_imbalance_experiment.png` | None vs class weights vs SMOTE |
| 11 | `outputs/11_threshold_metrics.csv`, `outputs/11_threshold_metrics.png` | Threshold scan |
| 12 | `outputs/12_confusion_matrix.png` | Test confusion matrix |
| 13 | `outputs/13_roc_curve.png` | Test ROC curve |
| 14 | `outputs/14_precision_recall_curve.png` | Test precision-recall curve |
| 15 | `outputs/15_permutation_importance.csv`, `outputs/15_permutation_importance.png` | Permutation importance |
| 16 | `outputs/16_decile_default_rate.png`, `outputs/16_decile_lift_table.csv`, `outputs/16_test_risk_scores.csv` | Decile lift chart, lift table, and row scores |
| 17 | Streamlit screenshots | Not saved. The app was not opened in a browser |
| 18 | `outputs/18_logistic_regression_coefficients.csv`, `outputs/18_logistic_regression_coefficients.png` | Logistic regression coefficients. The spec has no number for this plot, so it is 18, leaving 17 for screenshots |

`outputs/model_comparison_results.csv` is the same table as file 08. `outputs/final_metrics.json` holds the test metrics, the confusion-matrix counts, the chosen threshold, the best parameters, and the best cross-validated F2.

## Choices made

Where the spec left a choice, the simpler option that still matches it was used.

**Data and cleaning.** The project files live in this folder rather than a nested `loan-default-risk` directory. Exact duplicates are removed first (165 rows). Rows with `person_age > 100` (5) and `person_emp_length > 60` (2) are then counted separately and dropped together. No row failed both rules. Missing employment length is kept and imputed later. After cleaning, 32,409 rows remain, with a default rate of 21.87%.

**EDA.** Plots and correlations use the full cleaned table, before the split. Correlation is Pearson and pairwise complete, with no imputation. The heatmap includes the seven numeric columns and `loan_status`. Home-ownership and loan-intent bars are sorted by default rate. Loan grade stays in A to G order.

**Encoding and VIF.** `Y`/`N` is an ordinal encoding of `["N", "Y"]`, so N is 0 and Y is 1. One-hot encoding uses `handle_unknown="ignore"` and dense output. Unseen loan grades or Y/N values become −1. VIF uses only the seven numeric columns, after a median imputer fit on the training rows, with an intercept that is not a feature. A tie would be broken by feature name. Every VIF was at most 4.40, so no column was dropped. Categorical fields are not imputed. The modelling transformer is built in that section and fitted only inside later training pipelines.

**Model comparison.** The standard deviation is the sample standard deviation across the 10 folds (`ddof=1`). ROC AUC and PR AUC use predicted probabilities. `scale_pos_weight` for XGBoost is 20,257 / 5,670, computed once on the training labels. KNN and Gaussian Naive Bayes have no `random_state`. Every other stochastic step uses seed 42. The winner is the highest mean F2. If another model is within 0.01 F2, the higher mean ROC AUC wins. XGBoost led by 0.046 F2 (0.8015 versus 0.7553 for the decision tree), so the tie-break did not apply.

**Imbalance.** For XGBoost, class weights means `scale_pos_weight`. The no-handling and SMOTE arms use `scale_pos_weight = 1`. SMOTE sits after the preprocessor inside an imbalanced-learn pipeline (`random_state=42`, default `sampling_strategy="auto"`), so it is fit on each training fold only. Synthetic one-hot values can fall between 0 and 1. SMOTE’s mean F2 was 0.0308 below class weights, so the final model keeps class weights. KNN, Gaussian Naive Bayes, and gradient boosting have no class-weight setting in the spec. If one of them had won, that arm would have used the same estimator as no handling.

**Threshold, bands, and deciles.** Grid search scores F2 at the model’s default 0.5 cutoff. The operating threshold is chosen afterward from 10-fold out-of-fold probabilities, scanning 0.05 to 0.95 by 0.01. If several thresholds tie on F2, the lowest is kept. This run selected 0.35 with no tie. The 5-fold grid F2 (0.8017) and the 10-fold out-of-fold F2 at 0.50 (0.8042) are different checks. Score cutoffs are the 50th and 80th percentiles of out-of-fold probability times 100 (13.67 and 54.27). High is at or above the 80th, Medium is from the 50th up to the 80th, and Low is below the 50th. The displayed score is rounded to 2 decimals. The band uses the unrounded score. Decile 10 is the highest risk, which follows the detailed training section rather than the summary line that says decile 1 is highest. `row_id` is the row index after cleaning, because the file has no customer id. The risk table also stores the band.

**Explainability.** Permutation importance is computed on the encoded test matrix, with ROC AUC, 5 repeats, and seed 42. The chart shows the top 15. The logistic regression used for the coefficient plot is the balanced model from the bake-off (`max_iter=1000`, seed 42) with the same preprocessor, fit on the training set only. It is not the scored model.

**App.** The review flag uses probability at or above 0.35, the same rule as the test-set labels. A single applicant has no decile. In a batch, decile 10 is the highest risk in that file. Files with fewer than 10 rows cannot fill 10 equal groups, so ranks are scaled onto 1–10 and the highest score is still decile 10. Extra columns, including `loan_status`, are ignored. Blank employment length and interest rate are allowed, because the pipeline imputes them. Other missing values, unknown categories, and out-of-range numbers are rejected. Allowed ranges are age 18–100, income 1,000–2,500,000, employment length 0–60, loan amount 500–50,000, interest rate 0–40, loan as a fraction of income 0–1, and credit history 0–40 years. Employment length and credit history cannot exceed age. The band default rates shown in the app are read from `outputs/16_test_risk_scores.csv`.

## Limitations

`loan_grade` and `loan_int_rate` are assigned by the lender after its own risk assessment, so the model partly reproduces that view. `person_age` is a sensitive attribute. It is kept here, and a production system would need a fairness review. The data is a public sample, not one bank’s portfolio, so the numbers show the method rather than that bank’s performance. The score is a ranking. Class-weighted scores are not calibrated probabilities. The SMOTE comparison uses SMOTE on the encoded numeric matrix, not SMOTENC, so synthetic rows can contain fractional one-hot values. SMOTE was not selected for the final model.

## Acceptance checklist

Checked against the project acceptance criteria after the reproducibility run.

1. **Not met.** `python src/train.py` runs from start to finish with no manual steps and writes outputs 01–16, the metrics file, and the model. Output 17, the Streamlit screenshots, was not saved.
2. **Met.** The stratified split happens before any model is fit. Imputers, encoders, and the scaler sit inside the training pipeline. SMOTE, when it is used, sits inside an imbalanced-learn pipeline and is fit on training folds only.
3. **Met.** `outputs/08_model_comparison_results.csv` has all 7 models, with mean and standard deviation for F2, recall, precision, ROC AUC, PR AUC, and accuracy.
4. **Met.** Test metrics are at the tuned threshold 0.35. Recall is 0.8773, F2 is 0.8232, and ROC AUC is 0.9548. Those are above the soft targets of 0.70, 0.65, and 0.85. The confusion matrix is TN 4424, FP 640, FN 174, TP 1244.
5. **Met.** In `outputs/16_decile_lift_table.csv` the test default rate rises from decile 1 to decile 10: 0%, 0.3%, 1.7%, 2.6%, 2.8%, 6.8%, 11.9%, 16.7%, 75.9%, 100%.
6. **Met for scoring, not verified in the browser.** The app’s scoring function, loading `models/loan_default_model.joblib`, scored one applicant and a 12-row CSV. A file that also contained `loan_status` was accepted and that column was ignored. The Streamlit page was not opened, so the on-screen form and upload control were not clicked.
7. **Not met.** This item asks that every number in the course report match a saved file or the console. The report has not been written, so that check cannot be done. The figures in this README are taken from `outputs/final_metrics.json` and `outputs/16_decile_lift_table.csv`.
