# Loan Default Risk Prediction: MVP PRD

*Data Mining course-based project. Python, scikit-learn, XGBoost, Streamlit.*

## 1. Goal

Build a basic working system that predicts whether a loan applicant is likely to default and outputs a **default risk score** used to rank applicants. The project follows the same end-to-end shape as the churn reference report (data, preprocessing and diagnostics, multi-model comparison, tuning, evaluation, saved model) with two upgrades: explicit class-imbalance handling and a small working demo app.

**Done means:** one reproducible training script, one saved model file, one Streamlit demo, and every figure and number needed to write the 12-chapter report.

## 2. Decisions at a glance

All choices below are fixed for the MVP so nothing is left open.

| Area | Decision |
| --- | --- |
| Problem type | Binary classification. Target `loan_status`: 1 = default, 0 = non-default |
| Dataset | Kaggle *Credit Risk Dataset* (`credit_risk_dataset.csv`, about 32.5k rows, 12 columns, roughly 22% defaults). Verify column names after download |
| Fallback dataset | *Give Me Some Credit* (Kaggle), only if the main one is unavailable |
| Split | 80/20 stratified, `random_state=42`, done before any fitting |
| Preprocessing | One sklearn `Pipeline` with `ColumnTransformer` (impute, encode, scale) so nothing leaks from test data or CV folds |
| Imbalance handling | Class weights by default. SMOTE compared against it on the winning model |
| Models (7) | Logistic Regression, KNN, Gaussian Naive Bayes, Decision Tree, Random Forest, Gradient Boosting, XGBoost |
| Validation | Stratified 10-fold CV on the training set |
| Primary metric | F2-score. Also report Recall, Precision, ROC AUC, PR AUC, Accuracy |
| Model selection | Highest mean CV F2. If two models are within 0.01, pick the higher ROC AUC |
| Tuning | `GridSearchCV`, scoring = F2, 5-fold, small grids |
| Decision threshold | Tuned on out-of-fold predictions to maximise F2 (not fixed at 0.5) |
| Output | Risk score 0 to 100, risk decile (1 = highest), band: Low / Medium / High |
| Explainability | Permutation importance on the test set plus Logistic Regression coefficients. No SHAP in MVP |
| Demo | Streamlit app: single-applicant form and batch CSV scoring |
| Saved artifact | `models/loan_default_model.joblib` containing pipeline, threshold, score cutoffs, feature list |

## 3. Scope

**In scope (MVP)**

- Data loading, cleaning, EDA, VIF diagnostic
- Seven-model comparison with 10-fold CV
- Imbalance experiment (none vs class weight vs SMOTE)
- Tuning, threshold selection, final test evaluation
- Risk scoring, decile ranking, permutation importance
- Streamlit demo
- All figures saved to `outputs/`

**Out of scope (list under Future Scope in the report)**

- Time-based validation, survival analysis, probability calibration
- Per-applicant explanations (SHAP/LIME)
- Fairness audits, cost-sensitive optimisation with real loan amounts
- Deployment beyond a local Streamlit app, authentication, databases

## 4. Dataset specification

Expected columns (confirm after download):

| Column | Type | Group |
| --- | --- | --- |
| `person_age` | numeric | Demographics |
| `person_income` | numeric | Demographics / financial |
| `person_home_ownership` | categorical (RENT, OWN, MORTGAGE, OTHER) | Demographics |
| `person_emp_length` | numeric (years) | Demographics |
| `loan_intent` | categorical (6 values) | Loan terms |
| `loan_grade` | ordinal categorical (A to G) | Loan terms |
| `loan_amnt` | numeric | Loan terms |
| `loan_int_rate` | numeric | Loan terms |
| `loan_percent_income` | numeric | Loan terms |
| `cb_person_default_on_file` | binary (Y/N) | Credit history |
| `cb_person_cred_hist_length` | numeric (years) | Credit history |
| `loan_status` | binary target | Target |

**Target definition (state this in the report's Data Collection section):** `loan_status = 1` means the borrower defaulted on the loan (stopped repaying, typically 90+ days overdue or written off). `0` means the loan is being or was repaid.

**Known data issues to handle:** missing values in `person_emp_length` and `loan_int_rate`; impossible values (for example ages above 100 and employment length above 60 years); duplicate rows.

## 5. Functional requirements: training script

Single file `src/train.py`, organised in lettered sections like the reference script (A to G), printing progress and saving outputs.

**A. Load and clean**

- Read CSV, drop exact duplicates, remove rows with `person_age > 100` or `person_emp_length > 60`. Print how many rows were removed at each step (needed for the report).
- Print shape, dtypes, missing-value counts, and target distribution.

**B. EDA (save figures)**

- Target distribution (default rate), numeric histograms, default rate by `loan_grade`, `person_home_ownership`, `loan_intent`, correlation heatmap of numeric features.
- Print the top positive and negative correlations with the target.

**C. Split**

- Stratified 80/20 split on the cleaned data. Everything after this is fitted on the training set only.

**D. Preprocessing pipeline**

- Numeric: median imputation, then `StandardScaler`.
- `loan_grade`: `OrdinalEncoder` with order A to G.
- `cb_person_default_on_file`: map Y/N to 1/0.
- `person_home_ownership`, `loan_intent`: `OneHotEncoder(handle_unknown='ignore')`.
- **VIF diagnostic:** on the imputed numeric training features, compute VIF with `statsmodels`. Iteratively drop the feature with the highest VIF until all are at most 5. Log which features were dropped and why. Save the VIF table before and after.

**E. Baseline comparison (the model bake-off)**

| Model | Settings |
| --- | --- |
| Logistic Regression | `class_weight='balanced'`, `max_iter=1000` |
| KNN | `n_neighbors=15` |
| Gaussian NB | defaults |
| Decision Tree | `max_depth=8`, `class_weight='balanced'` |
| Random Forest | `n_estimators=200`, `class_weight='balanced_subsample'`, `n_jobs=-1` |
| Gradient Boosting | defaults |
| XGBoost | `scale_pos_weight = negatives / positives`, `eval_metric='logloss'` |

- Wrap each in `Pipeline([preprocessor, model])`. Run `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)` using `cross_validate` with scorers: accuracy, precision, recall, F2, ROC AUC, average precision.
- Save `outputs/model_comparison_results.csv` (mean and std per metric) and a boxplot of per-fold F2 and Recall.
- Select the winner using the rule in Section 2.

**F. Imbalance experiment**

- On the winning model only, compare with 10-fold CV: no handling, class weights, SMOTE (use `imblearn.pipeline.Pipeline` so SMOTE runs inside each training fold only). Save a small results table and bar chart.
- If SMOTE beats class weights on mean CV F2 by more than 0.01, use SMOTE for the final model. Otherwise keep class weights.

**G. Tuning, threshold, final evaluation**

- `GridSearchCV` on the winner (5-fold, scoring F2). Grids for every model type so the script tunes whichever wins:
  - Logistic Regression: `C` in \[0.01, 0.1, 1, 10\], `penalty` in \[l1, l2\] (solver `liblinear`)
  - Random Forest: `n_estimators` \[200, 400\], `max_depth` \[8, 12, None\], `min_samples_leaf` \[1, 5\]
  - XGBoost: `n_estimators` \[200, 400\], `max_depth` \[3, 5\], `learning_rate` \[0.05, 0.1\]
  - Gradient Boosting: `n_estimators` \[100, 200\], `max_depth` \[2, 3\], `learning_rate` \[0.05, 0.1\]
  - Decision Tree: `max_depth` \[4, 6, 8, 10\], `min_samples_leaf` \[1, 5, 20\]
  - KNN: `n_neighbors` \[5, 15, 25\]
  - Gaussian NB: `var_smoothing` \[1e-9, 1e-8, 1e-7\]
- **Threshold:** get out-of-fold probabilities on the training set with `cross_val_predict(method='predict_proba')`, scan thresholds 0.05 to 0.95 in steps of 0.01, choose the one with the highest F2. Save a threshold vs Precision/Recall/F2 plot.
- **Test evaluation (run once):** with the tuned model and chosen threshold, report Accuracy, Precision, Recall, F1, F2, ROC AUC and PR AUC. **Compute ROC AUC from predicted probabilities, never from hard 0/1 predictions.** Save confusion matrix, ROC curve, PR curve.
- **Risk output:** on the test set produce a table with `customer/row id`, actual, predicted, risk score (probability times 100, rounded to 2 decimals), and decile rank (10 = highest risk, using `pd.qcut` on the rank). Save a decile table showing actual default rate per decile (a lift table) and a bar chart of it.
- **Explainability:** permutation importance on the test set (top 15 features, bar chart). Also fit a plain Logistic Regression and plot its coefficients (sign shows direction of risk).
- **Save artifact:** `joblib.dump` a dict with `pipeline`, `threshold`, `score_cutoffs` (training-set 50th and 80th percentile of scores), `feature_columns`, and `test_metrics`.

**Note on the score:** class-weighted models do not give calibrated probabilities, so call the output a *default risk score*, not a true probability. Bands are percentile-based: High = score at or above the 80th percentile cutoff, Medium = between the 50th and 80th, Low = below the 50th.

## 6. Functional requirements: demo app

File `app.py` (Streamlit). Loads the joblib artifact. No retraining inside the app.

- **Tab 1, Single applicant:** form with one input per feature, with sensible ranges and dropdowns for categorical fields. On submit show risk score, band (colour-coded), the flag "Likely default: flag for review" if the score is above the saved threshold, and a short note on the top global risk drivers.
- **Tab 2, Batch scoring:** upload a CSV with the same columns, validate columns, show a table sorted by risk score with decile and band, with a download button.
- Input validation: show a clear error for missing columns or out-of-range values instead of crashing.

## 7. Deliverables and repo layout

```
loan-default-risk/
  data/credit_risk_dataset.csv
  src/train.py
  app.py
  models/loan_default_model.joblib
  outputs/               (all figures and CSV results)
  requirements.txt
  README.md
```

`requirements.txt`: pandas, numpy, scikit-learn, imbalanced-learn, xgboost, statsmodels, matplotlib, seaborn, joblib, streamlit.

`README.md`: how to install, run `python src/train.py`, then `streamlit run app.py`. Include the dataset download link and expected runtime.

All randomness uses seed 42 so numbers in the report can be reproduced.

## 8. Acceptance criteria

1. `python src/train.py` runs top to bottom with no manual steps and writes all outputs listed in Section 9.
2. No data leakage: split happens before fitting, and imputers, encoders, scalers and SMOTE are fitted only inside training folds.
3. The comparison table has all 7 models with mean and standard deviation for 6 metrics.
4. Final test metrics are reported at the tuned threshold. Target (soft) values: Recall at least 0.70, F2 at least 0.65, ROC AUC at least 0.85. Report whatever the code produces; do not adjust the report to hit the targets.
5. The decile table shows default rate rising from the lowest-risk to the highest-risk decile.
6. The Streamlit app scores a single applicant and a batch CSV using the saved artifact.
7. Every number quoted in the report matches the saved CSV or console output.

## 9. Outputs needed for the report's Output chapter

Save as numbered files in `outputs/`:

| # | File | Used for |
| --- | --- | --- |
| 01 | target distribution | Class imbalance evidence |
| 02 | numeric histograms | Data understanding |
| 03 | default rate by loan grade | Key driver |
| 04 | default rate by home ownership | Key driver |
| 05 | default rate by loan intent | Key driver |
| 06 | correlation heatmap | Feature relationships |
| 07 | VIF table before and after | Multicollinearity diagnostic |
| 08 | model comparison table (CSV) | Bake-off results |
| 09 | CV boxplots (F2 and Recall) | Model stability |
| 10 | imbalance experiment chart | Handling minority class |
| 11 | threshold vs metrics plot | Threshold choice |
| 12 | confusion matrix | Final result |
| 13 | ROC curve | Final result |
| 14 | Precision-Recall curve | Final result for imbalanced data |
| 15 | permutation importance | Why applicants default |
| 16 | decile default-rate chart | Ranking usefulness |
| 17 | Streamlit screenshots (form and batch) | Working prototype |

## 10. Report plan (same 12 chapters as the reference)

| Chapter | What to write |
| --- | --- |
| 1. Abstract | Problem, dataset, preprocessing (imputation, encoding, VIF), 7 models with 10-fold CV, imbalance handling, F2 focus, final model and key numbers, risk score output |
| 2. Introduction | Why defaults are costly, why early identification matters, project objective, structure of the report |
| 3. Literature Review | Credit scoring and default prediction, class imbalance techniques (class weights, SMOTE), model comparison studies, interpretability in lending, evaluation metrics for imbalanced data. Use only sources you have actually read |
| 4. Research Gap | Binary-only outputs with no ranking, accuracy used on imbalanced data, black-box models, leakage in preprocessing, no end-to-end deployable artifact. Say which gaps your project addresses and which it does not |
| 5. Existing System | Manual credit officer judgement and rule-based scorecards, simple statistical models, black-box commercial scoring tools, academic notebooks with no deployment |
| 6. Proposed System | System overview, key features, five-layer pipeline architecture (ingestion and cleaning, diagnostics, preparation, training and selection, evaluation and deployment) |
| 7. Methodology | KDD phases: problem understanding, data collection, preprocessing, model development, evaluation, deployment. Define every metric and explain why F2 and Recall lead |
| 8. Architecture and Implementation | Pipeline diagram plus code listings by section A to G, and the Streamlit app |
| 9. Output | Figures 01 to 17 with a short interpretation under each |
| 10. Conclusion | What was built, final model and test metrics, what the drivers of default are, what the score enables |
| 11. Future Scope | Probability calibration, cost-sensitive thresholds using loan amounts, SHAP explanations, time-based validation, fairness audit, deployment as an API, richer data |
| 12. References | Real, verified citations only |

**Safe references to start with:** Pedregosa et al. 2011 (scikit-learn, JMLR); Chen and Guestrin 2016 (XGBoost, KDD); Chawla et al. 2002 (SMOTE, JAIR); Breiman 2001 (Random Forests, Machine Learning); Belsley, Kuh and Welsch 1980 (collinearity diagnostics); RBI guidelines on NPA classification. Add papers on credit default prediction only after you have found and checked them.

**Quality rules learned from the reference report (do not copy these mistakes):**

- Keep the model count consistent everywhere (the reference says seven but lists six in places).
- Quote one set of final numbers. The reference conclusion cites 0.845 ROC AUC while its ROC figure shows 0.80, because the code computed AUC from hard predictions.
- Only claim artifacts the code actually saves.
- Remove stray or garbled entries from the reference list.

## 11. Risks and limitations (state these in the report)

- `loan_grade` and `loan_int_rate` are assigned by the lender after its own risk assessment, so the model partly reproduces the lender's view. Mention this as a limitation.
- `person_age` is a sensitive attribute in real lending. Keep it for the MVP and note that a production system would need a fairness review.
- The dataset is a public sample, not a specific bank's portfolio, so results show the method, not real-world lending performance.
- Class-weighted scores are rankings, not calibrated probabilities.
- Accuracy will look high because most borrowers repay. Use it only alongside Recall and F2.

## 12. Build order

1. Set up the repo, download the data, run Sections A to C and confirm cleaning counts.
2. Add the pipeline and VIF step (D) and EDA figures (B).
3. Add the 7-model bake-off (E) and save the comparison table.
4. Add the imbalance experiment (F).
5. Add tuning, threshold, final evaluation, risk table, importance, and artifact saving (G).
6. Build the Streamlit app, then take screenshots.
7. Write the README, then write the report from the saved outputs.

## 13. Build prompt (paste into your AI coding assistant with this PRD)

> Implement the project described in this PRD exactly as specified. Create the repo layout from Section 7. Write `src/train.py` with sections A to G in order, printing progress, saving every output listed in Section 9 into `outputs/`, and saving the artifact described in Section 5G. Use seed 42 everywhere, fit all transformers inside sklearn pipelines, and compute ROC AUC from probabilities. Then write `app.py` with the two tabs from Section 6, plus `requirements.txt` and `README.md`. Do not add features beyond the PRD. After each section, run it and fix errors before moving on, and show me the printed results.
