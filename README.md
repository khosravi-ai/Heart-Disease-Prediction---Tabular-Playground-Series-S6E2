This notebook is designed for the Tabular Playground Series S6E2 heart disease prediction task. It follows a competition-ready workflow from data understanding to model selection, with clear explanations and high-quality static visualizations. 

- Data loading and quick checks
- Exploratory analysis with professional static visualizations
- Baseline model with cross-validation
- Submission file generation

Evaluation metric: ROC-AUC.


---

## Project Structure

* `train.csv` — Training dataset
* `test.csv` — Test dataset
* `sample_submission.csv` — Submission format reference
* `submission.csv` — Final generated predictions

---

## Exploratory Data Analysis

The notebook builds a dynamic visualization dashboard including:

* Target class distribution
* Density plots for top high-variance numeric features
* Missing value analysis
* Boxplots and violin plots grouped by target
* Scatter plots between top numeric features
* Correlation heatmaps
* Log-transformed distributions
* Histogram grid with mean and median overlays

EDA is adaptive:

* Top numeric features are selected based on variance
* Missing-rate visualization only appears when needed
* Plots scale automatically with dataset structure

This ensures the analysis remains robust even if the dataset changes.

---

## Feature Engineering

A custom feature builder function `build_features()` creates:

### Row-wise statistical features

* Mean
* Standard deviation
* Minimum
* Maximum
* Median
* Sum

### Per-feature transformations

* Log transformation (`log1p`)
* Missing indicator flags

### Quantile binning

* Top 3 high-variance features are discretized using `qcut`
* Bin edges learned on training data and reused on test data

This approach increases feature richness while maintaining reproducibility.

---

## Preprocessing Pipelines

Two preprocessing strategies are used via `ColumnTransformer`.

### Logistic Regression Pipeline

* Numeric features:

  * Median imputation
* Categorical features:

  * Most frequent imputation
  * OneHotEncoding (handle_unknown="ignore")

### HistGradientBoosting Pipeline

* Numeric features:

  * Median imputation
* Categorical features:

  * Most frequent imputation
  * OrdinalEncoding (unknown_value = -1)

`sparse_threshold=0.0` ensures dense output for tree-based model compatibility.

---

## Models

Two models are trained and compared:

* `LogisticRegression`
* `HistGradientBoostingClassifier`

Both are wrapped inside full preprocessing pipelines to prevent data leakage.

---

## Validation Strategy

* Stratified train/validation split (80/20)
* Stratified K-Fold cross-validation (5 folds)
* Evaluation metric: ROC-AUC

Hyperparameter search is performed using `RandomizedSearchCV`.

### Logistic Regression Search Space

* `C`: [0.1, 1.0]

### HistGradientBoosting Search Space

* `learning_rate`: [0.1]
* `max_depth`: [3]

---

## Model Performance

Cross-Validation ROC-AUC:

* Logistic Regression: 0.9521
* HistGradientBoosting: 0.9538

Validation ROC-AUC:

* Logistic Regression: 0.9533
* HistGradientBoosting: 0.9548

Both models perform strongly, with HistGradientBoosting slightly outperforming Logistic Regression.

The final selected model is chosen automatically based on best cross-validation score.

---

## Final Training and Submission

The best model is retrained on the full training dataset and used to generate probability predictions on the test dataset.

Output format:

| id  | Heart Disease |
| --- | ------------- |
| ... | probability   |

The predictions are saved as:

```
submission.csv
```

---

## Key Design Principles

* No data leakage (strict pipeline separation)
* Reusable feature engineering
* Adaptive EDA
* Stratified validation
* Reproducible random states
* Model comparison before final selection

---

## Dependencies

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn

---

## How to Run

1. Update dataset paths.
2. Execute the script or notebook.
3. Review validation scores.
4. Submit the generated `submission.csv`.

---

## Summary

This project demonstrates a production-style structured ML workflow for tabular classification problems:

* Clean pipeline architecture
* Feature enrichment
* Proper validation
* Competitive ROC-AUC performance

It serves as a strong baseline template for structured binary classification tasks such as medical risk prediction, financial scoring, or customer analytics.
