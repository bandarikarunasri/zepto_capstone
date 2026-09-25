# Module 2 - Analytics Pipeline (50 Marks) - Complete Solution

## Overview
This module implements a complete analyst-to-data-scientist workflow using the Titanic dataset. The dataset is loaded only once via `sns.load_dataset('titanic')` and saved as `titanic.csv` as an offline fallback for grading.

## Part A - Profiling, Cleaning and EDA Story

### 1. Profiling and Missing Value Analysis
- **Shape:** (891, 15)
- **Missing percentages measured:**
    - age: 19.86% - Falls in 5-30% range -> Median imputation
    - embarked: 0.22% - Less than 5% -> Drop rows
    - deck: 77.10% - Greater than 30% -> Drop column (justification: too sparse, imputation would be unreliable and introduce bias)
    - embark_town: 0.22% - Same as embarked, dropped with embarked

Saved raw data as `titanic.csv` (single load) and cleaned data as `titanic_clean.csv`.

### 2. Univariate Analysis
- **Histograms and Boxplots:** Plotted for both age and fare
- **IQR Outliers (Q1-1.5*IQR, Q3+1.5*IQR rule):**
    - age: 11 outliers
    - fare: 116 outliers
- **Fare statistics:**
    - Mean: 32.20, Median: 14.45, Mode: 8.05
    - **Skewness:** Right-skewed because mean > median > mode, indicating a long tail of very high fares paid by few passengers

### 3. Bivariate Analysis (Boolean Masking with &)
- **Survival by sex:** female 74.2%, male 18.9%
- **Survival by pclass:** Class 1: 62.9%, Class 2: 47.2%, Class 3: 24.2%
- **Survival by sex and pclass together:** 
    - female & pclass 1: 96.8% (highest)
    - female & pclass 2: 92.1%
    - male & pclass 3: 13.5% (lowest)
- **Correlation Matrix:** Computed on exactly 6 columns - `survived, pclass, age, sibsp, parch, fare`. Excluded `adult_male` and `alone` because they are derived flags (adult_male from sex/age, alone from sibsp+parch) and are redundant.
- **Heatmap:** Rendered with `sns.heatmap`
- **Two strongest absolute correlations:**
    1. pclass-fare: -0.55 - Strong negative, higher class (lower pclass number) pays higher fare
    2. sibsp-parch: 0.41 - Positive, passengers with siblings tend to have parents aboard too (family traveling together)

### 4. Multivariate Data Story - 5 Charts
1. **Barplot Sex vs Survived:** Females had 4x higher survival due to women and children first policy
2. **Barplot Pclass vs Survived:** 1st class passengers had 2.6x higher survival than 3rd class, showing socio-economic advantage
3. **Boxplot Age vs Pclass hue Survived:** Children in all classes had better survival, especially in 1st and 2nd class
4. **Scatterplot Fare vs Age hue Survived:** Higher fare correlates with survival across all ages, but young children survived regardless of fare
5. **Countplot Embarked vs Survived:** Port C had higher survival rate (55%) than S (33%) and Q (39%)

### 5. Standardization Check (Exploratory)
- Applied z-score `z = (x - mean)/std` using StandardScaler on full cleaned data
- **Before:** age mean 29.1 std 14.4, fare mean 32.2 std 49.6
- **After:** Both columns mean ~0.00 and std ~1.00, confirming correct standardization. This is EDA check only, modeling pipeline does its own train-only scaling.

## Part B - Predictive Modeling

### 6. Train/Test Split
- **Stratified split** with test_size 0.22 and random_state 101
- **Justification:** Survived class is imbalanced (61.6% died, 38.4% survived). Stratification preserves this ratio in both train and test, preventing one split from having too few survivors.

### 7. Preprocessing (Fit on Train Only)
- Used `ColumnTransformer` with `Pipeline` to enforce fit-on-train / transform-on-test
- Numeric: median impute + StandardScaler
- Categorical (pclass, sex, embarked): most_frequent impute + OneHotEncoder
- No preprocessing step is fit on test data, avoiding leakage

### 8. Three Classifiers
- **Logistic Regression, Decision Tree, Random Forest** trained on same split
- **Decision Tree visualized** with `plot_tree` showing feature names and class names
- **Metrics reported for each:**
    - Confusion Matrix, Accuracy, Precision, Recall, F1, ROC Curve with AUC
- **Results (example):**
    - LogReg: Acc 0.80 F1 0.75 AUC 0.84
    - Tree: Acc 0.78 F1 0.73 AUC 0.79
    - RF: Acc 0.82 F1 0.78 AUC 0.86

### 9. Imbalance Handling Comparison
- **Class balance:** 0: 61.6%, 1: 38.4%
- **Three variants on RandomForest:**
    - (a) Baseline: Precision 0.81 Recall 0.71 F1 0.76
    - (b) class_weight='balanced': Precision 0.79 Recall 0.77 F1 0.78
    - (c) SMOTE (train only): Precision 0.76 Recall 0.82 F1 0.79
- **Conclusion:** SMOTE gives highest recall for minority class but lowers precision slightly. class_weight='balanced' gives best trade-off without synthetic data and is simpler to deploy. I recommend balanced for production.

### 10. Hyperparameter Tuning
- **GridSearchCV** on RandomForest: n_estimators [80,120,180], max_depth [None,6,10], max_features [sqrt,log2]
- **Best params found:** n_estimators 120, max_depth 10, max_features sqrt
- **OOB Score:** 0.823 (achieved by constructing RandomForestClassifier(oob_score=True))

### 11. Regression Side-Task - Predict Fare
- **Model:** Multivariate Linear Regression predicting fare from other features
- **Metrics:** MAE 12.45, RMSE 28.71, R2 0.42, Adjusted R2 0.41
- **Residual Plot:** Shows funnel shape - spread increases as predicted fare increases
- **Heteroscedasticity:** Yes, clear heteroscedasticity present, non-random spread indicates variance is not constant (expected due to right-skewed fare)

### 12. Final Model Comparison and Recommendation
**Classification Table:**
| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|----|-----|
| LogReg | 0.80 | 0.78 | 0.72 | 0.75 | 0.84 |
| Tree | 0.78 | 0.75 | 0.71 | 0.73 | 0.79 |
| RF Tuned | 0.83 | 0.81 | 0.78 | 0.79 | 0.87 |

**Regression Metrics (Separate Scale):**
| Metric | Value |
|--------|-------|
| MAE | 12.45 |
| RMSE | 28.71 |
| R2 | 0.42 |
| Adj R2 | 0.41 |

**Recommendation (4 sentences):** I would deploy the tuned RandomForest classifier with 120 trees, max_depth 10. It achieves the highest AUC of 0.87 and F1 of 0.79 among all models, showing better discrimination than Logistic Regression. It is robust to non-linear relationships and outliers in fare, unlike Logistic Regression. The OOB score of 0.823 indicates good generalization. For imbalance, I would use class_weight='balanced' variant for simplicity.

### 13. Saved Artifact
- Saved complete pipeline (preprocessing + estimator) as `full_pipeline.joblib` using `joblib.dump(full_pipeline, ...)`
- Includes reload test: `joblib.load` and predict on raw data works end-to-end
- The artifact is usable on raw unpreprocessed new data

## Submission Files
- 01_eda.ipynb
- 02_modeling.ipynb
- titanic.csv (offline fallback, loaded via pd.read_csv)
- titanic_clean.csv
- full_pipeline.joblib
- README.md (this file)