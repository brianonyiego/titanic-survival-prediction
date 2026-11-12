
# Titanic Survival Prediction

Predicting passenger survival on the RMS Titanic using machine learning for the Kaggle [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) competition.

This project explores passenger survival patterns, engineers predictive features, compares 16 classification models, tunes Random Forest and Gradient Boosting, and evaluates model performance using cross-validation, ROC-AUC, confusion matrices, learning curves, and feature importance analysis.

## Highlights

- **Best model:** XGBoost, achieving **83.61% mean cross-validation accuracy** across 5-fold cross-validation repeated 3 times.
- **Best ROC-AUC:** 0.8828 for XGBoost using out-of-fold predictions.
- **Models compared:** 16 classification models, including boosting algorithms, neural-inspired ensembles, and traditional machine learning methods.
- **Hyperparameter tuning:** RandomizedSearchCV with 30 iterations for Random Forest and Gradient Boosting.
- **Feature engineering:** Passenger titles, family size, ticket group size, cabin information, and fare transformations.
- **Final submission:** Generated predictions for all 418 passengers in the Kaggle test dataset.
- **Key finding:** Sex, passenger title, fare, age, and passenger class are among the strongest survival predictors.

## Dataset

The project uses the official Titanic competition dataset:

| Dataset | Observations | Features |
|---|---:|---:|
| Training set | 891 | 19 |
| Test set | 418 | 18 |

The training dataset contains passenger information such as:

- PassengerId
- Name
- Sex
- Age
- Pclass
- SibSp
- Parch
- Ticket
- Fare
- Cabin
- Embarked

The overall survival rate in the training dataset is **38.4%**.

### Missing Values

| Feature | Missing values |
|---|---:|
| Age | 177 |
| Cabin | 687 |
| Embarked | 2 |

The original competition data is not included in this repository. Download `train.csv`, `test.csv`, and `gender_submission.csv` from the [Kaggle competition page](https://www.kaggle.com/competitions/titanic).

## Approach

### 1. Exploratory Data Analysis

The exploratory analysis investigates survival patterns across passenger demographics, ticket classes, family groups, fares, and cabin information.

#### Survival by Sex

| Sex | Total | Died | Survived | Survival Rate |
|---|---:|---:|---:|---:|
| Female | 314 | 81 | 233 | 74.2% |
| Male | 577 | 468 | 109 | 18.9% |

Female passengers had a substantially higher survival rate than male passengers.

#### Survival by Passenger Class

| Class | Total | Died | Survived | Survival Rate |
|---|---:|---:|---:|---:|
| First | 216 | 80 | 136 | 63.0% |
| Second | 184 | 97 | 87 | 47.3% |
| Third | 491 | 372 | 119 | 24.2% |

Survival probability decreased considerably from first to third class.

#### Survival by Title

Titles were extracted from passenger names and grouped into five categories.

| Title | Passengers | Survival Rate |
|---|---:|---:|
| Master | 40 | 57.5% |
| Miss | 185 | 70.3% |
| Mr | 517 | 15.7% |
| Mrs | 126 | 79.4% |
| Rare | 23 | 34.8% |

Titles provide useful demographic information beyond the original sex and age variables.

#### Survival by Family Size

| Family Size | Passengers | Survival Rate |
|---|---:|---:|
| 1 | 537 | 30.4% |
| 2 | 161 | 55.3% |
| 3 | 102 | 57.8% |
| 4 | 29 | 72.4% |
| 5 | 15 | 20.0% |
| 6 | 22 | 13.6% |
| 7 | 12 | 33.3% |
| 8 | 6 | 0.0% |
| 11 | 7 | 0.0% |

Small family groups generally had better survival outcomes than passengers travelling alone or in very large groups.

#### Survival by Sex and Passenger Class

| Sex | First Class | Second Class | Third Class |
|---|---:|---:|---:|
| Female | 96.8% | 92.1% | 50.0% |
| Male | 36.9% | 15.7% | 13.5% |

The combination of sex and passenger class reveals substantial differences in survival outcomes.

#### Other Exploratory Findings

- **Age:** Passengers aged 0–5 had a survival rate of 70.5%, compared with 33.3% among passengers aged 18–25.
- **Fare:** The highest fare bracket had a survival rate of 67.9%, compared with approximately 18–21% in the three cheapest brackets.
- **Embarkation:** Passengers embarking at Cherbourg had a survival rate of 55.4%, compared with 33.7% at Southampton.
- **Cabin:** Passengers with recorded cabin information generally had better survival outcomes than passengers without recorded cabin information.

### 2. Feature Engineering

Several additional features were created to improve predictive performance.

| Feature | Description |
|---|---|
| `Title` | Extracted from passenger names and grouped into Mr, Mrs, Miss, Master, and Rare |
| `FamilySize` | Calculated as `SibSp + Parch + 1` |
| `IsAlone` | Binary indicator for passengers travelling alone |
| `TicketGroupSize` | Number of passengers sharing a ticket |
| `HasCabin` | Indicates whether cabin information is available |
| `Deck` | Extracted from the first letter of the cabin identifier |
| `LogFare` | Logarithmic transformation of fare using `log(1 + Fare)` |

Missing values were handled through preprocessing pipelines:

- Numerical features: median imputation.
- Categorical features: most frequent value imputation.
- Categorical encoding: one-hot encoding where appropriate.

Using pipelines ensures that preprocessing is fitted within the cross-validation training folds, reducing information leakage.

### 3. Model Development

A total of 16 classification models were evaluated.

**Traditional Machine Learning**

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (RBF)
- Naive Bayes
- Decision Tree

**Ensemble and Tree-Based Models**

- Random Forest
- Extra Trees
- AdaBoost
- Gradient Boosting
- Hist Gradient Boosting
- XGBoost
- LightGBM

**Ensemble Combinations**

- Soft Voting
- Stacking

**Tuned Models**

- Random Forest (tuned)
- Gradient Boosting (tuned)

### 4. Cross-Validation and Hyperparameter Tuning

Model comparison used stratified 5-fold cross-validation repeated 3 times.

Hyperparameter tuning was performed using `RandomizedSearchCV`, with 30 iterations for each of the two selected algorithms.

#### Random Forest (Tuned)

| Parameter | Selected Value |
|---|---|
| n_estimators | 800 |
| min_samples_split | 10 |
| min_samples_leaf | 1 |
| max_features | sqrt |
| max_depth | 8 |
| Best search CV accuracy | 0.8406 |

#### Gradient Boosting (Tuned)

| Parameter | Selected Value |
|---|---|
| subsample | 0.8 |
| n_estimators | 500 |
| min_samples_leaf | 5 |
| max_depth | 3 |
| learning_rate | 0.01 |
| Best search CV accuracy | 0.8496 |

The tuned Gradient Boosting model achieved the highest hyperparameter-search accuracy of 84.96%.

The tuned model's final repeated cross-validation accuracy was 83.43%. These figures represent different evaluation results.

## Results

### 1. Cross-Validation Results

The final five-fold evaluation produced the following fold-level results:

| Fold | Accuracy |
|---|---:|
| Fold 1 | 84.36% |
| Fold 2 | 80.90% |
| Fold 3 | 83.15% |
| Fold 4 | 81.46% |
| Fold 5 | 85.39% |
| **Mean** | **83.05%** |
| **Standard Deviation** | **1.69%** |

The model demonstrated relatively consistent performance across the five validation folds.

### 2. Model Comparison

The following table reports the results from repeated stratified 5-fold cross-validation (5 folds × 3 repeats).

| Rank | Model | Mean Accuracy | Std |
|---:|---|---:|---:|
| 1 | XGBoost | **0.8361** | 0.0202 |
| 2 | Gradient Boosting (tuned) | 0.8343 | 0.0220 |
| 3 | Gradient Boosting | 0.8335 | 0.0157 |
| 4 | Voting (soft) | 0.8328 | 0.0188 |
| 5 | Stacking | 0.8320 | 0.0184 |
| 6 | LightGBM | 0.8298 | 0.0193 |
| 7 | Random Forest | 0.8290 | 0.0192 |
| 8 | Random Forest (tuned) | 0.8283 | 0.0218 |
| 9 | SVM (RBF) | 0.8268 | 0.0193 |
| 10 | AdaBoost | 0.8268 | 0.0203 |
| 11 | Hist Gradient Boosting | 0.8257 | 0.0156 |
| 12 | Logistic Regression | 0.8241 | 0.0208 |
| 13 | Extra Trees | 0.8215 | 0.0204 |
| 14 | Decision Tree | 0.8096 | 0.0304 |
| 15 | KNN (k=15) | 0.8069 | 0.0248 |
| 16 | Naive Bayes | 0.7714 | 0.0275 |

XGBoost achieved the highest mean accuracy of **83.61%**, followed closely by tuned Gradient Boosting at 83.43%.

The difference between the first and second-ranked models was only 0.18 percentage points. The relatively small differences among the leading models, compared with their cross-validation variability, suggest that their predictive performances are broadly similar.

Tuned-model scores should be interpreted cautiously because hyperparameters were selected using the same dataset.

### 3. ROC-AUC Analysis

Out-of-fold ROC-AUC was calculated for the five leading models.

| Model | ROC-AUC |
|---|---:|
| XGBoost | **0.8828** |
| Gradient Boosting | 0.8791 |
| Gradient Boosting (tuned) | 0.8787 |
| Voting (soft) | 0.8773 |
| Stacking | 0.8757 |

XGBoost achieved the highest ROC-AUC of **0.8828**, indicating strong discrimination between surviving and deceased passengers.

#### XGBoost: True Positive Rate at Different False Positive Rates

| False Positive Rate | True Positive Rate |
|---|---:|
| 0.05 | 0.664 |
| 0.10 | 0.731 |
| 0.20 | 0.836 |
| 0.30 | 0.880 |
| 0.40 | 0.895 |
| 0.50 | 0.904 |

### 4. Confusion Matrix

The out-of-fold XGBoost predictions produced the following confusion matrix:

| Actual / Predicted | Died | Survived |
|---|---:|---:|
| Actual Died | 491 | 58 |
| Actual Survived | 84 | 258 |

The model correctly classified 491 deceased passengers and 258 survivors.

It incorrectly classified 58 deceased passengers as survivors and 84 survivors as deceased.

#### Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Died | 0.854 | 0.894 | 0.874 | 549 |
| Survived | 0.816 | 0.754 | 0.784 | 342 |
| **Macro Average** | **0.835** | **0.824** | **0.829** | **891** |
| **Weighted Average** | **0.840** | **0.841** | **0.839** | **891** |

**Overall accuracy: 84.1%**

The model performed better at identifying deceased passengers than survivors. Its recall for deceased passengers was 89.4%, compared with 75.4% for survivors.

### 5. Learning Curve

The XGBoost learning curve provides insight into how model performance changes as the training dataset increases.

| Training Samples | Training Accuracy | Validation Accuracy |
|---:|---:|---:|
| 71 | 0.9746 | 0.7665 |
| 162 | 0.9469 | 0.8013 |
| 254 | 0.9394 | 0.8204 |
| 345 | 0.9304 | 0.8283 |
| 437 | 0.9208 | 0.8428 |
| 528 | 0.9095 | 0.8440 |
| 620 | 0.9068 | 0.8451 |
| 712 | 0.9048 | 0.8372 |

Validation accuracy improved as the training sample increased, eventually reaching approximately 84–85%.

The remaining difference between training and validation accuracy suggests some overfitting, although the gap narrowed considerably as more training observations were introduced.

### 6. Feature Importance

Feature importance was extracted from the tuned Random Forest model.

#### Top 15 Features

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | Sex_male | 0.1432 |
| 2 | Title_Mr | 0.1428 |
| 3 | Sex_female | 0.1260 |
| 4 | LogFare | 0.0882 |
| 5 | Age | 0.0623 |
| 6 | TicketGroupSize | 0.0506 |
| 7 | Pclass_3 | 0.0490 |
| 8 | FamilySize | 0.0431 |
| 9 | Title_Miss | 0.0383 |
| 10 | Title_Mrs | 0.0357 |
| 11 | HasCabin | 0.0299 |
| 12 | Deck_U | 0.0297 |
| 13 | SibSp | 0.0256 |
| 14 | Pclass_1 | 0.0222 |
| 15 | Pclass_2 | 0.0163 |

Sex-related features and passenger titles were the most influential predictors.

Fare, age, ticket group size, and passenger class also contributed meaningfully to model predictions.

These importance values describe the fitted Random Forest model and should not be interpreted as causal effects.

### 7. Correlation Analysis

The strongest correlations with survival were:

| Feature | Correlation with Survival |
|---|---:|
| Female | 0.543 |
| LogFare | 0.330 |
| HasCabin | 0.317 |
| Parch | 0.082 |
| TicketGroupSize | 0.065 |
| FamilySize | 0.017 |
| SibSp | -0.035 |
| Age | -0.077 |
| IsAlone | -0.203 |
| Pclass | -0.338 |

Sex had the strongest positive correlation with survival, while passenger class had a negative correlation because larger class values represent lower passenger classes.

Family size had a near-zero linear correlation with survival despite the noticeable non-linear survival patterns across family groups. This demonstrates why feature relationships should not be assessed using correlation alone.

### 8. Agreement Between Model Predictions

The models generally produced similar predictions on the test dataset.

Several of the leading ensemble and boosting models had prediction agreement levels above 90%.

Examples include:

- Voting and SVM: 98%
- Random Forest and Stacking: 98%
- Gradient Boosting and tuned Gradient Boosting: 98%
- XGBoost and LightGBM: 96%

This suggests that many algorithms identified similar underlying passenger survival patterns despite using different modelling approaches.

### 9. Predicted Survival Rates on the Test Set

The models produced the following predicted survival proportions for the 418 test passengers.

| Model | Predicted Survival |
|---|---:|
| Naive Bayes | 51.7% |
| Logistic Regression | 40.7% |
| AdaBoost | 40.7% |
| Decision Tree | 40.0% |
| Stacking | 39.5% |
| Voting (soft) | 39.2% |
| SVM (RBF) | 38.8% |
| Random Forest | 38.5% |
| KNN (k=15) | 38.0% |
| Gradient Boosting | 37.8% |
| Gradient Boosting (tuned) | 37.6% |
| Extra Trees | 37.1% |
| XGBoost | 36.8% |
| Random Forest (tuned) | 36.4% |
| LightGBM | 36.1% |
| Hist Gradient Boosting | 35.6% |

XGBoost predicted that 36.8% of the test passengers survived, relatively close to the 38.4% survival rate in the training dataset.

## Final Kaggle Submission

XGBoost was selected as the final submission model based on its cross-validation accuracy and ROC-AUC performance.

### Prediction Summary

| Metric | Result |
|---|---:|
| Test observations | 418 |
| Predicted survivors | 154 |
| Predicted deceased | 264 |
| Predicted survival rate | 36.8% |
| Predicted death rate | 63.2% |
| Agreement with gender_submission baseline | 88.0% |

The generated submission has the expected Kaggle format:

- `PassengerId`
- `Survived`

The first ten predictions were:

| PassengerId | Survived |
|---:|---:|
| 892 | 0 |
| 893 | 0 |
| 894 | 0 |
| 895 | 0 |
| 896 | 0 |
| 897 | 0 |
| 898 | 1 |
| 899 | 0 |
| 900 | 1 |
| 901 | 0 |

The final submission was saved to:

`/kaggle/working/submission.csv`

**Kaggle public leaderboard score:** `0.XXXXX` (update after submitting the generated file).

Note: Agreement with `gender_submission.csv` measures prediction similarity, not Kaggle accuracy.

## Key Insights

1. **Sex was the strongest survival indicator.** Female passengers had a 74.2% survival rate, compared with 18.9% for male passengers.

2. **Passenger class mattered considerably.** First-class passengers had a 63.0% survival rate, compared with 24.2% for third-class passengers.

3. **Titles provided useful demographic information.** Mrs and Miss had substantially higher survival rates than Mr.

4. **Family size had a non-linear relationship with survival.** Small family groups generally performed better than passengers travelling alone or in large groups.

5. **Fare was an important predictor.** Passengers in the highest fare bracket had a survival rate of 67.9%.

6. **Age contributed to survival prediction.** Young children had notably higher survival rates.

7. **Different algorithms produced broadly comparable results.** The top five models achieved cross-validation accuracies between 83.20% and 83.61%.

8. **XGBoost provided the strongest overall result.** It achieved the highest repeated cross-validation accuracy and ROC-AUC among the evaluated models.

## Repository Structure

```text
.
├── titanic.ipynb        # Full analysis, feature engineering and modelling
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── data/
    ├── train.csv        # Download from Kaggle
    ├── test.csv         # Download from Kaggle
    └── gender_submission.csv
```

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/titanic-survival-prediction.git
cd titanic-survival-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

Download the competition files from:

https://www.kaggle.com/competitions/titanic

Place the downloaded CSV files inside the `data/` directory.

### 4. Run the Notebook

Open `titanic.ipynb` using Jupyter Notebook or JupyterLab.

Update the `DATA_DIR` variable to point to the local dataset directory.

The notebook was originally executed in Kaggle, where the dataset path was:

`/kaggle/input/competitions/titanic`

The final submission is generated at:

`/kaggle/working/submission.csv`

## Requirements

The project uses the following Python libraries:

```text
pandas
numpy
scikit-learn
xgboost
lightgbm
matplotlib
seaborn
jupyter
```

## Possible Improvements

Several extensions could potentially improve predictive performance:

- **Family survival features:** Construct family-level survival indicators within each cross-validation fold to prevent information leakage.
- **Improved age imputation:** Estimate missing ages using passenger title, sex, and passenger class.
- **Probability blending:** Combine XGBoost, tuned Gradient Boosting, and Logistic Regression predictions.
- **Threshold optimization:** Investigate classification thresholds using out-of-fold predictions rather than relying exclusively on the default 0.5 threshold.
- **Feature selection:** Evaluate whether removing redundant features improves generalization.
- **Nested cross-validation:** Use nested validation to obtain less optimistic estimates when hyperparameter tuning is involved.

## Acknowledgements

- Dataset and competition: [Kaggle Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)
- Machine learning framework: [Scikit-learn](https://scikit-learn.org/)
- Gradient boosting: [XGBoost](https://xgboost.readthedocs.io/) and [LightGBM](https://lightgbm.readthedocs.io/)
