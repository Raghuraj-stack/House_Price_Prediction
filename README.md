# 🏠 House Price Prediction

A regression project that predicts **median house value** for California districts using the housing census dataset. It builds a full scikit-learn preprocessing pipeline, compares linear and tree-based models with cross-validation, and tunes a Gradient Boosting model with grid search.

## 📌 Overview

| Item | Detail |
|---|---|
| Task | Regression (target: `median_house_value`) |
| Rows | 20,640 districts |
| Features | 8 numeric + 1 categorical (`ocean_proximity`) |
| Missing values | 207 in `total_bedrooms` |
| Metrics | RMSE, MAE, R² |

### Features

`longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`, `ocean_proximity`

## 📂 Dataset

The notebook expects `housing.csv` in the project root (the California Housing dataset, commonly available on Kaggle). Download it and place it next to the notebook.

## 🔬 Workflow

1. **EDA**: shape, dtypes, missing values, duplicates (none), summary statistics, histograms, boxplots, category counts and a correlation heatmap.
2. **Split**: 80/20 train/test, `random_state=42`.
3. **Preprocessing** (`ColumnTransformer`):
   - Numeric: median imputation → `StandardScaler`
   - Categorical: most-frequent imputation → `OneHotEncoder(handle_unknown="ignore")`
4. **Baseline**: Linear Regression.
5. **Model comparison** with 5-fold `KFold` CV: Linear Regression, Ridge, Lasso, Random Forest, HistGradientBoosting.
6. **Tuning**: `GridSearchCV` over HistGradientBoosting (288 candidates, 1,440 fits), scored by RMSE.
7. **Final model**: tuned HistGradientBoostingRegressor evaluated on the test set.
8. **Inference helper**: `predict_house_price()` takes raw feature values and returns a predicted price.

## 📊 Results

| Model | Split | RMSE | MAE | R² |
|---|---|---|---|---|
| Linear Regression (baseline) | Train | 68,434 | 49,595 | 0.650 |
| Linear Regression (baseline) | Test | 70,059 | 50,670 | 0.625 |
| **HistGradientBoosting (tuned)** | Train | 36,916 | 25,199 | 0.898 |
| **HistGradientBoosting (tuned)** | Test | **46,487** | **31,005** | **0.835** |

- Ridge and Lasso performed almost identically to plain Linear Regression (CV RMSE ≈ 68.6k), which suggests the linear baseline is limited by model capacity rather than overfitting.
- Tuned gradient boosting cut test RMSE by about **34%** versus the baseline.
- Best CV RMSE during grid search: **47,428**.

**Best hyperparameters**

```python
{
    "l2_regularization": 0.1,
    "learning_rate": 0.1,
    "max_bins": 255,
    "max_depth": 10,
    "max_iter": 200,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
}
```

## 🚀 Getting Started

```bash
git clone https://github.com/Raghuraj-stack/House_Price_Prediction.git
cd House_Price_Prediction
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
# place housing.csv in the project root
jupyter notebook House_Price_Prediction.ipynb
```

### Example prediction

```python
price = predict_house_price(
    model=hgb_best,
    longitude=-122.230,
    latitude=37.880,
    housing_median_age=41,
    total_rooms=880,
    total_bedrooms=129,
    population=322,
    households=126,
    median_income=8.3252,
    ocean_proximity="NEAR BAY",
)
print(round(price, 2))  # ≈ 456,078.69
```

## 🛠️ Tech Stack

Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn

