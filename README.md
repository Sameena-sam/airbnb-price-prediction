# Airbnb Price Prediction (New York City)

Predicts the nightly price of New York City Airbnb listings and compares seven regression model configurations. Built for the Applied Machine Learning module (CT046-3-M) of the MSc in Artificial Intelligence at Asia Pacific University (2026).

## Results

The best model was tuned XGBoost (learning rate 0.05, max depth 8, 200 trees, subsample 0.8):

| Metric | Value |
|---|---|
| Test MAE | $38.23 |
| Test RMSE | $58.12 |
| Test R² | 0.552 |
| Test MAPE | 29.6% |
| 5-fold cross-validated R² | 0.567 ± 0.006 |

Room type was the strongest predictor in both tree ensembles, and XGBoost trained more than 11 times faster than Random Forest.

## Approach

1. **Data:** AB_NYC_2019 from Kaggle, 48,895 listings with 16 features.
2. **Preprocessing:** missing-value imputation, outlier handling and a log-transformed target for the skewed prices.
3. **Exploration:** price distribution, room types, neighbourhoods, geographic spread and correlations.
4. **Feature engineering:** new predictors derived from the existing columns.
5. **Models:** Linear Regression, Random Forest, Gradient Boosting, XGBoost and a Voting ensemble, tuned with grid search and cross-validation.
6. **Evaluation:** MAE, RMSE, R² and MAPE on a held-out test set.

## Project structure

```
airbnb-price-prediction/
├── notebooks/
│   └── airbnb_price_prediction.ipynb   # full pipeline with outputs
├── data/                               # put AB_NYC_2019.csv here (not included)
├── requirements.txt
└── README.md
```

## Run

```bash
pip install -r requirements.txt
# download AB_NYC_2019.csv from Kaggle ("New York City Airbnb Open Data") into data/
cd notebooks
jupyter notebook airbnb_price_prediction.ipynb
```

## Tech

Python, pandas, scikit-learn, XGBoost, Matplotlib, seaborn.

## Author

Sameena Siddiqua
