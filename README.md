# House Prices - Advanced Regression Techniques

Kaggle competition solution for predicting house prices in Ames, Iowa using advanced regression techniques.

## Results

| Model | CV RMSE (log scale) |
|-------|---------------------|
| Ridge Baseline | 0.13496 |
| Ridge + Features | 0.13409 |
| LightGBM | 0.12916 |
| CatBoost | 0.12484 |
| **Ensemble** | **0.12412** |

## Approach

A disciplined, leakage-safe solution following 7 phases:

1. **Target Handling** - Log1p transform on SalePrice, 5-fold CV (shuffle=True, random_state=42)
2. **Preprocessing** - Median imputation for numeric, "None" for categorical, skew correction (|skew| > 0.75)
3. **Ordinal Encoding** - Quality features mapped: None < Po < Fa < TA < Gd < Ex
4. **Feature Engineering** - 4 proven features:
   - `TotalSF` = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
   - `Bathrooms` = FullBath + 0.5×HalfBath + BsmtFullBath + 0.5×BsmtHalfBath
   - `QualxArea` = OverallQual × GrLivArea
   - `GarageScore` = GarageArea × GarageCars
5. **Tree Models** - LightGBM and CatBoost with early stopping
6. **Ensemble** - Simple 50/50 average
7. **Submission** - Convert predictions back with expm1

## Usage

```bash
python3 solution.py
```

Generates `submission.csv` ready for Kaggle upload.

## Project Structure

```
├── data/
│   ├── train.csv          # Training data (1460 houses, 79 features)
│   ├── test.csv           # Test data (1459 houses)
│   └── sample_submission.csv
├── doc/
│   └── overview.md        # Competition description
├── solution.py            # Main solution script
├── submission.csv         # Generated predictions
└── README.md
```

## Requirements

- numpy
- pandas
- scikit-learn
- lightgbm
- catboost
- scipy
