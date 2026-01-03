"""
House Prices - Advanced Regression Techniques
Disciplined, leakage-safe solution using:
- Log-target regression
- Ordinal encoding
- Limited feature engineering
- LightGBM + CatBoost ensemble
"""

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PHASE 1: Setup & Target Handling
# =============================================================================
print("=" * 60)
print("PHASE 1: Setup & Target Handling")
print("=" * 60)

# Load data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Define log target
y = np.log1p(train['SalePrice'])
train_id = train['Id']
test_id = test['Id']

# Drop target and Id from features
train = train.drop(['SalePrice', 'Id'], axis=1)
test = test.drop(['Id'], axis=1)

# Combine for consistent preprocessing
ntrain = train.shape[0]
all_data = pd.concat([train, test], axis=0, ignore_index=True)

# Setup CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print(f"Target (log1p): mean={y.mean():.4f}, std={y.std():.4f}")

# =============================================================================
# PHASE 2: Preprocessing
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 2: Preprocessing")
print("=" * 60)

# Define ordinal mappings
ordinal_map = {
    'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5
}

ordinal_cols = [
    'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
    'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC'
]

# Additional ordinal columns with custom mappings
bsmt_exposure_map = {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
bsmt_fin_map = {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
garage_finish_map = {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
fence_map = {'None': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}
functional_map = {'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8}
lot_shape_map = {'IR3': 1, 'IR2': 2, 'IR1': 3, 'Reg': 4}
land_slope_map = {'Sev': 1, 'Mod': 2, 'Gtl': 3}
paved_drive_map = {'N': 0, 'P': 1, 'Y': 2}
central_air_map = {'N': 0, 'Y': 1}

# Identify numeric and categorical columns
numeric_cols = all_data.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = all_data.select_dtypes(include=['object']).columns.tolist()

print(f"Numeric columns: {len(numeric_cols)}")
print(f"Categorical columns: {len(categorical_cols)}")

# Fill missing values
# Numeric: median
for col in numeric_cols:
    all_data[col] = all_data[col].fillna(all_data[col].median())

# Categorical: "None"
for col in categorical_cols:
    all_data[col] = all_data[col].fillna('None')

# Apply ordinal encoding
for col in ordinal_cols:
    if col in all_data.columns:
        all_data[col] = all_data[col].map(ordinal_map).fillna(0).astype(int)

# Additional ordinal encodings
all_data['BsmtExposure'] = all_data['BsmtExposure'].map(bsmt_exposure_map).fillna(0).astype(int)
all_data['BsmtFinType1'] = all_data['BsmtFinType1'].map(bsmt_fin_map).fillna(0).astype(int)
all_data['BsmtFinType2'] = all_data['BsmtFinType2'].map(bsmt_fin_map).fillna(0).astype(int)
all_data['GarageFinish'] = all_data['GarageFinish'].map(garage_finish_map).fillna(0).astype(int)
all_data['Fence'] = all_data['Fence'].map(fence_map).fillna(0).astype(int)
all_data['Functional'] = all_data['Functional'].map(functional_map).fillna(8).astype(int)
all_data['LotShape'] = all_data['LotShape'].map(lot_shape_map).fillna(4).astype(int)
all_data['LandSlope'] = all_data['LandSlope'].map(land_slope_map).fillna(3).astype(int)
all_data['PavedDrive'] = all_data['PavedDrive'].map(paved_drive_map).fillna(2).astype(int)
all_data['CentralAir'] = all_data['CentralAir'].map(central_air_map).fillna(1).astype(int)

# Update column lists after ordinal encoding
encoded_ordinals = ordinal_cols + ['BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                                   'GarageFinish', 'Fence', 'Functional', 'LotShape',
                                   'LandSlope', 'PavedDrive', 'CentralAir']

# Remaining categorical columns for one-hot encoding
remaining_cat_cols = [col for col in categorical_cols if col not in encoded_ordinals]

# Skew correction for numeric features
numeric_feats = all_data.select_dtypes(include=[np.number]).columns.tolist()
skewed_feats = []
for col in numeric_feats:
    skewness = skew(all_data[col].dropna())
    if abs(skewness) > 0.75:
        skewed_feats.append(col)
        all_data[col] = np.log1p(all_data[col].clip(lower=0))

print(f"Skew-corrected features: {len(skewed_feats)}")

# One-hot encode remaining categoricals
all_data = pd.get_dummies(all_data, columns=remaining_cat_cols, drop_first=True)

print(f"Total features after preprocessing: {all_data.shape[1]}")

# Split back to train/test
X_train = all_data[:ntrain].copy()
X_test = all_data[ntrain:].copy()

# =============================================================================
# PHASE 3: Baseline Ridge Model
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 3: Baseline Ridge Model")
print("=" * 60)

# Standardize for Ridge
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Tune Ridge alpha
alphas = [0.1, 1, 10, 100, 1000]
best_alpha = None
best_score = float('inf')

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    scores = cross_val_score(ridge, X_train_scaled, y, cv=kf, scoring='neg_root_mean_squared_error')
    rmse = -scores.mean()
    if rmse < best_score:
        best_score = rmse
        best_alpha = alpha

print(f"Best Ridge alpha: {best_alpha}")
print(f"Ridge CV RMSE (baseline): {best_score:.5f}")

ridge_baseline = best_score

# =============================================================================
# PHASE 4: Feature Engineering
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 4: Feature Engineering")
print("=" * 60)

# Reload for feature engineering (before one-hot)
train_fe = pd.read_csv('data/train.csv')
test_fe = pd.read_csv('data/test.csv')

y_fe = np.log1p(train_fe['SalePrice'])
train_fe = train_fe.drop(['SalePrice', 'Id'], axis=1)
test_fe = test_fe.drop(['Id'], axis=1)

all_data_fe = pd.concat([train_fe, test_fe], axis=0, ignore_index=True)

# Fill missing for feature engineering columns
for col in ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'FullBath', 'HalfBath',
            'BsmtFullBath', 'BsmtHalfBath', 'OverallQual', 'GrLivArea',
            'GarageArea', 'GarageCars']:
    all_data_fe[col] = all_data_fe[col].fillna(0)

# Add engineered features
all_data_fe['TotalSF'] = all_data_fe['TotalBsmtSF'] + all_data_fe['1stFlrSF'] + all_data_fe['2ndFlrSF']
all_data_fe['Bathrooms'] = (all_data_fe['FullBath'] + 0.5 * all_data_fe['HalfBath'] +
                            all_data_fe['BsmtFullBath'] + 0.5 * all_data_fe['BsmtHalfBath'])
all_data_fe['QualxArea'] = all_data_fe['OverallQual'] * all_data_fe['GrLivArea']
all_data_fe['GarageScore'] = all_data_fe['GarageArea'] * all_data_fe['GarageCars']

print("Added features: TotalSF, Bathrooms, QualxArea, GarageScore")

# Re-apply full preprocessing with new features
numeric_cols_fe = all_data_fe.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols_fe = all_data_fe.select_dtypes(include=['object']).columns.tolist()

# Fill missing
for col in numeric_cols_fe:
    all_data_fe[col] = all_data_fe[col].fillna(all_data_fe[col].median())
for col in categorical_cols_fe:
    all_data_fe[col] = all_data_fe[col].fillna('None')

# Apply ordinal encoding
for col in ordinal_cols:
    if col in all_data_fe.columns:
        all_data_fe[col] = all_data_fe[col].map(ordinal_map).fillna(0).astype(int)

all_data_fe['BsmtExposure'] = all_data_fe['BsmtExposure'].map(bsmt_exposure_map).fillna(0).astype(int)
all_data_fe['BsmtFinType1'] = all_data_fe['BsmtFinType1'].map(bsmt_fin_map).fillna(0).astype(int)
all_data_fe['BsmtFinType2'] = all_data_fe['BsmtFinType2'].map(bsmt_fin_map).fillna(0).astype(int)
all_data_fe['GarageFinish'] = all_data_fe['GarageFinish'].map(garage_finish_map).fillna(0).astype(int)
all_data_fe['Fence'] = all_data_fe['Fence'].map(fence_map).fillna(0).astype(int)
all_data_fe['Functional'] = all_data_fe['Functional'].map(functional_map).fillna(8).astype(int)
all_data_fe['LotShape'] = all_data_fe['LotShape'].map(lot_shape_map).fillna(4).astype(int)
all_data_fe['LandSlope'] = all_data_fe['LandSlope'].map(land_slope_map).fillna(3).astype(int)
all_data_fe['PavedDrive'] = all_data_fe['PavedDrive'].map(paved_drive_map).fillna(2).astype(int)
all_data_fe['CentralAir'] = all_data_fe['CentralAir'].map(central_air_map).fillna(1).astype(int)

# Skew correction
numeric_feats_fe = all_data_fe.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_feats_fe:
    skewness = skew(all_data_fe[col].dropna())
    if abs(skewness) > 0.75:
        all_data_fe[col] = np.log1p(all_data_fe[col].clip(lower=0))

# One-hot encode
remaining_cat_cols_fe = [col for col in categorical_cols_fe if col not in encoded_ordinals]
all_data_fe = pd.get_dummies(all_data_fe, columns=remaining_cat_cols_fe, drop_first=True)

# Split
X_train_fe = all_data_fe[:ntrain].copy()
X_test_fe = all_data_fe[ntrain:].copy()

# Ridge with features
scaler_fe = StandardScaler()
X_train_fe_scaled = scaler_fe.fit_transform(X_train_fe)
X_test_fe_scaled = scaler_fe.transform(X_test_fe)

ridge_fe = Ridge(alpha=best_alpha)
scores_fe = cross_val_score(ridge_fe, X_train_fe_scaled, y, cv=kf, scoring='neg_root_mean_squared_error')
ridge_fe_rmse = -scores_fe.mean()

print(f"Ridge CV RMSE (with features): {ridge_fe_rmse:.5f}")
print(f"Delta from baseline: {ridge_baseline - ridge_fe_rmse:.5f}")

# =============================================================================
# PHASE 5: Tree Models
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 5: Tree Models")
print("=" * 60)

# Prepare data for LightGBM (uses one-hot encoded data)
X_lgb = X_train_fe.values
y_lgb = y.values

# LightGBM with early stopping via CV
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'n_estimators': 2000,
    'early_stopping_rounds': 50
}

# Manual CV for LightGBM with early stopping
lgb_oof = np.zeros(len(X_train_fe))
lgb_test_preds = np.zeros(len(X_test_fe))
lgb_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_lgb)):
    X_tr, X_val = X_lgb[train_idx], X_lgb[val_idx]
    y_tr, y_val = y_lgb[train_idx], y_lgb[val_idx]

    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        lgb_params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )

    lgb_oof[val_idx] = model.predict(X_val)
    lgb_test_preds += model.predict(X_test_fe.values) / 5

    fold_rmse = np.sqrt(np.mean((lgb_oof[val_idx] - y_val) ** 2))
    lgb_scores.append(fold_rmse)

lgb_cv_rmse = np.sqrt(np.mean((lgb_oof - y_lgb) ** 2))
print(f"LightGBM CV RMSE: {lgb_cv_rmse:.5f}")

# CatBoost - uses raw categoricals
# Reload data for CatBoost
train_cat = pd.read_csv('data/train.csv')
test_cat = pd.read_csv('data/test.csv')

y_cat = np.log1p(train_cat['SalePrice'])
train_cat = train_cat.drop(['SalePrice', 'Id'], axis=1)
test_cat = test_cat.drop(['Id'], axis=1)

all_data_cat = pd.concat([train_cat, test_cat], axis=0, ignore_index=True)

# Fill missing
numeric_cols_cat = all_data_cat.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols_cat = all_data_cat.select_dtypes(include=['object']).columns.tolist()

for col in numeric_cols_cat:
    all_data_cat[col] = all_data_cat[col].fillna(all_data_cat[col].median())
for col in categorical_cols_cat:
    all_data_cat[col] = all_data_cat[col].fillna('None')

# Add engineered features for CatBoost
all_data_cat['TotalSF'] = all_data_cat['TotalBsmtSF'] + all_data_cat['1stFlrSF'] + all_data_cat['2ndFlrSF']
all_data_cat['Bathrooms'] = (all_data_cat['FullBath'] + 0.5 * all_data_cat['HalfBath'] +
                             all_data_cat['BsmtFullBath'] + 0.5 * all_data_cat['BsmtHalfBath'])
all_data_cat['QualxArea'] = all_data_cat['OverallQual'] * all_data_cat['GrLivArea']
all_data_cat['GarageScore'] = all_data_cat['GarageArea'] * all_data_cat['GarageCars']

X_train_cat = all_data_cat[:ntrain].copy()
X_test_cat = all_data_cat[ntrain:].copy()

# Get categorical feature indices
cat_features = [X_train_cat.columns.get_loc(col) for col in categorical_cols_cat if col in X_train_cat.columns]

# CatBoost CV
cat_oof = np.zeros(len(X_train_cat))
cat_test_preds = np.zeros(len(X_test_cat))
cat_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_cat)):
    X_tr = X_train_cat.iloc[train_idx]
    X_val = X_train_cat.iloc[val_idx]
    y_tr = y_cat.iloc[train_idx]
    y_val = y_cat.iloc[val_idx]

    model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        loss_function='RMSE',
        cat_features=cat_features,
        early_stopping_rounds=50,
        verbose=0,
        random_state=42
    )

    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)

    cat_oof[val_idx] = model.predict(X_val)
    cat_test_preds += model.predict(X_test_cat) / 5

    fold_rmse = np.sqrt(np.mean((cat_oof[val_idx] - y_val.values) ** 2))
    cat_scores.append(fold_rmse)

cat_cv_rmse = np.sqrt(np.mean((cat_oof - y_cat.values) ** 2))
print(f"CatBoost CV RMSE: {cat_cv_rmse:.5f}")

# =============================================================================
# PHASE 6: Ensemble
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 6: Ensemble")
print("=" * 60)

# Simple average ensemble
ensemble_oof = 0.5 * lgb_oof + 0.5 * cat_oof
ensemble_test = 0.5 * lgb_test_preds + 0.5 * cat_test_preds

ensemble_cv_rmse = np.sqrt(np.mean((ensemble_oof - y.values) ** 2))
print(f"Ensemble CV RMSE: {ensemble_cv_rmse:.5f}")

# =============================================================================
# PHASE 7: Submission
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 7: Submission")
print("=" * 60)

# Convert back from log scale
final_predictions = np.expm1(ensemble_test)

# Create submission
submission = pd.DataFrame({
    'Id': test_id,
    'SalePrice': final_predictions
})

submission.to_csv('submission.csv', index=False)
print(f"Submission saved to submission.csv")
print(f"Predictions: min={final_predictions.min():.0f}, max={final_predictions.max():.0f}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Ridge Baseline CV RMSE:     {ridge_baseline:.5f}")
print(f"Ridge + Features CV RMSE:   {ridge_fe_rmse:.5f}")
print(f"LightGBM CV RMSE:           {lgb_cv_rmse:.5f}")
print(f"CatBoost CV RMSE:           {cat_cv_rmse:.5f}")
print(f"Ensemble CV RMSE:           {ensemble_cv_rmse:.5f}")
print("=" * 60)

if ensemble_cv_rmse <= 0.13:
    print("✓ Target achieved! Ensemble CV ≈ 0.12-0.13 RMSE")
else:
    print(f"Note: CV RMSE {ensemble_cv_rmse:.5f} - may need investigation")
