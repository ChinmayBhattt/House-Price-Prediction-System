"""
backend/model_trainer.py
Trains 5 Regression Models on the Pan-India Multi-City Housing Dataset:
1. Gradient Boosting (Top Performing Ensemble)
2. Random Forest
3. Linear Regression
4. Ridge Regression
5. Decision Tree

Evaluates with R², MAE, RMSE, and 5-Fold Cross Validation.
Exports pipelines to models/house_price_model.pkl and performance benchmarks to models/metrics.json.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

FEATURE_NAMES = [
    'City',
    'Locality',
    'Area_sqft',
    'BHK',
    'Bathrooms',
    'Age_years',
    'Floor',
    'Distance_km',
    'Parking',
    'Furnishing',
]
NUM_COLS = ['Area_sqft', 'BHK', 'Bathrooms', 'Age_years', 'Floor', 'Distance_km', 'Parking']
CAT_COLS = ['City', 'Locality', 'Furnishing']
TARGET_COL = 'Price_lakh'


def train_models(
    data_path: str = 'data/cleaned_house_data.csv',
    models_dir: str = 'models',
    random_state: int = 42
) -> dict:
    """Trains 5 regression models on the multi-city dataset and exports evaluation metrics."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cleaned dataset not found at {data_path}")
        
    df = pd.read_csv(data_path)
    X = df[FEATURE_NAMES]
    y = df[TARGET_COL]
    
    # 80:20 Train:Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state
    )
    
    pre = ColumnTransformer([
        ('n', StandardScaler(), NUM_COLS),
        ('c', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT_COLS)
    ])
    
    models = {
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, random_state=random_state),
        'Random Forest': RandomForestRegressor(n_estimators=150, random_state=random_state, n_jobs=-1),
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=random_state)
    }
    
    os.makedirs(models_dir, exist_ok=True)
    
    fitted_pipelines = {}
    metrics_summary = []
    test_predictions_dict = {'actual': y_test.tolist()}
    
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    
    best_name = None
    best_r2 = -1e9
    best_pipe = None
    
    for name, m in models.items():
        pipe = Pipeline([
            ('pre', pre),
            ('m', m)
        ])
        
        # Fit on training data
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe
        
        # Predict on test data
        pred = pipe.predict(X_test)
        test_predictions_dict[name] = [round(float(p), 2) for p in pred]
        
        # Evaluation metrics
        r2 = float(r2_score(y_test, pred))
        mae = float(mean_absolute_error(y_test, pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        
        # 5-fold cross validation
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=kf, scoring='r2')
        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())
        
        row_metrics = {
            'model': name,
            'r2_score': round(r2, 3),
            'mae': round(mae, 2),
            'rmse': round(rmse, 2),
            'cv_r2_mean': round(cv_mean, 3),
            'cv_r2_std': round(cv_std, 3)
        }
        metrics_summary.append(row_metrics)
        
        if r2 > best_r2:
            best_r2 = r2
            best_name = name
            best_pipe = pipe

    # Feature importances from best model
    gb_model = fitted_pipelines['Gradient Boosting'].named_steps['m']
    preprocessor = fitted_pipelines['Gradient Boosting'].named_steps['pre']
    cat_feature_names = preprocessor.named_transformers_['c'].get_feature_names_out(CAT_COLS)
    all_feature_names = list(NUM_COLS) + list(cat_feature_names)
    
    importances = gb_model.feature_importances_
    feat_imp = sorted(
        [{'feature': feat, 'importance': round(float(imp) * 100, 2)}
         for feat, imp in zip(all_feature_names, importances)],
        key=lambda x: x['importance'],
        reverse=True
    )
    
    # Save best pipeline
    best_model_path = os.path.join(models_dir, 'house_price_model.pkl')
    joblib.dump(best_pipe, best_model_path)
    joblib.dump(best_pipe, 'house_price_model.pkl')
    
    # Save all models
    all_models_path = os.path.join(models_dir, 'all_models.pkl')
    joblib.dump(fitted_pipelines, all_models_path)
    
    # Save metrics payload
    metrics_payload = {
        'best_model': best_name,
        'best_r2': round(best_r2, 3),
        'comparison_table': metrics_summary,
        'feature_importances': feat_imp,
        'test_predictions': test_predictions_dict,
        'dataset_info': {
            'total_samples': len(df),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'numeric_features': NUM_COLS,
            'categorical_features': CAT_COLS,
            'cities': sorted(df['City'].unique().tolist()),
            'localities_count': df['Locality'].nunique(),
            'furnishing_types': ['Unfurnished', 'Semi-Furnished', 'Furnished']
        }
    }
    
    metrics_file = os.path.join(models_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics_payload, f, indent=2)
        
    return metrics_payload


if __name__ == '__main__':
    print("Training Pan-India Multi-City Models...")
    res = train_models()
    print("\n=== Model Comparison on Test Data (Price in Lakh ₹) ===")
    df_metrics = pd.DataFrame(res['comparison_table'])
    print(df_metrics.to_string(index=False))
    print(f"\nBest Model: {res['best_model']} (R2 = {res['best_r2']})")
