"""
backend/predictor.py
Pan-India Real Estate Valuation Engine
Supports:
- 8 Major Indian Metro & Capital Markets (Mumbai, Delhi NCR, Bengaluru, Hyderabad, Pune, Jaipur, Ahmedabad, Kolkata)
- Universal / Custom City & Locality Valuations (Tier 1, Tier 2, Tier 3)
- Lakh and Crore formatted outputs
- Rate per sq.ft.
- Locality & City Benchmark Comparisons
- Home Loan and EMI Calculator
- Batch CSV Predictions
"""

import os
import joblib
import numpy as np
import pandas as pd

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

_CACHED_MODEL = None


def load_model(model_path: str = 'models/house_price_model.pkl'):
    """Loads and caches the trained Scikit-learn pipeline."""
    global _CACHED_MODEL
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL
        
    if not os.path.exists(model_path):
        if os.path.exists('house_price_model.pkl'):
            model_path = 'house_price_model.pkl'
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first.")
            
    _CACHED_MODEL = joblib.load(model_path)
    return _CACHED_MODEL


def format_inr_price(price_lakh: float) -> str:
    """Formats price into Lakhs or Crores."""
    if price_lakh >= 100:
        cr = price_lakh / 100.0
        return f"₹{cr:.2f} Cr"
    else:
        return f"₹{price_lakh:.2f} Lakh"


def calculate_emi(
    price_lakh: float,
    down_payment_pct: float = 20.0,
    annual_interest_rate: float = 8.5,
    tenure_years: int = 20
) -> dict:
    """Computes loan breakdown and monthly EMI."""
    total_price_inr = price_lakh * 100000.0
    down_payment_inr = total_price_inr * (down_payment_pct / 100.0)
    loan_amount_inr = total_price_inr - down_payment_inr
    
    monthly_r = (annual_interest_rate / 100.0) / 12.0
    n_months = tenure_years * 12
    
    if monthly_r > 0:
        emi = loan_amount_inr * monthly_r * ((1 + monthly_r) ** n_months) / (((1 + monthly_r) ** n_months) - 1)
    else:
        emi = loan_amount_inr / n_months
        
    total_payment = (emi * n_months) + down_payment_inr
    total_interest = (emi * n_months) - loan_amount_inr
    
    return {
        'loan_amount_lakh': round(loan_amount_inr / 100000.0, 2),
        'down_payment_lakh': round(down_payment_inr / 100000.0, 2),
        'monthly_emi_inr': int(round(emi)),
        'total_interest_lakh': round(total_interest / 100000.0, 2),
        'total_cost_lakh': round(total_payment / 100000.0, 2),
    }


def predict_price(
    locality: str,
    area: float,
    bhk: int,
    bath: int,
    age: float,
    floor: int,
    dist: float,
    parking: int,
    furnishing: str,
    city: str = 'Jaipur',
    model_path: str = 'models/house_price_model.pkl'
) -> float:
    """Standard prediction function matching pipeline signature."""
    row = pd.DataFrame(
        [[city, locality, area, bhk, bath, age, floor, dist, parking, furnishing]],
        columns=FEATURE_NAMES
    )
    model = load_model(model_path)
    pred = float(model.predict(row)[0])
    return round(max(5.0, pred), 2)


def predict_detailed(
    locality: str,
    area: float,
    bhk: int,
    bath: int,
    age: float,
    floor: int,
    dist: float,
    parking: int,
    furnishing: str,
    city: str = 'Jaipur',
    model_name: str = 'Gradient Boosting'
) -> dict:
    """Detailed valuation with benchmarks, EMI, rates, and confidence intervals."""
    all_models_path = 'models/all_models.pkl'
    pipe = None
    if os.path.exists(all_models_path):
        all_models = joblib.load(all_models_path)
        pipe = all_models.get(model_name)
        
    if pipe is None:
        pipe = load_model('models/house_price_model.pkl')
        
    row = pd.DataFrame(
        [[city, locality, area, bhk, bath, age, floor, dist, parking, furnishing]],
        columns=FEATURE_NAMES
    )
    
    predicted_lakh = float(pipe.predict(row)[0])
    predicted_lakh = round(max(5.0, predicted_lakh), 2)
    
    # Margin based on uncertainty (~6%)
    margin_lakh = round(predicted_lakh * 0.06, 2)
    min_lakh = max(5.0, round(predicted_lakh - margin_lakh, 2))
    max_lakh = round(predicted_lakh + margin_lakh, 2)
    
    rate_per_sqft = int(round((predicted_lakh * 100000.0) / max(1.0, area)))
    emi_info = calculate_emi(predicted_lakh)
    
    # Locality & City market comparison if dataset exists
    dataset_path = 'data/cleaned_house_data.csv'
    locality_stats = {}
    city_stats = {}
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        c_df = df[df['City'] == city]
        if len(c_df) > 0:
            city_med = float(c_df['Price_lakh'].median())
            city_avg_rate = int(round((c_df['Price_lakh'] * 100000.0 / c_df['Area_sqft']).median()))
            city_stats = {
                'median_price_lakh': city_med,
                'avg_rate_sqft': city_avg_rate
            }
            
            loc_df = c_df[c_df['Locality'] == locality]
            if len(loc_df) > 0:
                loc_median = float(loc_df['Price_lakh'].median())
                diff = round(predicted_lakh - loc_median, 2)
                pct = round((diff / loc_median) * 100.0, 1)
                locality_stats = {
                    'median_price_lakh': loc_median,
                    'min_price_lakh': float(loc_df['Price_lakh'].min()),
                    'max_price_lakh': float(loc_df['Price_lakh'].max()),
                    'avg_rate_sqft': int(round((loc_df['Price_lakh'] * 100000.0 / loc_df['Area_sqft']).median())),
                    'diff_from_median': diff,
                    'pct_from_median': pct
                }
                
    return {
        'city': city,
        'locality': locality,
        'price_lakh': predicted_lakh,
        'formatted_price': format_inr_price(predicted_lakh),
        'min_price_lakh': min_lakh,
        'max_price_lakh': max_lakh,
        'rate_per_sqft': rate_per_sqft,
        'area_sqft': area,
        'model_used': model_name,
        'emi': emi_info,
        'locality_stats': locality_stats,
        'city_stats': city_stats,
        'input_features': {
            'City': city,
            'Locality': locality,
            'Area_sqft': area,
            'BHK': bhk,
            'Bathrooms': bath,
            'Age_years': age,
            'Floor': floor,
            'Distance_km': dist,
            'Parking': parking,
            'Furnishing': furnishing,
        }
    }


def batch_predict(df: pd.DataFrame, model_name: str = 'Gradient Boosting') -> pd.DataFrame:
    """Runs batch predictions for a dataframe."""
    df_pred = df.copy()
    
    # If City column is missing in user CSV, default to Jaipur
    if 'City' not in df_pred.columns:
        df_pred['City'] = 'Jaipur'
        
    missing = [c for c in FEATURE_NAMES if c not in df_pred.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")
        
    all_models_path = 'models/all_models.pkl'
    pipe = None
    if os.path.exists(all_models_path):
        all_models = joblib.load(all_models_path)
        pipe = all_models.get(model_name)
        
    if pipe is None:
        pipe = load_model('models/house_price_model.pkl')
        
    preds = pipe.predict(df_pred[FEATURE_NAMES])
    df_pred['Predicted_Price_Lakh'] = [round(max(5.0, float(p)), 2) for p in preds]
    df_pred['Rate_per_sqft'] = [
        int(round((p * 100000.0) / max(1.0, a)))
        for p, a in zip(df_pred['Predicted_Price_Lakh'], df_pred['Area_sqft'])
    ]
    df_pred['Formatted_Price'] = [format_inr_price(p) for p in df_pred['Predicted_Price_Lakh']]
    return df_pred
