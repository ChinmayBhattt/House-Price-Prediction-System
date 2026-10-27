"""
backend/data_cleaner.py
Multi-City Data Cleaning Pipeline:
1. Deduplication (removes duplicate records)
2. Median Imputation for numeric gaps ('Area_sqft', 'Age_years', 'Distance_km') grouped by city
3. Mode Imputation for categorical gaps ('Furnishing')
4. Outlier removal for erroneous entries ('Area_sqft' <= 5500)
"""

import os
import pandas as pd


def clean_housing_data(
    raw_path: str = 'data/raw_house_data.csv',
    output_path: str = 'data/cleaned_house_data.csv'
) -> tuple[pd.DataFrame, dict]:
    """Cleans multi-city raw housing data and exports verified records."""
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found at {raw_path}")
        
    df = pd.read_csv(raw_path)
    initial_shape = df.shape
    initial_dups = int(df.duplicated().sum())
    initial_nulls = df.isnull().sum().to_dict()
    
    # 1. Drop duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    post_dup_shape = df.shape
    
    # 2. Impute numeric features with city-level or global medians
    imputation_values = {}
    for c in ['Area_sqft', 'Age_years', 'Distance_km']:
        med = float(df[c].median())
        imputation_values[c] = med
        df[c] = df.groupby('City')[c].transform(lambda s: s.fillna(s.median()))
        df[c] = df[c].fillna(med)
        
    # 3. Impute categorical mode
    furn_mode = str(df['Furnishing'].mode()[0])
    imputation_values['Furnishing'] = furn_mode
    df['Furnishing'] = df['Furnishing'].fillna(furn_mode)
    
    # 4. Outlier removal (e.g. 98,000 sq ft typing error)
    outliers_count = int((df['Area_sqft'] > 5500).sum())
    df = df[df['Area_sqft'] <= 5500].reset_index(drop=True)
    
    final_shape = df.shape
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    summary = {
        'initial_shape': initial_shape,
        'initial_duplicates': initial_dups,
        'initial_nulls': initial_nulls,
        'post_duplicate_shape': post_dup_shape,
        'imputation_values': imputation_values,
        'outliers_removed': outliers_count,
        'final_shape': final_shape,
        'final_nulls': df.isnull().sum().to_dict(),
        'output_path': output_path,
        'cities': df['City'].unique().tolist()
    }
    return df, summary


if __name__ == '__main__':
    cleaned_df, report = clean_housing_data()
    print("=== Multi-City Cleaning Summary ===")
    print(f"Raw shape: {report['initial_shape']} -> Cleaned shape: {report['final_shape']}")
    print(f"Cities covered: {report['cities']}")
    print(f"Missing values remaining: {sum(report['final_nulls'].values())}")
