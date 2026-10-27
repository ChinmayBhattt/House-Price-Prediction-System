"""
backend/data_generator.py
Pan-India Multi-City Real Estate Dataset Generator
Covers 8 Major Indian Real Estate Markets (80 Prime Localities):
- Mumbai (MMR)
- Delhi NCR
- Bengaluru
- Hyderabad
- Pune
- Jaipur
- Kolkata
- Ahmedabad
Plus support for Universal / Custom Cities.
"""

import os
import numpy as np
import pandas as pd

CITY_LOCALITY_STATS = {
    'Mumbai': {
        'tier': 'Tier 1 Metro (Financial Capital)',
        'localities': {
            'South Mumbai': {'base_rate': 42000, 'dist_mean': 2.0, 'dist_std': 0.8},
            'Bandra West': {'base_rate': 38000, 'dist_mean': 10.0, 'dist_std': 1.5},
            'Worli': {'base_rate': 36000, 'dist_mean': 6.0, 'dist_std': 1.2},
            'Juhu': {'base_rate': 35000, 'dist_mean': 14.0, 'dist_std': 1.5},
            'Andheri West': {'base_rate': 24000, 'dist_mean': 18.0, 'dist_std': 2.0},
            'Powai': {'base_rate': 22000, 'dist_mean': 20.0, 'dist_std': 2.0},
            'Goregaon': {'base_rate': 19000, 'dist_mean': 22.0, 'dist_std': 2.5},
            'Borivali': {'base_rate': 16500, 'dist_mean': 28.0, 'dist_std': 3.0},
            'Thane': {'base_rate': 13500, 'dist_mean': 30.0, 'dist_std': 3.5},
            'Navi Mumbai': {'base_rate': 12000, 'dist_mean': 28.0, 'dist_std': 4.0},
        }
    },
    'Delhi NCR': {
        'tier': 'Tier 1 Metro (National Capital Region)',
        'localities': {
            'Connaught Place': {'base_rate': 28000, 'dist_mean': 1.5, 'dist_std': 0.6},
            'Golf Course Road': {'base_rate': 24000, 'dist_mean': 18.0, 'dist_std': 2.5},
            'South Extension': {'base_rate': 22000, 'dist_mean': 8.0, 'dist_std': 1.5},
            'Cyber City (Gurgaon)': {'base_rate': 20000, 'dist_mean': 16.0, 'dist_std': 2.0},
            'Vasant Kunj': {'base_rate': 18500, 'dist_mean': 14.0, 'dist_std': 1.8},
            'Dwarka': {'base_rate': 12500, 'dist_mean': 19.0, 'dist_std': 2.5},
            'Noida Sector 62': {'base_rate': 11000, 'dist_mean': 18.0, 'dist_std': 2.5},
            'Rohini': {'base_rate': 9800, 'dist_mean': 16.0, 'dist_std': 2.2},
            'Indirapuram': {'base_rate': 8200, 'dist_mean': 16.0, 'dist_std': 2.0},
            'Greater Noida': {'base_rate': 6200, 'dist_mean': 32.0, 'dist_std': 4.0},
        }
    },
    'Bengaluru': {
        'tier': 'Tier 1 Metro (Tech Capital)',
        'localities': {
            'Indiranagar': {'base_rate': 18000, 'dist_mean': 5.0, 'dist_std': 1.0},
            'Koramangala': {'base_rate': 16500, 'dist_mean': 6.0, 'dist_std': 1.2},
            'HSR Layout': {'base_rate': 13000, 'dist_mean': 10.0, 'dist_std': 1.5},
            'Bellandur': {'base_rate': 11500, 'dist_mean': 13.0, 'dist_std': 1.8},
            'Hebbal': {'base_rate': 11000, 'dist_mean': 11.0, 'dist_std': 1.5},
            'Whitefield': {'base_rate': 10500, 'dist_mean': 18.0, 'dist_std': 2.5},
            'JP Nagar': {'base_rate': 9800, 'dist_mean': 9.0, 'dist_std': 1.5},
            'Marathahalli': {'base_rate': 8800, 'dist_mean': 14.0, 'dist_std': 2.0},
            'Yelahanka': {'base_rate': 7800, 'dist_mean': 17.0, 'dist_std': 2.5},
            'Electronic City': {'base_rate': 7200, 'dist_mean': 19.0, 'dist_std': 2.5},
        }
    },
    'Hyderabad': {
        'tier': 'Tier 1 Metro (Genome & IT Hub)',
        'localities': {
            'Jubilee Hills': {'base_rate': 16000, 'dist_mean': 8.0, 'dist_std': 1.2},
            'Banjara Hills': {'base_rate': 15000, 'dist_mean': 6.0, 'dist_std': 1.0},
            'HITEC City': {'base_rate': 11500, 'dist_mean': 12.0, 'dist_std': 1.5},
            'Gachibowli': {'base_rate': 10800, 'dist_mean': 14.0, 'dist_std': 1.8},
            'Kondapur': {'base_rate': 9200, 'dist_mean': 13.0, 'dist_std': 1.5},
            'Begumpet': {'base_rate': 8500, 'dist_mean': 4.0, 'dist_std': 1.0},
            'Manikonda': {'base_rate': 7800, 'dist_mean': 12.0, 'dist_std': 1.8},
            'Kukatpally': {'base_rate': 7200, 'dist_mean': 15.0, 'dist_std': 2.0},
            'Uppal': {'base_rate': 5800, 'dist_mean': 12.0, 'dist_std': 2.0},
            'Miyapur': {'base_rate': 5400, 'dist_mean': 19.0, 'dist_std': 2.2},
        }
    },
    'Pune': {
        'tier': 'Tier 1 / Oxford of the East',
        'localities': {
            'Koregaon Park': {'base_rate': 15000, 'dist_mean': 4.0, 'dist_std': 0.8},
            'Kalyani Nagar': {'base_rate': 13500, 'dist_mean': 6.0, 'dist_std': 1.0},
            'Viman Nagar': {'base_rate': 11000, 'dist_mean': 8.0, 'dist_std': 1.2},
            'Aundh': {'base_rate': 10800, 'dist_mean': 8.0, 'dist_std': 1.2},
            'Kothrud': {'base_rate': 10500, 'dist_mean': 7.0, 'dist_std': 1.1},
            'Baner': {'base_rate': 10200, 'dist_mean': 11.0, 'dist_std': 1.5},
            'Magarpatta': {'base_rate': 8800, 'dist_mean': 9.0, 'dist_std': 1.4},
            'Kharadi': {'base_rate': 8500, 'dist_mean': 11.0, 'dist_std': 1.5},
            'Wakad': {'base_rate': 8200, 'dist_mean': 16.0, 'dist_std': 2.0},
            'Hinjewadi': {'base_rate': 7200, 'dist_mean': 19.0, 'dist_std': 2.2},
        }
    },
    'Jaipur': {
        'tier': 'Tier 2 Heritage & Rapid Growth Hub',
        'localities': {
            'C-Scheme': {'base_rate': 13500, 'dist_mean': 2.5, 'dist_std': 0.8},
            'Raja Park': {'base_rate': 9800, 'dist_mean': 4.0, 'dist_std': 1.0},
            'Bani Park': {'base_rate': 9200, 'dist_mean': 3.5, 'dist_std': 0.9},
            'Malviya Nagar': {'base_rate': 8700, 'dist_mean': 9.0, 'dist_std': 1.5},
            'Vaishali Nagar': {'base_rate': 8400, 'dist_mean': 7.5, 'dist_std': 1.4},
            'Mansarovar': {'base_rate': 6500, 'dist_mean': 10.5, 'dist_std': 1.8},
            'Jagatpura': {'base_rate': 5500, 'dist_mean': 13.0, 'dist_std': 2.0},
            'Ajmer Road': {'base_rate': 5200, 'dist_mean': 11.5, 'dist_std': 2.2},
            'Vidhyadhar Nagar': {'base_rate': 4700, 'dist_mean': 9.5, 'dist_std': 1.5},
            'Sanganer': {'base_rate': 4200, 'dist_mean': 15.0, 'dist_std': 2.2},
        }
    },
    'Ahmedabad': {
        'tier': 'Tier 2 Mega Commercial Hub',
        'localities': {
            'Bodakdev': {'base_rate': 10500, 'dist_mean': 7.0, 'dist_std': 1.0},
            'Satellite': {'base_rate': 9500, 'dist_mean': 6.0, 'dist_std': 1.0},
            'SG Highway': {'base_rate': 8800, 'dist_mean': 9.0, 'dist_std': 1.4},
            'Vastrapur': {'base_rate': 8600, 'dist_mean': 5.0, 'dist_std': 0.9},
            'Prahlad Nagar': {'base_rate': 8400, 'dist_mean': 8.0, 'dist_std': 1.2},
            'Thaltej': {'base_rate': 8200, 'dist_mean': 8.0, 'dist_std': 1.2},
            'Navrangpura': {'base_rate': 8000, 'dist_mean': 3.0, 'dist_std': 0.8},
            'Bopal': {'base_rate': 5800, 'dist_mean': 13.0, 'dist_std': 1.8},
            'Chandkheda': {'base_rate': 4800, 'dist_mean': 12.0, 'dist_std': 1.6},
            'Maninagar': {'base_rate': 4500, 'dist_mean': 6.0, 'dist_std': 1.2},
        }
    },
    'Kolkata': {
        'tier': 'Tier 1 Metro (Cultural Capital)',
        'localities': {
            'Alipore': {'base_rate': 16500, 'dist_mean': 6.0, 'dist_std': 1.2},
            'Park Street': {'base_rate': 15500, 'dist_mean': 2.0, 'dist_std': 0.8},
            'Ballygunge': {'base_rate': 14000, 'dist_mean': 5.0, 'dist_std': 1.0},
            'Salt Lake': {'base_rate': 9500, 'dist_mean': 8.0, 'dist_std': 1.5},
            'New Town': {'base_rate': 7800, 'dist_mean': 14.0, 'dist_std': 2.0},
            'Jadavpur': {'base_rate': 6800, 'dist_mean': 9.0, 'dist_std': 1.5},
            'Rajarhat': {'base_rate': 6200, 'dist_mean': 16.0, 'dist_std': 2.2},
            'Garia': {'base_rate': 5200, 'dist_mean': 14.0, 'dist_std': 2.0},
            'Behala': {'base_rate': 4800, 'dist_mean': 10.0, 'dist_std': 1.8},
            'Howrah': {'base_rate': 4500, 'dist_mean': 7.0, 'dist_std': 1.5},
        }
    }
}

# General Fallback for Custom / Other Cities
FALLBACK_TIERS = {
    'Tier 1 (Metro / Megacity)': {'base_rate': 16000, 'dist_mean': 12.0, 'dist_std': 3.0},
    'Tier 2 (Emerging Industrial/Tech Hub)': {'base_rate': 7000, 'dist_mean': 8.0, 'dist_std': 2.0},
    'Tier 3 (Developing Town / Suburban)': {'base_rate': 4000, 'dist_mean': 6.0, 'dist_std': 1.5},
}

FURNISHING_MULTIPLIERS = {
    'Unfurnished': 0.97,
    'Semi-Furnished': 1.00,
    'Furnished': 1.08,
}


def generate_housing_data(
    records_per_city: int = 800,
    random_seed: int = 42,
    output_dir: str = 'data'
) -> tuple[pd.DataFrame, str]:
    """Generates an extensive multi-city Indian housing dataset with realistic market distributions."""
    np.random.seed(random_seed)
    
    all_dfs = []
    
    for city_name, city_data in CITY_LOCALITY_STATS.items():
        loc_dict = city_data['localities']
        loc_names = list(loc_dict.keys())
        
        # Sample records for this city
        n_city = records_per_city
        loc_choices = np.random.choice(loc_names, size=n_city)
        bhk_choices = np.random.choice([1, 2, 3, 4, 5], size=n_city, p=[0.10, 0.35, 0.38, 0.13, 0.04])
        
        # Area based on BHK
        area_means = {1: 650, 2: 1100, 3: 1650, 4: 2400, 5: 3400}
        area_stds = {1: 90, 2: 140, 3: 230, 4: 330, 5: 450}
        
        areas = []
        bathrooms = []
        for bhk in bhk_choices:
            a = int(np.random.normal(area_means[bhk], area_stds[bhk]))
            a = max(450, min(a, 4800))
            areas.append(a)
            
            if bhk == 1:
                bath = 1
            elif bhk == 2:
                bath = np.random.choice([1, 2], p=[0.2, 0.8])
            elif bhk == 3:
                bath = np.random.choice([2, 3, 4], p=[0.2, 0.7, 0.1])
            elif bhk == 4:
                bath = np.random.choice([3, 4, 5], p=[0.25, 0.65, 0.1])
            else:
                bath = np.random.choice([4, 5], p=[0.4, 0.6])
            bathrooms.append(bath)
            
        areas = np.array(areas)
        bathrooms = np.array(bathrooms)
        ages = np.random.choice(range(0, 26), size=n_city, p=[0.08] + [0.92 / 25] * 25)
        floors = np.random.choice(range(0, 16), size=n_city, p=[0.12] + [0.88 / 15] * 15)
        
        distances = []
        for loc in loc_choices:
            stats = loc_dict[loc]
            d = np.random.normal(stats['dist_mean'], stats['dist_std'])
            d = round(max(1.0, min(d, 35.0)), 1)
            distances.append(d)
        distances = np.array(distances)
        
        parkings = []
        for bhk in bhk_choices:
            if bhk <= 2:
                p = np.random.choice([0, 1, 2], p=[0.35, 0.55, 0.10])
            elif bhk == 3:
                p = np.random.choice([0, 1, 2], p=[0.10, 0.65, 0.25])
            else:
                p = np.random.choice([1, 2], p=[0.35, 0.65])
            parkings.append(p)
        parkings = np.array(parkings)
        
        furnishing_choices = np.random.choice(
            ['Unfurnished', 'Semi-Furnished', 'Furnished'],
            size=n_city,
            p=[0.28, 0.48, 0.24]
        )
        
        # Calculate realistic property price
        prices = []
        for i in range(n_city):
            loc = loc_choices[i]
            area = areas[i]
            bath = bathrooms[i]
            age = ages[i]
            flr = floors[i]
            dist = distances[i]
            park = parkings[i]
            furn = furnishing_choices[i]
            
            base_rate = loc_dict[loc]['base_rate']
            val_inr = area * base_rate
            
            age_factor = 1.0 - (min(age, 20) * 0.007)
            dist_factor = 1.0 - (dist * 0.004)
            amenity_bonus = (bath * 50000) + (park * 100000)
            
            floor_factor = 1.0
            if flr == 0:
                floor_factor = 0.98
            elif 1 <= flr <= 7:
                floor_factor = 1.02
            else:
                floor_factor = 0.99
                
            furn_factor = FURNISHING_MULTIPLIERS[furn]
            total_inr = (val_inr * age_factor * dist_factor * floor_factor * furn_factor) + amenity_bonus
            
            noise = np.random.normal(1.0, 0.04)
            total_inr *= noise
            price_lakh = round(max(8.0, total_inr / 100000.0), 2)
            prices.append(price_lakh)
            
        city_df = pd.DataFrame({
            'City': city_name,
            'Locality': loc_choices,
            'Area_sqft': areas.astype(float),
            'BHK': bhk_choices,
            'Bathrooms': bathrooms,
            'Age_years': ages.astype(float),
            'Floor': floors,
            'Distance_km': distances,
            'Parking': parkings,
            'Furnishing': furnishing_choices,
            'Price_lakh': prices
        })
        all_dfs.append(city_df)
        
    df = pd.concat(all_dfs, ignore_index=True)
    
    # Introduce deliberate dirty data for realism & cleaning demo
    # 1. Duplicate rows
    dup_indices = np.random.choice(range(len(df)), size=45, replace=False)
    duplicates = df.iloc[dup_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # 2. Outliers (typing error of 98,000 sqft in area)
    df.loc[14, 'Area_sqft'] = 98000.0
    df.loc[1050, 'Area_sqft'] = 75000.0
    
    # 3. Missing values
    clean_pool = [i for i in range(len(df) - 45) if i not in [14, 1050] and i not in dup_indices]
    for col, count in [('Area_sqft', 60), ('Age_years', 60), ('Distance_km', 60), ('Furnishing', 50)]:
        miss_idx = np.random.choice(clean_pool, size=count, replace=False)
        df.loc[miss_idx, col] = np.nan
        
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'raw_house_data.csv')
    df.to_csv(out_path, index=False)
    return df, out_path


# Legacy alias
def generate_jaipur_housing_data():
    return generate_housing_data()


if __name__ == '__main__':
    df, p = generate_housing_data()
    print(f"Generated multi-city dataset at {p}: {df.shape}")
    print("Cities included:", df['City'].unique())
