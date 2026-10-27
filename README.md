# 🏡 House Price Prediction System (Jaipur Real Estate)

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-green.svg)](https://fastapi.tiangolo.com/)

> **Industrial Training Project in Artificial Intelligence & Machine Learning (AIML)**  
> **Student:** Chinmay Bhatt (Roll No: `24EARAD034`)  
> **Degree:** B.Tech – Artificial Intelligence & Data Science  
> **College:** Arya College of Engineering & I.T., Kukas, Jaipur (Rajasthan Technical University, Kota)  
> **Training Organization:** DMV CoreTech, Silvassa, India  
> **Mentor:** Angothu Muni  

---

## 📌 Project Overview
The **House Price Prediction System** is an end-to-end Machine Learning web application designed to estimate residential property prices across 10 key localities in Jaipur. Built following standard industry lifecycle methodologies:
1. **Data Collection & Dirty Data Simulation** (1,218 raw records with missing values, duplicate entries, and typing anomalies)
2. **Data Cleaning & Preprocessing** (Imputations, outlier filtering to 1,199 verified records)
3. **Exploratory Data Analysis (EDA)** (Histograms, scatter plots, correlation heatmaps, locality benchmarks)
4. **Model Building & Cross-Validation** (Linear Regression, Ridge, Decision Tree, Random Forest, Gradient Boosting)
5. **Interactive UI & REST API** (Streamlit frontend + FastAPI backend service)

---

## 🏆 Model Benchmark Results (Table 4.1 from Training Report)

| Model | R² Score | MAE (Lakh ₹) | RMSE (Lakh ₹) | 5-Fold CV R² | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting** | **0.959** | **8.23** | **12.49** | **0.956** | 🏆 **Best Model** |
| Random Forest | 0.949 | 8.69 | 13.91 | 0.930 | Ensembles Runner-up |
| Linear Regression | 0.910 | 12.04 | 18.53 | 0.907 | Baseline |
| Ridge Regression | 0.910 | 12.00 | 18.54 | 0.907 | Regularized |
| Decision Tree | 0.882 | 14.57 | 21.21 | 0.871 | Single Tree |

### 🔑 Key Insight:
* **Gradient Boosting Regressor** achieved the highest accuracy (**R² = 95.9%**), lowest error, and steady 5-fold cross-validation (**0.956**), preventing overfitting.
* **Built-up Area** contributes over **54%** to price variation, followed by **City Centre Distance** and premium location tags like **C-Scheme** and **Vaishali Nagar**.

---

## 📁 Repository Structure

```text
├── app.py                      # Main Streamlit web application
├── run.sh                      # Shell launch script
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation
│
├── backend/                    # Python Backend & ML Engine
│   ├── __init__.py
│   ├── data_generator.py       # Raw dataset generator (1,218 records with deliberate errors)
│   ├── data_cleaner.py         # Cleaning pipeline (dedup, imputation, outlier filter -> 1,199 rows)
│   ├── model_trainer.py        # 5-Model training, 5-fold CV, and artifact exporter
│   ├── predictor.py            # High-performance valuation engine & EMI calculator
│   └── api.py                  # FastAPI REST endpoints (/predict, /models, /localities)
│
├── frontend/                   # Frontend assets
│   └── styles.css              # Glassmorphic modern executive stylesheet
│
├── data/                       # Datasets
│   ├── raw_house_data.csv      # Raw anomalous dataset (1,218 rows)
│   └── cleaned_house_data.csv  # Cleaned dataset (1,199 rows)
│
└── models/                     # Saved ML Pipelines & Metrics
    ├── house_price_model.pkl   # Gradient Boosting Scikit-learn Pipeline
    ├── all_models.pkl          # All 5 fitted pipelines
    └── metrics.json            # Model benchmark scores & feature importances
```

---

## 🚀 Quickstart Guide

### 1. Setup Virtual Environment
```bash
# Clone or navigate to the directory
cd /Users/chinmaybhatt/project

# Install dependencies using uv or pip
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Generate Data & Train Models (Automated)
```bash
# Generate raw dataset with deliberate anomalies
python backend/data_generator.py

# Clean data (duplicates, medians, outlier removal)
python backend/data_cleaner.py

# Train and compare 5 regression models
python backend/model_trainer.py
```

### 3. Launch Streamlit Application
```bash
# Direct launcher
./run.sh

# Or directly via streamlit
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser!

---

## 🌐 Optional: Launch FastAPI REST Server
If you'd like to use the backend as a decoupled REST API service:
```bash
uvicorn backend.api:app --reload --port 8000
```
Interactive Swagger Documentation available at: **[http://localhost:8000/docs](http://localhost:8000/docs)**

---

## 🛠️ Input Features Used by the Model

| Feature | Type | Description |
| :--- | :--- | :--- |
| **Locality** | Categorical | 10 Localities: C-Scheme, Vaishali Nagar, Malviya Nagar, Mansarovar, Raja Park, Bani Park, Jagatpura, Ajmer Road, Vidhyadhar Nagar, Sanganer |
| **Area_sqft** | Numerical | Built-up area in square feet (450 – 4,800 sq.ft) |
| **BHK** | Numerical | Number of bedrooms (1 to 5) |
| **Bathrooms** | Numerical | Number of bathrooms (1 to 5) |
| **Age_years** | Numerical | Age of the property in years (0 to 25) |
| **Floor** | Numerical | Floor number (0 = Ground floor to 14) |
| **Distance_km** | Numerical | Distance from City Centre / MI Road (km) |
| **Parking** | Numerical | Dedicated parking slots (0, 1, 2) |
| **Furnishing** | Categorical | Unfurnished, Semi-Furnished, Furnished |
| **Price_lakh** | Target (₹) | Residential price in Lakh ₹ (Target variable) |

---

## 📑 Streamlit Application Modules

1. **🎯 Property Valuation:** Real-time property pricing, confidence interval, rate per sq.ft., locality median comparison, home loan & EMI calculator, and comparable recent sales.
2. **📊 Market Trends & EDA:** Figures 4.2, 4.3, 4.4 from the report (Price distribution, Area vs. Price, Median locality rankings, Correlation heatmap, Rate/sqft box plots).
3. **🤖 Model Benchmark & Comparison:** Re-creation of Table 4.1 comparing 5 algorithms, actual vs. predicted parity plot, and top 10 feature importances.
4. **🧹 Data Cleaning Pipeline:** Transparent walkthrough of raw dirty data (1,218 rows) transformed into clean training data (1,199 rows).
5. **📁 Batch CSV Valuation:** Bulk estimation supporting CSV file upload and exportable valuation results.
6. **📜 Project Report & Info:** Complete academic and internship credentials with one-click download for datasets and `.pkl` models.

---
© 2026 Chinmay Bhatt | Arya College of Engineering & I.T., Jaipur
# House-Price-Prediction-System
