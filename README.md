# 🏙️ Pan-India House Price Prediction & Real Estate Analytics System

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-green.svg)](https://fastapi.tiangolo.com/)

> **Comprehensive Real Estate Valuation System (Pan-India Coverage)**  
> **Student:** Chinmay Bhatt (Roll No: `24EARAD034`)  
> **Degree:** B.Tech – Artificial Intelligence & Data Science  
> **College:** Arya College of Engineering & I.T., Kukas, Jaipur (Rajasthan Technical University, Kota)  
> **Training Organization:** DMV CoreTech, Silvassa, India  
> **Mentor:** Angothu Muni  

---

## 📌 Project Overview
The **Pan-India House Price Prediction System** is a production-grade machine learning platform that evaluates residential properties across all major Indian metropolitan areas and tier-based custom locations.

### 🌐 Cities & Coverage:
1. **🏙️ Mumbai (MMR):** South Mumbai, Bandra West, Worli, Juhu, Andheri West, Powai, Goregaon, Borivali, Thane, Navi Mumbai
2. **🏛️ Delhi NCR:** Connaught Place, Golf Course Road, Cyber City, South Extension, Vasant Kunj, Dwarka, Noida Sec 62, Rohini, Indirapuram, Greater Noida
3. **💻 Bengaluru:** Indiranagar, Koramangala, HSR Layout, Whitefield, Bellandur, JP Nagar, Hebbal, Electronic City, Marathahalli, Yelahanka
4. **🚀 Hyderabad:** Jubilee Hills, Banjara Hills, HITEC City, Gachibowli, Kondapur, Begumpet, Manikonda, Kukatpally, Uppal, Miyapur
5. **🌿 Pune:** Koregaon Park, Kalyani Nagar, Viman Nagar, Aundh, Kothrud, Baner, Magarpatta, Kharadi, Wakad, Hinjewadi
6. **🏰 Jaipur:** C-Scheme, Raja Park, Bani Park, Malviya Nagar, Vaishali Nagar, Mansarovar, Jagatpura, Ajmer Road, Vidhyadhar Nagar, Sanganer
7. **🌆 Ahmedabad:** Bodakdev, Satellite, SG Highway, Vastrapur, Prahlad Nagar, Thaltej, Navrangpura, Bopal, Chandkheda, Maninagar
8. **🌉 Kolkata:** Alipore, Park Street, Ballygunge, Salt Lake, New Town, Jadavpur, Rajarhat, Garia, Behala, Howrah
9. **🌍 Custom City Mode:** Universal evaluation for any Indian city or town based on Tier 1 / Tier 2 / Tier 3 dynamic calibration!

---

## 🏆 Model Benchmark Results (Pan-India Dataset: 6,398 Properties)

| Model | R² Score | MAE (Lakh ₹) | RMSE (Lakh ₹) | 5-Fold CV R² | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **0.963** | **15.04** | **26.99** | **0.951** | 🏆 **Best Generalization** |
| **Gradient Boosting** | **0.953** | **19.45** | **30.44** | **0.952** | High Accuracy Ensemble |
| Decision Tree | 0.900 | 30.05 | 44.62 | 0.883 | Tree Baseline |
| Ridge Regression | 0.870 | 28.81 | 50.89 | 0.860 | Regularized Linear |
| Linear Regression | 0.869 | 28.71 | 50.97 | 0.860 | Linear Baseline |

---

## 📁 Repository Structure

```text
├── app.py                      # Streamlit Pan-India Web Application
├── run.sh                      # Shell launch script (with auto port-freeing)
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation
│
├── backend/                    # Python Backend & ML Engine
│   ├── __init__.py
│   ├── data_generator.py       # Pan-India dataset generator (6,400+ rows across 8 cities)
│   ├── data_cleaner.py         # Multi-city cleaning pipeline (dedup, imputation, outlier removal)
│   ├── model_trainer.py        # 5 Regression models training + 5-fold cross-validation
│   ├── predictor.py            # Pan-India valuation engine & EMI calculator
│   └── api.py                  # FastAPI REST endpoints (/predict, /cities, /health)
│
├── frontend/                   # Frontend assets
│   └── styles.css              # Glassmorphic executive stylesheet
│
├── data/                       # Datasets
│   ├── raw_house_data.csv      # Raw dataset with deliberate anomalies
│   └── cleaned_house_data.csv  # Verified dataset (6,398 records)
│
└── models/                     # Saved ML Pipelines & Metrics
    ├── house_price_model.pkl   # Best trained Scikit-learn Pipeline
    ├── all_models.pkl          # All 5 fitted pipelines
    └── metrics.json            # Model benchmark scores & feature importances
```

---

## 🚀 Quickstart Guide

### 1. Launch the Application
```bash
./run.sh
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser!

### 2. Launch FastAPI REST Server (Optional)
```bash
uvicorn backend.api:app --reload --port 8000
```
Interactive Swagger Documentation available at: **[http://localhost:8000/docs](http://localhost:8000/docs)**

---
© 2026 Chinmay Bhatt | Arya College of Engineering & I.T., Jaipur
