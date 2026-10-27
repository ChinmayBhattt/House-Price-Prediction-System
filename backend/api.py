"""
backend/api.py
FastAPI REST Backend for Pan-India Real Estate Price Valuation
"""

from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import json
import os

from backend.predictor import (
    predict_price,
    predict_detailed,
    format_inr_price,
    batch_predict,
    FEATURE_NAMES
)

app = FastAPI(
    title="Pan-India House Price Prediction API",
    description="Machine Learning REST Backend for Residential Real Estate Valuation across Major Indian Cities",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HouseFeatures(BaseModel):
    city: Optional[str] = Field("Jaipur", example="Mumbai")
    locality: str = Field(..., example="Bandra West")
    area_sqft: float = Field(..., gt=100, lt=10000, example=1500.0)
    bhk: int = Field(..., ge=1, le=10, example=3)
    bathrooms: int = Field(..., ge=1, le=10, example=3)
    age_years: float = Field(..., ge=0, le=100, example=5.0)
    floor: int = Field(..., ge=0, le=50, example=2)
    distance_km: float = Field(..., ge=0.1, le=100.0, example=10.0)
    parking: int = Field(..., ge=0, le=5, example=1)
    furnishing: str = Field(..., example="Semi-Furnished")
    model_name: Optional[str] = Field("Gradient Boosting", example="Gradient Boosting")


class PredictionResponse(BaseModel):
    city: str
    locality: str
    area_sqft: float
    price_lakh: float
    formatted_price: str
    rate_per_sqft: int
    model_used: str


@app.get("/")
def root():
    return {
        "project": "Pan-India House Price Prediction System",
        "author": "Chinmay Bhatt",
        "coverage": ["Mumbai", "Delhi NCR", "Bengaluru", "Hyderabad", "Pune", "Jaipur", "Ahmedabad", "Kolkata", "Custom Cities"],
        "documentation": "/docs"
    }


@app.get("/health")
def health():
    model_exists = os.path.exists("models/house_price_model.pkl") or os.path.exists("house_price_model.pkl")
    return {
        "status": "healthy" if model_exists else "model_missing",
        "model_loaded": model_exists
    }


@app.post("/predict", response_model=PredictionResponse)
def get_prediction(house: HouseFeatures):
    try:
        c = house.city or "Jaipur"
        val = predict_price(
            city=c,
            locality=house.locality,
            area=house.area_sqft,
            bhk=house.bhk,
            bath=house.bathrooms,
            age=house.age_years,
            floor=house.floor,
            dist=house.distance_km,
            parking=house.parking,
            furnishing=house.furnishing
        )
        rate = int(round((val * 100000.0) / house.area_sqft))
        return PredictionResponse(
            city=c,
            locality=house.locality,
            area_sqft=house.area_sqft,
            price_lakh=val,
            formatted_price=format_inr_price(val),
            rate_per_sqft=rate,
            model_used=house.model_name or "Gradient Boosting"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/detailed")
def get_detailed_prediction(house: HouseFeatures):
    try:
        detailed = predict_detailed(
            city=house.city or "Jaipur",
            locality=house.locality,
            area=house.area_sqft,
            bhk=house.bhk,
            bath=house.bathrooms,
            age=house.age_years,
            floor=house.floor,
            dist=house.distance_km,
            parking=house.parking,
            furnishing=house.furnishing,
            model_name=house.model_name or "Gradient Boosting"
        )
        return detailed
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/cities")
def get_cities():
    data_path = "data/cleaned_house_data.csv"
    if not os.path.exists(data_path):
        raise HTTPException(status_code=404, detail="Dataset not found.")
    df = pd.read_csv(data_path)
    city_summary = df.groupby("City").agg(
        total_properties=("Price_lakh", "count"),
        median_price_lakh=("Price_lakh", "median"),
        avg_rate_sqft=("Price_lakh", lambda x: int(round((x * 100000.0 / df.loc[x.index, "Area_sqft"]).median())))
    ).reset_index()
    return city_summary.to_dict(orient="records")
