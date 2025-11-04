# import os
# import mlflow # type: ignore
# import pandas as pd
# from fastapi import FastAPI # type: ignore
# from pydantic import BaseModel

# # --- DEFINE THE INPUT DATA MODEL ---
# # Pydantic model for input data validation.
# # These feature names must match the training data columns.
# class HousingFeatures(BaseModel):
#     MedInc: float
#     HouseAge: float
#     AveRooms: float
#     AveBedrms: float
#     Population: float
#     AveOccup: float
#     Latitude: float
#     Longitude: float

# # --- LOAD THE TRAINED MODEL ---
# # This function finds the latest run in MLflow and loads its model.
# def load_latest_model():
    
# # Point to the MLflow tracking server, same as in train.py
#     mlflow.set_tracking_uri("http://127.0.0.1:5000")
    
#     # Search for runs in the default experiment (experiment_id='0')
#     runs = mlflow.search_runs(experiment_ids=['0'])
    
#     # Get the latest run's ID
#     latest_run_id = runs.iloc[0]['run_id']
    
#     # Construct the model URI
#     model_uri = f"runs:/{latest_run_id}/random-forest-model"
    
#     print(f"Loading model from: {model_uri}")
#     return mlflow.sklearn.load_model(model_uri)

# model = load_latest_model()

# # --- CREATE THE FASTAPI APP ---
# app = FastAPI(title="Real Estate Prediction API", version="1.0")

# @app.get("/")
# def read_root():
#     return {"message": "Welcome to the Real Estate Prediction API!"}

# @app.post("/predict/")
# def predict(features: HousingFeatures):
#     """
#     Accepts housing features and returns a prediction.
#     """
#     # Convert input data to a Pandas DataFrame
#     input_df = pd.DataFrame([features.model_dump()])
    
#     # Make a prediction
#     prediction = model.predict(input_df)
    
#     # Return the prediction in a JSON response
#     return {"predicted_median_house_value": prediction[0]}


import os
import mlflow
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator  # Updated import
from typing import Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HousingFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

    # Updated validators using field_validator
    @field_validator('MedInc')
    @classmethod  # Required for field_validator
    def validate_medinc(cls, v):
        if v < 0:
            raise ValueError('Median income cannot be negative')
        return v

    @field_validator('Latitude')
    @classmethod
    def validate_latitude(cls, v):
        if not (32.0 <= v <= 42.0):
            raise ValueError('Latitude must be between 32.0 and 42.0')
        return v

    @field_validator('Longitude')
    @classmethod
    def validate_longitude(cls, v):
        if not (-124.0 <= v <= -114.0):
            raise ValueError('Longitude must be between -124.0 and -114.0')
        return v


def load_preprocessing_artifacts():
    try:
        scaler = joblib.load(os.path.join("artifacts", "preprocessing", "scaler.joblib"))
        imputer = joblib.load(os.path.join("artifacts", "preprocessing", "imputer.joblib"))
        return scaler, imputer
    except Exception as e:
        logger.error(f"Error loading preprocessing artifacts: {str(e)}")
        raise

def load_latest_model():
    try:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        runs = mlflow.search_runs(experiment_ids=['0'])
        latest_run_id = runs.iloc[0]['run_id']
        model_uri = f"runs:/{latest_run_id}/random-forest-model"
        logger.info(f"Loading model from: {model_uri}")
        return mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

model = load_latest_model()
scaler, imputer = load_preprocessing_artifacts()

app = FastAPI(
    title="Real Estate Prediction API",
    version="1.0",
    description="API for predicting California housing prices"
)

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict/")
async def predict(features: HousingFeatures) -> Dict[str, float]:
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([features.dict()])
        
        # Preprocess the input
        input_imputed = imputer.transform(input_df)
        input_scaled = scaler.transform(input_imputed)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        
        # Calculate prediction interval (mock implementation)
        std_dev = np.std(prediction) if hasattr(model, 'estimators_') else 0.1
        lower_bound = prediction[0] - 1.96 * std_dev
        upper_bound = prediction[0] + 1.96 * std_dev
        
        return {
            "predicted_median_house_value": float(prediction[0]),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound)
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))