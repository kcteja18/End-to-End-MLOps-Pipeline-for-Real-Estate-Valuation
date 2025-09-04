import os
import mlflow # type: ignore
import pandas as pd
from fastapi import FastAPI # type: ignore
from pydantic import BaseModel

# --- DEFINE THE INPUT DATA MODEL ---
# Pydantic model for input data validation.
# These feature names must match the training data columns.
class HousingFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

# --- LOAD THE TRAINED MODEL ---
# This function finds the latest run in MLflow and loads its model.
def load_latest_model():
    
# Point to the MLflow tracking server, same as in train.py
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    
    # Search for runs in the default experiment (experiment_id='0')
    runs = mlflow.search_runs(experiment_ids=['0'])
    
    # Get the latest run's ID
    latest_run_id = runs.iloc[0]['run_id']
    
    # Construct the model URI
    model_uri = f"runs:/{latest_run_id}/random-forest-model"
    
    print(f"Loading model from: {model_uri}")
    return mlflow.sklearn.load_model(model_uri)

model = load_latest_model()

# --- CREATE THE FASTAPI APP ---
app = FastAPI(title="Real Estate Prediction API", version="1.0")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Real Estate Prediction API!"}

@app.post("/predict/")
def predict(features: HousingFeatures):
    """
    Accepts housing features and returns a prediction.
    """
    # Convert input data to a Pandas DataFrame
    input_df = pd.DataFrame([features.model_dump()])
    
    # Make a prediction
    prediction = model.predict(input_df)
    
    # Return the prediction in a JSON response
    return {"predicted_median_house_value": prediction[0]}