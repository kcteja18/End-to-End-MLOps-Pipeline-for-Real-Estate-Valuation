import os
import pytest # type: ignore
import pandas as pd
import numpy as np
from src.process_data import prepare_data

# @pytest.fixture(scope="module")
# def processed_data():
#     """Fixture to prepare data once for all tests"""
#     prepare_data()
#     train_df = pd.read_csv(os.path.join("data", "processed", "train.csv"))
#     test_df = pd.read_csv(os.path.join("data", "processed", "test.csv"))
#     return train_df, test_df
@pytest.fixture(scope="module")
def processed_data():
    """Fixture to prepare data once for all tests"""
    prepare_data()
    train_df = pd.read_csv(os.path.join("data", "processed", "train_unscaled.csv"))
    test_df = pd.read_csv(os.path.join("data", "processed", "test_unscaled.csv"))
    return train_df, test_df

def test_file_creation():
    """Test if all necessary files are created"""
    prepare_data()
    
    # Check data files
    assert os.path.exists(os.path.join("data", "processed", "train.csv")), "Training data file not created"
    assert os.path.exists(os.path.join("data", "processed", "test.csv")), "Test data file not created"
    
    # Check preprocessing artifacts
    assert os.path.exists(os.path.join("artifacts", "preprocessing", "scaler.joblib")), "Scaler not saved"
    assert os.path.exists(os.path.join("artifacts", "preprocessing", "imputer.joblib")), "Imputer not saved"

def test_data_structure(processed_data):
    """Test the structure and content of processed data"""
    train_df, test_df = processed_data
    
    # Check expected columns
    expected_columns = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                       'Population', 'AveOccup', 'Latitude', 'Longitude', 'MedHouseVal']
    assert all(col in train_df.columns for col in expected_columns), "Missing columns in training data"
    assert all(col in test_df.columns for col in expected_columns), "Missing columns in test data"

def test_data_quality(processed_data):
    """Test the quality of processed data"""
    train_df, test_df = processed_data
    
    # Check for missing values
    assert train_df.isnull().sum().sum() == 0, "Found missing values in training data"
    assert test_df.isnull().sum().sum() == 0, "Found missing values in test data"
    
    # Check for data types
    numeric_columns = train_df.select_dtypes(include=[np.number]).columns
    assert len(numeric_columns) == len(train_df.columns), "Non-numeric columns found"

def test_data_splits(processed_data):
    """Test the train-test split properties"""
    train_df, test_df = processed_data
    
    # Check split sizes (80-20 split)
    total_rows = len(train_df) + len(test_df)
    assert abs(len(train_df)/total_rows - 0.8) < 0.01, "Unexpected train split size"
    assert abs(len(test_df)/total_rows - 0.2) < 0.01, "Unexpected test split size"

# def test_value_ranges(processed_data):
#     """Test if values are within expected ranges"""
#     train_df, test_df = processed_data
    
#     # Test value ranges
#     assert train_df['MedInc'].min() >= 0, "Negative median income found"
#     assert train_df['HouseAge'].min() >= 0, "Negative house age found"
#     assert all(32.0 <= train_df['Latitude']) and all(train_df['Latitude'] <= 42.0), "Latitude out of range"
#     assert all(-124.0 <= train_df['Longitude']) and all(train_df['Longitude'] <= -114.0), "Longitude out of range"
def test_value_ranges(processed_data):
    """Test if values are within expected ranges"""
    train_df, test_df = processed_data
    
    # Test value ranges on unscaled data
    assert train_df['MedInc'].min() >= 0, "Negative median income found"
    assert train_df['HouseAge'].min() >= 0, "Negative house age found"
    assert all(32.0 <= train_df['Latitude']) and all(train_df['Latitude'] <= 42.0), "Latitude out of range"
    assert all(-124.0 <= train_df['Longitude']) and all(train_df['Longitude'] <= -114.0), "Longitude out of range"

if __name__ == "__main__":
    pytest.main([__file__, '-v'])