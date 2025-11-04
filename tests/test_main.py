import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.main import app

client = TestClient(app)

@pytest.fixture
def mock_model():
    with patch('src.main.model') as mock:
        mock.predict.return_value = [4.0]  # Mock prediction value
        yield mock

@pytest.fixture
def mock_preprocessing():
    with patch('src.main.scaler') as mock_scaler, \
         patch('src.main.imputer') as mock_imputer:
        mock_scaler.transform.return_value = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]
        mock_imputer.transform.return_value = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]
        yield mock_scaler, mock_imputer

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict_endpoint(mock_model, mock_preprocessing):
    test_input = {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.984127,
        "AveBedrms": 1.023810,
        "Population": 322.0,
        "AveOccup": 2.555556,
        "Latitude": 37.88,
        "Longitude": -122.23
    }
    
    response = client.post("/predict/", json=test_input)
    assert response.status_code == 200
    prediction_response = response.json()
    
    # Check response structure
    assert "predicted_median_house_value" in prediction_response
    assert "lower_bound" in prediction_response
    assert "upper_bound" in prediction_response
    
    # Check if mock was called
    mock_model.predict.assert_called_once()
    mock_preprocessing[0].transform.assert_called_once()  # scaler
    mock_preprocessing[1].transform.assert_called_once()  # imputer

def test_predict_endpoint_invalid_input():
    test_input = {
        "MedInc": -1.0,  # Invalid negative value
        "HouseAge": 41.0,
        "AveRooms": 6.984127,
        "AveBedrms": 1.023810,
        "Population": 322.0,
        "AveOccup": 2.555556,
        "Latitude": 37.88,
        "Longitude": -122.23
    }
    
    response = client.post("/predict/", json=test_input)
    assert response.status_code == 422  # Validation error