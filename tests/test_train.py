import os
import pytest # type: ignore
import mlflow # type: ignore
import numpy as np
from unittest.mock import patch, MagicMock
from src.train import train_model

@pytest.fixture
def mock_mlflow():
    """Fixture to mock MLflow operations"""
    with patch('mlflow.start_run') as mock_run, \
         patch('mlflow.log_metric') as mock_log_metric, \
         patch('mlflow.log_param') as mock_log_param, \
         patch('mlflow.sklearn.log_model') as mock_log_model:
        yield {
            'run': mock_run,
            'log_metric': mock_log_metric,
            'log_param': mock_log_param,
            'log_model': mock_log_model
        }

@pytest.fixture
def mock_data():
    """Fixture to mock training data"""
    with patch('pandas.read_csv') as mock_read:
        # Create mock training and test data
        mock_data = MagicMock()
        mock_data.drop.return_value = np.random.rand(100, 8)
        mock_data['MedHouseVal'] = np.random.rand(100)
        mock_read.return_value = mock_data
        yield mock_read

def test_model_artifacts_creation():
    """Test if model artifacts are created correctly"""
    train_model()
    
    # Check model file
    model_path = os.path.join("artifacts", "models", "random_forest_model.joblib")
    assert os.path.exists(model_path), "Model file not created"
    
    # Check feature importance file
    assert os.path.exists("feature_importance.csv"), "Feature importance file not created"

def test_mlflow_logging(mock_mlflow, mock_data):
    """Test if MLflow logging works correctly"""
    train_model()
    
    # Verify MLflow operations were called
    assert mock_mlflow['run'].called, "MLflow run not started"
    assert mock_mlflow['log_metric'].called, "Metrics not logged"
    assert mock_mlflow['log_param'].called, "Parameters not logged"
    assert mock_mlflow['log_model'].called, "Model not logged"

def test_model_metrics():
    """Test if model metrics are within acceptable ranges"""
    train_model()
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    runs = mlflow.search_runs(experiment_ids=['0'])
    assert len(runs) > 0, "No MLflow runs found"
    
    latest_run = runs.iloc[0]
    
    # Check presence of required metrics
    required_metrics = ['test_rmse', 'test_mae', 'test_r2', 'cv_mean_r2']
    for metric in required_metrics:
        assert metric in latest_run.keys(), f"Missing metric: {metric}"
    
    # Validate metric values
    assert 0 <= latest_run['test_r2'] <= 1, "R2 score out of valid range"
    assert latest_run['test_rmse'] >= 0, "RMSE cannot be negative"
    assert latest_run['test_mae'] >= 0, "MAE cannot be negative"
    assert 0 <= latest_run['cv_mean_r2'] <= 1, "CV R2 score out of valid range"

@pytest.mark.parametrize("metric", [
    'test_rmse',
    'test_mae',
    'test_r2',
    'cv_mean_r2'
])
def test_individual_metrics(metric):
    """Test each metric individually"""
    train_model()
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    runs = mlflow.search_runs(experiment_ids=['0'])
    latest_run = runs.iloc[0]
    
    assert metric in latest_run.keys(), f"Metric {metric} not found"
    assert not np.isnan(latest_run[metric]), f"Metric {metric} is NaN"
    assert latest_run[metric] is not None, f"Metric {metric} is None"

if __name__ == "__main__":
    pytest.main([__file__, '-v'])