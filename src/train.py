import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_model():
    """
    This function trains a model on the processed data, evaluates it,
    and logs all results to MLflow.
    """
    # Set the MLflow tracking URI. By default, it saves to a local 'mlruns' directory.
    # You can also set this to a remote server.
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    # NEW, CORRECTED LINE
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    # Start an MLflow run
    with mlflow.start_run():
        # --- 1. LOAD DATA ---
        train_path = os.path.join("data", "processed", "train.csv")
        test_path = os.path.join("data", "processed", "test.csv")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # Prepare data for Scikit-learn
        X_train = train_df.drop("MedHouseVal", axis=1)
        y_train = train_df["MedHouseVal"]
        X_test = test_df.drop("MedHouseVal", axis=1)
        y_test = test_df["MedHouseVal"]

        # --- 2. DEFINE & LOG PARAMETERS ---
        n_estimators = 500
        max_depth = 15
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        # --- 3. TRAIN THE MODEL ---
        print("Training RandomForestRegressor model...")
        rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        rf.fit(X_train, y_train)

        # --- 4. EVALUATE & LOG METRICS ---
        print("Evaluating model...")
        predictions = rf.predict(X_test)
        
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        print(f"  MSE: {mse}")
        print(f"  MAE: {mae}")
        print(f"  R2 Score: {r2}")
        
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        # --- 5. LOG THE MODEL ---
        print("Logging model to MLflow...")
        mlflow.sklearn.log_model(rf, "random-forest-model")
        
        print("Training run complete.")

if __name__ == "__main__":
    train_model()