# import os
# import pandas as pd
# import mlflow
# import mlflow.sklearn
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# def train_model():
#     """
#     This function trains a model on the processed data, evaluates it,
#     and logs all results to MLflow.
#     """
#     # Set the MLflow tracking URI. By default, it saves to a local 'mlruns' directory.
#     # You can also set this to a remote server.
#     # mlflow.set_tracking_uri("http://127.0.0.1:5000")
#     # NEW, CORRECTED LINE
#     mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
#     # Start an MLflow run
#     with mlflow.start_run():
#         # --- 1. LOAD DATA ---
#         train_path = os.path.join("data", "processed", "train.csv")
#         test_path = os.path.join("data", "processed", "test.csv")
#         train_df = pd.read_csv(train_path)
#         test_df = pd.read_csv(test_path)

#         # Prepare data for Scikit-learn
#         X_train = train_df.drop("MedHouseVal", axis=1)
#         y_train = train_df["MedHouseVal"]
#         X_test = test_df.drop("MedHouseVal", axis=1)
#         y_test = test_df["MedHouseVal"]

#         # --- 2. DEFINE & LOG PARAMETERS ---
#         n_estimators = 500
#         max_depth = 15
#         mlflow.log_param("n_estimators", n_estimators)
#         mlflow.log_param("max_depth", max_depth)
        
#         # --- 3. TRAIN THE MODEL ---
#         print("Training RandomForestRegressor model...")
#         rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
#         rf.fit(X_train, y_train)

#         # --- 4. EVALUATE & LOG METRICS ---
#         print("Evaluating model...")
#         predictions = rf.predict(X_test)
        
#         mse = mean_squared_error(y_test, predictions)
#         mae = mean_absolute_error(y_test, predictions)
#         r2 = r2_score(y_test, predictions)

#         print(f"  MSE: {mse}")
#         print(f"  MAE: {mae}")
#         print(f"  R2 Score: {r2}")
        
#         mlflow.log_metric("mse", mse)
#         mlflow.log_metric("mae", mae)
#         mlflow.log_metric("r2", r2)
        
#         # --- 5. LOG THE MODEL ---
#         print("Logging model to MLflow...")
#         mlflow.sklearn.log_model(rf, "random-forest-model")
        
#         print("Training run complete.")

# if __name__ == "__main__":
#     train_model()

import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import RandomizedSearchCV
import joblib

def train_model():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    with mlflow.start_run() as run:
        # Load data
        train_path = os.path.join("data", "processed", "train.csv")
        test_path = os.path.join("data", "processed", "test.csv")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        X_train = train_df.drop("MedHouseVal", axis=1)
        y_train = train_df["MedHouseVal"]
        X_test = test_df.drop("MedHouseVal", axis=1)
        y_test = test_df["MedHouseVal"]

        # Hyperparameter search space
        param_space = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [10, 20, 30, 40, 50, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt'],
            'bootstrap': [True, False]
        }

        # Initialize model
        rf = RandomForestRegressor(random_state=42)

        # Random search with cross-validation
        print("Performing hyperparameter tuning...")
        random_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_space,
            n_iter=20,
            cv=5,
            random_state=42,
            n_jobs=-1,
            verbose=1,
            scoring='neg_mean_squared_error'
        )
        
        random_search.fit(X_train, y_train)
        
        # Log best parameters
        print("Best parameters found:")
        for param, value in random_search.best_params_.items():
            print(f"{param}: {value}")
            mlflow.log_param(param, value)

        # Get best model
        best_model = random_search.best_estimator_

        # Perform k-fold cross-validation
        print("Performing cross-validation...")
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='r2')
        
        # Log cross-validation results
        mlflow.log_metric("cv_mean_r2", cv_scores.mean())
        mlflow.log_metric("cv_std_r2", cv_scores.std())

        # Final evaluation on test set
        print("Evaluating on test set...")
        predictions = best_model.predict(X_test)
        
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        print(f"Test set metrics:")
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  R2 Score: {r2}")
        
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_r2", r2)
        
        # Log feature importances
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_importance.to_csv("feature_importance.csv", index=False)
        mlflow.log_artifact("feature_importance.csv")
        
        # Log the model
        print("Logging model to MLflow...")
        mlflow.sklearn.log_model(best_model, "random-forest-model")
        
        # Save the model locally as well
        model_dir = os.path.join("artifacts", "models")
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(best_model, os.path.join(model_dir, "random_forest_model.joblib"))
        
        print("Training complete.")

if __name__ == "__main__":
    train_model()