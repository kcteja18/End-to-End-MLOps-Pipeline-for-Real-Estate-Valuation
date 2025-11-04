
# import os
# import pandas as pd
# import numpy as np
# from sklearn.datasets import fetch_california_housing
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer
# import joblib

# def prepare_data():
#     """
#     Data preparation with preprocessing pipeline
#     """
#     print("Loading California Housing dataset...")
#     housing = fetch_california_housing(as_frame=True)
#     df = housing.frame
#     # Data validation and cleaning
#     def clean_data(df):
#         # Handle negative values
#         df['MedInc'] = df['MedInc'].clip(lower=0)  # Set negative values to 0
#         df['HouseAge'] = df['HouseAge'].clip(lower=0)
#         df['Population'] = df['Population'].clip(lower=0)
#         df['AveOccup'] = df['AveOccup'].clip(lower=0)
        
#         # Validate latitude and longitude ranges for California
#         df = df[
#             (df['Latitude'] >= 32.0) & 
#             (df['Latitude'] <= 42.0) & 
#             (df['Longitude'] >= -124.0) & 
#             (df['Longitude'] <= -114.0)
#         ]
        
#         return df
    
#     df = clean_data(df)
    
#     # Create preprocessing directory
#     preprocess_dir = os.path.join("artifacts", "preprocessing")
#     os.makedirs(preprocess_dir, exist_ok=True)
    
#     # Handle outliers using IQR method
#     def remove_outliers(df):
#         Q1 = df.quantile(0.25)
#         Q3 = df.quantile(0.75)
#         IQR = Q3 - Q1
#         return df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
    
#     df = remove_outliers(df)
    
#     # Split the data
#     print("Splitting data into training and testing sets...")
#     train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
#     # Initialize preprocessors
#     scaler = StandardScaler()
#     imputer = SimpleImputer(strategy='median')
    
#     # Fit and transform training data
#     X_train = train_df.drop("MedHouseVal", axis=1)
#     X_test = test_df.drop("MedHouseVal", axis=1)
    
#     X_train_imputed = imputer.fit_transform(X_train)
#     X_train_scaled = scaler.fit_transform(X_train_imputed)
    
#     # Transform test data
#     X_test_imputed = imputer.transform(X_test)
#     X_test_scaled = scaler.transform(X_test_imputed)
    
#     # Save preprocessors
#     joblib.dump(scaler, os.path.join(preprocess_dir, "scaler.joblib"))
#     joblib.dump(imputer, os.path.join(preprocess_dir, "imputer.joblib"))
    
#     # Reconstruct DataFrames
#     X_train_processed = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
#     X_test_processed = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
#     # Add target variable back
#     train_df_processed = X_train_processed.copy()
#     test_df_processed = X_test_processed.copy()
#     train_df_processed['MedHouseVal'] = train_df['MedHouseVal']
#     test_df_processed['MedHouseVal'] = test_df['MedHouseVal']
    
#     # Save processed datasets
#     output_dir = os.path.join("data", "processed")
#     os.makedirs(output_dir, exist_ok=True)
    
#     train_df_processed.to_csv(os.path.join(output_dir, "train.csv"), index=False)
#     test_df_processed.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
#     print("Data processing complete.")

# if __name__ == "__main__":
#     prepare_data()

import os
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

def prepare_data():
    """
    Data preparation with preprocessing pipeline
    """
    print("Loading California Housing dataset...")
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    
    # Create preprocessing directory
    preprocess_dir = os.path.join("artifacts", "preprocessing")
    os.makedirs(preprocess_dir, exist_ok=True)
    
    # Data validation before scaling
    df['MedInc'] = df['MedInc'].clip(lower=0)
    df['HouseAge'] = df['HouseAge'].clip(lower=0)
    df['Population'] = df['Population'].clip(lower=0)
    df['AveOccup'] = df['AveOccup'].clip(lower=0)
    # Filter geographic bounds for California
    df = df[
        (df['Latitude'] >= 32.0) & 
        (df['Latitude'] <= 42.0) & 
        (df['Longitude'] >= -124.0) & 
        (df['Longitude'] <= -114.0)
    ].copy()
    
    if len(df) == 0:
        raise ValueError("No data points remain after geographic filtering")
    
    print(f"Retained {len(df)} records after geographic filtering")
    # Split the data before scaling
    print("Splitting data into training and testing sets...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Save unscaled versions for testing
    train_df.to_csv(os.path.join("data", "processed", "train_unscaled.csv"), index=False)
    test_df.to_csv(os.path.join("data", "processed", "test_unscaled.csv"), index=False)
    
    # Initialize preprocessors
    scaler = StandardScaler()
    imputer = SimpleImputer(strategy='median')
    
    # Scale the features
    X_train = train_df.drop("MedHouseVal", axis=1)
    X_test = test_df.drop("MedHouseVal", axis=1)
    
    X_train_imputed = imputer.fit_transform(X_train)
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    
    X_test_imputed = imputer.transform(X_test)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # Save preprocessors
    joblib.dump(scaler, os.path.join(preprocess_dir, "scaler.joblib"))
    joblib.dump(imputer, os.path.join(preprocess_dir, "imputer.joblib"))
    
    # Create final DataFrames
    train_df_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    test_df_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    train_df_scaled['MedHouseVal'] = train_df['MedHouseVal']
    test_df_scaled['MedHouseVal'] = test_df['MedHouseVal']
    
    # Save scaled versions
    train_df_scaled.to_csv(os.path.join("data", "processed", "train.csv"), index=False)
    test_df_scaled.to_csv(os.path.join("data", "processed", "test.csv"), index=False)
    
    print("Data processing complete.")
    return train_df_scaled, test_df_scaled

if __name__ == "__main__":
    prepare_data()