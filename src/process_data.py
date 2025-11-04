# import os
# import pandas as pd
# from sklearn.datasets import fetch_california_housing
# from sklearn.model_selection import train_test_split

# def prepare_data():
#     """
#     This function loads the California Housing dataset, splits it into
#     training and testing sets, and saves them to a 'data/processed' directory.
#     """
#     # Load the dataset
#     print("Loading California Housing dataset...")
#     housing = fetch_california_housing(as_frame=True)
#     df = housing.frame
    
#     # Define the output directory
#     output_dir = os.path.join("data", "processed")
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Split the data
#     print("Splitting data into training and testing sets...")
#     train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
#     # Define file paths
#     train_path = os.path.join(output_dir, "train.csv")
#     test_path = os.path.join(output_dir, "test.csv")
    
#     # Save the datasets
#     print(f"Saving training data to {train_path}")
#     train_df.to_csv(train_path, index=False)
    
#     print(f"Saving testing data to {test_path}")
#     test_df.to_csv(test_path, index=False)
    
#     print("Data processing complete.")

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
    
    # Handle outliers using IQR method
    def remove_outliers(df):
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        return df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
    
    df = remove_outliers(df)
    
    # Split the data
    print("Splitting data into training and testing sets...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Initialize preprocessors
    scaler = StandardScaler()
    imputer = SimpleImputer(strategy='median')
    
    # Fit and transform training data
    X_train = train_df.drop("MedHouseVal", axis=1)
    X_test = test_df.drop("MedHouseVal", axis=1)
    
    X_train_imputed = imputer.fit_transform(X_train)
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    
    # Transform test data
    X_test_imputed = imputer.transform(X_test)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # Save preprocessors
    joblib.dump(scaler, os.path.join(preprocess_dir, "scaler.joblib"))
    joblib.dump(imputer, os.path.join(preprocess_dir, "imputer.joblib"))
    
    # Reconstruct DataFrames
    X_train_processed = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_processed = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    # Add target variable back
    train_df_processed = X_train_processed.copy()
    test_df_processed = X_test_processed.copy()
    train_df_processed['MedHouseVal'] = train_df['MedHouseVal']
    test_df_processed['MedHouseVal'] = test_df['MedHouseVal']
    
    # Save processed datasets
    output_dir = os.path.join("data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    train_df_processed.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_df_processed.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    print("Data processing complete.")

if __name__ == "__main__":
    prepare_data()