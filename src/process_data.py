import os
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

def prepare_data():
    """
    This function loads the California Housing dataset, splits it into
    training and testing sets, and saves them to a 'data/processed' directory.
    """
    # Load the dataset
    print("Loading California Housing dataset...")
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    
    # Define the output directory
    output_dir = os.path.join("data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    # Split the data
    print("Splitting data into training and testing sets...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Define file paths
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    # Save the datasets
    print(f"Saving training data to {train_path}")
    train_df.to_csv(train_path, index=False)
    
    print(f"Saving testing data to {test_path}")
    test_df.to_csv(test_path, index=False)
    
    print("Data processing complete.")

if __name__ == "__main__":
    prepare_data()