# src/data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

# Define input and output file paths
RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "tax.csv")
CLEANED_DATA_PATH = os.path.join(os.path.dirname(__file__), "tax_cleaned.csv")

def preprocess_data():
    # Load data
    print("🔹 Loading data...")
    df = pd.read_csv(RAW_DATA_PATH)

    print("\n🔹 Raw Data Preview:")
    print(df.head())
    
    print("\n🔹 Column Names:")
    print(df.columns.tolist())

    # Handle missing values
    print("\n🔹 Handling missing values...")
    df = df.fillna({
        "income": df["income"].median(),
        "deductions": df["deductions"].median(),
        "paid_tax": df["paid_tax"].median(),
        "declared_tax": df["declared_tax"].median(),
        "filing_status": "Single"
    })

    # Add fraud detection features
    print("\n🔹 Adding fraud detection features...")
    df['tax_rate'] = df['declared_tax'] / df['income']
    df['tax_payment_ratio'] = df['paid_tax'] / df['declared_tax']
    df['deduction_ratio'] = df['deductions'] / df['income']
    
    # Scale numeric features
    print("\n🔹 Scaling numeric features...")
    numeric_features = ['income', 'deductions', 'declared_tax', 'paid_tax', 
                       'tax_rate', 'tax_payment_ratio', 'deduction_ratio']
    scaler = MinMaxScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    # One-hot encode categorical features
    print("\n🔹 One-hot encoding categorical features...")
    df = pd.get_dummies(df, columns=['filing_status'])
    
    # Keep original ID and string columns
    id_cols = ['taxpayer_id', 'name', 'year']
    
    # Save cleaned dataset
    print("\n🔹 Saving cleaned dataset...")
    df.to_csv(CLEANED_DATA_PATH, index=False)

    print(f"✅ Cleaned data saved to {CLEANED_DATA_PATH}")
    print("\n🔹 Processed Data Preview:")
    print(df.head())


if __name__ == "__main__":
    preprocess_data()
