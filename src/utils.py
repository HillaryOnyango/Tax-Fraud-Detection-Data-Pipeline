"""
Utility functions for the Tax Fraud Detection Pipeline.
This module contains reusable helper functions for:
- File I/O operations
- Data cleaning and preprocessing
- Model persistence
- Logging setup
"""

import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import joblib
from datetime import datetime

# Configure logging
def setup_logger(name: str) -> logging.Logger:
    """
    Set up a logger with consistent formatting across all modules.
    
    Args:
        name (str): Name of the logger (typically __name__)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:  # Avoid adding handlers multiple times
        logger.setLevel(logging.INFO)
        
        # Create console handler with formatting
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Add file handler
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(
            log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# File I/O Operations
def safe_read_csv(filepath: str, **kwargs) -> pd.DataFrame:
    """
    Safely read a CSV file with error handling.
    
    Args:
        filepath (str): Path to the CSV file
        **kwargs: Additional arguments to pass to pd.read_csv()
    
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    logger = setup_logger(__name__)
    try:
        logger.info(f"Reading CSV file: {filepath}")
        df = pd.read_csv(filepath, **kwargs)
        logger.info(f"Successfully loaded {len(df)} records")
        return df
    except Exception as e:
        logger.error(f"Error reading CSV file {filepath}: {str(e)}")
        raise

def safe_save_csv(df: pd.DataFrame, filepath: str, **kwargs) -> None:
    """
    Safely save a dataframe to CSV with error handling.
    
    Args:
        df (pd.DataFrame): Dataframe to save
        filepath (str): Output file path
        **kwargs: Additional arguments to pass to df.to_csv()
    """
    logger = setup_logger(__name__)
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        logger.info(f"Saving data to: {filepath}")
        df.to_csv(filepath, **kwargs)
        logger.info(f"Successfully saved {len(df)} records")
    except Exception as e:
        logger.error(f"Error saving CSV file {filepath}: {str(e)}")
        raise

# Data Cleaning Functions
def handle_missing_values(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    """
    Handle missing values in a dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        strategy (str): Strategy for handling missing values 
                       ('median', 'mean', 'mode', 'zero', or 'drop')
    
    Returns:
        pd.DataFrame: Dataframe with handled missing values
    """
    df_clean = df.copy()
    
    if strategy == 'drop':
        return df_clean.dropna()
    
    # Handle numeric columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    if strategy == 'median':
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
    elif strategy == 'mean':
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
    elif strategy == 'zero':
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(0)
    
    # Handle categorical columns
    categorical_cols = df_clean.select_dtypes(exclude=[np.number]).columns
    df_clean[categorical_cols] = df_clean[categorical_cols].fillna(df_clean[categorical_cols].mode().iloc[0])
    
    return df_clean

def scale_numeric_features(df: pd.DataFrame, columns: list = None) -> tuple:
    """
    Scale numeric features using StandardScaler.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (list): List of columns to scale. If None, scales all numeric columns
    
    Returns:
        tuple: (scaled_df, scaler) - Scaled dataframe and fitted scaler
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[columns] = scaler.fit_transform(df[columns])
    
    return df_scaled, scaler

# Encoding Functions
def encode_categorical_features(df: pd.DataFrame, columns: list = None, 
                             method: str = 'onehot') -> tuple:
    """
    Encode categorical features using specified method.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (list): List of columns to encode. If None, encodes all categorical columns
        method (str): Encoding method ('onehot' or 'label')
    
    Returns:
        tuple: (encoded_df, encoder) - Encoded dataframe and fitted encoder
    """
    if columns is None:
        columns = df.select_dtypes(include=['object', 'category']).columns
    
    df_encoded = df.copy()
    
    if method == 'onehot':
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoded_features = encoder.fit_transform(df[columns])
        feature_names = encoder.get_feature_names_out(columns)
        
        # Replace original columns with encoded ones
        df_encoded = df_encoded.drop(columns=columns)
        for i, col in enumerate(feature_names):
            df_encoded[col] = encoded_features[:, i]
            
    elif method == 'label':
        encoder = LabelEncoder()
        for col in columns:
            df_encoded[col] = encoder.fit_transform(df[col])
    
    return df_encoded, encoder

# Model Persistence Functions
def save_model(model, filepath: str, scaler=None) -> None:
    """
    Save a model and its scaler to disk.
    
    Args:
        model: Trained model instance
        filepath (str): Path to save the model
        scaler: Optional fitted scaler instance
    """
    logger = setup_logger(__name__)
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        model_path = filepath
        joblib.dump(model, model_path)
        logger.info(f"Model saved to: {model_path}")
        
        # Save scaler if provided
        if scaler is not None:
            scaler_path = filepath.replace('.joblib', '_scaler.joblib')
            joblib.dump(scaler, scaler_path)
            logger.info(f"Scaler saved to: {scaler_path}")
            
    except Exception as e:
        logger.error(f"Error saving model to {filepath}: {str(e)}")
        raise

def load_model(filepath: str, load_scaler: bool = False) -> tuple:
    """
    Load a model and optionally its scaler from disk.
    
    Args:
        filepath (str): Path to the saved model
        load_scaler (bool): Whether to load the associated scaler
    
    Returns:
        tuple: (model, scaler) if load_scaler=True, else just model
    """
    logger = setup_logger(__name__)
    try:
        # Load model
        model = joblib.load(filepath)
        logger.info(f"Model loaded from: {filepath}")
        
        if load_scaler:
            scaler_path = filepath.replace('.joblib', '_scaler.joblib')
            scaler = joblib.load(scaler_path)
            logger.info(f"Scaler loaded from: {scaler_path}")
            return model, scaler
        
        return model
        
    except Exception as e:
        logger.error(f"Error loading model from {filepath}: {str(e)}")
        raise
