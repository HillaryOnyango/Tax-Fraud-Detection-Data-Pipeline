# src/anomaly_detection.py

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import joblib
import os

class AnomalyDetector:
    def __init__(self, model_type='isolation_forest', contamination=0.1, eps=0.5, min_samples=5):
        """Initialize the anomaly detector.
        
        Args:
            model_type (str): Type of model to use ('isolation_forest' or 'dbscan')
            contamination (float): Expected proportion of outliers (Isolation Forest only)
            eps (float): DBSCAN neighborhood radius
            min_samples (int): Minimum samples per cluster (DBSCAN)
        """
        self.model_type = model_type
        self.contamination = contamination
        self.eps = eps
        self.min_samples = min_samples
        self.model = None
        self.scaler = StandardScaler()
        
    def get_model(self):
        """Get the appropriate model based on model_type."""
        if self.model_type == 'isolation_forest':
            return IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
        elif self.model_type == 'dbscan':
            return DBSCAN(
                eps=self.eps,
                min_samples=self.min_samples,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def get_feature_columns(self, df):
        """Get numerical and encoded categorical columns for modeling."""
        exclude_cols = ['taxpayer_id', 'name', 'year']
        feature_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in feature_cols if col not in exclude_cols 
                       and not col.startswith('is_') 
                       and not col.startswith('risk_')]
        return feature_cols
        
    def calculate_risk_factors(self, df):
        """Calculate additional risk factors for fraud detection."""
        df = df.copy()
        df = df.replace([np.inf, -np.inf], np.nan)
        
        eps = 1e-10
        df['tax_rate'] = np.where(df['income'] > eps,
                                 df['declared_tax'] / df['income'], 0)
        df['tax_payment_ratio'] = np.where(df['declared_tax'] > eps,
                                         df['paid_tax'] / df['declared_tax'], 0)
        df['deduction_ratio'] = np.where(df['income'] > eps,
                                       df['deductions'] / df['income'], 0)
        
        df['tax_rate'] = df['tax_rate'].clip(0, 1)
        df['tax_payment_ratio'] = df['tax_payment_ratio'].clip(0, 1)
        df['deduction_ratio'] = df['deduction_ratio'].clip(0, 1)
        
        df['zero_tax_high_income'] = (df['paid_tax'] == 0) & (df['income'] > df['income'].median())
        df['high_deductions'] = df['deductions'] > df['income'] * 0.5
        df['negative_deductions'] = df['deductions'] < 0
        df['tax_rate_too_high'] = df['tax_rate'] > 0.35
        
        return df.fillna(0)
    
    def fit(self, df):
        """Fit the anomaly detection model."""
        df = self.calculate_risk_factors(df)
        feature_cols = self.get_feature_columns(df)
        X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(df[feature_cols].median())
        
        for col in X.columns:
            mask = np.abs((X[col] - X[col].mean()) / X[col].std()) < 5
            X[col] = X[col].where(mask, X[col].median())
        
        X_scaled = self.scaler.fit_transform(X)
        self.model = self.get_model()
        if self.model_type == 'isolation_forest':
            self.model.fit(X_scaled)
        else:
            self.model.fit(X_scaled)  # DBSCAN doesn't "train" but we keep for consistency
        self.X_scaled = X_scaled  # store for DBSCAN labels
    
    def predict(self, df):
        """Predict anomalies and add risk scores."""
        results_df = self.calculate_risk_factors(df)
        feature_cols = self.get_feature_columns(results_df)
        X = results_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(results_df[feature_cols].median())
        
        for col in X.columns:
            mask = np.abs((X[col] - X[col].mean()) / X[col].std()) < 5
            X[col] = X[col].where(mask, X[col].median())
        
        X_scaled = self.scaler.transform(X)
        
        if self.model_type == 'isolation_forest':
            predictions = self.model.predict(X_scaled)
            scores = self.model.score_samples(X_scaled)
            results_df['risk_score'] = 1 - (scores - scores.min()) / (scores.max() - scores.min())
            results_df['is_fraud'] = predictions == -1
        
        elif self.model_type == 'dbscan':
            cluster_labels = self.model.fit_predict(X_scaled)
            results_df['cluster'] = cluster_labels
            results_df['is_fraud'] = cluster_labels == -1
            results_df['risk_score'] = results_df['is_fraud'].astype(int)
        
        results_df['risk_factors'] = (
            results_df['zero_tax_high_income'].astype(int) +
            results_df['high_deductions'].astype(int) +
            results_df['negative_deductions'].astype(int) +
            results_df['tax_rate_too_high'].astype(int)
        )
        
        return results_df
    
    def save_model(self, model_dir):
        """Save the model and scaler to disk."""
        os.makedirs(model_dir, exist_ok=True)
        if self.model_type == 'isolation_forest':
            joblib.dump(self.model, os.path.join(model_dir, f"{self.model_type}_model.joblib"))
        joblib.dump(self.scaler, os.path.join(model_dir, f"{self.model_type}_scaler.joblib"))
        
    def load_model(self, model_dir):
        """Load the model and scaler from disk."""
        if self.model_type == 'isolation_forest':
            self.model = joblib.load(os.path.join(model_dir, f"{self.model_type}_model.joblib"))
        self.scaler = joblib.load(os.path.join(model_dir, f"{self.model_type}_scaler.joblib"))

def main():
    input_file = os.path.join(os.path.dirname(__file__), "tax_cleaned.csv")
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    output_file = os.path.join(os.path.dirname(__file__), "tax_predictions.csv")
    
    df = pd.read_csv(input_file)
    
    # Run both Isolation Forest and DBSCAN for comparison
    print("\n🔹 Isolation Forest Results")
    iso = AnomalyDetector(model_type='isolation_forest', contamination=0.1)
    iso.fit(df)
    iso_results = iso.predict(df)
    print(f" - Fraud cases detected: {iso_results['is_fraud'].sum()}")
    
    print("\n🔹 DBSCAN Results")
    db = AnomalyDetector(model_type='dbscan', eps=0.7, min_samples=10)
    db.fit(df)
    db_results = db.predict(df)
    print(f" - Fraud cases detected (noise points): {db_results['is_fraud'].sum()}")
    
    # Save Isolation Forest predictions as main output
    iso_results.to_csv(output_file, index=False)
    iso.save_model(model_dir)
    print(f"\n✅ Predictions saved to {output_file}")

if __name__ == "__main__":
    main()
