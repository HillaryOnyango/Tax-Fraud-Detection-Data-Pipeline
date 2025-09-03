# src/anomaly_detection.py

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

class AnomalyDetector:
    def __init__(self, model_type='isolation_forest', contamination=0.1):
        """Initialize the anomaly detector with optional supervised fraud detection."""
        self.model_type = model_type
        self.contamination = contamination
        self.model = None
        self.scaler = StandardScaler()
        self.supervised_model = None  # For optional fraud classification

    def get_model(self):
        if self.model_type == 'isolation_forest':
            return IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
        raise ValueError(f"Unknown model type: {self.model_type}")

    def get_feature_columns(self, df):
        exclude_cols = ['taxpayer_id', 'name', 'year']
        feature_cols = df.select_dtypes(include=[np.number]).columns
        return [col for col in feature_cols if col not in exclude_cols]

    def calculate_risk_factors(self, df):
        """Adds engineered fraud risk features."""
        df = df.copy()
        df = df.replace([np.inf, -np.inf], np.nan)

        eps = 1e-10
        df['tax_rate'] = np.where(df['income'] > eps,
                                  df['declared_tax'] / (df['income'] + eps), 0)
        df['tax_payment_ratio'] = np.where(df['declared_tax'] > eps,
                                           df['paid_tax'] / (df['declared_tax'] + eps), 0)
        df['deduction_ratio'] = np.where(df['income'] > eps,
                                         df['deductions'] / (df['income'] + eps), 0)

        # Clip ratios
        df['tax_rate'] = df['tax_rate'].clip(0, 1)
        df['tax_payment_ratio'] = df['tax_payment_ratio'].clip(0, 1)
        df['deduction_ratio'] = df['deduction_ratio'].clip(0, 1)

        # Fraud red flags
        df['zero_tax_high_income'] = (df['paid_tax'] == 0) & (df['income'] > df['income'].median())
        df['high_deductions'] = df['deductions'] > df['income'] * 0.5
        df['negative_deductions'] = df['deductions'] < 0
        df['tax_rate_too_high'] = df['tax_rate'] > 0.35

        return df.fillna(0)

    def fit(self, df):
        df = self.calculate_risk_factors(df)
        feature_cols = self.get_feature_columns(df)
        X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(df.median(numeric_only=True))

        X_scaled = self.scaler.fit_transform(X)
        self.model = self.get_model()
        self.model.fit(X_scaled)

        # Fake fraud labels for supervised model (optional)
        df['fraud_label'] = (
            (df['zero_tax_high_income']) |
            (df['high_deductions']) |
            (df['negative_deductions']) |
            (df['tax_rate_too_high'])
        ).astype(int)

        self.supervised_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.supervised_model.fit(X_scaled, df['fraud_label'])

    def predict(self, df):
        results_df = self.calculate_risk_factors(df)
        feature_cols = self.get_feature_columns(results_df)
        X = results_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(results_df.median(numeric_only=True))

        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        scores = self.model.score_samples(X_scaled)

        results_df['risk_score'] = 1 - (scores - scores.min()) / (scores.max() - scores.min())
        results_df['is_fraud'] = predictions == -1

        if self.supervised_model:
            results_df['fraud_probability'] = self.supervised_model.predict_proba(X_scaled)[:, 1]

        return results_df

    def save_model(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(model_dir, f"{self.model_type}_model.joblib"))
        joblib.dump(self.scaler, os.path.join(model_dir, f"{self.model_type}_scaler.joblib"))
        if self.supervised_model:
            joblib.dump(self.supervised_model, os.path.join(model_dir, "fraud_classifier.joblib"))

    def load_model(self, model_dir):
        self.model = joblib.load(os.path.join(model_dir, f"{self.model_type}_model.joblib"))
        self.scaler = joblib.load(os.path.join(model_dir, f"{self.model_type}_scaler.joblib"))
        fraud_clf = os.path.join(model_dir, "fraud_classifier.joblib")
        if os.path.exists(fraud_clf):
            self.supervised_model = joblib.load(fraud_clf)

def main():
    input_file = os.path.join(os.path.dirname(__file__), "tax_cleaned.csv")
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    output_file = os.path.join(os.path.dirname(__file__), "tax_predictions.csv")

    print("🔹 Loading cleaned data...")
    df = pd.read_csv(input_file)

    print("\n🔹 Training anomaly & fraud detector...")
    detector = AnomalyDetector()
    detector.fit(df)

    print("\n🔹 Predicting anomalies & fraud risk...")
    results = detector.predict(df)

    detector.save_model(model_dir)
    results.to_csv(output_file, index=False)

    n_fraud = results['is_fraud'].sum()
    print(f"\n✅ Processed {len(results)} records, flagged {n_fraud} as potential fraud.")
    print("\n🔹 Top 5 risky cases:")
    for _, case in results.nlargest(5, 'risk_score').iterrows():
        flags = []
        if case['zero_tax_high_income']: flags.append("High income but zero tax")
        if case['high_deductions']: flags.append("Excessive deductions")
        if case['negative_deductions']: flags.append("Negative deductions")
        if case['tax_rate_too_high']: flags.append("Unusually high tax rate")

        print(f"   - {case['name']} | Risk={case['risk_score']:.2f}, Fraud Probability={case.get('fraud_probability', 0):.2f}")
        print(f"     Income: {case['income']:,.0f}, Declared Tax: {case['declared_tax']:,.0f}, Paid Tax: {case['paid_tax']:,.0f}, Deductions: {case['deductions']:,.0f}")
        print(f"     Flags: {', '.join(flags) if flags else 'Multiple anomalous patterns'}\n")

if __name__ == "__main__":
    main()
