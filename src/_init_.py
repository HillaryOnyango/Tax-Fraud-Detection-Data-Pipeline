"""
Tax Fraud Detection Data Pipeline Package

This package contains modules for:
- Data ingestion and generation
- Data preprocessing
- Anomaly detection for tax fraud
"""

try:
    # When used as a package
    from .data_ingestion import generate_csv
    from .data_preprocessing import preprocess_data
    from .anomaly_detection import AnomalyDetector
except ImportError:
    # When run directly
    from data_ingestion import generate_csv
    from data_preprocessing import preprocess_data
    from anomaly_detection import AnomalyDetector

__version__ = '1.0.0'
__author__ = 'Hillary Onyango'

# Export main components
__all__ = [
    'generate_csv',
    'preprocess_data',
    'AnomalyDetector'
]