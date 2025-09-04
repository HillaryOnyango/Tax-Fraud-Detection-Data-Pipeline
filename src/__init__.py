"""
Tax Fraud Detection Data Pipeline Package

This package contains modules for:
- Data ingestion and generation: Generates synthetic tax data
- Data preprocessing: Cleans and prepares data for analysis
- Anomaly detection: Identifies potential tax fraud cases
"""

# Version and author information
__version__ = '1.0.0'
__author__ = 'Hillary Onyango'

# Import core functionality
try:
    # Import from installed package
    from data_ingestion import generate_csv
    from data_processing import preprocess_data
    from anomaly_detection import AnomalyDetector
except ImportError:
    # Import relative to this file
    from .data_ingestion import generate_csv
    from .data_processing import preprocess_data
    from .anomaly_detection import AnomalyDetector

__version__ = '1.0.0'
__author__ = 'Hillary Onyango'

# Export main components
__all__ = [
    'generate_csv',
    'preprocess_data',
    'AnomalyDetector'
]