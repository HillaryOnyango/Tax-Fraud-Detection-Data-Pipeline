from setuptools import setup, find_packages

setup(
    name="tax_fraud_detection",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "sqlalchemy",
        "psycopg2-binary"
    ],
)
