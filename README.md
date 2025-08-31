# Tax-Fraud-Detection-Data-Pipeline
# 🏦 Tax Fraud Detection Data Pipeline

This project simulates a **real-world tax fraud detection system** by building a scalable **data engineering + machine learning pipeline**.  
It ingests synthetic taxpayer data from multiple sources, cleans & processes it using PySpark, stores it in a PostgreSQL data warehouse, applies **anomaly detection algorithms** to identify suspicious filings, and sends alerts to compliance officers.

---

## 🚀 Features

- **Data Ingestion**
  - CSV extracts (synthetic taxpayer filings)
  - Database extracts (taxpayer master data)
  - Scanned PDF filings (via OCR → structured data)

- **Data Processing**
  - ETL orchestration with **Apache Airflow**
  - Data cleaning & standardization with **PySpark**
  - Warehouse schema in **PostgreSQL**

- **Machine Learning**
  - Unsupervised anomaly detection using **Isolation Forest** and **DBSCAN**
  - Fraud scoring (income/expense ratios, deductions anomalies, underreporting)

- **Alerting**
  - Suspicious filings trigger notifications (email/Slack)

---

## 🏗️ Architecture

![Architecture Diagram](docs/architecture.png)

**Pipeline Flow:**
1. Ingest tax data (CSV, DB, PDFs → OCR) → **Raw Zone (Bronze)**
2. Clean & transform with PySpark → **Clean Zone (Silver)**
3. Load to PostgreSQL Warehouse → **Analytics Zone (Gold)**
4. Apply ML anomaly detection models → flag suspicious filings
5. Send alerts → compliance officers

---

## 📂 Project Structure

