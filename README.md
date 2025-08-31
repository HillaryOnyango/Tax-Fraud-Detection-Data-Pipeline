
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




<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/c6246654-9fdd-4141-9fd0-c811eb1664bc" />



**Pipeline Flow:**
1. Ingest tax data (CSV, DB, PDFs → OCR) → **Raw Zone (Bronze)**
2. Clean & transform with PySpark → **Clean Zone (Silver)**
3. Load to PostgreSQL Warehouse → **Analytics Zone (Gold)**
4. Apply ML anomaly detection models → flag suspicious filings
5. Send alerts → compliance officers

---

## 📂 Project Structure

tax-fraud-detection-pipeline/

├── dags/ # Airflow DAG

├── jobs/ # ETL + ML jobs

├── data/ # Raw/Silver/Warehouse data

├── sql/ # Warehouse schema

├── docker/ # Infra setup

├── notebooks/ # EDA & ML experiments

├── docs/ # Documentation & diagrams

├── tests/ # Data quality tests

└── README.md


---

## ⚙️ Tech Stack

- **Orchestration:** Apache Airflow  
- **Processing:** PySpark, Pandas  
- **Storage/Warehouse:** PostgreSQL  
- **Machine Learning:** scikit-learn (Isolation Forest, DBSCAN)  
- **OCR:** Tesseract OCR (`pytesseract`, `pdf2image`)  
- **Containerization:** Docker Compose  

---

## 📊 Synthetic Data

Since real taxpayer data is confidential, the pipeline uses **synthetic data** generated with Python’s [Faker](https://faker.readthedocs.io/).  
Fraudulent cases are injected intentionally (e.g., deductions > income, unusually low tax paid) to test anomaly detection.

- Example schema (`synthetic_tax_filings.csv`):
  - `taxpayer_id`, `name`, `national_id`, `industry_code`
  - `tax_year`, `declared_income`, `claimed_expenses`, `deductions`, `tax_paid`
  - `fraud_flag` (for validation only)


## 📦 Setup & Run

### 1. Clone Repo
```bash
git clone https://github.com/HillarOnyango/Tax-Fraud-Detection-Pipeline
cd Tax-Fraud-Detection-Pipeline

**2. Start Environment (Docker Compose)**
cd docker
docker-compose up -d

This will spin up:

Airflow webserver & scheduler

PostgreSQL

Spark (optional, or run PySpark locally)



**3. Initialize Airflow**

docker exec -it airflow-webserver airflow db init

docker exec -it airflow-webserver airflow users create \

--username admin --password admin --role Admin --firstname First --lastname Last --email admin@example.com

**4. Generate Fake Data**

python jobs/data_generator.py

**5. Run Pipeline**

Open Airflow UI at: http://localhost:8080

Trigger tax_fraud_detection_pipeline DAG

**🔍 Example Output**

Warehouse (Postgres):

fact_tax_filing: Cleaned taxpayer filings

fact_fraud_score: Fraud detection scores

**Alerts:**

JSON payload with top suspicious cases

Email/Slack notifications

Example alert:

{
  "date": "2025-09-01",
  "suspicious_filings": [
    {"filing_id": 91822, "taxpayer_id": 12345, "score": 0.93, "exp_ratio": 2.1, "ded_ratio": 1.4}
  ]
}

**🧪 Testing**

Data Quality Checks:

Non-negative income

Deductions ≤ 150% income

Unique (taxpayer_id, tax_year)

Run tests:
pytest tests/


📌** Next Steps
**
Add dashboarding with Apache Superset / Metabase

Deploy to cloud (AWS/GCP/Azure)

Integrate model monitoring (drift detection, retraining)

Add CI/CD for Airflow DAGs + tests

**📜 License

MIT License – free to use and modify.**



---

# 3. Documentation Outline (`/docs/`)

Inside `docs/`, include:

1. **`design.md`**  
   - Data pipeline design (bronze/silver/gold layers)  
   - Airflow DAG task descriptions  
   - Data flow diagrams  

2. **`data_dictionary.md`**  
   - Table-by-table explanation (dim_taxpayer, fact_tax_filing, fact_fraud_score)  
   - Field descriptions & types  

3. **`architecture.png`**  
   - A simple diagram showing ingestion → ETL → warehouse → ML → alerts  

4. **`setup.md`**  
   - Step-by-step environment setup (Docker Compose, local dev)  
   - Airflow + Postgres config  

---

⚡ With this, you’ll have a **portfolio-ready project**: clear structure, professional README, and good documentation.  

Do you want me to also **generate the architecture diagram (`architecture.png`)** so you can drop it into the docs folder?
