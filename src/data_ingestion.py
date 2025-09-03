
import csv
import os
import random
from faker import Faker

# Initialize faker for multiple locales
fake = Faker()

# Common Kenyan names
kenyan_first_names = [
    "Wanjiku", "Kamau", "Njeri", "Odhiambo", "Otieno", "Akinyi", "Kipchoge", "Cherono",
    "Mutua", "Njoroge", "Wambui", "Muthoni", "Gathoni", "Omondi", "Kibet", "Chepkoech",
    "Kiplangat", "Chebet", "Rotich", "Ruto", "Kimani", "Maina", "Nyambura", "Auma",
    "Kariuki", "Wangari", "Odinga", "Kosgei", "Cheruiyot", "Keter", "Kiprop", "Jelimo"
]

kenyan_last_names = [
    "Kamau", "Ochieng", "Wanjala", "Ngugi", "Korir", "Ouma", "Karanja", "Mwangi",
    "Kimutai", "Sang", "Kipkemoi", "Nyong'o", "Owino", "Njuguna", "Kiprono", "Mutai",
    "Rotich", "Kuria", "Ondiek", "Muchiri", "Gitau", "Kibaki", "Kenyatta", "Moi",
    "Kipchoge", "Lagat", "Rudisha", "Kirui", "Kiprotich", "Bett", "Sigei", "Kemboi"
]

# Number of records to generate
NUM_RECORDS = 10000

# CSV file path - save in the src directory
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "tax.csv")

# Possible filing statuses
filing_statuses = ["Single", "Married", "Head of Household"]

# Function to generate a Kenyan name
def generate_kenyan_name():
    return f"{random.choice(kenyan_first_names)} {random.choice(kenyan_last_names)}"

# Function to generate a single record with fraud patterns
def generate_record(taxpayer_id):
    # Base case - normal taxpayer
    fraud_type = random.choices(
        ['none', 'zero_tax', 'negative_deductions', 'income_mismatch', 'excessive_deductions'],
        weights=[0.7, 0.1, 0.05, 0.1, 0.05]  # 70% normal cases, 30% fraudulent
    )[0]
    
    # Generate base income (in Kenyan Shillings)
    if random.random() < 0.05:  # 5% high-income individuals
        income = random.randint(5000000, 50000000)  # 5M to 50M KES
    else:
        income = random.randint(200000, 5000000)    # 200K to 5M KES
    
    if fraud_type == 'none':
        # Normal case
        deductions = random.randint(0, int(min(income * 0.3, 1000000)))  # Maximum 1M KES or 30% of income
        declared_tax = int(income * random.uniform(0.1, 0.3))  # 10-30% tax rate
        paid_tax = declared_tax
        
    elif fraud_type == 'zero_tax':
        # High income but zero tax paid
        income = random.randint(10000000, 50000000)  # 10M to 50M KES
        deductions = random.randint(0, int(income * 0.3))
        declared_tax = int(income * random.uniform(0.1, 0.3))
        paid_tax = 0
        
    elif fraud_type == 'negative_deductions':
        # Negative deductions (impossible case)
        deductions = random.randint(-1000000, -100000)
        declared_tax = int(income * random.uniform(0.1, 0.3))
        paid_tax = declared_tax
        
    elif fraud_type == 'income_mismatch':
        # Declared tax much higher than possible from income
        deductions = random.randint(0, int(income * 0.3))
        declared_tax = int(income * random.uniform(0.4, 0.8))  # Suspicious high tax rate
        paid_tax = declared_tax
        
    else:  # excessive_deductions
        # Deductions higher than income
        deductions = random.randint(income + 100000, income + 1000000)
        declared_tax = int(income * random.uniform(0.1, 0.3))
        paid_tax = declared_tax
    
    return {
        "taxpayer_id": taxpayer_id,
        "name": generate_kenyan_name(),
        "income": income,
        "declared_tax": declared_tax,
        "paid_tax": paid_tax,
        "deductions": deductions,
        "filing_status": random.choice(filing_statuses),
        "year": random.choice([2022, 2023, 2024])
    }

# Generate dataset and save to CSV
def generate_csv():
    with open(OUTPUT_FILE, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "taxpayer_id",
            "name",
            "income",
            "declared_tax",
            "paid_tax",
            "deductions",
            "filing_status",
            "year"
        ])
        writer.writeheader()
        
        for i in range(1, NUM_RECORDS + 1):
            writer.writerow(generate_record(i))
    
    print(f"✅ Generated {NUM_RECORDS} records in {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_csv()

