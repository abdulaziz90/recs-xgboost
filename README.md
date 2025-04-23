# RECS 2020 Electricity Usage Prediction


## 1. Project Overview

The U.S. Residential Energy Consumption Survey (RECS) is a rich source of data on housing characteristics and household energy usage. Accurately modeling and predicting a household’s annual electricity demand can help utilities, policymakers, and home‑energy auditors identify efficiency opportunities, plan capacity, and design targeted conservation programs.

This project presents an end‑to‑end pipeline that:

1. **Ingests** raw RECS 2020 data from the EIA website into Amazon S3
2. **Explores** the dataset with interactive EDA
3. **Engineers** leakage‑safe features and partitions into train/test splits
4. **Trains** an XGBoost regressor both locally and in SageMaker
5. **Evaluates** model performance (RMSE, feature importance, SHAP)
6. **Serves** batch predictions via a Flask HTTP API

---

## 3. Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Directory Structure](#directory-structure)
4. [Notebooks](#notebooks)
5. [Scripts](#scripts)
6. [Usage Examples](#usage-examples)
7. [Contributing](#contributing)
8. [License](#license)

---

## 4. Prerequisites

- **AWS Account** with permissions for SageMaker & S3
- **Python 3.9+**
- `git`, `curl` or `wget`

---

## 5. Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/your‑org/recs‑xgboost.git
   cd recs-xgboost
   ```

2. **Conda environment**

   ```bash
   conda env create -f environment.yml
   conda activate recs-prediction
   ```

   *Or* with **pip**:

   ```bash
   pip install -r requirements.txt
   ```

---

## 6. Directory Structure

```text
├── notebooks/
│   ├── 01_data_ingest.ipynb       # Download & upload RECS CSV to S3
│   ├── 02_eda.ipynb               # Exploratory Data Analysis
│   ├── 03_features.ipynb          # Feature engineering & train/test split
│   ├── 04_train.ipynb             # Local XGBoost training & evaluation
│   ├── 04_train-cloud.ipynb       # SageMaker XGBoost training job
│   └── 05_deploy.ipynb            # Batch inference against Flask endpoint
├── train.py                       # Entry point for SageMaker training
├── serve_flask.py                 # Flask app for model serving
├── requirements.txt               # pip dependencies
├── environment.yml                # conda environment spec
└── README.md                      # This file
```

---

## 7. Notebooks

### 01_data_ingest.ipynb  
**Ingest RECS 2020 CSV into S3**  
Downloads the official CSV, initializes an AWS SageMaker session, and uploads the raw file to your S3 bucket under `recs/`.

### 02_eda.ipynb  
**Exploratory Data Analysis**  
Loads the ingested CSV from S3, inspects schema and distributions, and computes summary statistics to understand data quality and variable ranges.

### 03_features.ipynb  
**Feature Engineering & Splits**  
Defines the regression target (`KWH`), filters out identifiers and leakage columns, one-hot encodes categorical variables, computes feature–target correlations, and writes Parquet train/test splits back to S3.

### 04_train.ipynb  
**Local XGBoost Training & Evaluation**  
Trains an XGBoost model locally with early stopping, calculates RMSE on the hold-out test set, generates feature importance and SHAP plots, and packages the model for S3 upload.

### 04_train-cloud.ipynb  
**Cloud Training with SageMaker**  
Launches a managed SageMaker training job using the built-in XGBoost container, monitors metrics, then downloads and extracts the resulting model artifact for in-notebook inference.

### 05_deploy.ipynb  
**Batch Inference Demo**  
Loads a small sample of the test set from S3, serializes it to JSON, calls the Flask `/predict` endpoint, and prints the predicted electricity usages.

---

## 8. Scripts

### train.py  
Entry point for SageMaker’s XGBoost container.  
- Reads Parquet files from `/opt/ml/input/data/train`  
- Parses hyperparameters via `argparse`  
- Trains with early stopping  
- Writes `model.bst` to `$SM_MODEL_DIR`

### serve_flask.py  
Lightweight Flask application exposing a `POST /predict` endpoint:  
- Loads `model.bst` at startup  
- Accepts JSON batches of feature dicts  
- Returns back-transformed predictions
  
---

## 9. Deployment & Inference Example

This example demonstrates how to serve the trained model with Flask and make a batch prediction request.

```bash
# 1) Run the Flask Server
python serve_flask.py
# By default, the server listens on http://0.0.0.0:8080

# 2) Generate a JSON payload from a sample of the test split and send the request
generated_payload=$(python - << 'EOF'
import pandas as pd, json
# Update the S3 path below
df = pd.read_parquet('s3://<your-bucket>/recs/test/X_test.parquet')
print(json.dumps(df.head(2).to_dict(orient='records')))
EOF
)

curl -X POST http://localhost:8080/predict \
     -H "Content-Type: application/json" \
     -d "$generated_payload"

# 3) Expected Response
# The server returns a JSON object with predicted annual electricity usages:
# { "predictions": [13452.23, 9823.47] }

---

## 10. Contact

- **Contact:** aazizai@proton.me  (**Academic correspondence:** a.abdulaziz@hw.ac.uk)

---

## 11. License

This project is licensed under the [MIT License](LICENSE).

