# 🛡️ Insurance Fraud Detection System (End-to-End ML Pipeline)

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Framework](https://img.shields.io/badge/Framework-Flask-green)
![ML](https://img.shields.io/badge/Machine%20Learning-XGBoost%20%7C%20SVM-orange)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)

**Architected by:** Jyoti Choudhary  
**Institution:** IGDTUW

---

## 📖 Project Overview

This project is a production-grade **Machine Learning System** designed to detect fraudulent insurance claims. Unlike traditional monolithic models, this system employs a **Clustering-Based Ensemble Architecture** to handle the high heterogeneity of insurance data.

It is designed to solve real-world engineering problems such as **Class Imbalance** (99% non-fraud vs 1% fraud), **Data Leakage**, and **Schema Skew** in production environments.

### 🚀 Key Features
* **Divide & Conquer Strategy:** Uses **K-Means Clustering** to segment claims into behavioral groups before classification.
* **Dynamic Model Selection:** Automatically runs a tournament between **XGBoost** and **SVM** for *each* cluster and selects the winner based on AUC scores.
* **Production Safety:** Includes a custom **Schema Alignment** layer to handle missing columns in prediction data without crashing.
* **Imbalance Handling:** Utilizes `RandomOverSampler` to synthetically balance the dataset, ensuring the model doesn't just bias towards the majority class.
* **Deployment:** Exposed via a **Flask API** for real-time inference.

---

## 🏗️ System Architecture

The pipeline follows a strict lifecycle:

1.  **Data Ingestion & Validation:**
    * Validates file names using Regex.
    * Rejects files with missing columns or incorrect types.
2.  **Preprocessing Layer:**
    * **Imputation:** Custom logic (`preprocessing.py`) to handle missing values based on data types.
    * **Scaling:** Saves `StandardScaler` statistics during training to prevent **Data Leakage** during prediction.
3.  **Clustering Layer:**
    * Uses **K-Means** with the **Elbow Method** (programmatic detection using `KneeLocator`) to find the optimal number of clusters.
4.  **Model Training (The Ensemble):**
    * For every cluster i, the system trains both an **XGBoost** and an **SVM** model.
    * It compares them using `ROC-AUC` and `F1-Score`.
    * The best model is saved as `Model_Cluster_i.sav`.
5.  **Prediction Layer:**
    * New data is cleaned, scaled, and assigned a cluster.
    * It is then routed to the specific model trained for that cluster.

---

## 📂 Project Structure

    ├── best_model_finder/      # Logic for Model Selection (XGBoost vs SVM)
    │   └── tuner.py
    ├── data_loader/            # CSV ingestion logic
    ├── data_preprocessing/     # Core logic (Cleaning, Clustering, Imbalance)
    │   ├── clustering.py
    │   └── preprocessing.py
    ├── file_operations/        # Saving/Loading models (w/ Version Patching)
    ├── models/                 # Binary model files (.sav) stored here
    ├── Prediction_Logs/        # Runtime logs for debugging
    ├── Training_Logs/
    ├── main.py                 # Entry point (Flask App)
    ├── trainingModel.py        # Controller for Training Pipeline
    ├── predictFromModel.py     # Controller for Inference Pipeline
    ├── requirements.txt        # Dependencies
    └── Procfile                # Heroku deployment config

---

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.8 or higher (Tested on Python 3.13)
* pip

### Step 1: Clone the Repository

    git clone https://github.com/cs1230020/MLFINALFRAUDDETECTION.git
    cd MLFINALFRAUDDETECTION

### Step 2: Install Dependencies

    pip install -r requirements.txt

---

## 🏃‍♂️ How to Run

### 1. Start the Server
Run the Flask application:

    python main.py

*You will see a message: `Running on http://127.0.0.1:5000/`*

### 2. Access the Web Interface
Open your browser and go to `http://localhost:5000`.

### 3. Training the Model
* Click on the **"Train"** button in the UI (or send a POST request to `/train`).
* The system will ingest data, preprocess, cluster, and tune models.
* Check `Training_Logs/ModelTrainingLog.txt` to see the real-time metrics (Precision/Recall).

### 4. Prediction
* Upload your dataset CSV file.
* The system will generate a `Predictions.csv` file indicating `Y` (Fraud) or `N` (Legit) for each row.

---

## 📊 Performance Results

The system was evaluated on a highly imbalanced insurance dataset. The **Clustering + Ensemble** approach significantly outperformed the baseline SVM model.

| Metric | XGBoost (Selected) | SVM (Baseline) |
| :--- | :--- | :--- |
| **Accuracy** | **99.8% - 100%** | ~40% - 65% |
| **Precision** | **High (>0.95)** | Low / Inconsistent |
| **Recall (Fraud)** | **1.0 (100%)** | Variable |
| **AUC Score** | **1.0** | 0.5 - 0.6 |

**Why XGBoost Won?**
The analysis proved that fraud patterns are **highly non-linear**. SVM (a linear separator) struggled to separate the classes effectively, often yielding low precision or failing to converge on a decision boundary. XGBoost's ensemble decision trees successfully identified fraud cases with near-perfect precision after oversampling.

---

## 🔧 Engineering Highlights (Interview Defense)

* **Thread-Safety Patch:** Implemented a runtime patch in `file_methods.py` to fix a `_n_threads` attribute error caused by Scikit-Learn version mismatch during migration.
* **Schema Alignment:** The `predictFromModel.py` includes a dynamic check that injects missing columns (with 0s) into prediction data if they don't match the training schema, preventing "Dimension Mismatch" errors in production.

---

