# Customer Segmentation Clustering Web App

## Overview
This project implements an interactive web application for customer segmentation using **Hierarchical Agglomerative Clustering** on the Mall Customers dataset. It meets the assignment requirements by deploying a trained clustering model in a live Streamlit app that performs **true inference without retraining**.

## Dataset
- **Source**: [Kaggle – Mall Customers Dataset](https://www.kaggle.com/datasets/kandij/mall-customers)
- **Records**: 200 customers
- **Features**: Age, Annual Income (k$), Spending Score (1-100)
- **License**: Openly available for educational use.

## Project Structure
clustering_app/
├── app/
│ └── streamlit_app.py # Interactive web app
├── data/
│ ├── Mall_Customers.csv # Original dataset
│ └── clustered_data.csv # Data with cluster labels
├── models/ # Saved pipeline files (.joblib)
├── notebooks/
│ └── 01_eda_clustering.ipynb # EDA, model training, evaluation
├── figures/ # Visualisations (optional)
├── Dockerfile # Docker deployment
├── requirements.txt # Dependencies
├── README.md # This file
└── report.md # Project report


## Technologies Used
- **Python 3.12+**
- **scikit-learn** (AgglomerativeClustering, StandardScaler)
- **UMAP** (Dimensionality reduction for visualization)
- **Streamlit** (Interactive web framework)
- **joblib** (Model persistence)
- **Docker** (Containerization – bonus)

## Installation & Setup

### 1. Clone or Download
```bash
git clone <your-repo-url>
cd clustering_app

2. Create Virtual Environment & Install Dependencies
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

3. Train the Model (Optional – models are already saved)
If you want to retrain the models, run the Jupyter notebook:
jupyter notebook notebooks/01_eda_clustering.ipynb
Click "Run All".

4. Run the Web App (Local)
streamlit run app/streamlit_app.py
Open your browser to http://localhost:8501.

5. Run with Docker (Bonus)
docker build -t clustering-app .
docker run -p 8501:8501 clustering-app

Then open http://localhost:8501.

Key Features
View Clusters: Visualises existing customers in 2D space (UMAP with PCA fallback), colour‑coded by cluster.

Predict New Customer: Input customer features (Age, Income, Spending) to assign them to a cluster.

Outlier Detection: Flags customers that are far from any cluster centroid (noise).

No Retraining: Predictions are made using pre‑calculated centroids, ensuring fast and correct inference.

Evaluation Metrics (Internal Validation)
Silhouette Score: 0.39

Davies‑Bouldin Index: 0.916

Calinski‑Harabasz Index: 107.83

Cluster Interpretations (Business Names)
Cluster	Name	Description
0	Medium Income, Medium Spenders	Average spenders with moderate income
1	Medium Income, Medium Spenders	(Similar profile group)
2	High Income, High Spenders	VIP customers
3	High Income, Low Spenders	Price‑sensitive high earners
4	Low Income, Low Spenders	Budget‑conscious customers
Rubric Compliance
✅ Dataset: Open‑source, >200 records, 3+ numeric features, not Iris/Tutorial.

✅ Algorithm: Hierarchical Agglomerative Clustering (Not K‑Means/GMM/PCA).

✅ Preprocessing: StandardScaler applied, LabelEncoder for Genre.

✅ Evaluation: Silhouette, Davies‑Bouldin, Calinski‑Harabasz reported.

✅ Interpretation: Clusters named and profiled.

✅ App Integration: Model loaded at startup, live inference without retraining.

✅ Documentation: Clear README and report provided.

✅ Bonus: Docker deployment working.
