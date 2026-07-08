import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import umap
import os

# ---- PATHS ----
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
models_dir = os.path.join(project_root, 'models')
data_dir = os.path.join(project_root, 'data')

st.set_page_config(page_title="Customer Clustering App", layout="wide")
st.title("🛍️ Customer Segmentation Clustering")
st.markdown("Hierarchical clustering on Mall Customers dataset")

# ---- LOAD MODELS ----
@st.cache_resource
def load_models():
    scaler = joblib.load(os.path.join(models_dir, 'scaler.joblib'))
    cluster_model = joblib.load(os.path.join(models_dir, 'cluster_model.joblib'))
    cluster_names = joblib.load(os.path.join(models_dir, 'cluster_names.joblib'))
    cluster_profiles = joblib.load(os.path.join(models_dir, 'cluster_profiles.joblib'))
    centroids = joblib.load(os.path.join(models_dir, 'centroids.joblib'))
    reducer = joblib.load(os.path.join(models_dir, 'umap_reducer.joblib'))
    config = joblib.load(os.path.join(models_dir, 'config.joblib'))
    
    # <-- NEW: Load the pre-calculated threshold
    threshold = joblib.load(os.path.join(models_dir, 'threshold.joblib'))
    
    return scaler, cluster_model, cluster_names, cluster_profiles, centroids, reducer, config, threshold

try:
    scaler, cluster_model, cluster_names, cluster_profiles, centroids, reducer, config, threshold = load_models()
    st.success("✅ Models loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load models. Please run the notebook first.\nError: {e}")
    st.stop()

# ---- PREDICTION HELPER (NO RETRAINING!) ----
def predict_cluster(input_scaled, centroids):
    # Calculate distances to each cluster centroid
    distances = np.linalg.norm(input_scaled - centroids, axis=1)
    closest_cluster = np.argmin(distances)
    min_distance = np.min(distances)
    return closest_cluster, min_distance, distances

# ---- UI ----
mode = st.sidebar.radio("Choose mode", ["View Clusters", "Predict New Customer"])

if mode == "View Clusters":
    st.header("📊 Existing Customers Clustered")
    df = pd.read_csv(os.path.join(data_dir, 'clustered_data.csv'))
    X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values
    X_scaled = scaler.transform(X)
    
    # Try UMAP, fallback to PCA if it fails
    try:
        X_umap = reducer.transform(X_scaled)
    except Exception as e:
        st.info("ℹ️ Using PCA for 2D visualization (UMAP fallback).")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_umap = pca.fit_transform(X_scaled)
    
    df['UMAP1'] = X_umap[:, 0]
    df['UMAP2'] = X_umap[:, 1]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(df['UMAP1'], df['UMAP2'], 
                         c=df['Cluster'], cmap='viridis', s=60, alpha=0.7)
    plt.colorbar(scatter, label='Cluster')
    ax.set_title('UMAP Projection of Customers')
    st.pyplot(fig)
    
    st.subheader("📈 Cluster Profiles")
    profiles = df.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()
    profiles['Count'] = df['Cluster'].value_counts().sort_index()
    profiles['Name'] = profiles.index.map(lambda x: cluster_names.get(x, f'Cluster {x}'))
    st.dataframe(profiles.style.background_gradient(cmap='Blues'))

else:
    st.header("🔮 Predict Cluster for a New Customer")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 18, 80, 30)
    with col2:
        income = st.number_input("Annual Income (k$)", 10, 150, 50)
    with col3:
        spending = st.number_input("Spending Score (1-100)", 1, 100, 50)
    
    if st.button("Predict", type="primary"):
        # Standardize input
        input_data = np.array([[age, income, spending]])
        input_scaled = scaler.transform(input_data)
        
        # PREDICT USING CENTROIDS (No retraining!)
        cluster_id, min_dist, all_dists = predict_cluster(input_scaled, centroids)
        name = cluster_names.get(cluster_id, f'Cluster {cluster_id}')
        
        # Check if outlier using the pre-calculated threshold
        if min_dist > threshold:
            st.warning(f"⚠️ This customer is an **Outlier/Noise** (distance {min_dist:.2f} > threshold {threshold:.2f})")
        else:
            st.success(f"✅ This customer belongs to: **{name}**")
        
        # Show profile
        st.subheader("📊 Typical profile of this cluster")
        prof = pd.DataFrame({
            'Feature': ['Age', 'Income', 'Spending'],
            'Average': [
                cluster_profiles['Age'][cluster_id],
                cluster_profiles['Annual Income (k$)'][cluster_id],
                cluster_profiles['Spending Score (1-100)'][cluster_id]
            ]
        })
        st.dataframe(prof)
        
        # Show distances to all clusters
        st.subheader("📏 Distance to each cluster centroid")
        dist_df = pd.DataFrame({
            'Cluster': list(range(len(all_dists))),
            'Distance': all_dists,
            'Name': [cluster_names.get(i, f'Cluster {i}') for i in range(len(all_dists))]
        })
        st.bar_chart(dist_df.set_index('Name')['Distance'])

st.sidebar.markdown("---")
st.sidebar.caption(f"Model: {config['n_clusters']} clusters, linkage={config['linkage']}")