import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Page Config
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

# Title
st.title("🟢 Customer Segmentation Dashboard")
st.write("This system uses **K-Means Clustering** to group customers based on purchasing behavior.")
st.markdown("👉 *Discover hidden customer groups without predefined labels.*")

# Load Dataset
df = pd.read_csv("Wholesale customers data.csv") 

# Preview Data
st.subheader("📂 Dataset Preview")
st.dataframe(df.head())

# Spending Features
spending_features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']

# Sidebar Controls
st.sidebar.header("🔧 Clustering Controls")

feature_1 = st.sidebar.selectbox("Select Feature 1", spending_features)
feature_2 = st.sidebar.selectbox("Select Feature 2", spending_features, index=1)

k = st.sidebar.slider("Number of Clusters (K)", 2, 10, 3)
random_state = st.sidebar.number_input("Random State", value=42)
run_clustering = st.sidebar.button("🟦 Run Clustering")

if run_clustering:

    # Data Prep
    X = df[spending_features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans Model
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df["Cluster"] = clusters

    # Visualization
    st.subheader("📈 Customer Clusters Visualization")
    fig, ax = plt.subplots(figsize=(8,6))

    ax.scatter(df[feature_1], df[feature_2], c=df["Cluster"], cmap="viridis", s=60)

    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    ax.scatter(
        centers[:, spending_features.index(feature_1)],
        centers[:, spending_features.index(feature_2)],
        s=300, c="red", marker="X", label="Cluster Centers"
    )

    ax.set_xlabel(feature_1)
    ax.set_ylabel(feature_2)
    ax.set_title("Customer Segmentation using K-Means")
    ax.legend()
    st.pyplot(fig)

    # Cluster Profiling
    st.subheader("📊 Cluster Summary")
    summary = df.groupby("Cluster")[spending_features].mean().reset_index()
    summary["Count"] = df.groupby("Cluster").size().values
    st.dataframe(summary)

    # Business Interpretation
    st.subheader("💡 Business Insights")
    for cluster in summary["Cluster"]:
        st.write(f"🔹 Cluster {cluster}: High spending in {feature_1}/{feature_2}. Recommended targeted promotions and inventory focus.")

    # Limitation
    st.warning("⚠️ Limitation: K-Means assumes spherical clusters and is sensitive to scaling and initial centroids.")

else:
    st.warning("⬅️ Select features and click Run Clustering.")
