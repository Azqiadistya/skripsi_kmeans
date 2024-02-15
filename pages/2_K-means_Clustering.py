import pandas as pd
from pandas import read_csv
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_samples, silhouette_score
import streamlit as st
import time
import numpy as np
import plotly.express as px
from streamlit import session_state as state
from sklearn.preprocessing import MinMaxScaler
st.set_page_config(page_title="Kmeans Clustering", layout="wide")

st.title("K-Means Clustering")

def load_uploaded_data(uploaded_file):
    data = pd.read_csv(uploaded_file) 
    return data

def main():
    uploaded_file = st.file_uploader("Unggah dataset", type=["csv"])

    if uploaded_file is not None:
        uploaded_data = load_uploaded_data(uploaded_file)
        st.write("Data yang diunggah:")
        # st.write(uploaded_data)

    data = st.write(uploaded_data)

    # Cek data kosong
    df = uploaded_data.copy()
    df.isnull().sum()

    # normalisasi data
    st.subheader("""Normalisasi Data""")
    # Inisialisasi MinMaxScaler
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df)
    dataScaled = pd.DataFrame(normalized_data, columns=df.columns)
    dataScaled

    # Menginput n cluster

    n_cluster = st.number_input(
        'Jumlah Cluster Yang Diinginkan', min_value=2, max_value=15, value=3, step=1)
    st.write('*silhouette score requires more than 1 cluster labels')

    # Menerapkan algoritma K-means dengan n klaster
    st.subheader("""Hasil Cluster""")

    n_cluster_fix = n_cluster
    kmeans = KMeans(n_clusters=n_cluster_fix)
    kmeans.fit(dataScaled)
    cluster_centers = kmeans.cluster_centers_
    predicted_labels = kmeans.fit_predict(dataScaled)

    # Menambah kolom cluster
    df['Cluster_int'] = predicted_labels
    df

    # mencari presentase sebaran data
    arrayCluster = pd.DataFrame(predicted_labels)

    unique_values, counts = np.unique(arrayCluster, return_counts=True)
    total_data = len(arrayCluster)

    percentage_distribution = (counts / total_data) * 100

    # Membuat diagram lingkaran
    labels = [f'Cluster {value + 1}' for value in unique_values]
    fig, ax = plt.subplots()
    ax.pie(percentage_distribution, labels=labels,
           autopct='%1.1f%%', startangle=90)
    ax.axis('equal')

    st.subheader("Diagram Presentase Cluster ")
    st.pyplot(fig)

    # Evaluation
    st.subheader("Silhouette Score")
    st.write("""Akurasi Clustering menggunakan Silhouette Score, Nilai Silhouette Score makin mendekati 1 makin BAGUS""")
    silhouette_avg_kmeans = silhouette_score(
        dataScaled.iloc[:, 0:6], predicted_labels)
    silhouette_avg_kmeans

    st.session_state['scoreKmeans'] = silhouette_avg_kmeans


if __name__ == "__main__":
    main()
