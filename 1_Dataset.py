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
from streamlit import session_state as state

st.set_page_config(page_title="Clustering App", page_icon="📈", layout="wide")

st.markdown("# Dataset")

# Definisikan variabel global
uploaded_data = None

def main():
    global uploaded_data

    uploaded_file = st.file_uploader("Unggah dataset", type=["csv"])

    if uploaded_file is not None:
        uploaded_data = pd.read_csv(uploaded_file)

    if uploaded_data is not None:
        st.write("Data yang diunggah:")
        st.write(uploaded_data)

    st.button("Re-run")


if __name__ == "__main__":
    main()

