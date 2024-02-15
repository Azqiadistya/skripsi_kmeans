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
st.set_page_config(page_title="Evaluation", layout="wide")
st.title("Evaluation - Silhouette Score")

st.subheader("Silhouette Score K-Means Murni")
st.write(st.session_state.scoreKmeans)
st.subheader("Silhouette Score K-Means + Elbow")
st.write(st.session_state.scoreKmeansElbow)
st.write("""NOTE :""")
st.write("""Nilai Silhouette Score antara -1 sampai 1, makin mendekati 1 makin BAGUS""")
