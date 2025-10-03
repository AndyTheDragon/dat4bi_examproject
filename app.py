import streamlit as st
import pandas as pd
import numpy as np

from DataExploration import *
from DataExploration import descriptive_statistics as ds
from DataExploration import plots as pl
import ShowLinearRegression as slr
import ShowClassification as sc
import ShowClustering.clustering as scc

st.title("Restaurant Sales Data Analysis")

pages = {
    "Data Exploration": [
        st.Page("app_data_cleaning.py", title="Data cleaning and engineering"),
        st.Page("app_data_exploration.py", title="Data exploration"),
    ],
    "Models": [
        st.Page("app_linear.py", title="Linear Regression"),
        #st.Page("app_classification.py", title="Classification"),
        #st.Page("app_clustering.py", title="Clustering"),
    ],
}

pg = st.navigation(pages, position="sidebar", expanded=True)
pg.run()

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df = df.drop(['datetime', 'order_id'], axis=1, inplace=False)
    df = df.drop(df[df.lt(0).any(axis=1)].index)
    return df

# load once (cached) and store raw dataframe in session_state
if 'raw_data' not in st.session_state:
    st.session_state['raw_data'] = load_data("data/cleaned_sales_data.csv")

data = st.session_state['raw_data']
st.dataframe(data, use_container_width=True)

