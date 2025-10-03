import streamlit as st
import pandas as pd
import numpy as np
#import ShowLinearRegression as slr
#import ShowClassification as sc
import ShowClustering.clustering as scc

st.title("Restaurant Sales Clustering Model")

df = st.session_state['raw_data']

st.write("""
A clustering model has been made in order to group similar orders based on their features.
""")

res, X, fig = scc.show_cluster_sizes(df)
st.pyplot(fig)
st.write("The bar plot shows the sizes of each cluster. This can help identify if there are any dominant clusters or if the data is evenly distributed among clusters.")
res, X, fig = scc.show_pca_scatter_2d(df)
st.pyplot(fig)
res, X, fig = scc.show_feature_scatter_2d(df,'order_total_log1p','number_of_drinks')
st.pyplot(fig)