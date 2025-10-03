import streamlit as st
import pandas as pd
import numpy as np
#import ShowLinearRegression as slr
import ShowClassification as sc
#import ShowClustering.clustering as scc

st.title("Restaurant Sales Classification Model")

df = st.session_state['raw_data']

st.write("""
A decision tree classification model has been made in order to predict the sales channel of the order.
""")

cl= sc.show_decision_tree(df, 'sales_channel', max_depth=4)