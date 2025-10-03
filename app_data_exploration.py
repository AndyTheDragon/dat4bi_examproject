import streamlit as st
import pandas as pd
import numpy as np

from DataExploration import *
from DataExploration import descriptive_statistics as ds
from DataExploration import plots as pl


st.title("Restaurant Sales Data Exploration")


df = st.session_state['raw_data']


st.write("In this section Whiskers Plot, Histograms and Correlation Heatmap is shown in order to understand the data, and to identify whether normal distribution is present or if there is correlation between columns. ")
# Find a way to do a multiple select of columns
columns = st.multiselect("Select columns for analysis", options=df.columns.tolist(), default=df.columns.tolist())
if columns:
    df = df[columns]  # Filter dataframe to selected columns
fig_boxplot, axes = pl.show_boxplots(df, layout='grid')
st.pyplot(fig_boxplot)
st.write("""
        Looking at the Whiskers plot it is clear that the amount of soups and extras sold by the restaurant are so few, that it has no impact on the analysis.
        In number_of_drinks, number_of_maindishes and order_total there are outliers that are significantly higher than the other values. These outliers are attempted removed in the codeblock below.
        """)

fig_hist, axes = pl.show_histograms(df, layout='grid', bell_curve=True, bins=30)
st.pyplot(fig_hist)