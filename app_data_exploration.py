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
box_columns = st.multiselect("Select columns for analysis", options=df.columns.tolist(), default=['order_total', 'number_of_maindishes', 'number_of_snacks','number_of_drinks', 'number_of_soups', 'number_of_extras'])
if box_columns:
    box_df = df[box_columns]  # Filter dataframe to selected columns
fig_boxplot, axes = pl.show_boxplots(box_df, layout='grid')
st.pyplot(fig_boxplot)
st.write("Look at the Whiskers plot to determine if there are any outliers, and decide whether to remove them or not.")

fig_hist, axes = pl.show_histograms(box_df, layout='grid', bell_curve=True, bins=30)
st.pyplot(fig_hist)
st.write("Look at the Histograms to determine if the data is normally distributed or skewed.")

fig_corr = pl.show_correlation_heatmap(df)
st.pyplot(fig_corr)
st.write("Look at the Correlation Heatmap to determine if there is correlation between columns.")
