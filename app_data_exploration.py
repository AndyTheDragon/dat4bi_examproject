import streamlit as st
import pandas as pd
import numpy as np

from DataExploration import *
from DataExploration import descriptive_statistics as ds
from DataExploration import plots as pl


st.title("Restaurant Sales Data Exploration")


df = st.session_state['raw_data']


fig_boxplot, axes = pl.show_boxplots(df, layout='grid')
st.pyplot(fig_boxplot)

fig_hist, axes = pl.show_histograms(df, layout='grid', bell_curve=True, bins=30)
st.pyplot(fig_hist)