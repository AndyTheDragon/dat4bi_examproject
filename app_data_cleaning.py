import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Restaurant Sales Data Analysis")

st.write("""
This app analyzes sales data from a restaurant using various data exploration techniques and machine learning models.
It includes sections on data cleaning and engineering, data exploration, linear regression, classification, and clustering.
         """)

st.write("""
         The original dataset contains sales data where each row represents a single line in an order.
         The dataset has been cleaned to remove erroneous data and engineered to create useful features for analysis.
         The cleaned dataset is stored in `data/cleaned_sales_data.csv`.

         TODO: Enable user to load their own dataset in the future.
         """)

st.write("""
         The app is structured into multiple pages, each focusing on a specific aspect of the analysis.
         Use the sidebar to navigate between pages.
         """)

df = st.session_state['raw_data']
st.write("Here is a preview of the cleaned sales data:")
st.dataframe(df)
# pie chart showing distribution of sales channels
sales_channel_counts = df['sales_channel'].value_counts()
fig, ax = plt.subplots()
ax.pie(sales_channel_counts, labels=['Inhouse', 'Wolt', 'Mealo'], autopct='%1.1f%%', startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
st.pyplot(fig)
st.write("The pie chart above shows the distribution of sales channels in the dataset.")