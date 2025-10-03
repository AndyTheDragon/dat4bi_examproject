import streamlit as st
import pandas as pd
import numpy as np
import ShowLinearRegression as slr
#import ShowClassification as sc
#import ShowClustering.clustering as scc

st.title("Restaurant Sales Linear Regression Model")

df = st.session_state['raw_data']

st.write("""
Multiple linear regression have been made in order to shape a model that can predict when customers will order an extra drink.
         In the report we describe the process of building a model with an R^2 score of 0.64. You can use that model to predict the number of drinks ordered based on the number of main dishes, snacks, whether the order is takeaway, the sales channel, and the total order amount.
""")

#test_size input slider
test_size = st.slider("Test size (as a fraction)", 0.1, 0.5, 0.2, step=0.05)
#random_state input slider 
random_state = st.slider("Random state (for reproducibility)", 0, 100, 42, step=1)
lin = slr.show_model(df, ['number_of_maindishes', 'number_of_snacks', 'is_takeaway', 'sales_channel', 'order_total'], ['number_of_drinks'], test_size=test_size, random_state=random_state)
#st.write(f"The model coefficients are: {lin[0]} and the intercept is: {lin[1]}")
st.write(f"The model equation is: {lin[9]}")
st.write(f"Mean Absolute Error: {lin[2]}")
st.write(f"Mean Squared Error: {lin[3]}")
st.write(f"Root Mean Squared Error: {lin[4]}")
st.write(f"Explained Variance Score: {lin[5]}")
st.write(f"R2 Score: {lin[6]}")
st.write("The closer the R2 score is to 1, the better the model fits the data.")
st.write("The model can be used to predict the number of drinks ordered based on the number of main dishes, snacks, whether the order is takeaway, the sales channel, and the total order amount.")
st.write("Try changing the test size and random state to see how it affects the model performance.")
st.write("Note: If the model is not a good fit, consider adding more features or using a different model.")
st.plotly_chart(lin[8], use_container_width=True)
