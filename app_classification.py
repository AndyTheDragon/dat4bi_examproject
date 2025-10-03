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

cl = sc.show_decision_tree(df, 'sales_channel', max_depth=4)
classifier, accuracy, confusion_mat, importances_series, graph, cffig, impfig = cl
st.graphviz_chart(graph.source)
st.write(f"Accuracy: {accuracy}")
st.write("The accuracy is the proportion of correct predictions made by the model. The closer the accuracy is to 1, the better the model fits the data.")

st.pyplot(cffig)
st.write("The confusion matrix shows the number of correct and incorrect predictions made by the model for each class.")

st.pyplot(impfig)
st.write("The feature importances show the relative importance of each feature in making predictions.")

