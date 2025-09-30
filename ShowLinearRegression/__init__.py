import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as sm

__all__ = ['show_model']

def show_model(data, feature_cols, target_col, test_size=0.2, random_state=42):
    X = data[feature_cols].values.reshape(-1, len(feature_cols))
    y = data[target_col].values.reshape(-1, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=test_size)
    myreg = LinearRegression()
    myreg.fit(X_train, y_train)
    a = myreg.coef_
    b = myreg.intercept_
    y_predicted = myreg.predict(X_test)
    mae = sm.mean_absolute_error(y_test, y_predicted)
    mse = sm.mean_squared_error(y_test, y_predicted)
    rmse = np.sqrt(sm.mean_squared_error(y_test, y_predicted))
    eV = round(sm.explained_variance_score(y_test, y_predicted), 2)
    r2 = sm.r2_score(y_test, y_predicted)

    if X.shape[1] == 1:
        plt.title('Simple Linear Regression: Monthly Income vs Total Working Years')
        plt.scatter(X, y, color='green')
        plt.plot(X_train, a * X_train + b, color='blue')
        plt.plot(X_test, y_predicted, color='orange')
        plt.xlabel('Total Working Years')
        plt.ylabel('Monthly Income')
        plt.show()
        print(f"The model is a line \n\ty = a * x + b, or\n\ty = {a} * x + {b}")
    else:
        fig = px.scatter_3d(data, x = feature_cols[0], y = feature_cols[1], z = target_col[0])
        fig.show()
        eq = " + ".join([f"{a[0][i]:.2f}*{feature_cols[i]}" for i in range(len(feature_cols))])
        print(f"The model is a hyperplane:\n\ty = {eq} + {b[0]:.2f}")



    print('Mean Absolute Error ',mae)
    print('Mean Squared Error ',mse)
    print('Root Mean Squared Error ',rmse)
    print('Explained variance score ',eV )
    print('R2 score ',r2)

    return list(a), b, mae, mse, rmse, eV, r2, myreg