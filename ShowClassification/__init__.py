import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import sklearn.metrics as sm
from sklearn.metrics import accuracy_score, confusion_matrix
import graphviz
from IPython.display import display

__all__ = ['show_decision_tree']

def show_decision_tree(df, target_col, test_size=0.2, random_state=42, max_depth=None):
    """
    Preprocesses the DataFrame, moves the target column to the last position,
    splits the data into training and testing sets, fits a Decision Tree Classifier,
    and visualizes the tree.

    Parameters:
    - df: pandas DataFrame
    - target_col: name of the target column
    - test_size: proportion of the dataset to include in the test split (default: 0.2)
    - random_state: random seed for reproducibility (default: 42)
    - max_depth: maximum depth of the decision tree (default: None)

    Returns:
    - classifier: trained DecisionTreeClassifier
    - accuracy: accuracy score on the test set
    - confusion_mat: confusion matrix
    """

    # Move the target column to the last position
    columns = [col for col in df.columns if col != target_col] + [target_col]
    df = df[columns]

    # Split into features and labels
    X = df.iloc[:, :-1].values  # All columns except the last one
    y = df.iloc[:, -1].values  # Only the last column

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Train the Decision Tree Classifier
    classifier = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    classifier.fit(X_train, y_train)

    # Predict on the test set
    y_pred = classifier.predict(X_test)

    # Calculate accuracy and confusion matrix
    accuracy = sm.accuracy_score(y_test, y_pred)
    confusion_mat = sm.confusion_matrix(y_test, y_pred)

    # Visualize the decision tree
    dot_data = export_graphviz(
        classifier,
        out_file=None,
        feature_names=df.columns[:-1],  # Feature names are all columns except the last one
        class_names=[str(cls) for cls in np.unique(y)],
        filled=True,
        rounded=True,
        special_characters=True
    )
    graph = graphviz.Source(dot_data)
    graph.render("Documents/decision_tree")  # Save the tree as a file
    display(graph)


    

    # Plot confusion matrix as a heatmap
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

        # Print metrics
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", confusion_mat)
    print("Classifier:", classifier)