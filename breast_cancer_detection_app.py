import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

import matplotlib
matplotlib.use("Agg")

st.set_option('deprecation.showPyplotGlobalUse', False)



st.write("""
# Breast Cancer Prediction App

Predict whether a tumor is malignant or benign!
""")
st.header("By Erick Rudelman")

st.sidebar.header('Input Tumor Parameters')

cancer = load_breast_cancer()
X = cancer.data
Y = cancer.target

def user_input_features():
    data = {}

    for col in range(X.shape[1]):
        feature_name = cancer.feature_names[col]
        feature_values = X[:, col]
        
        min_value = np.floor(np.min(feature_values)) - 2.0
        max_value = np.floor(np.max(feature_values)) + 2.0
        median_value = np.median(feature_values)
        
        # Ensure the minimum value is at least 0
        min_value = max(min_value, 0.0)
        
        default_value = median_value
        
        # Create number input for each feature
        feature_value = st.sidebar.number_input(
            feature_name,
            min_value,
            max_value,
            default_value,
            step=0.1  # You can adjust the step value as needed
        )
        
        data[feature_name] = feature_value
    
    features = pd.DataFrame(data, index=[0])
    return features





df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

print("Breast Cancer Dataset Information:")
print("Number of Samples:", X.shape[0])
print("Number of Features:", X.shape[1])
print("Class Labels:", np.unique(Y))

# Convert to DataFrame
df_cancer = pd.DataFrame(data=np.c_[cancer['data'], cancer['target']],
                         columns=np.append(cancer['feature_names'], ['target']))

# Display the DataFrame
st.write("Breast Cancer Dataset Overview:")
st.write(df_cancer)

# Column names of the Breast Cancer dataset
column_names = cancer.feature_names

# Display column names
st.write("Column Names of the Breast Cancer Dataset:")
st.write(column_names)



clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels')
st.write(['Benign', 'Malignant'])

st.subheader('Prediction')
prediction_labels = ['Benign', 'Malignant']
st.write(prediction_labels[int(prediction)])



st.subheader('Prediction Probability')
st.write(prediction_proba)

##The app essentially allows users to select sepal and petal dimensions using sliders,
#and based on these inputs, it predicts the type of Iris flower and displays the prediction
#along with the prediction probabilities.

# Create different types of visualizations
df_cancer = pd.DataFrame(data=np.c_[X, Y], columns=np.append(cancer.feature_names, 'target'))

st.write(""""

Here's what you're seeing in the histogram visualization:

X-Axis (Feature Values): The x-axis represents the values of the selected feature. In this case, it's the 'mean radius' feature.

Y-Axis (Frequency): The y-axis represents the frequency or count of data points that fall within each bin.

Bins: The histogram divides the range of the feature's values into several bins. Each bin represents a range of values, and the height of the bar in the bin indicates how many data points fall within that range.

KDE Curve: The histogram bars are accompanied by a Kernel Density Estimation (KDE) curve, which is a smoothed estimate of the underlying probability distribution of the data. It helps you visualize the general shape of the distribution.


""")

# Visualization Functions
def plot_histogram(data, feature_name):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(data[feature_name], bins=20, kde=True, ax=ax)
    plt.xlabel(feature_name)
    plt.ylabel('Frequency')
    plt.title(f'{feature_name} Distribution')
    st.pyplot(fig)

# Streamlit App
st.write("# Breast Cancer Data Visualization")

# Display Histogram
plot_histogram(df_cancer, 'mean radius')

st.write("""
The data appears to have a peak around a specific "mean radius" value. This suggests that there is a common range of "mean radius" values that are more frequent in the dataset.

The histogram is slightly skewed, with a longer tail on the right side. This indicates that there might be a few data points with higher "mean radius" values that are more spread out from the main cluster.

There are no extremely tall bars or outliers, indicating that there are no dramatic deviations from the main distribution.

The histogram is not bimodal or multimodal, as it doesn't show multiple distinct peaks.
""")
