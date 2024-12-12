# Predicting House Prices Using Python

This project demonstrates how to predict house prices using Python and machine learning techniques. We'll start with **Linear Regression** and improve the model using **Gradient Boosting Regression** to achieve better accuracy.

## Prerequisites

- Basic knowledge of Python and libraries such as `pandas`, `numpy`, `matplotlib`, and `seaborn`.
- Familiarity with machine learning concepts like regression and model evaluation.
- Python environment set up with the necessary libraries installed.

## Dataset Overview

The dataset contains information about houses, such as:
- Location (latitude and longitude)
- Price
- Number of bedrooms
- Square footage (including basement area)

Our goal is to predict the **price** of houses based on these features.

## Project Steps

### 1. Import Libraries and Load Data

Start by importing the necessary libraries and loading the dataset. Use `.head()` and `.describe()` to explore the data.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("house_data.csv")
print(data.head())
print(data.describe())

2. Explore the Data
Key Insights:
A house with 33 bedrooms might be an outlier worth investigating.
Square footage ranges from 290 to 13,450, indicating diverse property sizes.
Use .describe() and visualizations to uncover patterns.

3. Visualize Key Insights
Common Number of Bedrooms
Visualize the most common house types:
sns.countplot(data['bedrooms'])
plt.title("Number of Bedrooms vs Count")
plt.show()
