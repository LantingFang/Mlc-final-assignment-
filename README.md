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
```

### 2. Explore the Data
Key Insights:
A house with 33 bedrooms might be an outlier worth investigating.
Square footage ranges from 290 to 13,450, indicating diverse property sizes.
Use .describe() and visualizations to uncover patterns.

### 3. Visualize Key Insights
Common Number of Bedrooms
Visualize the most common house types:
```python
sns.countplot(data['bedrooms'])
plt.title("Number of Bedrooms vs Count")
plt.show()
```
House Locations
Plot the density of houses by latitude and longitude:
```python
sns.jointplot(x='longitude', y='latitude', data=data, kind="hex")
plt.title("House Location Density")
plt.show()
```

Factors Affecting House Prices
Explore how features like square footage and location impact house prices:
```python
sns.scatterplot(x='sqft_living', y='price', data=data)
plt.title("Square Footage vs Price")
plt.show()

sns.scatterplot(x='longitude', y='price', data=data)
plt.title("Longitude vs Price")
plt.show()
```
### 4. Build a Linear Regression Model
Linear regression predicts outcomes using the equation 
𝑌
=
𝑚
𝑋
+
𝑐
Y=mX+c. Here's how to implement it:
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Preparing the data
X = data[['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long']]
y = data['price']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Training the model
model = LinearRegression()
model.fit(X_train, y_train)

# Testing the model
y_pred = model.predict(X_test)
accuracy = r2_score(y_test, y_pred)
print(f"Linear Regression Accuracy: {accuracy * 100:.2f}%")
```
Result:
Accuracy: ~73%

### 5. Improve Accuracy with Gradient Boosting
Gradient Boosting combines multiple weak models (e.g., decision trees) to improve predictions.
```python
from sklearn.ensemble import GradientBoostingRegressor

# Initializing Gradient Boosting Regressor
gbr = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, max_depth=5, random_state=42)

# Training the model
gbr.fit(X_train, y_train)

# Testing the model
y_pred_gbr = gbr.predict(X_test)
accuracy_gbr = r2_score(y_test, y_pred_gbr)
print(f"Gradient Boosting Accuracy: {accuracy_gbr * 100:.2f}%")
```
Result:
Accuracy: ~91.94%
Key Takeaways
Linear Regression is simple but may underperform on complex datasets.
Gradient Boosting significantly improves accuracy, especially for datasets with multiple features.
Visualization is critical for understanding data relationships.
