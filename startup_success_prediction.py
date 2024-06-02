import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('startup_data.csv')  # Replace with the path to your dataset

# Explore the data
print(data.head())
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Preprocess the data
# Assume the dataset has columns: 'R&D Spend', 'Administration', 'Marketing Spend', 'State', and 'Profit'
# One-hot encode the 'State' column
data = pd.get_dummies(data, columns=['State'], drop_first=True)

# Define the features and the target variable
X = data.drop('Profit', axis=1)
y = data['Profit']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Measured vs Predicted Profit')
plt.show()

# Visualize feature importance
feature_importance = pd.Series(model.coef_, index=X.columns)
feature_importance.nlargest(10).plot(kind='barh')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()
