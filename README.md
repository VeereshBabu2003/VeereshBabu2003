# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset (replace 'your_dataset.csv' with your dataset file)
data = pd.read_csv('your_dataset.csv')

# Assuming your dataset has columns like 'bedrooms', 'bathrooms', 'sqft', 'year_built', 'price', etc.
# Select the features (independent variables) and the target (dependent variable)
X = data[['bedrooms', 'bathrooms', 'sqft', 'year_built']]  # Features
y = data['price']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# You can also use the model to make predictions for new data
# For example, if you have new data in a DataFrame called 'new_data':
# new_predictions = model.predict(new_data[['bedrooms', 'bathrooms', 'sqft', 'year_built']])
