#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load data from a CSV file
data = pd.read_csv("battery_data.csv")

# Split the data into training and testing sets
train_data = data.sample(frac=0.8, random_state=1)
test_data = data.drop(train_data.index)

# Extract the input features (cycle number and temperature) and the output variable (SoH)
train_X = train_data[["cycle_number", "temperature"]]
train_Y = train_data["SoH"]
test_X = test_data[["cycle_number", "temperature"]]
test_Y = test_data["SoH"]

# Create a linear regression model and fit it to the training data
model = LinearRegression()
model.fit(train_X, train_Y)

# Predict the SoH values for the test data
predictions = model.predict(test_X)

# Calculate the root mean squared error (RMSE) to evaluate the model's performance
mse = np.mean((predictions - test_Y) ** 2)
rmse = np.sqrt(mse)
print("RMSE:", rmse)

# Use the model to predict the future SoH of a battery given its cycle number and temperature
future_cycle_number = 200
future_temperature = 30
future_soh = model.predict([[future_cycle_number, future_temperature]])
print("Predicted future SoH:", future_soh[0])

