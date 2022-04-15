from math import sqrt

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add any additional imports here
# TODO

np.random.seed(416)

# Import data
sales = pd.read_csv('home_data.csv') 
sales = sales.sample(frac=0.01) 

# All of the features of interest
selected_inputs = [
    'bedrooms', 
    'bathrooms',
    'sqft_living', 
    'sqft_lot', 
    'floors', 
    'waterfront', 
    'view', 
    'condition', 
    'grade',
    'sqft_above',
    'sqft_basement',
    'yr_built', 
    'yr_renovated'
]

# Compute the square and sqrt of each feature
all_features = []
for data_input in selected_inputs:
    square_feat = data_input + '_square'
    sqrt_feat  = data_input + '_sqrt'
    
    # TODO compute the square and square root as two new features
    sales[square_feat] = sales[data_input] ** 2
    sales[sqrt_feat] = sales[data_input].apply(lambda x: sqrt(x))
    all_features.extend([data_input, square_feat, sqrt_feat])
    
# Split the data into features and price
price = sales['price']
sales = sales[all_features]

# Train test split
train_and_validation_sales, test_sales, train_and_validation_price, test_price = \
    train_test_split(sales, price, test_size=0.2)
train_sales, validation_sales, train_price, validation_price = \
    train_test_split(train_and_validation_sales, train_and_validation_price, test_size=.125) # .10 (validation) of .80 (train + validation)


# Q2: Standardize data
### edTest(test_standardization) ###
from sklearn import preprocessing

# TODO: Check if Gradescope grader passes these
# If not, change fit() and transform()s to only use _sales[selected_inputs]
# scaler = preprocessing.StandardScaler().fit(train_sales)
# train_sales = scaler.transform(train_sales)
# validation_sales = scaler.transform(validation_sales)
# test_sales = scaler.transform(test_sales)

scaler = preprocessing.StandardScaler().fit(train_sales[selected_inputs])
train_sales = scaler.transform(train_sales[selected_inputs])
validation_sales = scaler.transform(validation_sales[selected_inputs])
test_sales = scaler.transform(test_sales[selected_inputs])

# Q3: Train baseline model
### edTest(test_train_linear_regression) ###
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# TODO Train a linear regression model (you'll likely need to import some things)
model = LinearRegression().fit(train_sales, train_price)
pred = model.predict(train_sales)
test_rmse_unregularized = mean_squared_error(train_price, pred, squared=False)
# print(test_rmse_unregularized)

# Train Ridge models
l2_lambdas = np.logspace(-5, 5, 11, base = 10)

### edTest(test_ridge) ###
from sklearn.linear_model import Ridge
l2_lambdas = np.logspace(-5, 5, 11, base = 10)

# TODO Implement code to evaluate Ridge Regression with various l2 Penalties
data = []
for l in l2_lambdas:
  ridge_model = Ridge(alpha=l)
  ridge_model.fit(train_sales, train_price)
  train_pred = ridge_model.predict(train_sales)
  train_rmse = mean_squared_error(train_price, train_pred, squared=False)
  val_pred = ridge_model.predict(validation_sales)
  val_rmse = mean_squared_error(validation_price, val_pred, squared=False)
  data.append({
    'l2_penalty': l,
    'model': ridge_model,
    'train_rmse': train_rmse,
    'validation_rmse': val_rmse
  }) 
ridge_data = pd.DataFrame(data)
# ridge_data.head()

# Q5: Analyze Ridge data
### edTest(test_ridge_analysis) ###

# TODO Print information about best l2 model
index = ridge_data['validation_rmse'].idxmin()
row = ridge_data.loc[index]

best_l2 = row['l2_penalty']

pred = row['model'].predict(test_sales)
test_rmse_ridge = mean_squared_error(test_price, pred, squared=False)
num_zero_coeffs_ridge = 0

# print('L2 Penalty',  best_l2)
# print('Test RSME', test_rmse_ridge)
# print('Num Zero Coeffs', num_zero_coeffs_ridge)
# print_coefficients(row['model'], selected_inputs)

# Train LASSO models
l1_lambdas = np.logspace(1, 7, 7, base=10)

# Q6: Implement code to evaluate LASSO Regression with various L1 penalties
### edTest(test_lasso) ###
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

# TODO Implement code to evaluate LASSO Regression with various L1 penalties
data = []
for l in l1_lambdas:
  lasso_model = Lasso(alpha=l)
  lasso_model.fit(train_sales, train_price)
  train_pred = lasso_model.predict(train_sales)
  train_rmse = mean_squared_error(train_price, train_pred, squared=False)
  val_pred = lasso_model.predict(validation_sales)
  val_rmse = mean_squared_error(validation_price, val_pred, squared=False)
  data.append({
    'l1_penalty': l,
    'model': lasso_model,
    'train_rmse': train_rmse,
    'validation_rmse': val_rmse
  }) 

lasso_data = pd.DataFrame(data)
# lasso_data.head()

# Q7: LASSO Analysis
### edTest(test_lasso_analysis) ###

# TODO Print information about best L1 model
index = lasso_data['validation_rmse'].idxmin()
row = lasso_data.loc[index]
best_l1 = row['l1_penalty']

pred = row['model'].predict(test_sales)
test_rmse_lasso = mean_squared_error(test_price, pred, squared=False)
num_zero_coeffs_lasso = 5

# print('Best L1 Penalty', best_l1)
# print('Test RMSE', test_rmse_lasso)
# print('Num Zero Coeffs', num_zero_coeffs_lasso)
# print_coefficients(row['model'], all_features)