import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import numpy as np
sales = pd.read_csv('home_data.csv')

# Q1
num_rows = sales['id'].count()
y = sales['price']
num_inputs = sales.drop('price', axis=1).count().sum()

# Q2
sales_3_bedrooms = sales[sales['bedrooms'] == 3]
avg_price_3_bed = sales_3_bedrooms['price'].mean()

# Q3
sqft_living_2000_to_4000 = sales[(sales['sqft_living'] >= 2000) & (sales['sqft_living'] < 4000)]
percent_q3 = sqft_living_2000_to_4000['id'].count() / sales['id'].count()

# Q4
# Set seed to create pseudo-randomness
np.random.seed(416)

# Split data into 80% train and 20% validation
train_data, val_data = train_test_split(sales, test_size=0.2)

basic_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']

advanced_features = basic_features + [
    'condition',      # condition of the house
    'grade',          # measure of qality of construction
    'waterfront',     # waterfront property
    'view',           # type of view
    'sqft_above',     # square feet above ground
    'sqft_basement',  # square feet in basementab
    'yr_built',       # the year built
    'yr_renovated',   # the year renovated
    'lat',            # the longitude of the parcel
    'long',           # the latitide of the parcel
    'sqft_living15',  # average sq.ft. of 15 nearest neighbors
    'sqft_lot15',     # average lot size of 15 nearest neighbors
]
from sklearn.linear_model import LinearRegression
basic_model = LinearRegression().fit(train_data[basic_features], train_data['price'])
advanced_model = LinearRegression().fit(train_data[advanced_features], train_data['price'])

# Q5
from sklearn.metrics import mean_squared_error

train_basic_pred = basic_model.predict(train_data[basic_features])
train_rmse_basic = mean_squared_error(train_data['price'], train_basic_pred, squared=False)
# print('Train RMSE Basic: ' + str(train_rmse_basic))

train_advanced_pred = advanced_model.predict(train_data[advanced_features])
train_rmse_advanced = mean_squared_error(train_data['price'], train_advanced_pred, squared=False)
# print('Train RMSE Advanced: ' + str(train_rmse_advanced))

# Q6
val_basic_pred = basic_model.predict(val_data[basic_features])
val_rmse_basic = mean_squared_error(val_data['price'], val_basic_pred, squared=False)
# print('Val RMSE Basic: ' + str(val_rmse_basic))

val_advanced_pred = advanced_model.predict(val_data[advanced_features])
val_rmse_advanced = mean_squared_error(val_data['price'], val_advanced_pred, squared=False)
# print('Val RMSE Advanced: ' + str(val_rmse_advanced))