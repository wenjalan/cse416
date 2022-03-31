# Intro to Python & Pandas
Python is an interpreted language, commonly used for data analysis and machine learning. The library pandas will be used in CSE 416 to handle data processing.

## Setup
First, import `numpy` and `pandas`. Look up instructions to intsall these packages on the internet if you're running locally.

```py
import numpy as np
import pandas as pd
```

## Pandas
### Load a CSV
Pandas is a library that implements dataframes in Python. To create a dataframe from a `.csv` file:

```py
# read filename.csv from the disk into a dataframe
dataframe = pd.read_csv("filename.csv")

# display the contents of the dataframe
dataframe

# display the first few rows of the dataframe
dataframe.head()
```

### Accessing Columns
To access a certain column of the dataframe, use bracket notation. Several aggregate functions can be used with columns.

```py
# display the contents of column col_a in dataframe
dataframe['col_a']

# get the minimum value of column price in dataframe
min_price = dataframe['price'].min()

# get the maximum profit
max_profit = dataframe['profit'].max()

# get the mean height
mean_height = dataframe['height'].mean()
```

The `dtypes` property of a dataframe will provide the data types of the columns.

```py
dataframe.dtypes
```

### Accessing Rows & Cells
The `iloc` property, used with bracket notation, can be used to find a specific row of a column. Rows are 0-indexed.

```py
# get the row at index 8
row = dataframe.iloc[8]

# get the cell at col 'price' and row 8
# alternative syntaxes provided
cell = dataframe['price'].iloc[8]
cell = dataframe[8]['price'] # probably gonna use this one
cell = dataframe.iloc[8]['price']
```

### Filtering
Pandas also supports filtering with conditions.

```py
# get all rows where recipient == James
emailsToJames = dataframe[dataframe['recipient'] == 'James']

# get all rows where price < 100 and country = USA
cheapUSA = dataframe[
    (dataframe['price'] < 100) & 
    (dataframe['country'] == 'USA')
]

# get the shape (dimensions) of the dataframe provided by the above query
cheapUSA.shape
```

### Sorting
Pandas lets you sort data with `sort_values()`.

```py
# Sort rows by name
dataframe.sort_values(by = ['name'])

# Sort rows by price, descending
dataframe.sort_values(by = ['price'], ascending = False)
```
