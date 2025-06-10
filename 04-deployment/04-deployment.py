#!/usr/bin/env python
# coding: utf-8



import pandas as pd



df = pd.read_parquet('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet')
df.head()


# Q1. Downloading the data
# We'll use the same NYC taxi dataset, but instead of "Green Taxi Trip Records", we'll use "Yellow Taxi Trip Records".
# Read the data for January. How many columns are there?

len(df.columns) # 19



# Q2. Computing duration
# Now let's compute the duration variable. It should contain the duration of a ride in minutes.
# What's the standard deviation of the trips duration in January?


df['duration'] = df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
df['duration'] = df['duration'].dt.total_seconds() / 60
df['duration'].std() # 42.59



# Q3. Dropping outliers
# Next, we need to check the distribution of the duration variable. There are some outliers. Let's remove them and keep only the records where the duration was between 1 and 60 minutes (inclusive).

# What fraction of the records left after you dropped the outliers?

df_filtered = df[(df['duration'] >= 1) & (df['duration'] <= 60)]

fraction = f'{(len(df_filtered) / len(df)) * 100}%'
fraction # 98.12%



# Q4. One-hot encoding
# Let's apply one-hot encoding to the pickup and dropoff location IDs. We'll use only these two features for our model.

# Turn the dataframe into a list of dictionaries (remember to re-cast the ids to strings - otherwise it will label encode them)
# Fit a dictionary vectorizer. Get a feature matrix from it.
# What's the dimensionality of this matrix (number of columns)?

df_filtered['pickup_ids'] = df_filtered['PULocationID'].astype(str)
df_filtered['dropoff_ids'] = df_filtered['DOLocationID'].astype(str)

df_dict = df_filtered[['pickup_ids', 'dropoff_ids']].to_dict(orient='records')

from sklearn.feature_extraction import DictVectorizer

dv = DictVectorizer()
matrix = dv.fit_transform(df_dict)
matrix.shape  # 515



# Q5. Training a model
# Now let's use the feature matrix from the previous step to train a model.
# Train a plain linear regression model with default parameters, where duration is the response variable
# Calculate the RMSE of the model on the training data
# What's the RMSE on train?

X_train = matrix
y_train = df_filtered['duration'].values

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error
import numpy as np

y_pred = lr.predict(X_train)
rmse = mean_squared_error(y_train, y_pred) # it does not accept the argument 'squared' so I use numpy to do that!
rmse = np.sqrt(rmse) 
rmse # 7.64



# Q6. Evaluating the model
# Now let's apply this model to the validation dataset (February 2023).
# What's the RMSE on validation?

df_feb = pd.read_parquet('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-02.parquet')

df_feb['duration'] = df_feb['tpep_dropoff_datetime'] - df_feb['tpep_pickup_datetime']
df_feb['duration'] = df_feb['duration'].dt.total_seconds() / 60

df_feb_filtered = df_feb[(df_feb['duration'] >= 1) & (df_feb['duration'] <= 60)]

df_feb_filtered['pickup_ids'] = df_feb_filtered['PULocationID'].astype(str)
df_feb_filtered['dropoff_ids'] = df_feb_filtered['DOLocationID'].astype(str)

df_feb_dict = df_feb_filtered[['pickup_ids', 'dropoff_ids']].to_dict(orient='records')

X_feb = dv.transform(df_feb_dict)
y_val = df_feb_filtered['duration'].values
y_pred = lr.predict(X_feb)

rmse_feb = mean_squared_error(y_pred, y_val) # it does not accept the argument 'squared' so I use numpy to do that!
rmse_feb = np.sqrt(rmse_feb)
rmse_feb # 7.81



import pickle

with open('model.bin', 'wb') as f_out:
    pickle.dump((dv, lr), f_out)

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)



# Q1. Notebook
# We'll start with the same notebook we ended up with in homework 1. 
# We cleaned it a little bit and kept only the scoring part. You can find the initial notebook here.
# Run this notebook for the March 2023 data.
# What's the standard deviation of the predicted duration for this dataset?

import pickle
import pandas as pd


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

df = read_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet')

df['pickup_ids'] = df['PULocationID'].astype(str)
df['dropoff_ids'] = df['DOLocationID'].astype(str)
dicts = df[['pickup_ids', 'dropoff_ids']].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)
y_pred.std() # 6.24


# Q2. What's the size of the output file?

year = 2023
month = 3

# Create a new column 'ride_id'
df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

# Create a new DataFrame to hold only the output columns
df_result = pd.DataFrame()

# Copy the ride_id to the new DataFrame
df_result['ride_id'] = df['ride_id']

# Add the predicted duration values to the new DataFrame
df_result['predicted_duration'] = y_pred

# Define the output filename based on the year and month
output_file = f'output_{year:04d}-{month:02d}.parquet'

# Save the result as a Parquet file using pyarrow
# No compression, and we don't include the index in the file
df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)

# 65.5MB ~ 66MB


# Q5. Parametrize the script 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--year', type=int, required=True)
parser.add_argument('--month', type=int, required=True)

args = parser.parse_args()
year = args.year
month = args.month

print("Mean predicted duration:", y_pred.mean())




