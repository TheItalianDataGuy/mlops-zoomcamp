#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import argparse
import pickle

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--year', type=int, required=True)
parser.add_argument('--month', type=int, required=True)
args = parser.parse_args()
year = args.year
month = args.month

# Load the model
with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

# Define the URL
url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'

# Load the data
df = pd.read_parquet(url)

# Compute duration
df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
df['duration'] = df.duration.dt.total_seconds() / 60
df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

# Prepare features
df['pickup_ids'] = df['PULocationID'].astype(str)
df['dropoff_ids'] = df['DOLocationID'].astype(str)
dicts = df[['pickup_ids', 'dropoff_ids']].to_dict(orient='records')

# Transform and predict
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)

# Print result
print("Mean predicted duration:", y_pred.mean())

# Mean predicted duration: 14.29228498722522