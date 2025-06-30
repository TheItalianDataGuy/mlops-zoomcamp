import pandas as pd
from datetime import datetime
import os

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

# Use the raw data used in Q3 **before prepare_data**
data = [
    (None, None, dt(1, 1), dt(1, 10)),         # 9 min
    (1, 1, dt(1, 2), dt(1, 10)),               # 8 min
    (1, None, dt(1, 2, 0), dt(1, 2, 59)),      # 59 sec (gets dropped later)
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),          # 1441 min (gets dropped later)
]
columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
df = pd.DataFrame(data, columns=columns)

# Save to Localstack S3
year = 2023
month = 1

input_file = f"s3://nyc-duration/in/{year:04d}-{month:02d}.parquet"
s3_endpoint_url = os.getenv("S3_ENDPOINT_URL", "http://localhost:4566")

options = {
    "client_kwargs": {
        "endpoint_url": s3_endpoint_url
    }
}

df.to_parquet(
    input_file,
    engine='pyarrow',
    compression=None,
    index=False,
    storage_options=options
)


# Set env vars so batch.py reads from Localstack
os.environ["INPUT_FILE_PATTERN"] = "s3://nyc-duration/in/{year:04d}-{month:02d}.parquet"
os.environ["OUTPUT_FILE_PATTERN"] = "s3://nyc-duration/out/{year:04d}-{month:02d}.parquet"
os.environ["S3_ENDPOINT_URL"] = "http://localhost:4566"

# Run batch.py
os.system("python batch.py 2023 1")

# Read the prediction result
output_path = "s3://nyc-duration/out/2023-01.parquet"
options = {
    "client_kwargs": {
        "endpoint_url": os.getenv("S3_ENDPOINT_URL")
    }
}

df_result = pd.read_parquet(output_path, storage_options=options)

print(df_result)
print("Sum of predicted durations:", round(df_result["predicted_duration"].sum(), 2))
