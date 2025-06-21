import pandas as pd

# Load March 2024 Green Taxi data
df = pd.read_parquet("green_tripdata_2024-03.parquet")

# Convert datetime columns
df["lpep_pickup_datetime"] = pd.to_datetime(df["lpep_pickup_datetime"])
df["lpep_dropoff_datetime"] = pd.to_datetime(df["lpep_dropoff_datetime"])

# ✅ Filter only March 2024
df = df[(df["lpep_pickup_datetime"].dt.month == 3) & (df["lpep_pickup_datetime"].dt.year == 2024)]

# ✅ Compute trip duration in minutes
df["duration"] = (df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]).dt.total_seconds() / 60

# ✅ Apply duration and fare filters
df = df[(df["duration"] >= 1) & (df["duration"] <= 60)].copy()
df = df[(df["fare_amount"] >= 1) & (df["fare_amount"] <= 60)].copy()

# ✅ Extract pickup date for grouping
df["pickup_date"] = df["lpep_pickup_datetime"].dt.date

# ✅ Compute 0.5 quantile (median) of fare_amount by day
daily_medians = df.groupby("pickup_date")["fare_amount"].quantile(0.5)

# ✅ Print all daily medians
print(daily_medians)

# ✅ Print the maximum
max_median = daily_medians.max()
print(f"\n🚨 Max daily median fare_amount in March 2024: {max_median:.2f}")
