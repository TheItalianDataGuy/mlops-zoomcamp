import pandas as pd
from evidently.report import Report
from evidently.metrics import ColumnQuantileMetric, ColumnDriftMetric

# Load reference and current data
ref_df = pd.read_parquet("green_tripdata_2024-02.parquet")
curr_df = pd.read_parquet("green_tripdata_2024-03.parquet")

# Create the report
report = Report(metrics=[
    ColumnQuantileMetric(column_name="fare_amount", quantile=0.5),
    ColumnDriftMetric(column_name="fare_amount")
])

# Run and save
report.run(reference_data=ref_df, current_data=curr_df)
report.save_html("fare_monitoring_report.html")

print("Report saved as fare_monitoring_report.html")