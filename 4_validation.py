import kagglehub
import os
import pandas as pd

path = kagglehub.dataset_download("pavan1512/ieee-cis-fraud-detection")
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
print(f"Files: {csv_files}")

df_paysim = pd.read_csv(os.path.join(path, csv_files[0]))
print(f"Shape: {df_paysim.shape}")
print(f"Columns: {df_paysim.columns.tolist()}")
print(df_paysim.head(3))
print(df_paysim["isFraud"].value_counts())