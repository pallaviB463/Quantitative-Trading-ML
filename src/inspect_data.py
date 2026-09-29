import pandas as pd

# Load the original research dataset
file_path = "data/raw/databaseFrankfurtComplete.parquet"

df = pd.read_parquet(file_path)

# Display dataset dimensions
print("Dataset shape:")
print(df.shape)

# Display column names
print("\nColumn names:")
print(df.columns.tolist())

# Display first five rows
print("\nFirst 5 rows:")
print(df.head())

# Display data types
print("\nDataset information:")
df.info()

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check the index
print("\nIndex information:")
print(df.index)