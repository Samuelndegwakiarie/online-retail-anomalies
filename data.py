import pandas as pd
from sklearn.datasets import fetch_openml

# Fetch dataset from UCI Machine Learning Repository
online_retail = fetch_openml(data_id=524)

# Convert the fetched data to a pandas DataFrame
df_online_retail = pd.DataFrame(data=online_retail.data, columns=online_retail.feature_names)

# Display the first few rows of the online retail dataset
print("First few rows of the online retail dataset:")
print(df_online_retail.head())

# Accessing data (as pandas DataFrame)
X = df_online_retail  # Features are stored in X
y = online_retail.target  # Target variable is stored in y

# Metadata
print("\nMetadata:")
print(online_retail.details)

# Variable information
print("\nVariable information:")
print(online_retail.feature_names)

# Define the file path to the local CSV file
file_path = "path_to_your_dataset/OnlineRetail.csv"  # Replace "path_to_your_dataset" with the actual path

# Load the dataset from the local CSV file into a pandas DataFrame
df_local_csv = pd.read_csv(file_path, encoding="ISO-8859-1")

# Display the first few rows of the local CSV dataset
print("\nFirst few rows of the local CSV dataset:")
print(df_local_csv.head())

# Check the dimensions of the local CSV dataset (number of rows and columns)
print("\nDimensions of the local CSV dataset:")
print(df_local_csv.shape)

# Check the data types of each column in the local CSV dataset
print("\nData types of each column in the local CSV dataset:")
print(df_local_csv.dtypes)

# Check for missing values in the local CSV dataset
print("\nMissing values in the local CSV dataset:")
print(df_local_csv.isnull().sum())

# Check for duplicate entries in the local CSV dataset
print("\nDuplicate entries in the local CSV dataset:")
print(df_local_csv.duplicated().sum())
