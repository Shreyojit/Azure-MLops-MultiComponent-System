from azureml.core import Workspace, Dataset, Datastore
import pandas as pd
import numpy as np
import os
import argparse
from datetime import datetime, date, timedelta
from sklearn.preprocessing import QuantileTransformer
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core import Run

#------------------------------Auth-------------------------------#
print("Authenticating AzureML workspace...")
ws = Workspace(subscription_id="7b8ef4c6-77cc-453a-81db-0c0c47f97eca",
               workspace_name="mlops-wp",
               resource_group="mlops-learn")
print("Workspace authenticated successfully.")

#------------------------------Data Import-------------------------------#
data_store_name = "workspaceblobstore"
container_name = os.getenv("BLOB_CONTAINER", "blobstoreagemlops")
account_name = os.getenv("BLOB_ACCOUNTNAME", "blobstoreagemlops")
print(f"Datastore Name: {data_store_name}, Container Name: {container_name}")

datastore = Datastore.get(ws, data_store_name)
print(f"Datastore '{data_store_name}' retrieved successfully.")

#------------------------------Argparser-------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument("--prep", type=str)
args = parser.parse_args()
print(f"Argument received: --prep={args.prep}")

#------------------------------Run-------------------------------#
print("Getting AzureML Run context...")
run = Run.get_context()
print("Run context retrieved.")

#------------------------------Read_data-------------------------------#
print("Reading dataset from datastore...")
try:
    df = Dataset.Tabular.from_delimited_files(path=[(datastore, args.prep)]).to_pandas_dataframe()
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise

#------------------------------Transf/prep-------------------------------#
print("Dropping duplicates from dataset...")
df = df.drop_duplicates()
print(f"Dataset shape after dropping duplicates: {df.shape}")

# Handling missing or zero values
print("Replacing zero values in specific columns with mean/median...")
df['Glucose'] = df['Glucose'].replace(0, df['Glucose'].mean())
df['BloodPressure'] = df['BloodPressure'].replace(0, df['BloodPressure'].mean())
df['SkinThickness'] = df['SkinThickness'].replace(0, df['SkinThickness'].median())
df['Insulin'] = df['Insulin'].replace(0, df['Insulin'].median())
df['BMI'] = df['BMI'].replace(0, df['BMI'].median())
print("Zero value replacement completed.")

# Transformations
print("Performing quantile transformation on selected columns...")
df_selected = df[['Pregnancies', 'Glucose', 'SkinThickness', 'BMI', 'Age', 'Outcome']]
quantile = QuantileTransformer()
X = quantile.fit_transform(df_selected)
df_new = pd.DataFrame(X, columns=['Pregnancies', 'Glucose', 'SkinThickness', 'BMI', 'Age', 'Outcome'])
print("Transformation completed.")

#------------------------------Export Data-------------------------------#
path = "tmp/"
print(f"Checking/creating export directory: {path}")
try:
    os.mkdir(path)
    print(f"Directory {path} created successfully.")
except OSError as e:
    print(f"Error creating directory {path}: {e}")

temp_path = os.path.join(path, "preprocessed.csv")
print(f"Saving transformed dataset to {temp_path}...")
df_new.to_csv(temp_path, index=False)
print("Dataset saved successfully.")

#------------------------------To Datastore-------------------------------#
print(f"Uploading preprocessed dataset to datastore '{data_store_name}'...")
try:
    datastr = Datastore.get(ws, data_store_name)
    datastr.upload(src_dir=path, target_path="", overwrite=True)
    print("Dataset uploaded to datastore successfully.")
except Exception as e:
    print(f"Error uploading dataset to datastore: {e}")
    raise

print("Data preprocessing Completed!")
