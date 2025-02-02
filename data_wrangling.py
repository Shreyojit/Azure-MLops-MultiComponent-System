from azureml.core import Workspace, Dataset, Datastore
import pandas as pd
import numpy as np
import os
import argparse
from datetime import datetime, date, timedelta
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core import Run

#------------------------------Auth-------------------------------#


ws = Workspace(subscription_id = "7b8ef4c6-77cc-453a-81db-0c0c47f97eca",
                workspace_name = "mlops-wp",
                resource_group = "mlops-learn",
)

#------------------------------End Auth-------------------------------#
#------------------------------Data Import-------------------------------#

data_store_name = "workspaceblobstore"
container_name = os.getenv("BLOB_CONTAINER", "blobstoreagemlops")
account_name = os.getenv("BLOB_ACCOUNTNAME", "blobstoreagemlops")

datastore = Datastore.get(ws, data_store_name)

#------------------------------Argparser-------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument("--input-data", type=str)
args = parser.parse_args()
#------------------------------Run-------------------------------#
run = Run.get_context()
#------------------------------Read_data-------------------------------#
#df = Dataset.Tabular.from_delimited_files(path=[(datastore, "diabetes.csv")]).to_pandas_dataframe()
df = Dataset.Tabular.from_delimited_files(path=[(datastore, args.input_data)]).to_pandas_dataframe()

print("Shape of df", df.shape)

#------------------------------Export Data-------------------------------#
path = "tmp/"
try:
    os.mkdir(path)
except OSError:
    print("Creation of directory %s failed!" % path)
else:
    print("Sucessfully created the directory %s" % path)

temp_path = path + "wrangled.csv"
df.to_csv(temp_path)

#------------------------------To Datastore-------------------------------#
datastr = Datastore.get(ws, data_store_name)
datastr.upload(src_dir = path, target_path = "", overwrite=True)

print("Data Wrangling Completed!")