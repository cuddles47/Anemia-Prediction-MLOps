import pandas as pd
import numpy as np
from io import BytesIO
import json
import io
from typing import List, Dict

from ump.modules.airflow_connections import _get_minio_connection
from ump.modules.utlis import get_unique_values, check_keywords, check_type, check_view

"""
#TODO
Có thể define dưới dạng 1 class
Class này sẽ define các thông tin của 1 bucket
- bucket-name
- Object path, raw, intermediate, processed,...
- client connection
- secret (class sẽ tự retrieve từ đâu đó)
- password (class sẽ tự retrieve từ đâu đó)
- Có thể code theo hướng thừa kế, 1 parent class định nghĩa các
functions phải có khi làm việc với MinIO
"""

def minio_object_to_dataframe(minio_object):
    """_summary_

    Args:
        minio_object (_type_): _description_

    Returns:
        _type_: _description_
    """
    df = pd.read_csv(io.BytesIO(minio_object.read()), low_memory=False)
    return df

def minio_object_to_dataframe_with_column(minio_object, columns_to_read: List):
    df = pd.read_csv(io.BytesIO(minio_object.read()), 
                     usecols=columns_to_read,
                     low_memory=False)
    return df

def drop_duplicate_row(df: pd.DataFrame):
    return df.drop_duplicates()

def drop_duplicate_column(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(axis=1, how='all')

def drop_null(df: pd.DataFrame, column_to_drop: List[str]) -> pd.DataFrame:
    return df.dropna(subset=column_to_drop)

def get_minio_object(BUCKET_NAME, OBJECT_PATH):
    minio_client = _get_minio_connection()
    response = minio_client.get_object(bucket_name=BUCKET_NAME,
                      object_name=OBJECT_PATH)
    return response

def put_object_to_minio(client, put_object, BUCKET_NAME, OBJECT_PATH):
    client.put_object(
        BUCKET_NAME, OBJECT_PATH, io.BytesIO(put_object), 5,
        )
    return 0

def put_file_to_minio(BUCKET_NAME, OBJECT_PATH, OUTPUT_FILE_PATH):
    minio_client = _get_minio_connection()
    minio_client.fput_object(bucket_name=BUCKET_NAME,
                            object_name=OBJECT_PATH,
                            file_path=OUTPUT_FILE_PATH,
                            content_type="application/csv")
    return 0

def list_minio_object(client, BUCKET_NAME, OBJECT_PATH):
    response = client.list_objects(
        bucket_name=BUCKET_NAME,
        prefix=f"{OBJECT_PATH}/",
        # recursive=True
    )  # ['ResponseMetadata', 'IsTruncated', 'Contents', 'Name', 'Prefix', 'MaxKeys', 'EncodingType', 'KeyCount']

    csv_files = []
    for obj in response:
        serialize_obj = json.loads(json.dumps(
            obj.__dict__, indent=4, default=str))
        prefix = serialize_obj['_object_name']
        file_path = f"s3://{BUCKET_NAME}/{prefix}"
        csv_files.append(file_path)
    return csv_files

def concatenate_files(BUCKET_NAME: str, OBJECT_PATH: str, axis=0) -> pd.DataFrame:
    minio_client = _get_minio_connection()
    file_paths = list_minio_object(minio_client, BUCKET_NAME, OBJECT_PATH)
    
    dfs = []
    for path in file_paths:
        path_parts = path.replace("s3://", "").split("/", 1)
        bucket_name = path_parts[0]
        object_name = path_parts[1]
        response = minio_client.get_object(bucket_name, object_name)
        df  = minio_object_to_dataframe(response)
        dfs.append(df)
    print(file_paths)
    if axis==0:
        final_df = pd.concat(dfs, ignore_index=False)
    elif axis==1:
        final_df = pd.concat(dfs, ignore_index=False, axis=1)
    
    print(final_df.columns)
    return final_df

def exclude_unused_data(df):
    df = drop_duplicate_row(df)
    df = drop_duplicate_column(df)
    null_column_to_drop = ["streetAddress", "addressLocality", "addressRegion"]
    df = drop_null(df, column_to_drop=null_column_to_drop)
    return df

def join_data(BUCKET_NAME, OBJECT_PATH):
    concatenate_df = concatenate_files(BUCKET_NAME, OBJECT_PATH, axis=1)
    return concatenate_df

def drop_old_columns(df: pd.DataFrame):
    """_summary_

    Args:
        df (pd.DataFrame): _description_

    Returns:
        _type_: _description_
    """    
    df = df.drop(columns=['streetAddress', 'postalCode', 'description', 'priceCurrency', 
                        "Parking Features", 'Basement', 'Exterior', 'Features', 'Fireplace', 
                        'Garage', 'Heating', 'MLS® #', 'Roof', 'Sewer', 'Waterfront', 
                        'Parking', 'Flooring', 'Fireplace Features', 'Subdivision', 
                        'Square Footage', 'Bath', 'Property Tax'], axis=1)
    
    
    return df

def rename_column(df):
    """
    Generates a dictionary mapping old column names to new ones based on a predefined renaming rule.

    Args:
        columns_list (List[str]): A list of original column names.

    Returns:
        Dict[str, str]: A dictionary where keys are original column names and values are the renamed versions.
    """
        #TODO Nhớ check tên cột

    df = df.rename(columns={'addressLocality': 'City', 
                            'addressRegion': 'Province', 
                            'latitude': 'Latitude', 
                            'longitude': 'Longitude', 
                            'price': 'Price',
                            'property-baths': 'Bathrooms', 
                            'property-beds': 'Bedrooms', 
                            'property-sqft': 'Square_Footage', 
                            'Garage new': 'Garage',	
                            'Parking_new': 'Parking',	
                            'Basement_new': 'Basement', 
                            'Exterior_new': 'Exterior', 
                            'Fireplace_new': 'Fireplace', 
                            'Heating_new': 'Heating', 
                            'Flooring_new': 'Flooring', 
                            'Roof_new': 'Roof', 
                            'Waterfront_new': 'Waterfront', 
                            'Sewer_new': 'Sewer'})
    return df

# Function to replace NaN values with the mean of the column
def reloc_nan(x):
    """
    Replace NaN values in a pandas Series with the mean of the column.

    """
    mean_value = x.mean()  # Calculate the mean value of the column
    return x.fillna(mean_value)  # Replace NaN values with the mean
