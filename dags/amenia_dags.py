import logging
from dags.ump.modules.airflow_connections import _get_postgres_connection
from pendulum import datetime
import pandas as pd
import boto3
from minio.error import S3Error
from minio import Minio
from astro.sql.table import Table
from astro import sql as aql
from airflow.utils.task_group import TaskGroup
from airflow.operators.empty import EmptyOperator
from airflow.hooks.base import BaseHook
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.models import Variable
from airflow.decorators import dag, task, task_group

from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import mlflow
from mlflow.models import infer_signature
from mlflow import MlflowClient

from ump.modules.data_processing import *
BUCKET_NAME = "children-anemia"
RAW_OBJECT_PATH = 'data/raw/files'
INTERMEDIATE_OBJECT_PATH = "data/intermediate/files"
PROCESSED_OBJECT_PATH = "data/processed/files"
FINAL_OBJECT_PATH = "data/final/files"

logger = logging.getLogger(__name__)

@dag(
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    doc_md=__doc__,
    default_args={"owner": "Astro", "retries": 2},
    tags=["children-anemia-dev1"],
)
def Quan_children_anemia():
    
        # OK
        @task
        def rename_columns():
            response = get_minio_object(BUCKET_NAME, 
                                        RAW_OBJECT_PATH + "/children_anemia.csv")
            rename_columns_df = minio_object_to_dataframe(response)
            OUTPUT_PATH = "/tmp/rename_columns.csv"
            
            rename_columns_df.rename(columns={'Age in 5-year groups': 'age',
                                              'Type of place of residence': 'residence',
                                              'Highest educational level': 'highest_educational',
                                              'Wealth index combined': 'wealth_index',
                                              'Births in last five years': 'births_5_years',
                                              'Age of respondent at 1st birth': 'respondent_1st_birth',
                                              'Hemoglobin level adjusted for altitude and smoking (g/dl - 1 decimal)': 'hemoglobin_altitude_smoking',
                                              'Anemia level': 'anemia_level_target',
                                              'Have mosquito bed net for sleeping (from household questionnaire)': 'mosquito_bed_sleeping',
                                              'Smokes cigarettes': 'smoking',
                                              'Current marital status': 'status',
                                              'Currently residing with husband/partner': 'residing_husband_partner',
                                              'When child put to breast': 'child_put_breast',
                                              'Had fever in last two weeks': 'fever_two_weeks',
                                              'Hemoglobin level adjusted for altitude (g/dl - 1 decimal)': 'hemoglobin_altitude',
                                              'Anemia level.1': 'anemia_level_1',
                                              'Taking iron pills, sprinkles or syrup': 'iron_pills'}, inplace=True)
            
            rename_columns_df.to_csv(OUTPUT_PATH, index=False)
            put_file_to_minio(BUCKET_NAME,
                            INTERMEDIATE_OBJECT_PATH + "/rename_columns.csv",
                            OUTPUT_PATH)
            return 0

        @task_group
        def task_group_1():
            
            @task
            def hemoglobin_altitude_smoking():   
                response = get_minio_object(BUCKET_NAME, 
                                        INTERMEDIATE_OBJECT_PATH + "/rename_columns.csv")
                smoking_df = minio_object_to_dataframe_with_column(response, columns_to_read=["hemoglobin_altitude_smoking"])
                OUTPUT_PATH = "/tmp/g1_hemoglobin_altitude_smoking.csv"
                
                # Replace NaN values in the 'hemoglobin_altitude_smoking' column with the mean of the column
                smoking_df['hemoglobin_altitude_smoking'] = reloc_nan(smoking_df['hemoglobin_altitude_smoking'])
                
                smoking_df.to_csv(OUTPUT_PATH, index=False)
                put_file_to_minio(BUCKET_NAME,
                            PROCESSED_OBJECT_PATH + "/g1_hemoglobin_altitude_smoking.csv",
                            OUTPUT_PATH)
                return 0
            
            # OK
            @task
            def child_put_breast():
                response = get_minio_object(BUCKET_NAME, 
                                        INTERMEDIATE_OBJECT_PATH + "/rename_columns.csv")
                child_put_breast_df = minio_object_to_dataframe_with_column(response, columns_to_read=["child_put_breast"])
                OUTPUT_PATH = "/tmp/g1_child_put_breast.csv"
                
                def func_child_put_breast(dot):
                    values = dot['child_put_breast'].values
                    for idx, val in enumerate(values):
                        values[idx] = 0 if val == 'Immediately' else (0.5 if val == 'Hours: 1' else (1 if val == 'Days: 1' else val))
                    dot['child_put_breast'] = values
                    return dot

                # Apply function to convert categorical values in 'child_put_breast' column
                dot = func_child_put_breast(child_put_breast_df)

                # Convert the 'child_put_breast' column to the float64 data type
                dot['child_put_breast'] = dot['child_put_breast'].astype('float64')

                # Replace NaN values in the 'child_put_breast' column with the mean of the column
                dot['child_put_breast'] = reloc_nan(dot['child_put_breast'])

                dot.to_csv(OUTPUT_PATH, index=False)
                put_file_to_minio(BUCKET_NAME,
                                PROCESSED_OBJECT_PATH + "/g1_child_put_breast.csv",
                                OUTPUT_PATH)
                return 0

            # OK
            @task
            def hemoglobin_altitude():
                response = get_minio_object(BUCKET_NAME, 
                                        INTERMEDIATE_OBJECT_PATH + "/rename_columns.csv")
                hemoglobin_altitude_df = minio_object_to_dataframe_with_column(response, columns_to_read=["hemoglobin_altitude"])
                OUTPUT_PATH = "/tmp/g1_hemoglobin_altitude.csv"
                
                # Replace NaN values in 'hemoglobin_altitude' column with the mean and round to 1 decimal place
                hemoglobin_altitude_df['hemoglobin_altitude'] = reloc_nan(hemoglobin_altitude_df['hemoglobin_altitude']).apply(lambda x: round(x, 1))
                
                hemoglobin_altitude_df.to_csv(OUTPUT_PATH, index=False)
                put_file_to_minio(BUCKET_NAME,
                                PROCESSED_OBJECT_PATH + "/g1_hemoglobin_altitude.csv",
                                OUTPUT_PATH)
                return 0
         
            t1 = hemoglobin_altitude_smoking()
            t2 = child_put_breast()
            t3 = hemoglobin_altitude()
            [t1, t2, t3]

        @task_group
        def task_group_2():
            
            # OK
            @task
            def residing_husband_partner():
                response = get_minio_object(BUCKET_NAME, 
                                            INTERMEDIATE_OBJECT_PATH + "/rename_columns.csv")
                husband_df = minio_object_to_dataframe_with_column(response, columns_to_read=["residing_husband_partner"])
                OUTPUT_PATH = "/tmp/g2_residing_husband_partner.csv"
                
                def replace_nan_residing(x):
                    """
                    Replace NaN values in a pandas Series with a specific word.

                    """
                    word = 'Staying elsewhere'  # Specify the word to replace NaN values
                    return x.fillna(word)  # Replace NaN values with the specified word

                # Apply the function to replace NaN values in the 'residing_husband_partner' column
                husband_df['residing_husband_partner'] = replace_nan_residing(husband_df['residing_husband_partner'])
                husband_df.to_csv(OUTPUT_PATH, index=False)
                put_file_to_minio(BUCKET_NAME,
                                PROCESSED_OBJECT_PATH + "/g2_residing_husband_partner.csv",
                                OUTPUT_PATH)
                return 0
            
            # OK
            @task
            def convert_text_value():
                response = get_minio_object(BUCKET_NAME, 
                                            INTERMEDIATE_OBJECT_PATH + "/rename_columns.csv")
                columns = ["fever_two_weeks", "anemia_level_1", "iron_pills", "anemia_level_target"]
                fever_df = minio_object_to_dataframe_with_column(response, columns_to_read=columns)
                OUTPUT_PATH = "/tmp/g2_convert_text_value.csv"            
                fever_df[columns] = fever_df[columns].fillna("Dont know")
                
                fever_df.to_csv(OUTPUT_PATH, index=False)
                put_file_to_minio(BUCKET_NAME,
                                PROCESSED_OBJECT_PATH + "/g2_convert_text_value.csv",
                                OUTPUT_PATH)
                return 0
        
            t1 = residing_husband_partner()
            t2 = convert_text_value()
            [t1, t2]
        
        
        @task
        def calculate_mean_age():
            response = get_minio_object(BUCKET_NAME, 
                                            INTERMEDIATE_OBJECT_PATH + "/rename_columns.csv")
            calculate_mean_age_df = minio_object_to_dataframe_with_column(response, columns_to_read=["age"])
            OUTPUT_PATH = "/tmp/single_calculate_mean_age.csv"
         
         # Define a function to calculate the mean of age groups represented as strings   
            def func_mean_column_age(x):
                if isinstance(x, str):
                    start, end = map(int, x.split('-'))
                    return (start + end) / 2
                else:
                    return x

            # Apply the function to calculate the mean of age groups in the 'age' column
            calculate_mean_age_df['age'] = calculate_mean_age_df['age'].apply(lambda x: func_mean_column_age(x) if isinstance(x, str) else x)            
        
            calculate_mean_age_df.to_csv(OUTPUT_PATH, index=False)
            put_file_to_minio(BUCKET_NAME,
                                PROCESSED_OBJECT_PATH + "/single_calculate_mean_age.csv",
                                OUTPUT_PATH)
            return 0
        
        @task
        def value_mapping_0_1():
            response = get_minio_object(BUCKET_NAME, 
                                        INTERMEDIATE_OBJECT_PATH + "/rename_columns.csv")
            columns = ["mosquito_bed_sleeping", "smoking", "fever_two_weeks", "iron_pills"]
            value_mapping_df = minio_object_to_dataframe_with_column(response, columns_to_read=columns)
            OUTPUT_PATH = "/tmp/single_value_mapping_0_1.csv"
            
            # Replace values in the specified columns with 0 and 1
            value_mapping_df[columns] = value_mapping_df[columns].replace({'No': 0, 'Yes': 1})
            
            value_mapping_df.to_csv(OUTPUT_PATH, index=False)
            put_file_to_minio(BUCKET_NAME,
                            PROCESSED_OBJECT_PATH + "/single_value_mapping_0_1.csv",
                            OUTPUT_PATH)
            return 0
        
        @task
        def replace_no_longer():
            response = get_minio_object(BUCKET_NAME, 
                                        INTERMEDIATE_OBJECT_PATH + "/rename_columns.csv")
            replace_no_longer_df = minio_object_to_dataframe_with_column(response, columns_to_read=["status"])
            OUTPUT_PATH = "/tmp/single_replace_no_longer_with_separate.csv"
            
            # Replace 'No longer' with 'No' in the specified columns
            replace_no_longer_df['status'] = replace_no_longer_df['status'].replace({'No longer': 'No'})
            
            replace_no_longer_df.to_csv(OUTPUT_PATH, index=False)
            put_file_to_minio(BUCKET_NAME,
                            PROCESSED_OBJECT_PATH + "/single_replace_no_longer_with_separate.csv",
                            OUTPUT_PATH)
            return 0
        
        @task
        def create_dummy_variables():
            response = get_minio_object(BUCKET_NAME, 
                                        INTERMEDIATE_OBJECT_PATH + "/rename_columns.csv")
            columns = ["residence", "highest_educational", "wealth_index"]
            create_dummy_variables_df = minio_object_to_dataframe_with_column(response, columns_to_read=columns)
            print(create_dummy_variables_df.head())
            OUTPUT_PATH = "/tmp/single_create_dummy_and_drop_columns.csv"
            
            # Create dummy variables for specified categorical columns and concatenate them to the DataFrame
            col = ['residence', 'highest_educational', 'wealth_index']

            for column in col:
                # Generate dummy variables and drop the first category to avoid multicollinearity
                status = pd.get_dummies(create_dummy_variables_df[column], prefix=column, drop_first=False)
                
                # Concatenate dummy variables to the original DataFrame
                create_dummy_variables_df = pd.concat([create_dummy_variables_df, status], axis=1)

                # Drop specified columns 'residence', 'highest_educational', 'wealth_index' from the DataFrame
            create_dummy_variables_df = create_dummy_variables_df.drop(columns=['residence', 'highest_educational', 'wealth_index'], axis=1)
            
            create_dummy_variables_df.to_csv(OUTPUT_PATH, index=False)
            put_file_to_minio(BUCKET_NAME,
                            PROCESSED_OBJECT_PATH + "/single_create_dummy_and_drop_columns.csv",
                            OUTPUT_PATH)
            return 0
        
        @task
        def join_data():
        # Danh sách các file trung gian đã xử lý
            file_paths = [
                "g1_hemoglobin_altitude_smoking.csv",
                "g1_child_put_breast.csv",
                "g1_hemoglobin_altitude.csv",
                "g2_residing_husband_partner.csv",
                "g2_convert_text_value.csv",
                "single_calculate_mean_age.csv",
                "single_value_mapping_0_1.csv",
                "single_replace_no_longer_with_separate.csv",
                "single_create_dummy_and_drop_columns.csv"
            ]

            # Đọc lại file gốc đã rename để giữ thứ tự gốc ban đầu
            response = get_minio_object(BUCKET_NAME, INTERMEDIATE_OBJECT_PATH + "/rename_columns.csv")
            rename_columns_df = minio_object_to_dataframe(response)

            # Đọc từng file trong danh sách và nối theo chiều ngang
            dfs = []
            for file in file_paths:
                response = get_minio_object(BUCKET_NAME, f"{PROCESSED_OBJECT_PATH}/{file}")
                df = minio_object_to_dataframe(response)
                dfs.append(df)

            # Gộp tất cả các dataframe theo chiều ngang
            join_data_df = pd.concat(dfs, axis=1)

            OUTPUT_PATH = "/tmp/single_joined_data.csv"
            join_data_df.to_csv(OUTPUT_PATH, index=False)
            put_file_to_minio(BUCKET_NAME,
                            PROCESSED_OBJECT_PATH + "/single_joined_data.csv",
                            OUTPUT_PATH)
            return 0
        
        @task
        def rearrange_columns():
            try:
                # Lấy thông tin về cấu trúc của file raw data (file sau khi đã rename columns)
                response_original = get_minio_object(BUCKET_NAME, INTERMEDIATE_OBJECT_PATH + "/rename_columns.csv")
                original_df = minio_object_to_dataframe(response_original)
                
                # Lấy danh sách cột từ file gốc theo đúng thứ tự
                original_columns = original_df.columns.tolist()
                
                # Lấy dữ liệu từ file đã join
                response_joined = get_minio_object(BUCKET_NAME, PROCESSED_OBJECT_PATH + "/single_joined_data.csv")
                joined_df = minio_object_to_dataframe(response_joined)
                
                # Lấy danh sách các cột mới tạo ra (không có trong file gốc)
                new_columns = [col for col in joined_df.columns if col not in original_columns]
                
                # Tạo DataFrame kết quả với các cột được sắp xếp theo đúng thứ tự file gốc
                result_df = pd.DataFrame()
                
                # Sao chép từng cột từ file gốc theo đúng thứ tự
                for col in original_columns:
                    if col in joined_df.columns:
                        result_df[col] = joined_df[col]
                    else:
                        logger.warning(f"Column '{col}' from original data not found in joined data!")
                
                # Thêm các cột mới vào cuối
                for col in new_columns:
                    result_df[col] = joined_df[col]
                
                # Kiểm tra số lượng cột
                if len(result_df.columns) != len(joined_df.columns):
                    logger.warning(f"Column count mismatch! Original: {len(original_df.columns)}, Joined: {len(joined_df.columns)}, Result: {len(result_df.columns)}")
                
                # Lưu kết quả
                OUTPUT_PATH = "/tmp/single_rearranged_data.csv"
                result_df.to_csv(OUTPUT_PATH, index=False)
                put_file_to_minio(BUCKET_NAME,
                                INTERMEDIATE_OBJECT_PATH + "/single_rearranged_data.csv",
                                OUTPUT_PATH)
                
                logger.info(f"Data columns rearranged exactly as in raw data. Total columns: {len(result_df.columns)}")
                return 0
            except Exception as e:
                logger.error(f"Failed to rearrange data columns: {e}")
                raise

        @task
        def standard_text_value():
            response = get_minio_object(BUCKET_NAME, 
                                        INTERMEDIATE_OBJECT_PATH + "/single_rearranged_data.csv")
            standard_text_value_df = minio_object_to_dataframe(response)
            OUTPUT_PATH = "/tmp/single_standard_text_value.csv"
            
            # Define a function to replace specific values in the DataFrame with standardized values
            def func_replace(dot):
                def val_replace(value):
                    if value == "Don't know":
                        return 'Dont know'
                    else:
                        return value
                # Apply the value replacement function to all elements in the DataFrame
                dot = dot.applymap(val_replace)
                return dot

            # Apply the function to replace specific values in the DataFrame
            standard_text_value_df = func_replace(standard_text_value_df)
            
            standard_text_value_df.to_csv(OUTPUT_PATH, index=False)
            put_file_to_minio(BUCKET_NAME,
                            PROCESSED_OBJECT_PATH + "/single_standard_text_value.csv",
                            OUTPUT_PATH)
            return 0

        @task
        def drop_na():
            response = get_minio_object(BUCKET_NAME, 
                                        INTERMEDIATE_OBJECT_PATH + "/single_rearranged_data.csv")
            drop_na_df = minio_object_to_dataframe(response)
            OUTPUT_PATH = "/tmp/single_drop_na.csv"
            
            drop_na_df = drop_na_df.dropna()

            drop_na_df.to_csv(OUTPUT_PATH, index=False)
            put_file_to_minio(BUCKET_NAME,
                            PROCESSED_OBJECT_PATH + "/single_drop_na.csv",
                            OUTPUT_PATH)
            put_file_to_minio(BUCKET_NAME,
                            FINAL_OBJECT_PATH + "/children_anemia_clean.csv",
                            OUTPUT_PATH)
            return 0

        @task
        def upload_to_postgres():
            try:
                # Lấy dữ liệu từ MinIO
                response = get_minio_object(BUCKET_NAME, FINAL_OBJECT_PATH + "/children_anemia_clean.csv")
                upload_df = minio_object_to_dataframe(response)

                # Kiểm tra dữ liệu
                if upload_df.empty:
                    logger.warning("DataFrame is empty. No data to upload.")
                    return 1
                logger.info(f"DataFrame shape: {upload_df.shape}")

                # Kết nối tới PostgreSQL
                postgres_conn = _get_postgres_connection("formatted_zone_ump")
                table_name = "children_anemia_clean"

                # Đẩy dữ liệu vào PostgreSQL
                upload_df.to_sql(
                    name=table_name,
                    con=postgres_conn,
                    if_exists="replace",
                    index=False
                )
                logger.info(f"Data uploaded to PostgreSQL database 'formatted_zone_ump', table '{table_name}' successfully.")
                return 0
            except Exception as e:
                logger.error(f"Failed to upload data to PostgreSQL: {e}")
                raise
        
        @task
        def training_model():
            mlflow.sklearn.autolog()
            
            mlflow.set_tracking_uri(uri="http://192.168.88.216:5000")
            mlflow.set_experiment("MLflow Quickstart")
            
            response = get_minio_object(BUCKET_NAME, FINAL_OBJECT_PATH + "/children_anemia_clean.csv")
            dot = minio_object_to_dataframe(response)
            X = dot.drop(columns=['anemia_level_target'], axis=1)  # Features
            y = dot['anemia_level_target']  # Target variable            
            obj = [col for col in X.columns if dot[col].dtype == 'object']
            for i in obj:
                lr = LabelEncoder()  # Initialize a LabelEncoder for the current column
                X[i] = lr.fit_transform(X[i])  # Transform the categorical values to numerical labels
            y = y.map({'Dont know': 0, 'Moderate': 1, 'Mild': 2, 'Not anemic': 3, 'Severe': 4})
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
            
            std = StandardScaler()
            X_train = std.fit_transform(X_train)
            y_train = std.fit_transform(y_train)
            
            lrg = LogisticRegression()
            
            lrg.fit(X_train, y_train)
            y_pred = lrg.predict(X_test)
            def fetch_logged_data(run_id):
                client = MlflowClient()
                data = client.get_run(run_id).data
                tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
                artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
                return data.params, data.metrics, tags, artifacts
            
            
            
            return 0
        
        
        
        t1 = rename_columns()
        t2 = task_group_1()
        t3 = task_group_2()
        t4 = calculate_mean_age()
        t5 = value_mapping_0_1()
        t6 = replace_no_longer()
        t7 = create_dummy_variables()
        t8 = join_data()
        t9 = rearrange_columns()
        t10 = standard_text_value()
        t11 = drop_na()
        t12 = upload_to_postgres()
        t13 = training_model()
        
        t1 >> [t2, t3, t4, t5, t6, t7] >> t8  >> t9 >> t10 >> t11 >> t12 >> t13
        
Quan_children_anemia()