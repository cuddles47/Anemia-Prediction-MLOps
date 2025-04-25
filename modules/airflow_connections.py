from airflow import DAG
from airflow.models import Variable
from airflow.hooks.base import BaseHook
from datetime import datetime
from minio import Minio
from sqlalchemy import create_engine

def _get_minio_connection():
    conn = BaseHook.get_connection("minio_connection")
    minio_endpoint = conn.host  # Should include http:// or https://
    access_key = conn.login
    secret_key = conn.password
    minio_client = Minio(
        endpoint=f"{minio_endpoint}:9000",
        access_key=access_key,
        secret_key=secret_key,
        secure=False
    )
    return minio_client

def _get_postgres_connection(schema=None):
    postgres_conn_id = "postgres_connection"
    conn = BaseHook.get_connection(postgres_conn_id)
    
    # Nếu schema được cung cấp, sử dụng nó; nếu không, dùng schema từ connection
    schema_to_use = schema if schema else conn.schema
    
    engine = create_engine(f"postgresql://{conn.login}:{conn.password}@{conn.host}:{conn.port}/{schema_to_use}")
    return engine.connect()
