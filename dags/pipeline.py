from airflow import models
from airflow.operators.python_operator import PythonOperator
from airflow.contrib.sensors.gcs_sensor import GoogleCloudStoragePrefixSensor
from airflow.contrib.operators.gcs_delete_operator import GoogleCloudStorageDeleteOperator
from pathlib import Path
from datetime import datetime
from config import config
from utils.loader import perekiduvach, predictor, retrainer

animals = config.animals
dag_name = Path(__file__).stem

with models.DAG(dag_name,
                default_args = {},
                start_date = datetime(2021, 8, 8),
                schedule_interval = '30 14 * * *',
                catchup = False) as dag:

    image_transfer = PythonOperator(
        task_id = f'transfer_and_sort_animals',
        name = f'transfer_and_sort_animals',
        python_callable = perekiduvach
    )

    image_predictor = PythonOperator(
        task_id = f'predict_animals',
        name = f'predict_animals',
        python_callable = predictor
    )

    model_retrain = PythonOperator(
        task_id = f'retrain_model',
        name = f'retrain_model',
        python_callable = retrainer
    )
    image_transfer >> image_predictor >> model_retrain