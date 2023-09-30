# import io
import json
import logging
# import numpy as np
import pandas as pd
import pickle

from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.models import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago


DEFAULT_ARGS = {
    'owner': 'Dmitry Bulgakov',
    'email': 'dmitrii.bulghakov@gmail.com',
    'email_on_failure': True,
    'email_on_retry': False,
    'retry': 3,
    'retry_delay': timedelta(minutes=1)
}

dag = DAG(
    dag_id='dummy_train_dag',
    schedule_interval='0 1 * * *',
    start_date=days_ago(2),
    catchup=False,
    tags=['mlops'],
    default_args=DEFAULT_ARGS
)


_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())

BUCKET = 'mlops-project'
DATA_PATH = 'mlops_project/test_project/california_housing.pkl'
FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", 
            "Population", "AveOccup", "Latitude", "Longitude"]
TARGET = "MedHouseVal"


def init() -> None:
    _LOG.info("Train pipeline started.")


def get_data_from_postgres() -> None:
    _LOG.info("Postgres connection initializing.")
    # Использовать созданный ранее PG connection
    pg_hook = PostgresHook('pg_connection')
    con = pg_hook.get_conn()
    # Прочитать все данные из таблицы california_housing
    _LOG.info("Taking data from Postgres.")
    data = pd.read_sql('SELECT * FROM california_housing', con)
    # Использовать созданный ранее S3 connection
    _LOG.info("S3 connection initializing.")
    s3_hook = S3Hook('s3_connector')
    session = s3_hook.get_session('ru-central1')
    resource = session.resource('s3')
    # Сохранить файл в формате pkl на S3
    _LOG.info("S3 data uploading.")
    pickle_byte_obj = pickle.dumps(obj=data)
    resource.Object(BUCKET, DATA_PATH).put(Body=pickle_byte_obj)
    _LOG.info("Data uploading finished.")


def prepare_data() -> None:
    # Использовать созданный ранее S3 connection
    s3_hook = S3Hook('s3_connector')
    file = s3_hook.download_file(key=DATA_PATH, bucket_name=BUCKET)
    # Сделать препроцессинг
    data = pd.read_pickle(file)
    # Разделить на фичи и таргет
    X, y = data[FEATURES], data[TARGET]
    # Разделить данные на обучение и тест
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Обучить стандартизатор на train
    scaler = StandardScaler()
    X_train_fitted = scaler.fit_transform(X_train)
    X_test_fitted = scaler.fit_transform(X_test)
    # Сохранить готовые данные на S3
    session = s3_hook.get_session('ru-central1')
    resource = session.resource('s3')

    for name, data in zip(['X_train', 'X_test', 'y_train', 'y_test'],
                          [X_train_fitted, X_test_fitted, y_train, y_test]):
        pickle_byte_obj = pickle.dumps(obj=data)
        resource.Object(BUCKET, f'mlops_project/test_project/dataset/{name}.pkl').put(Body=pickle_byte_obj)

    _LOG.info('Data preparation finished!')


def train_model() -> None:
    # Использовать созданный ранее S3 connection
    s3_hook = S3Hook('s3_connector')
    # Загрузить готовые данные с S3
    data = dict()
    for name in ['X_train', 'X_test', 'y_train', 'y_test']:
        file = s3_hook.download_file(key=f'mlops_project/test_project/dataset/{name}.pkl', bucket_name=BUCKET)
        data[name] = pd.read_pickle(file)
    # Обучить модель
    model = RandomForestRegressor()
    model.fit(data['X_train'], data['y_train'])
    prediction = model.predict(data['X_test'])
    # Посчитать метрики
    result = dict()
    result['r2_score'] = r2_score(data['y_test'], prediction)
    result['rmse'] = mean_squared_error(data['y_test'], prediction) ** 0.5
    result['mae'] = median_absolute_error(data['y_test'], prediction)
    # Сохранить результат на S3
    date = datetime.now().strftime('%Y_%m_%d_%H')
    session = s3_hook.get_session('ru-central1')
    resource = session.resource('s3')
    json_byte_object = json.dumps(result)
    resource.Object(BUCKET, f'mlops_project/test_project/results/{date}.json').put(Body=json_byte_object)

    _LOG.info('Model train finished!')


def save_results() -> None:
    _LOG.info("Success.")


task_init = PythonOperator(
    task_id='init',
    python_callable=init,
    dag=dag
)

task_get_data = PythonOperator(
    task_id='data_load',
    python_callable=get_data_from_postgres,
    dag=dag
)

task_prepare_data = PythonOperator(
    task_id='data_preprocessing',
    python_callable=prepare_data,
    dag=dag
)

task_train_model = PythonOperator(
    task_id='model_training',
    python_callable=train_model,
    dag=dag
)

task_save_results = PythonOperator(
    task_id='save_results',
    python_callable=save_results,
    dag=dag
)

task_init >> task_get_data >> task_prepare_data >> task_train_model >> task_save_results
