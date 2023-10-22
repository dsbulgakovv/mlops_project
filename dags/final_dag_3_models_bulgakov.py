"""
  - BUCKET заменить на свой;
  - EXPERIMENT_NAME и DAG_ID оставить как есть (ссылками на переменную NAME);
  - имена коннекторов: pg_connection и s3_connection;
  - данные должны читаться из таблицы с названием california_housing;
  - данные на S3 должны лежать в папках {NAME}/datasets/ и {NAME}/results/.
"""
import json
import logging
import mlflow
import numpy as np
import pandas as pd
import pickle

from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error
from typing import Any, Dict

from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.models import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

from mlflow.models.signature import infer_signature
from mlflow.exceptions import MlflowException

_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())

NAME = "dm1trybu"  # свой ник в телеграме
BUCKET = "mlops-project"  # свой бакет
FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup",
    "Latitude", "Longitude"
]
TARGET = "MedHouseVal"
EXPERIMENT_NAME = NAME
DAG_ID = NAME

models = {
    "rf": RandomForestRegressor(),
    "lr": LinearRegression(),
    "hgb": HistGradientBoostingRegressor()
}  # словарь моделей

DEFAULT_ARGS = {
    "owner": "Dmitry Bulgakov",
    "email": "dmitrii.bulghakov@bk.ru",
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=1)
}

dag = DAG(
    dag_id=DAG_ID,
    schedule_interval="45 9 * * *",
    start_date=days_ago(2),
    catchup=False,
    tags=["mlops"],
    default_args=DEFAULT_ARGS
)


def init() -> Dict[str, Any]:
    # TO-DO 1 metrics: В этом шаге собрать start_tiemstamp, run_id, experiment_name, experiment_id.
    _LOG.info("Initializing dag and collecting meta-data...")
    metrics = dict()
    metrics["start_tiemstamp"] = datetime.now().strftime("%Y%m%d %H:%M")
    metrics["parent_run_name"] = 'parent_3_models'
    metrics["experiment_name"] = EXPERIMENT_NAME

    # TO-DO 2 mlflow: Создать эксперимент с experiment_name=NAME.
    # Добавить проверку на уже созданный эксперимент!
    # your code here.
    try:
        _LOG.info("Trying to set new experiment...")
        experiment_id = mlflow.create_experiment(
            name=metrics["experiment_name"],
            artifact_location=f"s3://{BUCKET}/{metrics['experiment_name']}"
        )
        metrics["experiment_id"] = experiment_id
        _LOG.info(f"Experiment '{metrics['experiment_name']}' created successfully!")
    except MlflowException:
        experiment_id = mlflow.get_experiment_by_name(metrics['experiment_name']).experiment_id
        metrics["experiment_id"] = experiment_id
        _LOG.info(f"Experiment '{metrics['experiment_name']}' already exists in MLFlow.")
        pass

    mlflow.set_experiment(metrics["experiment_name"])

    # TO-DO 3 mlflow: Создать parent run.
    _LOG.info("Creating parent run in MLFlow.")
    with mlflow.start_run(
            run_name=metrics["parent_run_name"],
            experiment_id=metrics["experiment_id"],
            description="parent"
    ) as parent_run:
        metrics['parent_run_id'] = parent_run.info.run_id

    return metrics


def get_data_from_postgres(**kwargs) -> Dict[str, Any]:
    # TO-DO 1 metrics: В этом шаге собрать data_download_start, data_download_end.
    ti = kwargs["ti"]
    metrics = ti.xcom_pull(task_ids="init")
    metrics["data_download_start"] = datetime.now().strftime("%Y%m%d %H:%M")

    # TO-DO 2 connections: Создать коннекторы.
    _LOG.info("Connecting to Postgres...")
    pg_hook = PostgresHook("pg_connection")
    con = pg_hook.get_conn()

    # TO-DO 3 Postgres: Прочитать данные.
    _LOG.info("Data downloading from Postgres...")
    data = pd.read_sql_query("SELECT * FROM california_housing", con)

    # TO-DO 4 Postgres: Сохранить данные на S3 в формате pickle в папку {NAME}/datasets/.
    file_name = f"{NAME}/datasets/california_housing.pkl"
    _LOG.info("Connecting to S3...")
    s3_hook = S3Hook("s3_connection")
    session = s3_hook.get_session("ru-central1")
    resource = session.resource("s3")
    pickle_byte_obj = pickle.dumps(data)
    resource.Object(BUCKET, file_name).put(Body=pickle_byte_obj)
    _LOG.info("Data is uploaded to S3.")

    metrics["data_download_end"] = datetime.now().strftime("%Y%m%d %H:%M")

    return metrics


def prepare_data(**kwargs) -> Dict[str, Any]:
    # TO-DO 1 metrics: В этом шаге собрать data_preparation_start, data_preparation_end.
    ti = kwargs["ti"]
    metrics = ti.xcom_pull(task_ids="get_data_from_postgres")
    metrics["data_preparation_start"] = datetime.now().strftime("%Y%m%d %H:%M")

    # TO-DO 2 connections: Создать коннекторы.
    # your code here.
    _LOG.info("Connecting to S3...")
    s3_hook = S3Hook("s3_connection")
    session = s3_hook.get_session("ru-central1")
    resource = session.resource("s3")

    # TO-DO 3 S3: Прочитать данные с S3.
    _LOG.info("Downloading data from S3.")
    file_name = f"{NAME}/datasets/california_housing.pkl"
    file = s3_hook.download_file(key=file_name, bucket_name=BUCKET)
    data = pd.read_pickle(file)
    
    # TO-DO 4 Сделать препроцессинг.
    # Разделить данные на train/test.
    # Подготовить 4 обработанных датасета.
    _LOG.info("Data preparation started...")
    X, y = data[FEATURES], data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.25,
                                                        random_state=45)
    scaler = StandardScaler()
    X_train_fitted = scaler.fit_transform(X_train)
    X_test_fitted = scaler.transform(X_test)

    # Сохранить данные на S3 в папку {NAME}/datasets/.
    _LOG.info("Saving prepared data to S3...")
    for name, data in zip(["X_train", "X_test", "y_train", "y_test"],
                          [X_train_fitted, X_test_fitted, y_train, y_test]):
        pickle_byte_obj = pickle.dumps(data)
        resource.Object(BUCKET,
                        f"{NAME}/datasets/{name}.pkl").put(Body=pickle_byte_obj)
        _LOG.info(f"Saved: {name}.")

    _LOG.info("Data preparation finished.")

    metrics["data_preparation_end"] = datetime.now().strftime("%Y%m%d %H:%M")

    return metrics


def train_mlflow_model(model: Any, name: str, X_train: np.array,
                       X_test: np.array, y_train: pd.Series,
                       y_test: pd.Series) -> None:

    # TO-DO 1: Обучить модель.
    model.fit(X_train, y_train)

    # TO-DO 2: Сделать predict.
    prediction = model.predict(X_test)
    # Посчитать метрики
    r_2_score = r2_score(y_test, prediction)
    rmse = mean_squared_error(y_test, prediction) ** 0.5
    mae = median_absolute_error(y_test, prediction)

    # TO-DO 3: Сохранить результаты обучения с помощью MLFlow.
    # Сохранить метрики
    mlflow.log_metric(f'r2_score_{name}', r_2_score)
    mlflow.log_metric(f'rmse_{name}', rmse)
    mlflow.log_metric(f'mae_{name}', mae)
    # Сохранить модель
    signature = infer_signature(X_test, prediction)
    mlflow.sklearn.log_model(model, name, signature=signature)
    mlflow.sklearn.save_model(model, name)


def train_model(**kwargs) -> Dict[str, Any]:
    # TO-DO 1 metrics: В этом шаге собрать f"train_start_{model_name}" и f"train_end_{model_name}".
    ti = kwargs["ti"]
    metrics = ti.xcom_pull(task_ids="prepare_data")
    model_name = kwargs["model_name"]
    metrics[f"train_start_{model_name}"] = datetime.now().strftime("%Y%m%d %H:%M")

    # TO-DO 2 connections: Создать коннекторы.
    s3_hook = S3Hook("s3_connection")

    # TO-DO 3 S3: Прочитать данные с S3 из папки {NAME}/datasets/.
    data = dict()
    for name in ["X_train", "X_test", "y_train", "y_test"]:
        file = s3_hook.download_file(key=f"{NAME}/datasets/{name}.pkl",
                                     bucket_name=BUCKET)
        data[name] = pd.read_pickle(file)

    # TO-DO 4: Обучить модели и залогировать обучение с помощью MLFlow.
    with mlflow.start_run(run_id=metrics["parent_run_id"]):  # parent run
        with mlflow.start_run(
                run_name=model_name,
                experiment_id=metrics["experiment_id"],
                nested=True
        ):  # child run
            train_mlflow_model(
                models[model_name], model_name,
                data["X_train"], data["X_test"], data["y_train"], data["y_test"]
            )

    metrics[f"train_end_{model_name}"] = datetime.now().strftime("%Y%m%d %H:%M")

    return metrics


def save_results(**kwargs) -> None:
    # TO-DO 1 metrics: В этом шаге собрать end_timestamp.
    ti = kwargs["ti"]
    models_metrics = ti.xcom_pull(task_ids=["train_rf", "train_lr", "train_hgb"])
    result = {}
    for model_metrics in models_metrics:
        result.update(model_metrics)

    result["end_timestamp"] = datetime.now().strftime("%Y%m%d %H:%M")

    # TO-DO 2: сохранить результаты обучения на S3 в файл {NAME}/results/{date}.json.
    date = datetime.now().strftime("%Y%m%d")
    file_name = f"{NAME}/results/{date}.json"
    _LOG.info("Connecting to S3...")
    s3_hook = S3Hook("s3_connection")
    session = s3_hook.get_session("ru-central1")
    resource = session.resource("s3")
    json_byte_object = json.dumps(result)
    resource.Object(
        BUCKET, file_name).put(Body=json_byte_object)
    _LOG.info("Meta data uploaded to S3 successfully!")
    _LOG.info("Pipeline is finished!")


# INIT DAG TASKS

task_init = PythonOperator(
    task_id="init",
    python_callable=init,
    dag=dag
)

task_get_data_from_postgres = PythonOperator(
    task_id="get_data_from_postgres",
    python_callable=get_data_from_postgres,
    dag=dag,
    provide_context=True
)

task_prepare_data = PythonOperator(
    task_id="prepare_data",
    python_callable=prepare_data,
    dag=dag,
    provide_context=True
)

training_model_tasks = [
    PythonOperator(
        task_id=f"train_{model_name}",
        python_callable=train_model,
        dag=dag,
        provide_context=True,
        op_kwargs={"model_name": model_name}
    )
    for model_name in models.keys()
]

task_save_results = PythonOperator(
    task_id="save_results",
    python_callable=save_results,
    dag=dag,
    provide_context=True
)

# DAG ARCHITECTURE
# TO-DO: Прописать архитектуру DAG'a.
task_init >> task_get_data_from_postgres >> task_prepare_data >> training_model_tasks >> task_save_results
