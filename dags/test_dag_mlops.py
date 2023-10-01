from datetime import timedelta
from typing import NoReturn

from airflow.models import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator

DEFAULT_ARGS = {
    'owner': 'Dmitry Bulgakov',
    'email': 'dmitrii.bulghakov@gmail.com',
    'email_on_failure': True,
    'email_on_retry': False,
    'retry': 3,
    'retry_delay': timedelta(minutes=1)
}

dag = DAG(
    dag_id='test_dag_mlops',
    schedule_interval='0 1 * * *',
    start_date=days_ago(2),
    catchup=False,
    tags=['mlops'],
    default_args=DEFAULT_ARGS
)


def init() -> NoReturn:
    print('Hello, world!')


task_init = PythonOperator(
    task_id='init',
    python_callable=init,
    dag=dag)


task_init
