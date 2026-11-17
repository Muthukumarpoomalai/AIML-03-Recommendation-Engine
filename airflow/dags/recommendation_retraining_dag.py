
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator


PROJECT_DIR = "/opt/airflow/project"

default_args = {
    "owner": "recommendation-engine",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5)
}

with DAG(
    dag_id="movie_recommendation_retraining",
    default_args=default_args,
    description="Scheduled retraining of the movie recommendation models",
    start_date=datetime(2026, 11, 17),
    schedule="0 2 * * 1",
    catchup=False,
    tags=["AIML-03", "recommendation"]
) as dag:

    train_two_tower = BashOperator(
        task_id="train_two_tower",
        bash_command=f"python {PROJECT_DIR}/train_two_tower.py"
    )

    rebuild_faiss = BashOperator(
        task_id="rebuild_faiss_index",
        bash_command=f"python {PROJECT_DIR}/build_faiss_index.py"
    )

    train_dcn = BashOperator(
        task_id="train_dcn",
        bash_command=f"python {PROJECT_DIR}/train_dcn.py"
    )

    train_two_tower >> rebuild_faiss >> train_dcn
