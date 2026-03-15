from airflow import DAG
from airflow.operators.python import PythonOperator
import pendulum

def task_ingest():
    import sys
    sys.path.insert(0, "/usr/local/airflow/include")
    from pipeline.data_ingestion import run_ingestion
    from pipeline.db_writer import write_prices_to_db, write_returns_to_db
    close, volume, returns, market_returns = run_ingestion()
    write_prices_to_db(close, volume)
    write_returns_to_db(returns, market_returns)

def task_features():
    import sys
    sys.path.insert(0, "/usr/local/airflow/include")
    from pipeline.feature_engineering import run_feature_engineering
    run_feature_engineering()

def task_models():
    import sys
    sys.path.insert(0, "/usr/local/airflow/include")
    from pipeline.main import run_models
    run_models()

def task_optimize():
    import sys
    sys.path.insert(0, "/usr/local/airflow/include")
    from pipeline.main import run_optimizer
    run_optimizer()

with DAG(
    dag_id="portfolio_pipeline",
    schedule="0 11 * * 1-5",  # 4:30 PM IST, weekdays only
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    tags=["portfolio", "nse", "ml"],
) as dag:

    ingest   = PythonOperator(task_id="ingest_data",          python_callable=task_ingest)
    features = PythonOperator(task_id="feature_engineering",  python_callable=task_features)
    models   = PythonOperator(task_id="run_ml_models",        python_callable=task_models)
    optimize = PythonOperator(task_id="optimize_portfolio",   python_callable=task_optimize)

    ingest >> features >> models >> optimize