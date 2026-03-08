"""
Run this once:  python setup.py
It will:
  1. Create the PostgreSQL database
  2. Write a .env file
  3. Run Django migrations (creates all tables)
  4. Create Django superuser
  5. Print next steps
"""

import subprocess
import sys
import os

# ── Load user config ──────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT,
    AIRFLOW_ADMIN_USERNAME, AIRFLOW_ADMIN_PASSWORD, AIRFLOW_ADMIN_EMAIL,
)


def run(cmd, cwd=None, env=None):
    result = subprocess.run(
        cmd, shell=True, cwd=cwd,
        env={**os.environ, **(env or {})},
    )
    if result.returncode != 0:
        print(f"Error: {cmd}")
        sys.exit(1)


def create_database():
    commands = f"""
        SELECT 'CREATE DATABASE {DB_NAME}'
        WHERE NOT EXISTS (
            SELECT FROM pg_database WHERE datname = '{DB_NAME}'
        )\\gexec
    """
    run(f'psql -U {DB_USER} -h {DB_HOST} -p {DB_PORT} -c "{commands}"')





def run_django_setup():
    print("Running Django migrations...")
    django_dir = os.path.join(os.path.dirname(__file__), "django_app")
    run("python manage.py migrate", cwd=django_dir)

    print("Creating Django superuser...")
    create_su = (
        f"from django.contrib.auth import get_user_model; "
        f"U = get_user_model(); "
        f"U.objects.filter(username='{AIRFLOW_ADMIN_USERNAME}').exists() or "
        f"U.objects.create_superuser('{AIRFLOW_ADMIN_USERNAME}', "
        f"'{AIRFLOW_ADMIN_EMAIL}', '{AIRFLOW_ADMIN_PASSWORD}')"
    )
    run(f'python manage.py shell -c "{create_su}"', cwd=django_dir)
    print("Django superuser ready.")




if __name__ == "__main__":
    create_database()
    run_django_setup()
