import os
import sys
import django
from django.conf import settings as django_settings
import pandas as pd
import importlib.util

def _setup_django():
       # Force override even if already set
    os.environ["DJANGO_SETTINGS_MODULE"] = "ml_portfolio.settings"
   
    print("Salam Waleikum")
    django_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../ml_portfolio"))
    if django_path not in sys.path:
        sys.path.insert(0, django_path)

    # Force unconfigure if already configured
    if django_settings.configured:
        return

    spec = importlib.util.find_spec("ml_portfolio.settings")
    print("Settings file path:", spec.origin)
    print("DB_HOST env var:", os.environ.get("DB_HOST", "NOT SET"))
    django.setup()
    
    #  Auto-run migrations if tables don't exist
    from django.db import connection
    from django.core.management import call_command
    existing_tables = connection.introspection.table_names()
    if "stocks_stock" not in existing_tables:
        call_command("migrate", "--run-syncdb", verbosity=0)

def write_prices_to_db(close: pd.DataFrame, volume: pd.DataFrame):
    _setup_django()
    from stocks.models import Stock, StockPrice

    for symbol in close.columns:
        stock, _ = Stock.objects.get_or_create(symbol=symbol)
        records = [
            StockPrice(
                stock=stock,
                date=date.date(),
                close=row[symbol],
                volume=int(volume.loc[date, symbol]) if symbol in volume.columns else None,
            )
            for date, row in close[[symbol]].iterrows()
        ]
        StockPrice.objects.bulk_create(
            records,
            update_conflicts=True,
            unique_fields=["stock", "date"],
            update_fields=["close", "volume"],
        )

def write_returns_to_db(returns: pd.DataFrame, market_returns: pd.Series):
    _setup_django()
    from stocks.models import Stock, StockReturn

    for symbol in returns.columns:
        stock, _ = Stock.objects.get_or_create(symbol=symbol)
        records = [
            StockReturn(
                stock=stock,
                date=date.date(),
                log_return=value,
                market_return=float(market_returns.get(date, None) or 0),
            )
            for date, value in returns[symbol].items()
        ]
        StockReturn.objects.bulk_create(
            records,
            update_conflicts=True,
            unique_fields=["stock", "date"],
            update_fields=["log_return", "market_return"],
        )