import pandas as pd
import os
import shutil
from pathlib import Path

data_dir = Path('../data/raw').resolve()
output_dir = Path('../data/incoming').resolve()
dvc_file = output_dir.parent / f"{output_dir.name}.dvc"

if output_dir.exists():
    shutil.rmtree(output_dir)

if dvc_file.exists():
    os.remove(dvc_file)

(output_dir / 'base').mkdir(parents=True, exist_ok=True)
(output_dir / 'oil').mkdir(parents=True, exist_ok=True)
(output_dir / 'transactions').mkdir(parents=True, exist_ok=True)
(output_dir / 'holidays_events').mkdir(parents=True, exist_ok=True)

cutoff_date = pd.to_datetime('2016-05-31')
last_batch_end_date = pd.to_datetime('2017-08-15')

# деление на батчи train
train = pd.read_csv(f'{data_dir}/train.csv', parse_dates=['date'])

train_base = train[train['date'] <= cutoff_date].copy()
train_base.to_csv(f'{output_dir}/base/train_base.csv', index=False)

train_future = train[(train['date'] > cutoff_date) & (train['date'] <= last_batch_end_date)].copy()
train_future['year_month'] = train_future['date'].dt.to_period('M')
unique_months = sorted(train_future['year_month'].unique())

for i, month in enumerate(unique_months, start=1):
    batch = train_future[train_future['year_month'] == month].drop(columns=['year_month'])
    batch.to_csv(f'{output_dir}/base/batch_{i:02d}.csv', index=False)

# лагирование + деление на батчи oil
oil = pd.read_csv(f'{data_dir}/oil.csv', parse_dates=['date'])
oil = oil[oil['date'] <= last_batch_end_date].copy()

oil = oil.sort_values('date').reset_index(drop=True)
oil['dcoilwtico'] = oil['dcoilwtico'].shift(1)

oil_base = oil[oil['date'] <= cutoff_date].copy()
oil_base.to_csv(f'{output_dir}/oil/oil_base.csv', index=False)

oil_future = oil[(oil['date'] > cutoff_date) & (oil['date'] <= last_batch_end_date)].copy()
oil_future['year_month'] = oil_future['date'].dt.to_period('M')
unique_months = sorted(oil_future['year_month'].unique())

for i, month in enumerate(unique_months, start=1):
    batch = oil_future[oil_future['year_month'] == month].drop(columns=['year_month'])
    batch.to_csv(f'{output_dir}/oil/oil_batch_{i:02d}.csv', index=False)

# лагирование + деление на батчи transactions
transactions = pd.read_csv(f'{data_dir}/transactions.csv', parse_dates=['date'])
transactions = transactions[transactions['date'] <= last_batch_end_date].copy()

# Создаем лаг
transactions = transactions.sort_values(['store_nbr', 'date']).reset_index(drop=True)
transactions['transactions'] = transactions.groupby('store_nbr')['transactions'].shift(1)

# Финальная сортировка по дате и номеру магазина (для читаемости)
transactions = transactions.sort_values(['date', 'store_nbr']).reset_index(drop=True)

# Делим на base и future
trans_base = transactions[transactions['date'] <= cutoff_date].copy()
trans_base.to_csv(f'{output_dir}/transactions/transactions_base.csv', index=False)

trans_future = transactions[(transactions['date'] > cutoff_date) & (transactions['date'] <= last_batch_end_date)].copy()
trans_future['year_month'] = trans_future['date'].dt.to_period('M')
unique_months = sorted(trans_future['year_month'].unique())

for i, month in enumerate(unique_months, start=1):
    batch = trans_future[trans_future['year_month'] == month].drop(columns=['year_month'])
    batch.to_csv(f'{output_dir}/transactions/transactions_batch_{i:02d}.csv', index=False)

# деление holidays_events
holidays = pd.read_csv(f'{data_dir}/holidays_events.csv', parse_dates=['date'])

holidays_base = holidays[holidays['date'].dt.year < 2017].copy()
holidays_base.to_csv(f'{output_dir}/holidays_events/holidays_base.csv', index=False)

holidays_2017 = holidays[holidays['date'].dt.year == 2017].copy()
holidays_2017.to_csv(f'{output_dir}/holidays_events/holidays_2017.csv', index=False)