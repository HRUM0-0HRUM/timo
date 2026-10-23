import pandas as pd
import os

def split_data_by_month_correct(train_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(train_path, parse_dates=['date'])
    
    min_date = df['date'].min()
    max_date = df['date'].max()
    total_days = (max_date - min_date).days
    
    days_60 = int(total_days * 0.60)
    approx_60_date = min_date + pd.Timedelta(days=days_60)
    
    cutoff_month = approx_60_date.to_period('M')
    cutoff_date = cutoff_month.end_time.normalize() 
    
    train_base = df[df['date'] <= cutoff_date]
    train_base.to_csv(os.path.join(output_dir, 'train_base.csv'), index=False)
    
    new_data = df[df['date'] > cutoff_date].copy()
    
    new_data['year_month'] = new_data['date'].dt.to_period('M')
    months = sorted(new_data['year_month'].unique())
    
    for i, month in enumerate(months):
        batch_name = f'batch_{str(i+1).zfill(2)}.csv'
        batch_data = new_data[new_data['year_month'] == month].drop(columns=['year_month'])
        batch_data.to_csv(os.path.join(output_dir, batch_name), index=False)

if __name__ == "__main__":
    split_data_by_month_correct('../data/raw/train.csv', '../data/incoming')