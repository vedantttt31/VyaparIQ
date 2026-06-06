# VyaparIQ Daily Update Pipeline
# Run every morning to fetch latest agmarknet data and retrain model
# Usage: python update_pipeline.py

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from io import StringIO
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

DATA_PATH  = 'data_clean.csv'
MODEL_PATH = 'model_lr.pkl'
ENC_PATH   = 'label_encoders.pkl'
FEAT_PATH  = 'features.pkl'
MSP        = 2425

def fetch_agmarknet(date_str):
    url = "https://api.agmarknet.gov.in/v1/daily-price-arrival/report"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/plain, */*",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "en-US,en;q=0.5",
        "Origin": "https://agmarknet.gov.in",
        "Referer": "https://agmarknet.gov.in/",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36",
        "Sec-Ch-Ua": '"Brave";v="149", "Chromium";v="149", "Not)A;Brand";v="24"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site"
    }
    payload = {
        "from_date": date_str,
        "to_date": date_str,
        "commodity": "1",
        "data_type": "100006",
        "district": "[293,295,306,307,308,301,326,335]",
        "download": "true",
        "download_type": "csv",
        "grade": "[100003]",
        "group": "1",
        "limit": "10",
        "market": "[100002]",
        "page": "1",
        "state": "[19]",
        "variety": "[100007]"
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code != 200:
        print(f"❌ Failed for {date_str}: {response.status_code}")
        return None
    lines = response.text.split('\n')
    header_idx = next(i for i, l in enumerate(lines) if l.startswith('State,'))
    csv_clean = '\n'.join(lines[header_idx:])
    df = pd.read_csv(StringIO(csv_clean))
    df.columns = [c.strip() for c in df.columns]
    return df

def clean_new_data(raw_df):
    df = raw_df.copy()
    df = df.rename(columns={
        'District': 'district', 'Market': 'market',
        'Variety': 'variety', 'Grade': 'grade',
        'Min Price': 'min_price', 'Max Price': 'max_price',
        'Modal Price': 'modal_price', 'Arrival Quantity': 'arrival_qty_mt',
        'Arrival Date': 'date'
    })
    drop_cols = ['State','Commodity Group','Commodity','Price Unit','Arrival Unit']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    for col in ['min_price', 'max_price', 'modal_price']:
        df[col] = df[col].astype(str).str.replace(',','').astype(float)
    df['date']         = pd.to_datetime(df['date'], format='%d-%m-%Y')
    df['msp']          = MSP
    df['rainfall_mm']  = 0.0
    df['temp_max_c']   = np.nan
    df['rained_today'] = 0
    df['year']         = df['date'].dt.year
    df['month']        = df['date'].dt.month
    df['week']         = df['date'].dt.isocalendar().week.astype(int)
    df['day_of_year']  = df['date'].dt.dayofyear
    df['price_vs_msp']    = df['modal_price'] - MSP
    df['log_arrival']     = np.log1p(df['arrival_qty_mt'])
    df['log_modal_price'] = np.log(df['modal_price'])
    df['log_rainfall']    = np.log1p(df['rainfall_mm'])
    df['is_sharbati']     = df['variety'].astype(str).str.contains('Sharbati').astype(int)
    df['variety_group']   = df['variety'].apply(
        lambda x: 'Sharbati' if 'Sharbati' in str(x) else 'Regular Wheat')
    df = df[df['is_sharbati'] == 0]
    target_districts = ['Betul','Bhopal','Dewas','Harda',
                        'Hoshangabad','Sehore','Ujjain','Vidisha']
    df = df[df['district'].isin(target_districts)]
    return df

def append_to_csv(new_df, data_path):
    existing       = pd.read_csv(data_path, parse_dates=['date'])
    existing_dates = set(existing['date'].dt.date.astype(str))
    new_dates      = set(new_df['date'].dt.date.astype(str))
    overlap        = existing_dates & new_dates
    if overlap:
        print(f"⚠️ Skipping existing dates: {overlap}")
        new_df = new_df[~new_df['date'].dt.date.astype(str).isin(overlap)]
    if len(new_df) == 0:
        print("No new rows to add.")
        return existing
    for col in existing.columns:
        if col not in new_df.columns:
            new_df[col] = np.nan
    new_df   = new_df[existing.columns]
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.sort_values(
        ['district','market','variety','grade','date']).reset_index(drop=True)
    combined.to_csv(data_path, index=False)
    print(f"✅ Added {len(new_df)} rows — CSV now has {len(combined)} total rows")
    return combined

def recompute_features(df):
    group_cols = ['district','market','variety','grade']
    df = df.sort_values(group_cols + ['date']).reset_index(drop=True)
    df['price_lag1']   = df.groupby(group_cols, observed=True)['modal_price'].shift(1)
    df['price_lag3']   = df.groupby(group_cols, observed=True)['modal_price'].shift(3)
    df['arrival_lag1'] = df.groupby(group_cols, observed=True)['arrival_qty_mt'].shift(1)
    df['arrival_lag3'] = df.groupby(group_cols, observed=True)['arrival_qty_mt'].shift(3)
    df['state_price_lag1']   = df.groupby('date', observed=True)['modal_price'].transform('mean').shift(1)
    df['state_arrival_lag1'] = df.groupby('date', observed=True)['arrival_qty_mt'].transform('mean').shift(1)
    df['price_roll3_mean']   = (df.groupby(group_cols, observed=True)['modal_price']
                                  .transform(lambda x: x.shift(1).rolling(3).mean()))
    df['arrival_roll3_mean'] = (df.groupby(group_cols, observed=True)['arrival_qty_mt']
                                  .transform(lambda x: x.shift(1).rolling(3).mean()))
    df['days_since_season_start'] = df.apply(
        lambda row: (row['date'] - pd.Timestamp(f"{row['date'].year}-04-01")).days
        if row['is_sharbati']
        else (row['date'] - pd.Timestamp(f"{row['date'].year}-03-01")).days, axis=1)
    df['price_above_msp']      = df['price_lag1'] - df['msp']
    df['target_price_t1']      = df.groupby(group_cols, observed=True)['modal_price'].shift(-1)
    df['target_price_t3']      = (df.groupby(group_cols, observed=True)['modal_price']
                                    .transform(lambda x: x.shift(-1).rolling(3).mean()))
    df['days_since_last_trade'] = (df.groupby(group_cols, observed=True)['date']
                                     .transform(lambda x: x.diff().dt.days)
                                     .fillna(1).clip(upper=30))
    return df

def retrain_model(df):
    FEATURES = joblib.load(FEAT_PATH)
    df_model = df[df['is_sharbati'] == 0].copy()
    df_model = df_model[df_model['modal_price'] <= 3200].copy()
    le_dict  = {}
    for col in ['district','variety','grade']:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col].astype(str))
        le_dict[col]  = le
    df_model = df_model.dropna(subset=FEATURES + ['target_price_t1'])
    X = df_model[FEATURES]
    y = df_model['target_price_t1']
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model,   MODEL_PATH)
    joblib.dump(le_dict, ENC_PATH)
    print(f"✅ Model retrained on {len(df_model)} rows")

# ── MAIN ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("VyaparIQ Daily Update Pipeline")
    print(f"Running at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    raw = fetch_agmarknet(yesterday)
    if raw is not None:
        cleaned  = clean_new_data(raw)
        combined = append_to_csv(cleaned, DATA_PATH)
        combined = recompute_features(combined)
        combined.to_csv(DATA_PATH, index=False)
        retrain_model(combined)
        print(f"\n✅ Pipeline complete!")
        print(f"Data: {combined['date'].min().date()} → {combined['date'].max().date()}")
