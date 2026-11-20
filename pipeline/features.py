"""Пайплайн подготовки данных и признаков (ПР № 3). Прогноз sales, метрика RMSLE."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, TargetEncoder

RAW_DIR = Path("data/raw")
LAG = 30

KEY = ["date", "store_nbr", "family"]
HOL_TYPES = {"Holiday", "Transfer", "Additional", "Bridge", "Work Day", "Event"}
HOL_LOCALES = {"Local", "Regional", "National"}

NUM_BASE = ["onpromotion", "dcoilwtico", "lag_30", "roll7_lag30", "roll28_lag30"]
CAT_BASE = ["store_nbr", "family", "city", "state", "type", "cluster"]

NEW_NUM = [
    "is_weekend",
    "is_payday",
    "is_promo",
    "is_holiday",
    "month",
    "median_trans_store_dow",
]
NEW_CAT = ["dow", "type_promo_interaction"]


def clean_sales(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "id" in df.columns:
        df = df.dropna(subset=["id"]).drop_duplicates(subset=["id"])
    df = df.drop_duplicates(subset=KEY)
    df = df[df["store_nbr"].between(1, 54)]
    df = df[(df["sales"] >= 0) & (df["onpromotion"] >= 0)]
    return df.dropna(subset=KEY + ["sales", "onpromotion"])


def load_tables(tables_dir=RAW_DIR):
    stores = pd.read_csv("data/raw/stores.csv")
    stores = stores[
        stores["type"].isin(list("ABCDE"))
        & stores["cluster"].between(1, 17)
        & stores["store_nbr"].between(1, 54)
    ]

    oil = pd.read_csv("data/incoming/oil/oil_base.csv", parse_dates=["date"])
    oil = oil.drop_duplicates("date").sort_values("date")

    hol = pd.read_csv(
        "data/incoming/holidays_events/holidays_base.csv", parse_dates=["date"]
    )
    hol = hol[hol["type"].isin(HOL_TYPES) & hol["locale"].isin(HOL_LOCALES)].copy()
    hol["description"] = hol["description"].astype(str).str.lower()
    hol["transferred"] = hol["transferred"].astype(str).str.lower().eq("true")

    tr = pd.read_csv(
        "data/incoming/transactions/transactions_base.csv", parse_dates=["date"]
    )
    tr = tr.drop_duplicates(["date", "store_nbr"])
    tr["transactions"] = tr["transactions"].fillna(0).clip(lower=0)

    return stores, oil, hol, tr


def _holiday_flags(hol):
    real = hol[
        hol["type"].isin(["Holiday", "Transfer", "Additional", "Bridge"])
        & ~hol["transferred"]
    ]
    nat = (
        real[real["locale"] == "National"][["date"]]
        .drop_duplicates()
        .assign(hol_national=1)
    )
    reg = (
        real[real["locale"] == "Regional"][["date", "locale_name"]]
        .drop_duplicates()
        .rename(columns={"locale_name": "state"})
        .assign(hol_regional=1)
    )
    loc = (
        real[real["locale"] == "Local"][["date", "locale_name"]]
        .drop_duplicates()
        .rename(columns={"locale_name": "city"})
        .assign(hol_local=1)
    )
    return nat, reg, loc


def prepare(sales_path, history_path=None, tables_dir=RAW_DIR):
    stores, oil, hol, tr = load_tables(tables_dir)
    new = clean_sales(
        pd.read_csv("data/incoming/base/train_base.csv", parse_dates=["date"])
    ).assign(_new=True)

    if history_path is not None:
        hist = clean_sales(pd.read_csv(history_path)).assign(_new=False)
        df = pd.concat([hist, new], ignore_index=True)
        df = df.drop_duplicates(KEY, keep="last")
    else:
        df = new
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    df["log_sales"] = np.log1p(df["sales"])
    g = df.groupby(["store_nbr", "family"], sort=False)["log_sales"]
    df["r7"] = g.transform(lambda s: s.rolling(7, min_periods=1).mean())
    df["r28"] = g.transform(lambda s: s.rolling(28, min_periods=1).mean())
    lag = df[KEY + ["log_sales", "r7", "r28"]].copy()
    lag["date"] = lag["date"] + pd.Timedelta(days=LAG)
    lag = lag.rename(
        columns={"log_sales": "lag_30", "r7": "roll7_lag30", "r28": "roll28_lag30"}
    )
    df = df.drop(columns=["r7", "r28"]).merge(lag, on=KEY, how="left")

    df = df.merge(
        stores[["store_nbr", "city", "state", "type", "cluster"]],
        on="store_nbr",
        how="left",
    )
    days = pd.date_range(
        min(oil["date"].min(), df["date"].min()),
        max(oil["date"].max(), df["date"].max()),
    )
    oil_full = (
        oil.set_index("date")["dcoilwtico"]
        .reindex(days)
        .ffill()
        .bfill()
        .rename_axis("date")
        .reset_index()
    )
    df = df.merge(oil_full, on="date", how="left")
    nat, reg, loc = _holiday_flags(hol)
    df = (
        df.merge(nat, on="date", how="left")
        .merge(reg, on=["date", "state"], how="left")
        .merge(loc, on=["date", "city"], how="left")
    )
    for c in ["hol_national", "hol_regional", "hol_local"]:
        df[c] = df[c].fillna(0).astype(int)
    df = df.merge(tr, on=["date", "store_nbr"], how="left")
    df["transactions"] = df["transactions"].fillna(0)

    df = df[df["_new"]].drop(columns="_new")
    df = df.dropna(subset=["city"])
    df = df.sort_values(KEY).reset_index(drop=True)
    y = df["log_sales"].copy()
    X = df.drop(columns=["log_sales", "sales", "id"], errors="ignore")
    return X, y


def add_features(df):
    df = df.copy()
    d = pd.to_datetime(df["date"])
    df["dow"] = d.dt.dayofweek.astype(str)
    df["month"] = d.dt.month
    df["is_weekend"] = (d.dt.dayofweek >= 5).astype(int)
    df["is_payday"] = ((d.dt.day == 15) | d.dt.is_month_end).astype(int)
    df["is_promo"] = (df["onpromotion"] > 0).astype(int)
    df["is_holiday"] = (
        (df["hol_national"] + df["hol_regional"] + df["hol_local"]) > 0
    ).astype(int)
    df["type_promo_interaction"] = (
        df["type"].astype(str)
        + "_"
        + np.where(df["is_promo"] == 1, "promo", "no_promo")
    )
    for c in CAT_BASE:
        df[c] = df[c].astype(str)
    return df


class StoreDowMedian(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self.map_ = X.groupby(["store_nbr", "dow"])["transactions"].median()
        self.global_ = float(X["transactions"].median())
        return self

    def transform(self, X):
        X = X.copy()
        idx = pd.MultiIndex.from_frame(X[["store_nbr", "dow"]])
        X["median_trans_store_dow"] = (
            self.map_.reindex(idx).fillna(self.global_).to_numpy()
        )
        return X


class SeasonalMedian(BaseEstimator):

    def fit(self, X, y):
        d = X[["store_nbr", "family", "dow"]].copy()
        d["y"] = np.asarray(y)
        self.map_ = d.groupby(["store_nbr", "family", "dow"])["y"].median()
        self.global_ = float(np.median(y))
        return self

    def predict(self, X):
        idx = pd.MultiIndex.from_frame(X[["store_nbr", "family", "dow"]])
        return self.map_.reindex(idx).fillna(self.global_).to_numpy()


def make_pipeline(model, new_features=True):
    num = NUM_BASE + (NEW_NUM if new_features else [])
    cat = CAT_BASE + (NEW_CAT if new_features else [])
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = TargetEncoder(target_type="continuous", random_state=42)
    prep = ColumnTransformer([("num", numeric, num), ("cat", categorical, cat)])
    steps = [("features", FunctionTransformer(add_features))]
    if new_features:
        steps.append(("trans_median", StoreDowMedian()))
    steps += [("prep", prep), ("model", model)]
    return Pipeline(steps)


def make_baseline():
    return Pipeline(
        [("features", FunctionTransformer(add_features)), ("model", SeasonalMedian())]
    )
