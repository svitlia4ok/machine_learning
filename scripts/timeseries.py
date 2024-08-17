from typing import Dict, Any
import pandas as pd
import numpy as np

def enrich_with_date_features(df: pd.DataFrame, params: Dict[str, bool], dateColumn: str) -> pd.DataFrame:
    if params.get('hour', False):
        df["hour"] = df[dateColumn].dt.hour

    if params.get('dayofweek', False):
        df["dayofweek"] = df[dateColumn].dt.dayofweek

    if params.get('quarter', False):
        df["quarter"] = df[dateColumn].dt.quarter

    if params.get('month', False):
        df["month"] = df[dateColumn].dt.month

    if params.get('year', False):
        df["year"] = df[dateColumn].dt.year

    if params.get('dayofyear', False):
        df["dayofyear"] = df[dateColumn].dt.dayofyear

    if params.get('sin_day', False):
        df["sin_day"] = np.sin(df["dayofyear"])

    if params.get('cos_day', False):
        df["cos_day"] = np.cos(df["dayofyear"])

    if params.get('dayofmonth', False):
        df["dayofmonth"] = df[dateColumn].dt.day

    return df