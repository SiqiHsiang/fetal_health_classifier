# data.py
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

def load_csv(path):
    path = Path(path)
    df = pd.read_csv(path)
    return df

def make_xy(df, target="fetal_health"):
    X = df.drop(columns=[target]).values
    y = df[target].values
    return X, y

def split_data(X, y, test_size=0.2, val_size=0.2, random_state=42, stratify=True):
    strat = y if stratify else None
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )
    val_ratio = val_size / (1 - test_size)
    strat_temp = y_temp if stratify else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - val_ratio, random_state=random_state, stratify=strat_temp
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def scale_sets(X_train, X_val, X_test):
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test), scaler

def resample(X, y, method=None, random_state=42):
    if method is None or method.lower() == "none":
        return X, y
    method = method.lower()
    if method == "random":
        sampler = RandomOverSampler(random_state=random_state)
    elif method == "smote":
        sampler = SMOTE(random_state=random_state)
    elif method == "adasyn":
        sampler = ADASYN(random_state=random_state)
    else:
        raise ValueError(f"Unknown resampling method: {method}")
    X_res, y_res = sampler.fit_resample(X, y)
    return X_res, y_res
