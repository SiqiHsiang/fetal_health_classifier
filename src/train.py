# train.py
from pathlib import Path
import json
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

def build_pipeline(estimator):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("selector", VarianceThreshold()),
        ("clf", estimator)
    ])

def grid_search_cv(pipeline, param_grid, X, y, outer_splits=2, inner_splits=2, random_state=42):
    outer = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    scores = []
    best_models = []
    for tr_idx, te_idx in outer.split(X, y):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
        gs = GridSearchCV(pipeline, param_grid=param_grid, cv=inner, n_jobs=-1)
        gs.fit(X_tr, y_tr)
        best = gs.best_estimator_
        y_pred = best.predict(X_te)
        scores.append(accuracy_score(y_te, y_pred))
        best_models.append(best)
    return float(np.mean(scores)), best_models

def evaluate(model, X, y):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    rep = classification_report(y, y_pred, digits=3)
    return acc, rep

def save_model(model, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return str(path)

def save_text(text, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text)
    return str(path)
