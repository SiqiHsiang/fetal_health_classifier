# model.py
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def build_model(name="rf", **kwargs):
    name = name.lower()
    if name in ("rf", "randomforest", "random_forest"):
        return RandomForestClassifier(random_state=42, **kwargs)
    if name in ("ada", "adaboost"):
        return AdaBoostClassifier(random_state=42, **kwargs)
    if name in ("logreg", "lr", "logistic"):
        return LogisticRegression(max_iter=1000, **kwargs)
    if name in ("svc", "svm"):
        return SVC(probability=True, **kwargs)
    raise ValueError(f"Unknown model name: {name}")
