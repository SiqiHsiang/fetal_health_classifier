# main.py
import argparse
from pathlib import Path
from . import data as D
from . import model as M
from . import train as T

def main():
    ap = argparse.ArgumentParser(description="Fetal Health Classifier (clean split)")
    ap.add_argument("--csv", default="data/raw/fetal_health.csv")
    ap.add_argument("--model", default="rf", help="rf|ada|logreg|svc")
    ap.add_argument("--resample", default="none", help="none|random|smote|adasyn")
    ap.add_argument("--save", default="models/fetal_health_model.joblib")
    ap.add_argument("--report", default="results/classification_report.txt")
    args = ap.parse_args()

    df = D.load_csv(args.csv)
    X, y = D.make_xy(df)
    (Xtr, ytr), (Xval, yval), (Xte, yte) = D.split_data(X, y)
    # optional resampling on train only
    Xtr, ytr = D.resample(Xtr, ytr, method=args.resample)

    est = M.build_model(args.model)

    # simple pipeline + grid search over a couple of params
    pipe = T.build_pipeline(est)
    # very small grids just as placeholders
    grid = {
        "clf__n_estimators": [100, 200] if args.model in ("rf", "randomforest", "random_forest") else [50],
    } if args.model.startswith("rf") else {}

    # if no grid, just fit once
    if grid:
        acc, models = T.grid_search_cv(pipe, grid, Xtr, ytr)
        best = models[0]
        print(f"Nested-CV mean accuracy: {acc:.3f}")
    else:
        best = pipe.fit(Xtr, ytr)

    acc_te, rep_te = T.evaluate(best, Xte, yte)
    print(f"Test accuracy: {acc_te:.3f}\\n")
    print(rep_te)

    T.save_model(best, args.save)
    T.save_text(rep_te, args.report)

if __name__ == "__main__":
    main()
