"""Benchmark de familias de modelos sobre los descriptores de docking.

Validacion cruzada 5-fold repetida 4 veces, identica para todos los modelos.
Ningun hiperparametro se ajusta mirando el fold de test: los modelos con sufijo
(CV) ajustan su regularizacion internamente sobre el fold de entrenamiento.

Uso:  python scripts/02_benchmark.py
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.dummy import DummyRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from common import load, feature_sets, RESULTS, FIGURES

warnings.filterwarnings("ignore")
SEED = 0
SCALE = ("scaler", StandardScaler())


def pipe(*steps):
    return lambda: Pipeline(list(steps))


MODELS = {
    "Dummy (media)":       pipe(("m", DummyRegressor())),
    "OLS (reg. multiple)": pipe(SCALE, ("m", LinearRegression())),
    "Ridge (CV)":          pipe(SCALE, ("m", RidgeCV(alphas=np.logspace(-2, 5, 60)))),
    "Lasso (CV)":          pipe(SCALE, ("m", LassoCV(n_alphas=60, max_iter=5000, random_state=SEED, n_jobs=-1))),
    "ElasticNet (CV)":     pipe(SCALE, ("m", ElasticNetCV(l1_ratio=[.5, .9, 1], n_alphas=40, max_iter=5000, random_state=SEED, n_jobs=-1))),
    "PLS (3 comp)":        pipe(SCALE, ("m", PLSRegression(n_components=3))),
    "PLS (6 comp)":        pipe(SCALE, ("m", PLSRegression(n_components=6))),
    "PLS (12 comp)":       pipe(SCALE, ("m", PLSRegression(n_components=12))),
    "SVR RBF C=1":         pipe(SCALE, ("m", SVR(C=1))),
    "SVR RBF C=10":        pipe(SCALE, ("m", SVR(C=10))),
    "SVR RBF C=100":       pipe(SCALE, ("m", SVR(C=100))),
    "KNN k=5":             pipe(SCALE, ("m", KNeighborsRegressor(5, weights="distance"))),
    "RandomForest":        pipe(("m", RandomForestRegressor(600, min_samples_leaf=2, random_state=SEED, n_jobs=-1))),
    "ExtraTrees":          pipe(("m", ExtraTreesRegressor(600, min_samples_leaf=2, random_state=SEED, n_jobs=-1))),
    "HistGradientBoosting": pipe(("m", HistGradientBoostingRegressor(max_depth=3, learning_rate=.05, max_iter=300, l2_regularization=1., random_state=SEED))),
}


def score_model(make, X, y, folds):
    r2, mae, r = [], [], []
    for train, test in folds:
        est = make()
        est.fit(X[train], y[train])
        pred = np.asarray(est.predict(X[test])).ravel()
        r2.append(r2_score(y[test], pred))
        mae.append(mean_absolute_error(y[test], pred))
        r.append(pearsonr(y[test], pred)[0] if np.std(pred) > 1e-9 else 0.0)
    return dict(R2=np.mean(r2), R2_sd=np.std(r2), MAE=np.mean(mae), pearson_r=np.mean(r))


def main():
    X_df, y_s, _ = load()
    y = y_s.values
    all_rows = []

    for set_name, cols in feature_sets(X_df).items():
        X = X_df[cols].values
        folds = list(RepeatedKFold(n_splits=5, n_repeats=4, random_state=SEED).split(X))
        print(f"\n{'=' * 78}\nFEATURE SET: {set_name}  (p={len(cols)}, n={len(y)})\n{'=' * 78}", flush=True)
        rows = [dict(feature_set=set_name, modelo=name, **score_model(mk, X, y, folds))
                for name, mk in MODELS.items()]
        table = pd.DataFrame(rows).sort_values("R2", ascending=False)
        print(table.drop(columns="feature_set").to_string(index=False, float_format=lambda v: f"{v:8.3f}"), flush=True)
        all_rows.extend(rows)

    RESULTS.mkdir(exist_ok=True)
    out = pd.DataFrame(all_rows)
    out.to_csv(RESULTS / "benchmark.csv", index=False)
    print(f"\nGuardado: {RESULTS / 'benchmark.csv'}")
    plot(out)


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    d = out[out.feature_set == "ALL"].sort_values("R2")
    d = d[d.modelo != "OLS (reg. multiple)"]  # fuera de escala (R2 ~ -794)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = ["#c44536" if v < 0 else "#2a6f97" for v in d.R2]
    ax.barh(d.modelo, d.R2, color=colors)
    ax.axvline(0, color="#444", lw=1)
    ax.set_xlabel("R2 en validacion cruzada (5-fold x 4)")
    ax.set_title("Ningun modelo supera R2 = 0.13  (OLS omitido: R2 = -794)", loc="left")
    ax.grid(axis="x", alpha=.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "benchmark_r2.png", dpi=150)
    print(f"Guardado: {FIGURES / 'benchmark_r2.png'}")


if __name__ == "__main__":
    main()
