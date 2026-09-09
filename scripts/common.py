"""Carga de datos y definicion de bloques de descriptores."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "all_8Yvs7F_mix_pKi.xlsx"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TARGET = "Experimental pKi"
ID_COL = "Compound"


def load(drop_constant=True):
    """Devuelve (X, y, df) con X numerico y sin la columna identificadora."""
    df = pd.read_excel(DATA)
    y = df[TARGET]
    X = df.drop(columns=[ID_COL, TARGET])
    if drop_constant:
        X = X.loc[:, X.std() > 0]
    return X, y, df


def blocks(X):
    """Los cuatro bloques de descriptores presentes en la matriz."""
    return {
        "8Y": [c for c in X if c.endswith("_8Y")],
        "7F": [c for c in X if c.endswith("_7F")],
        "Dif": [c for c in X if c.startswith("Dif_")],
        "Rel": [c for c in X if c.startswith("Rel_")],
    }


def feature_sets(X):
    """Subconjuntos de features que se comparan en el benchmark."""
    b = blocks(X)
    return {
        "ALL": list(X.columns),
        "RAW_8Y_7F": b["8Y"] + b["7F"],
        "REL": b["Rel"],
        "RAW_PLUS_DIF": b["8Y"] + b["7F"] + b["Dif"],
    }
