"""Diagnostico de la matriz: redundancia entre bloques, dimensionalidad efectiva
y senal univariante de cada descriptor frente al pKi.

Uso:  python scripts/01_diagnostics.py
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from common import load, blocks, RESULTS


def redundancy(X):
    """Comprueba si el bloque Dif_ es exactamente 8Y - 7F."""
    hits, checked = 0, 0
    for c in blocks(X)["Dif"]:
        stem = c[4:].rstrip("_")
        a, b = f"{stem}_8Y", f"{stem}_7F"
        if a in X and b in X:
            checked += 1
            if abs(np.corrcoef(X[c], X[a] - X[b])[0, 1]) > 0.999:
                hits += 1
    return hits, checked


def effective_rank(X):
    Z = ((X - X.mean()) / X.std().replace(0, 1)).fillna(0).values
    s = np.linalg.svd(Z, compute_uv=False)
    ev = s**2 / (s**2).sum()
    comps = {t: int(np.searchsorted(np.cumsum(ev), t) + 1) for t in (0.90, 0.95, 0.99)}
    return comps, int((s > s[0] * 1e-8).sum())


def univariate(X, y):
    rows = []
    for c in X.columns:
        r, p = pearsonr(X[c], y)
        rho, _ = spearmanr(X[c], y)
        rows.append((c, r, rho, p))
    out = pd.DataFrame(rows, columns=["feature", "pearson", "spearman", "p_value"])
    return out.assign(abs_r=out.pearson.abs()).sort_values("abs_r", ascending=False)


def main():
    X, y, df = load()
    b = blocks(X)

    print(f"n = {len(df)} moleculas, p = {X.shape[1]} descriptores")
    print("tamano de bloques:", {k: len(v) for k, v in b.items()})
    print(f"\npKi: media {y.mean():.2f}  sd {y.std():.2f}  rango [{y.min()}, {y.max()}]")

    hits, checked = redundancy(X)
    print(f"\nRedundancia: {hits}/{checked} columnas Dif_ reproducen 8Y - 7F (|r| > 0.999)")

    comps, rank = effective_rank(X)
    print("\nDimensionalidad efectiva:")
    for t, k in comps.items():
        print(f"  componentes para {t:.0%} de varianza: {k}")
    print(f"  rango de la matriz: {rank}")

    u = univariate(X, y)
    print("\nTop 15 correlaciones univariantes con pKi:")
    print(u.head(15).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n|r| maximo = {u.abs_r.max():.3f} -> R2 del mejor descriptor solo = {u.abs_r.max()**2:.3f}")
    print(f"descriptores con |r| > 0.3: {(u.abs_r > 0.3).sum()} de {len(u)}")

    print("\nScores nativos de gnina frente a pKi:")
    for c in ["CNN affinity_8Y", "CNN affinity_7F", "Affinity_8Y", "Affinity_7F",
              "CNN score pose_8Y", "CNN score pose_7F"]:
        print(f"  {c:22s} pearson {pearsonr(X[c], y)[0]:+.3f}  spearman {spearmanr(X[c], y)[0]:+.3f}")

    RESULTS.mkdir(exist_ok=True)
    u.to_csv(RESULTS / "univariate_correlations.csv", index=False)
    print(f"\nGuardado: {RESULTS / 'univariate_correlations.csv'}")


if __name__ == "__main__":
    main()
