"""Evaluationsmetriken für Zeitreihenprognosen.

MAE   — Mean Absolute Error (robust, gleiche Einheit wie y)
RMSE  — Root Mean Squared Error (bestraft große Fehler)
MAPE  — Mean Absolute Percentage Error (dimensionslos, problematisch bei y≈0)
sMAPE — Symmetric MAPE (besser bei kleinen y)
MASE  — Mean Absolute Scaled Error (skaleninvariant, vergleichbar über Reihen!)
"""
import numpy as np


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mape(y, p) -> float:
    y, p = np.asarray(y), np.asarray(p)
    mask = y != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y[mask] - p[mask]) / y[mask])) * 100)


def smape(y, p) -> float:
    y, p = np.asarray(y), np.asarray(p)
    denom = (np.abs(y) + np.abs(p)) + 1e-9
    return float(np.mean(2 * np.abs(p - y) / denom) * 100)


def mase(y, p, y_train, m: int = 1) -> float:
    """Skaleninvariante Metrik: MAE / MAE des saisonalen Naive-Modells."""
    y_train = np.asarray(y_train)
    naive = np.mean(np.abs(np.diff(y_train, n=m)))
    return float(mae(y, p) / naive) if naive > 0 else float("nan")


def all_metrics(y, p, y_train=None, m: int = 1) -> dict:
    out = {"MAE": mae(y, p), "RMSE": rmse(y, p),
           "MAPE": mape(y, p), "sMAPE": smape(y, p)}
    if y_train is not None:
        out["MASE"] = mase(y, p, y_train, m)
    return out
