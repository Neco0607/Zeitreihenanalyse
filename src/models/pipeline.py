"""Automatisierte Forecast-Pipeline über mehrere Zeitreihen.

Vergleicht ARIMA, SARIMA, ETS, Naive und SeasonalNaive mit identischer
Schnittstelle. Unterstützt Holdout-Evaluation und Rolling-Origin
Cross-Validation. Wählt pro Reihe das beste Modell anhand wählbarer Metrik.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import TimeSeriesSplit

from src.evaluation.metrics import all_metrics

warnings.filterwarnings("ignore")


# ─── Modell-Wrapper (einheitliche API) ────────────────────────────────────
class NaiveModel:
    def fit(self, y):
        self.last = y.iloc[-1]
        return self
    def forecast(self, h):
        return np.repeat(self.last, h)


class SeasonalNaive:
    def __init__(self, m: int = 12):
        self.m = m
    def fit(self, y):
        self.last_season = y.iloc[-self.m:].values
        return self
    def forecast(self, h):
        return np.array([self.last_season[i % self.m] for i in range(h)])


class ARIMAModel:
    def __init__(self, seasonal: bool = False, m: int = 1):
        self.seasonal, self.m = seasonal, m
    def fit(self, y):
        self.model = pm.auto_arima(
            y, seasonal=self.seasonal, m=self.m,
            stepwise=True, suppress_warnings=True,
            error_action="ignore", information_criterion="aic",
        )
        return self
    def forecast(self, h):
        return np.asarray(self.model.predict(n_periods=h))


class ETSModel:
    def __init__(self, m: int = 12):
        self.m = m
    def fit(self, y):
        try:
            self.model = ExponentialSmoothing(
                y, trend="add", seasonal="add", seasonal_periods=self.m
            ).fit(optimized=True)
        except Exception:
            self.model = ExponentialSmoothing(y, trend="add").fit()
        return self
    def forecast(self, h):
        return np.asarray(self.model.forecast(h))


MODEL_REGISTRY = {
    "Naive":         lambda m: NaiveModel(),
    "SeasonalNaive": lambda m: SeasonalNaive(m=m),
    "ARIMA":         lambda m: ARIMAModel(seasonal=False),
    "SARIMA":        lambda m: ARIMAModel(seasonal=True, m=m),
    "ETS":           lambda m: ETSModel(m=m),
}


# ─── Evaluation ──────────────────────────────────────────────────────────
def evaluate_holdout(
    series_dict: dict[str, pd.Series],
    horizon: int = 12,
    m: int = 12,
) -> pd.DataFrame:
    """Train/Test-Split: letzten `horizon` Perioden als Test."""
    rows = []
    for name, y in series_dict.items():
        if len(y) <= horizon + m:
            continue  # zu kurz
        train, test = y.iloc[:-horizon], y.iloc[-horizon:]
        for mname, factory in MODEL_REGISTRY.items():
            try:
                model = factory(m).fit(train)
                pred  = model.forecast(horizon)
                rows.append({
                    "series": name, "model": mname,
                    **all_metrics(test.values, pred,
                                  y_train=train.values, m=m),
                })
            except Exception as e:
                rows.append({"series": name, "model": mname,
                             "error": str(e)[:80]})
    return pd.DataFrame(rows)


def evaluate_cv(
    series_dict: dict[str, pd.Series],
    horizon: int = 12,
    m: int = 12,
    n_splits: int = 4,
) -> pd.DataFrame:
    """Rolling-Origin Cross-Validation (TimeSeriesSplit)."""
    rows = []
    for name, y in series_dict.items():
        if len(y) <= (n_splits + 1) * horizon:
            continue
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=horizon)
        for fold, (tr, te) in enumerate(tscv.split(y)):
            train, test = y.iloc[tr], y.iloc[te]
            for mname, factory in MODEL_REGISTRY.items():
                try:
                    pred = factory(m).fit(train).forecast(horizon)
                    rows.append({
                        "series": name, "model": mname, "fold": fold,
                        **all_metrics(test.values, pred,
                                      y_train=train.values, m=m),
                    })
                except Exception:
                    continue
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return (
        df.groupby(["series", "model"])[
            ["MAE", "RMSE", "MAPE", "sMAPE", "MASE"]
        ]
        .mean().reset_index()
    )


def select_best(results: pd.DataFrame, metric: str = "RMSE") -> pd.DataFrame:
    """Pro Zeitreihe Modell mit niedrigstem Score wählen."""
    valid = results.dropna(subset=[metric])
    return valid.loc[valid.groupby("series")[metric].idxmin()].reset_index(drop=True)
