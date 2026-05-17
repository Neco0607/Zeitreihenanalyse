"""
Box-Jenkins ARIMA-Modellierung fuer ein einzelnes Asset.

Pipeline (analog zur Vorlesung Menden TSA Advanced):
  1. Daten laden, Log-Transformation, Train/Test-Split (80/20)
  2. ARIMA-Ordnung (p, d, q) automatisch finden via AIC
  3. Modell auf Trainings-Set fitten
  4. Residuen-Diagnostik: Ljung-Box-Test, QQ-Plot, ACF der Residuen
  5. Forecast auf Test-Horizont mit 95%-Konfidenzintervallen
  6. Evaluation: MSE, RMSE, MAE, MAPE
  7. Time-Series-Cross-Validation (expanding window)

Aufruf vom Repo-Root:
    python -m src.models.box_jenkins                  # default: DAX
    python -m src.models.box_jenkins --asset Gold
    python -m src.models.box_jenkins --asset Bitcoin

Outputs landen in reports/box_jenkins/<asset>/:
    - model_summary.txt        statsmodels ARIMA-Summary
    - forecast.png             Forecast + 95%-KI vs. Actual
    - residuals.png            Residuen-Diagnostik (4 Subplots)
    - evaluation.csv           Metriken auf Test-Set
    - cv_results.csv           Cross-Validation-Ergebnisse
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
from statsmodels.tsa.arima.model import ARIMA

# pmdarima fuer auto_arima
import pmdarima as pm

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=ValueWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Pfade
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
REPORTS_BASE = REPO_ROOT / "reports" / "box_jenkins"

# Einheitliche matplotlib-Konfiguration
plt.rcParams.update({
    "figure.dpi": 110,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})

ASSET_COLORS = {
    "DAX": "#1f77b4",
    "Gold": "#d4a017",
    "Bitcoin": "#ff7f0e",
}


# ---------------------------------------------------------------------------
# 1. Daten laden + Train/Test-Split
# ---------------------------------------------------------------------------

def load_asset(asset: str) -> pd.Series:
    """Bereinigte Reihe aus data/processed/ laden."""
    path = PROCESSED_DIR / f"{asset.lower()}_clean.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Bereinigte Daten fuer {asset} fehlen: {path}\n"
            f"Bitte erst ausfuehren: python -m src.data.clean_data"
        )
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return df.iloc[:, 0].astype(float)


def train_test_split(series: pd.Series, test_ratio: float = 0.2) -> tuple[pd.Series, pd.Series]:
    """
    Chronologischer Split (KEIN Shuffle bei Zeitreihen!).
    Default 80/20 - laut Vorlesung typisch fuer Train/Test.
    """
    split_idx = int(len(series) * (1 - test_ratio))
    return series.iloc[:split_idx], series.iloc[split_idx:]


# ---------------------------------------------------------------------------
# 2. ARIMA-Ordnung automatisch finden
# ---------------------------------------------------------------------------

def find_best_order(
    train: pd.Series,
    max_p: int = 5,
    max_q: int = 5,
    criterion: str = "aic",
) -> tuple[int, int, int]:
    """
    Auto-ARIMA via pmdarima. Differenzierungsgrad d wird automatisch
    via KPSS-Test bestimmt. p und q per Grid-Search nach AIC oder BIC.
    """
    print(f"  Suche optimale ARIMA-Ordnung (max p={max_p}, max q={max_q}, "
          f"Kriterium={criterion.upper()})...")
    model = pm.auto_arima(
        train,
        start_p=0, max_p=max_p,
        start_q=0, max_q=max_q,
        d=None,                # automatisch via KPSS
        seasonal=False,
        information_criterion=criterion,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        trace=False,
    )
    return model.order


# ---------------------------------------------------------------------------
# 3. Modell fitten
# ---------------------------------------------------------------------------

def fit_arima(train: pd.Series, order: tuple[int, int, int]):
    """ARIMA-Modell mit statsmodels fitten (hat schoenere Summary als pmdarima)."""
    model = ARIMA(train, order=order)
    fitted = model.fit()
    return fitted


# ---------------------------------------------------------------------------
# 4. Residuen-Diagnostik
# ---------------------------------------------------------------------------

def residual_diagnostics(fitted_model, asset: str, outdir: Path) -> dict:
    """
    Vier Standard-Diagnostiken:
      a) Residuen-Plot ueber die Zeit (sollte wie weisses Rauschen aussehen)
      b) Histogramm + Normal-Overlay
      c) ACF der Residuen (sollte keine signifikanten Lags zeigen)
      d) QQ-Plot gegen Normal
    Plus: Ljung-Box-Test auf Autokorrelation der Residuen.
    """
    # Initial-Condition-Artefakte ausschliessen: die ersten max(p,d,q)+1
    # Residuen sind bei ARIMA oft extreme Ausreisser, weil das Modell
    # noch keine vollstaendige Historie hat. Standardpraxis: skippen.
    p, d, q = fitted_model.model.order
    n_burn = max(p, d, q) + 1
    residuals = fitted_model.resid.iloc[n_burn:].dropna()

    # Ljung-Box-Test: H0 = keine Autokorrelation in den Residuen.
    # Wenn p > 0.05 -> Residuen sind unkorreliert -> Modell ist OK.
    lb_test = acorr_ljungbox(residuals, lags=[10, 20], return_df=True)

    color = ASSET_COLORS.get(asset, "#444444")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # a) Residuen ueber die Zeit
    axes[0, 0].plot(residuals.index, residuals.values, color=color, lw=0.5)
    axes[0, 0].axhline(0, color="black", lw=0.5)
    axes[0, 0].set_title("Residuen über die Zeit")
    axes[0, 0].set_xlabel("Datum")
    axes[0, 0].set_ylabel("Residuum")

    # b) Histogramm + Normal-Overlay
    axes[0, 1].hist(residuals, bins=60, density=True, color=color,
                    alpha=0.7, edgecolor="white")
    x = np.linspace(residuals.min(), residuals.max(), 200)
    axes[0, 1].plot(x, stats.norm.pdf(x, residuals.mean(), residuals.std()),
                    "k--", lw=1.5, label="Normalverteilung")
    axes[0, 1].set_title("Verteilung der Residuen")
    axes[0, 1].set_xlabel("Residuum")
    axes[0, 1].set_ylabel("Dichte")
    axes[0, 1].legend()

    # c) ACF der Residuen
    plot_acf(residuals, lags=40, ax=axes[1, 0], color=color)
    axes[1, 0].set_title("ACF der Residuen")

    # d) QQ-Plot
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("QQ-Plot vs. Normalverteilung")
    axes[1, 1].get_lines()[0].set_markerfacecolor(color)
    axes[1, 1].get_lines()[0].set_markeredgecolor(color)
    axes[1, 1].get_lines()[0].set_markersize(3)

    fig.suptitle(f"{asset} - Residuen-Diagnostik", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "residuals.png")
    plt.close(fig)

    return {
        "ljung_box_lag10_p": float(lb_test["lb_pvalue"].iloc[0]),
        "ljung_box_lag20_p": float(lb_test["lb_pvalue"].iloc[1]),
        "residuals_mean": float(residuals.mean()),
        "residuals_std": float(residuals.std()),
        "residuals_skew": float(stats.skew(residuals)),
        "residuals_kurtosis": float(stats.kurtosis(residuals)),
    }


# ---------------------------------------------------------------------------
# 5. Forecast mit Konfidenzintervallen
# ---------------------------------------------------------------------------

def forecast_with_ci(fitted_model, horizon: int, alpha: float = 0.05):
    """
    h-step-ahead Forecast mit (1-alpha)-Konfidenzintervall.
    Default: alpha=0.05 -> 95%-Intervall (Multiplier c=1.96 laut Vorlesung).
    """
    fc = fitted_model.get_forecast(steps=horizon)
    point_forecast = fc.predicted_mean
    ci = fc.conf_int(alpha=alpha)
    ci.columns = ["lower", "upper"]
    return point_forecast, ci


def plot_forecast(
    train: pd.Series,
    test: pd.Series,
    forecast: pd.Series,
    ci: pd.DataFrame,
    asset: str,
    order: tuple[int, int, int],
    outdir: Path,
) -> None:
    """Train + Test + Forecast + Konfidenzintervall in einem Plot."""
    color = ASSET_COLORS.get(asset, "#444444")

    fig, ax = plt.subplots(figsize=(13, 6))

    # Train (nur die letzten ~250 Tage zur Lesbarkeit)
    last_train = train.iloc[-250:]
    ax.plot(last_train.index, last_train.values, color="black",
            lw=0.7, label="Train (letzte 250 Tage)")

    # Test (Actual)
    ax.plot(test.index, test.values, color=color, lw=1.2, label="Test (Actual)")

    # Forecast (Mittelwert)
    ax.plot(forecast.index, forecast.values, color="red", lw=1.4,
            linestyle="--", label="Forecast")

    # Konfidenzintervall
    ax.fill_between(ci.index, ci["lower"], ci["upper"],
                    color="red", alpha=0.15, label="95%-Konfidenzintervall")

    ax.set_title(f"{asset} - ARIMA{order} Forecast vs. Actual")
    ax.set_xlabel("Datum")
    ax.set_ylabel("Log-Preis")
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(outdir / "forecast.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Evaluation
# ---------------------------------------------------------------------------

def evaluate(actual: pd.Series, forecast: pd.Series) -> dict:
    """MSE, RMSE, MAE, MAPE auf gemeinsamen Indizes."""
    aligned = pd.concat([actual.rename("a"), forecast.rename("f")], axis=1).dropna()
    a, f = aligned["a"].values, aligned["f"].values
    errors = a - f
    return {
        "MSE": float(np.mean(errors ** 2)),
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
        "MAE": float(np.mean(np.abs(errors))),
        "MAPE_pct": float(np.mean(np.abs(errors / a)) * 100),
    }


# ---------------------------------------------------------------------------
# 7. Time-Series-Cross-Validation (expanding window)
# ---------------------------------------------------------------------------

def time_series_cv(
    series: pd.Series,
    order: tuple[int, int, int],
    n_splits: int = 5,
    horizon: int = 20,
) -> pd.DataFrame:
    """
    Expanding-Window-CV fuer Zeitreihen.

    Im Gegensatz zur klassischen k-fold-CV duerfen wir hier NICHT shufflen,
    weil Zeitreihen kausale Struktur haben (Vorlesung Folie 19).
    """
    n = len(series)
    min_train_size = int(n * 0.5)  # erste 50% bilden den Initial-Train

    splits = []
    step = (n - min_train_size - horizon) // n_splits
    for k in range(n_splits):
        train_end = min_train_size + k * step
        test_start = train_end
        test_end = test_start + horizon
        if test_end > n:
            break
        splits.append((train_end, test_start, test_end))

    results = []
    for k, (train_end, test_start, test_end) in enumerate(splits, 1):
        tr = series.iloc[:train_end]
        te = series.iloc[test_start:test_end]
        try:
            model = ARIMA(tr, order=order).fit()
            fc, _ = forecast_with_ci(model, horizon=len(te))
            fc.index = te.index
            metrics = evaluate(te, fc)
            metrics["fold"] = k
            metrics["train_size"] = train_end
            results.append(metrics)
        except Exception as exc:
            print(f"    Fold {k} fehlgeschlagen: {exc}")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Hauptpipeline
# ---------------------------------------------------------------------------

def run_box_jenkins(asset: str = "DAX", test_ratio: float = 0.2) -> None:
    print(f"\n{'='*70}")
    print(f"  Box-Jenkins-Pipeline fuer {asset}")
    print(f"{'='*70}\n")

    outdir = REPORTS_BASE / asset.lower()
    outdir.mkdir(parents=True, exist_ok=True)

    # --- 1. Laden + Log-Transformation + Split ---
    prices = load_asset(asset)
    log_prices = np.log(prices)
    train, test = train_test_split(log_prices, test_ratio=test_ratio)
    print(f"  Train: {len(train):>5} Beobachtungen ({train.index.min().date()} bis {train.index.max().date()})")
    print(f"  Test : {len(test):>5} Beobachtungen ({test.index.min().date()} bis {test.index.max().date()})")
    print(f"  Modellierung auf Log-Preisen (numerisch stabiler, Returns implizit via d=1)")

    # --- 2. Ordnung finden ---
    print()
    order = find_best_order(train)
    print(f"  -> Gewaehlte Ordnung: ARIMA{order}")

    # --- 3. Fit ---
    print(f"\n  Fitte ARIMA{order} auf Trainingsdaten ...")
    fitted = fit_arima(train, order)

    summary_path = outdir / "model_summary.txt"
    summary_path.write_text(str(fitted.summary()), encoding="utf-8")
    print(f"  Summary -> {summary_path.relative_to(REPO_ROOT)}")

    # --- 4. Residuen-Diagnostik ---
    print("\n  Residuen-Diagnostik ...")
    diag = residual_diagnostics(fitted, asset, outdir)
    print(f"    Ljung-Box p (Lag 10): {diag['ljung_box_lag10_p']:.4f}  "
          f"{'OK (keine Autokorrelation)' if diag['ljung_box_lag10_p'] > 0.05 else 'PROBLEM (Restautokorrelation)'}")
    print(f"    Ljung-Box p (Lag 20): {diag['ljung_box_lag20_p']:.4f}")
    print(f"    Residuen Std: {diag['residuals_std']:.5f}  |  "
          f"Skew: {diag['residuals_skew']:.3f}  |  Kurt: {diag['residuals_kurtosis']:.3f}")

    # --- 5. Forecast ---
    print(f"\n  Forecast ueber {len(test)} Tage mit 95%-Konfidenzintervall ...")
    forecast, ci = forecast_with_ci(fitted, horizon=len(test))
    forecast.index = test.index
    ci.index = test.index
    plot_forecast(train, test, forecast, ci, asset, order, outdir)

    # --- 6. Evaluation auf Test-Set ---
    metrics = evaluate(test, forecast)
    # In Originalpreisen umrechnen (e^log_price), um interpretierbare Metriken zu haben
    metrics_orig = evaluate(np.exp(test), np.exp(forecast))
    eval_df = pd.DataFrame({
        "Metrik": list(metrics.keys()),
        "Log-Preise": list(metrics.values()),
        "Original-Preise": list(metrics_orig.values()),
    })
    eval_path = outdir / "evaluation.csv"
    eval_df.to_csv(eval_path, index=False)
    print(f"\n  Evaluation auf Test-Set:")
    print(eval_df.to_string(index=False))
    print(f"  -> {eval_path.relative_to(REPO_ROOT)}")

    # --- 7. Cross-Validation ---
    print(f"\n  Time-Series-Cross-Validation (5 Folds, Horizon=20) ...")
    cv = time_series_cv(log_prices, order, n_splits=5, horizon=20)
    cv_path = outdir / "cv_results.csv"
    cv.to_csv(cv_path, index=False)
    print(f"  CV-Mittelwert RMSE: {cv['RMSE'].mean():.5f}  (Std: {cv['RMSE'].std():.5f})")
    print(f"  CV-Mittelwert MAPE: {cv['MAPE_pct'].mean():.3f}%  (Std: {cv['MAPE_pct'].std():.3f}%)")
    print(f"  -> {cv_path.relative_to(REPO_ROOT)}")

    print(f"\n  Outputs in: {outdir.relative_to(REPO_ROOT)}")
    print(f"{'='*70}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Box-Jenkins ARIMA-Pipeline.")
    parser.add_argument("--asset", default="DAX", choices=["DAX", "Gold", "Bitcoin"],
                        help="Welches Asset modelliert wird (default: DAX).")
    parser.add_argument("--test-ratio", type=float, default=0.2,
                        help="Anteil der Daten im Test-Set (default: 0.2).")
    args = parser.parse_args()
    run_box_jenkins(asset=args.asset, test_ratio=args.test_ratio)


if __name__ == "__main__":
    main()
