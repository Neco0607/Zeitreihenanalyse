"""
Univariate Zeitreihenanalyse – ARIMA Modell
Zeitreihen: Apple (AAPL), Microsoft (MSFT), Google (GOOGL)
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import yfinance as yf
import os

# ── Ordner für Ergebnisse ───────────────────────────────────────────────────
os.makedirs("results/plots", exist_ok=True)
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# ── Konfiguration ───────────────────────────────────────────────────────────
TICKERS   = {"Antonio": "AAPL", "Person2": "MSFT", "Person3": "GOOGL"}
START     = "2020-01-01"
END       = "2024-12-31"
FORECAST  = 10          # Perioden voraus
ALPHA     = 0.05        # Signifikanzniveau


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATEN LADEN
# ═══════════════════════════════════════════════════════════════════════════
def load_data(ticker: str) -> pd.Series:
    """Lädt Schlusskurse von Yahoo Finance und speichert als CSV."""
    print(f"\n📥 Lade {ticker} ...")
    df = yf.download(ticker, start=START, end=END, progress=False)
    series = df["Close"].squeeze()
    series.name = ticker
    series.to_csv(f"data/raw/{ticker}.csv")
    print(f"   {len(series)} Beobachtungen geladen ({series.index[0].date()} – {series.index[-1].date()})")
    return series


# ═══════════════════════════════════════════════════════════════════════════
# 2. STATIONARITÄTSTEST (ADF)
# ═══════════════════════════════════════════════════════════════════════════
def adf_test(series: pd.Series, label: str = "") -> bool:
    """Augmented Dickey-Fuller Test. Gibt True zurück wenn stationär."""
    result = adfuller(series.dropna())
    stationary = result[1] < ALPHA
    print(f"\n📊 ADF-Test {label}")
    print(f"   Teststatistik : {result[0]:.4f}")
    print(f"   p-Wert        : {result[1]:.4f}  →  {'✅ stationär' if stationary else '❌ nicht stationär'}")
    return stationary


# ═══════════════════════════════════════════════════════════════════════════
# 3. TRANSFORMATION (Differenzieren bis Stationarität)
# ═══════════════════════════════════════════════════════════════════════════
def make_stationary(series: pd.Series) -> tuple[pd.Series, int]:
    """Differenziert die Reihe solange bis sie stationär ist."""
    d = 0
    s = series.copy()
    while not adf_test(s, label=f"(d={d})"):
        s = s.diff().dropna()
        d += 1
        if d > 3:
            print("⚠️  Warnung: d > 3, breche ab.")
            break
    print(f"   → Integrationsordnung d = {d}")
    return s, d


# ═══════════════════════════════════════════════════════════════════════════
# 4. ACF / PACF PLOT
# ═══════════════════════════════════════════════════════════════════════════
def plot_acf_pacf(series: pd.Series, ticker: str):
    """Erstellt ACF und PACF Plot."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"ACF & PACF – {ticker} (transformiert)", fontsize=14)
    plot_acf(series.dropna(),  ax=axes[0], lags=30, title="ACF")
    plot_pacf(series.dropna(), ax=axes[1], lags=30, title="PACF")
    plt.tight_layout()
    path = f"results/plots/{ticker}_acf_pacf.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"   💾 ACF/PACF gespeichert: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# 5. MODELLSELEKTION (AIC-Grid-Search)
# ═══════════════════════════════════════════════════════════════════════════
def select_arima(series: pd.Series, d: int) -> tuple[int, int, int]:
    """Sucht bestes ARIMA(p,d,q) per AIC über ein kleines Grid."""
    best_aic = np.inf
    best_order = (1, d, 1)
    print(f"\n🔍 Modellselektion für d={d} ...")
    for p in range(0, 4):
        for q in range(0, 4):
            try:
                m = ARIMA(series, order=(p, d, q)).fit()
                if m.aic < best_aic:
                    best_aic   = m.aic
                    best_order = (p, d, q)
            except Exception:
                continue
    print(f"   Bestes Modell: ARIMA{best_order}  AIC={best_aic:.2f}")
    return best_order


# ═══════════════════════════════════════════════════════════════════════════
# 6. MODELL SCHÄTZEN & DIAGNOSTIK
# ═══════════════════════════════════════════════════════════════════════════
def fit_and_diagnose(series: pd.Series, order: tuple, ticker: str):
    """Schätzt ARIMA, gibt t-Statistiken aus und testet Residuen."""
    model = ARIMA(series, order=order).fit()

    print(f"\n📋 Koeffizienten – ARIMA{order} ({ticker})")
    print(f"{'Parameter':<12} {'Koeffizient':>14} {'Std-Fehler':>12} {'t-Statistik':>13} {'p-Wert':>10}")
    print("─" * 65)
    for name, coef, se in zip(
        model.param_names,
        model.params,
        model.bse,
    ):
        t_stat = coef / se if se != 0 else np.nan
        p_val  = model.pvalues[name]
        sig    = "**" if p_val < 0.05 else ("*" if p_val < 0.1 else "")
        print(f"{name:<12} {coef:>14.4f} {se:>12.4f} {t_stat:>13.4f} {p_val:>10.4f} {sig}")

    # Ljung-Box Test auf Residuen
    lb = acorr_ljungbox(model.resid, lags=[10], return_df=True)
    lb_p = lb["lb_pvalue"].values[0]
    print(f"\n🧪 Ljung-Box (Lag 10): p={lb_p:.4f}  →  {'✅ keine Autokorrelation' if lb_p > ALPHA else '⚠️  Autokorrelation vorhanden'}")

    # Residuen-Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Residuen-Diagnostik – ARIMA{order} ({ticker})", fontsize=14)
    model.plot_diagnostics(fig=fig)
    plt.tight_layout()
    path = f"results/plots/{ticker}_diagnostics.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"   💾 Diagnostik gespeichert: {path}")

    return model


# ═══════════════════════════════════════════════════════════════════════════
# 7. PROGNOSE (10 Perioden)
# ═══════════════════════════════════════════════════════════════════════════
def forecast_and_plot(model, series: pd.Series, ticker: str):
    """Erstellt 10-Perioden-Prognose mit Konfidenzintervall."""
    fc = model.get_forecast(steps=FORECAST)
    mean_fc = fc.predicted_mean
    ci      = fc.conf_int(alpha=ALPHA)

    # Letzten 60 Tage + Prognose
    hist = series.iloc[-60:]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(hist.index, hist.values, label="Historisch", color="#2563eb", linewidth=1.5)
    ax.plot(mean_fc.index, mean_fc.values, label="Prognose", color="#dc2626", linewidth=2, linestyle="--")
    ax.fill_between(
        ci.index,
        ci.iloc[:, 0],
        ci.iloc[:, 1],
        alpha=0.2,
        color="#dc2626",
        label=f"{int((1-ALPHA)*100)}% Konfidenzintervall",
    )
    ax.set_title(f"10-Perioden-Prognose – {ticker}", fontsize=14)
    ax.set_xlabel("Datum")
    ax.set_ylabel("Preis (USD)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    path = f"results/plots/{ticker}_forecast.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"   💾 Prognose gespeichert: {path}")

    print(f"\n📈 Prognose {ticker} (nächste {FORECAST} Handelstage):")
    print(f"{'Datum':<14} {'Prognose':>10} {'CI unten':>10} {'CI oben':>10}")
    print("─" * 48)
    for date, m, lo, hi in zip(mean_fc.index, mean_fc, ci.iloc[:, 0], ci.iloc[:, 1]):
        print(f"{str(date.date()):<14} {m:>10.2f} {lo:>10.2f} {hi:>10.2f}")

    return mean_fc, ci


# ═══════════════════════════════════════════════════════════════════════════
# 8. HAUPTSCHLEIFE – UNIVARIATE ANALYSE
# ═══════════════════════════════════════════════════════════════════════════
results = {}

for person, ticker in TICKERS.items():
    print(f"\n{'═'*60}")
    print(f"  {person} → {ticker}")
    print(f"{'═'*60}")

    # Daten laden
    series = load_data(ticker)

    # Stationarität & Transformation
    stationary_series, d = make_stationary(series)

    # ACF / PACF
    plot_acf_pacf(stationary_series, ticker)

    # Modellselektion
    order = select_arima(series, d)

    # Modell schätzen & Diagnostik
    model = fit_and_diagnose(series, order, ticker)

    # Prognose
    mean_fc, ci = forecast_and_plot(model, series, ticker)

    # Ergebnisse speichern
    results[ticker] = {
        "series"  : series,
        "model"   : model,
        "order"   : order,
        "d"       : d,
        "forecast": mean_fc,
        "ci"      : ci,
        "aic"     : model.aic,
        "bic"     : model.bic,
    }

print(f"\n{'═'*60}")
print("✅ Univariate Analyse abgeschlossen!")
print(f"{'═'*60}")

# Zusammenfassung
print("\n📊 Modellübersicht:")
print(f"{'Ticker':<10} {'Modell':<16} {'AIC':>10} {'BIC':>10}")
print("─" * 48)
for ticker, res in results.items():
    print(f"{ticker:<10} ARIMA{str(res['order']):<11} {res['aic']:>10.2f} {res['bic']:>10.2f}")
