"""
Multivariate Zeitreihenanalyse – Modellvergleich & beste Prognose
Zeitreihen: Apple (AAPL), Microsoft (MSFT), Google (GOOGL)
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.varmax import VARMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import yfinance as yf
import os

os.makedirs("results/plots", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# ── Konfiguration ───────────────────────────────────────────────────────────
TICKERS   = ["AAPL", "MSFT", "GOOGL"]
START     = "2020-01-01"
END       = "2024-12-31"
TEST_SIZE = 30      # letzte 30 Tage als Test-Set
FORECAST  = 10      # Perioden voraus
ALPHA     = 0.05


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATEN LADEN
# ═══════════════════════════════════════════════════════════════════════════
print("📥 Lade alle Zeitreihen ...")
raw = yf.download(TICKERS, start=START, end=END, progress=False)["Close"]
raw.columns = TICKERS
raw.dropna(inplace=True)
raw.to_csv("data/processed/all_stocks.csv")
print(f"   {len(raw)} Beobachtungen × {len(TICKERS)} Zeitreihen")

# Train / Test Split
train = raw.iloc[:-TEST_SIZE]
test  = raw.iloc[-TEST_SIZE:]
print(f"   Train: {len(train)} | Test: {len(test)}")


# ═══════════════════════════════════════════════════════════════════════════
# 2. EVALUATIONSMETRIKEN
# ═══════════════════════════════════════════════════════════════════════════
def evaluate(actual: pd.Series, predicted: np.ndarray) -> dict:
    """Berechnet MAE, RMSE und MAPE."""
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual.values - predicted) / actual.values)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


# ═══════════════════════════════════════════════════════════════════════════
# 3. MODELL-LOOP (ARIMA Varianten pro Zeitreihe)
# ═══════════════════════════════════════════════════════════════════════════
ORDERS = [(1,1,0), (1,1,1), (2,1,1), (0,1,1), (2,1,2)]

print(f"\n{'═'*70}")
print("🔁 Modell-Loop: alle Zeitreihen × alle Modelle")
print(f"{'═'*70}")

all_metrics = []

for ticker in TICKERS:
    print(f"\n── {ticker} ──")
    s_train = train[ticker]
    s_test  = test[ticker]

    ticker_results = []
    for order in ORDERS:
        try:
            model = ARIMA(s_train, order=order).fit()
            fc    = model.forecast(steps=TEST_SIZE)
            m     = evaluate(s_test, fc.values)
            m["Ticker"] = ticker
            m["Modell"] = f"ARIMA{order}"
            m["AIC"]    = model.aic
            ticker_results.append(m)
            print(f"   ARIMA{order}  MAE={m['MAE']:7.2f}  RMSE={m['RMSE']:7.2f}  MAPE={m['MAPE']:6.2f}%  AIC={m['AIC']:8.1f}")
        except Exception as e:
            print(f"   ARIMA{order}  ❌ Fehler: {e}")

    all_metrics.extend(ticker_results)

# DataFrame mit allen Ergebnissen
metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv("results/model_comparison.csv", index=False)
print(f"\n💾 Modellvergleich gespeichert: results/model_comparison.csv")


# ═══════════════════════════════════════════════════════════════════════════
# 4. ÜBERSICHT DER EVALUATIONSMETRIKEN
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*70}")
print("📊 Übersicht Evaluationsmetriken (gemittelt über alle Zeitreihen)")
print(f"{'═'*70}")

summary = (
    metrics_df.groupby("Modell")[["MAE", "RMSE", "MAPE"]]
    .mean()
    .sort_values("RMSE")
)
print(summary.to_string(float_format="{:.2f}".format))

# Bestes Modell bestimmen
best_model_name = summary.index[0]
best_order_str  = best_model_name.replace("ARIMA", "")
best_order      = eval(best_order_str)
print(f"\n🏆 Bestes Modell: {best_model_name}  (niedrigstes RMSE)")


# ═══════════════════════════════════════════════════════════════════════════
# 5. HEATMAP DER METRIKEN
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Modellvergleich: Evaluationsmetriken", fontsize=14)

for ax, metric in zip(axes, ["MAE", "RMSE", "MAPE"]):
    pivot = metrics_df.pivot(index="Modell", columns="Ticker", values=metric)
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title(metric)
    plt.colorbar(im, ax=ax)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i,j]:.1f}",
                    ha="center", va="center", fontsize=8, color="black")

plt.tight_layout()
path = "results/plots/model_comparison_heatmap.png"
plt.savefig(path, dpi=150)
plt.close()
print(f"💾 Heatmap gespeichert: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# 6. PROGNOSE MIT BESTEM MODELL (alle Zeitreihen)
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*70}")
print(f"📈 Prognose mit {best_model_name} für alle Zeitreihen")
print(f"{'═'*70}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
fig.suptitle(f"10-Perioden-Prognose mit {best_model_name}", fontsize=14)

for ax, ticker in zip(axes, TICKERS):
    s_full  = raw[ticker]
    model   = ARIMA(s_full, order=best_order).fit()
    fc      = model.get_forecast(steps=FORECAST)
    mean_fc = fc.predicted_mean
    ci      = fc.conf_int(alpha=ALPHA)

    hist = s_full.iloc[-60:]
    ax.plot(hist.index, hist.values, label="Historisch", color="#2563eb", linewidth=1.5)
    ax.plot(mean_fc.index, mean_fc.values, label="Prognose",
            color="#dc2626", linewidth=2, linestyle="--")
    ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1],
                    alpha=0.2, color="#dc2626",
                    label=f"{int((1-ALPHA)*100)}% KI")
    ax.set_title(ticker, fontsize=13)
    ax.set_xlabel("Datum")
    ax.set_ylabel("Preis (USD)")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    print(f"\n  {ticker} – Prognose:")
    print(f"  {'Datum':<14} {'Prognose':>10} {'CI unten':>10} {'CI oben':>10}")
    print("  " + "─" * 48)
    for date, m, lo, hi in zip(mean_fc.index, mean_fc, ci.iloc[:, 0], ci.iloc[:, 1]):
        print(f"  {str(date.date()):<14} {m:>10.2f} {lo:>10.2f} {hi:>10.2f}")

plt.tight_layout()
path = "results/plots/multivariate_forecast.png"
plt.savefig(path, dpi=150)
plt.close()
print(f"\n💾 Multivariate Prognose gespeichert: {path}")

print(f"\n{'═'*70}")
print("✅ Multivariate Analyse abgeschlossen!")
print(f"   Alle Plots:  results/plots/")
print(f"   Metriken:    results/model_comparison.csv")
print(f"{'═'*70}")
