"""
═══════════════════════════════════════════════════════════════════════════════
 02_box_jenkins_dax.py
 ─────────────────────
 Vollständige Box-Jenkins-Analyse für den DAX (monatliche Schlusskurse).
 Zweistufig:  (1) Preise zeigen Random-Walk-Charakter (I(1))
              (2) Box-Jenkins auf Log-Returns

 Direkt ausführbar oder Block-für-Block in Jupyter (jeder „# %%" = neue Zelle).
 Autor: Nico Hirsch · THWS · SoSe 2026
═══════════════════════════════════════════════════════════════════════════════
"""

# %% Imports ──────────────────────────────────────────────────────────────────
from pathlib import Path
from itertools import product
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from scipy import stats as scs
import pmdarima as pm
from pmdarima.arima.utils import ndiffs, nsdiffs

from src.data.load import load_asset, load_returns

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
FIG_DIR = Path("reports/figures"); FIG_DIR.mkdir(parents=True, exist_ok=True)
ASSET = "DAX"

# %% 1) Daten laden ───────────────────────────────────────────────────────────
prices = load_asset(ASSET, start="2015-01-01", freq="ME")
print(f"{ASSET}: n={len(prices)} | "
      f"{prices.index.min().date()} → {prices.index.max().date()}")
print(prices.describe().round(2))

returns = load_returns(prices, log=True)   # Log-Returns
print(f"\nLog-Returns: μ={returns.mean():.4f}  "
      f"σ={returns.std():.4f}  "
      f"Skew={returns.skew():.2f}  Kurt={returns.kurtosis():.2f}")

# %% 2) Visualisierung Preise + Returns ───────────────────────────────────────
fig, ax = plt.subplots(2, 2, figsize=(14, 8))
prices.plot(ax=ax[0, 0], color="#264653", linewidth=1.5)
ax[0, 0].axvline(pd.Timestamp("2020-03-01"), color="crimson", ls="--", alpha=0.7)
ax[0, 0].text(pd.Timestamp("2020-03-01"), prices.max()*0.95, " COVID",
              color="crimson")
ax[0, 0].axvline(pd.Timestamp("2022-02-24"), color="crimson", ls="--", alpha=0.7)
ax[0, 0].text(pd.Timestamp("2022-02-24"), prices.max()*0.85, " Ukraine",
              color="crimson")
ax[0, 0].set_title(f"{ASSET} — monatliche Schlusskurse")

returns.plot(ax=ax[0, 1], color="#2A9D8F")
ax[0, 1].set_title("Log-Returns")
ax[0, 1].axhline(0, color="grey", lw=0.5)

# Volatilitätscluster sichtbar machen
returns.abs().plot(ax=ax[1, 0], color="#E76F51")
returns.abs().rolling(6).mean().plot(ax=ax[1, 0], color="black", lw=2,
                                     label="6-M Rolling-|Return|")
ax[1, 0].set_title("Volatilitäts-Cluster (|Return|)")
ax[1, 0].legend()

# Saisonalität (gibt es eine?)
sns.boxplot(x=returns.index.month, y=returns.values, ax=ax[1, 1],
            palette="viridis")
ax[1, 1].set_title("Returns nach Monat ('Sell in May'?)")
ax[1, 1].axhline(0, color="grey", lw=0.5); ax[1, 1].set_xlabel("Monat")

plt.tight_layout(); plt.savefig(FIG_DIR/f"02_{ASSET}_overview.png", dpi=120)
plt.show()

# %% 3) Stationaritätstests Preise ────────────────────────────────────────────
def stationarity(s, name="") -> pd.DataFrame:
    s = s.dropna()
    adf = adfuller(s, autolag="AIC")
    kp  = kpss(s, regression="c", nlags="auto")
    return pd.DataFrame({
        "Test": ["ADF", "KPSS"],
        "H0":   ["Einheitswurzel", "Stationarität"],
        "Statistik": [round(adf[0], 3), round(kp[0], 3)],
        "p-Wert":    [round(adf[1], 4), round(kp[1], 4)],
        "Ergebnis":  [
            "stationär" if adf[1] < 0.05 else "nicht stationär",
            "stationär" if kp[1] >= 0.05 else "nicht stationär"
        ]
    }).assign(Reihe=name)

print("\nPreise (Niveau):")
print(stationarity(prices, "prices").to_string(index=False))

print("\nLog-Preise:")
print(stationarity(np.log(prices), "log prices").to_string(index=False))

print("\nLog-Returns:")
print(stationarity(returns, "log returns").to_string(index=False))

# Ordnungsempfehlung
print(f"\nEmpfohlenes d für log(Preise): "
      f"ADF={ndiffs(np.log(prices), test='adf')}, "
      f"KPSS={ndiffs(np.log(prices), test='kpss')}")
print(f"Saisonale Differenz D (m=12): "
      f"{nsdiffs(returns, m=12, test='ocsb')}")

# %% 4) Transformation ────────────────────────────────────────────────────────
# Aktienpreise sind klassisch I(1) → Returns sind die geeignete Reihe für ARIMA
y = returns.copy()
print(f"\nAnalysiere weiter: Log-Returns ({ASSET}), n={len(y)}")

# %% 5) ACF / PACF der Returns ────────────────────────────────────────────────
fig, ax = plt.subplots(2, 2, figsize=(14, 8))
plot_acf(y, lags=36, ax=ax[0, 0]);  ax[0, 0].set_title(f"ACF — {ASSET} Returns")
plot_pacf(y, lags=36, ax=ax[0, 1], method="ywm")
ax[0, 1].set_title(f"PACF — {ASSET} Returns")

# Wichtig für Finance-Daten: ACF der quadrierten Returns (Vola-Cluster!)
plot_acf(y**2, lags=36, ax=ax[1, 0])
ax[1, 0].set_title("ACF — Returns² (zeigt ARCH-Effekte!)")
plot_pacf(y**2, lags=36, ax=ax[1, 1], method="ywm")
ax[1, 1].set_title("PACF — Returns²")

plt.tight_layout(); plt.savefig(FIG_DIR/f"02_{ASSET}_acf_pacf.png", dpi=120)
plt.show()

# Engle's ARCH-Test
arch = het_arch(y, nlags=12)
print(f"\nEngle ARCH-Test (H0 = keine ARCH-Effekte, Vola konstant):")
print(f"  LM-Statistik = {arch[0]:.2f}, p = {arch[1]:.4f}")
if arch[1] < 0.05:
    print("  → ARCH-Effekte vorhanden! Diskussion: GARCH wäre angemessen.")

# %% 6) Modellselektion ARIMA auf Returns ─────────────────────────────────────
results = []
for p, q in product(range(0, 4), range(0, 4)):
    try:
        m = ARIMA(y, order=(p, 0, q)).fit()  # d=0, weil Returns schon stationär
        results.append({"order": (p, 0, q), "AIC": m.aic, "BIC": m.bic,
                        "LogLik": m.llf})
    except Exception:
        continue
grid = pd.DataFrame(results).sort_values("AIC").reset_index(drop=True)
print("\nTop-5 nach AIC:"); print(grid.head().round(2))

auto = pm.auto_arima(
    y, seasonal=True, m=12,
    d=0, D=0,
    start_p=0, start_q=0, max_p=3, max_q=3, max_P=2, max_Q=2,
    stepwise=True, trace=False, suppress_warnings=True,
    error_action="ignore", information_criterion="aic",
)
print(f"\nauto_arima Wahl: {auto.order} × {auto.seasonal_order}")

best_order = grid.iloc[0]["order"]
best = ARIMA(y, order=best_order).fit()
print(f"\nGewähltes Modell: ARIMA{best_order}")
print(best.summary())

# %% 7) Residuendiagnose ──────────────────────────────────────────────────────
resid = best.resid

fig, ax = plt.subplots(2, 2, figsize=(14, 8))
resid.plot(ax=ax[0, 0]); ax[0, 0].set_title("Residuen über Zeit")
ax[0, 0].axhline(0, color="grey", lw=0.5)

ax[0, 1].hist(resid, bins=30, density=True, color="#2A9D8F", alpha=0.7,
              edgecolor="white")
xs = np.linspace(resid.min(), resid.max(), 200)
ax[0, 1].plot(xs, scs.norm.pdf(xs, resid.mean(), resid.std()),
              color="crimson", lw=2, label="N(μ,σ)")
ax[0, 1].set_title(
    f"Histogramm  (Skew={scs.skew(resid):.2f}, Kurt={scs.kurtosis(resid):.2f})"
)
ax[0, 1].legend()

plot_acf(resid, lags=36, ax=ax[1, 0]); ax[1, 0].set_title("ACF Residuen")
sm.qqplot(resid, line="s", ax=ax[1, 1]); ax[1, 1].set_title("Q-Q-Plot")
plt.tight_layout(); plt.savefig(FIG_DIR/f"02_{ASSET}_residuen.png", dpi=120)
plt.show()

lb = acorr_ljungbox(resid, lags=[10, 20, 30], return_df=True)
jb = scs.jarque_bera(resid)
print("\nLjung-Box (H0 = keine Autokorrelation, p>0.05 = gut):")
print(lb.round(4))
print(f"\nJarque-Bera: stat={jb.statistic:.2f}, p={jb.pvalue:.4f} "
      f"({'normal' if jb.pvalue > 0.05 else 'NICHT normal — Fat Tails!'})")

# %% 8) Koeffizienten-Tabelle ────────────────────────────────────────────────
coef = pd.DataFrame({
    "Koeffizient":  best.params,
    "Std-Fehler":   best.bse,
    "t-Statistik":  best.tvalues,
    "p-Wert":       best.pvalues,
    "signifikant_5%": best.pvalues < 0.05,
})
print("\nKoeffizienten-Tabelle:"); print(coef.round(4))

# %% 9) Forecast 10 Monate ────────────────────────────────────────────────────
fc       = best.get_forecast(steps=10)
mean_r   = fc.predicted_mean
ci_r     = fc.conf_int(alpha=0.05)

# Rück-Integration zu Preis-Forecast
last_price = prices.iloc[-1]
price_path = last_price * np.exp(mean_r.cumsum())
ci_lower   = last_price * np.exp(ci_r.iloc[:, 0].cumsum())
ci_upper   = last_price * np.exp(ci_r.iloc[:, 1].cumsum())

fig, ax = plt.subplots(2, 1, figsize=(13, 10))

# Returns-Forecast
y.iloc[-48:].plot(ax=ax[0], label="Historie", color="#264653", linewidth=1.5)
mean_r.plot(ax=ax[0], label="Forecast Returns", color="#E76F51", linewidth=2)
ax[0].fill_between(ci_r.index, ci_r.iloc[:, 0], ci_r.iloc[:, 1],
                   color="#E76F51", alpha=0.25, label="95%-KI")
ax[0].axhline(0, color="grey", lw=0.5)
ax[0].set_title(f"ARIMA{best_order} — Returns-Forecast {ASSET}")
ax[0].legend(); ax[0].grid(alpha=0.3)

# Preis-Forecast (rück-integriert)
prices.iloc[-48:].plot(ax=ax[1], label="Historie", color="#264653", linewidth=2)
price_path.plot(ax=ax[1], label="Forecast", color="#E76F51", linewidth=2.5)
ax[1].fill_between(price_path.index, ci_lower, ci_upper,
                   color="#E76F51", alpha=0.25, label="95%-KI")
ax[1].set_title(f"Implizierter Preis-Forecast {ASSET} (rück-integriert)")
ax[1].legend(); ax[1].grid(alpha=0.3)

plt.tight_layout(); plt.savefig(FIG_DIR/f"02_{ASSET}_forecast.png", dpi=120)
plt.show()

print("\nForecast-Tabelle:")
print(pd.DataFrame({
    "Return Forecast": mean_r.round(4),
    "implizierter Preis": price_path.round(2),
    "Untergrenze Preis": ci_lower.round(2),
    "Obergrenze Preis":  ci_upper.round(2),
}).to_string())

# %% 10) Diskussion (für die Slides!) ─────────────────────────────────────────
print("""
═══════════════════════════════════════════════════════════════════
DISKUSSION (für Termin 2 / Final-Präsentation)
═══════════════════════════════════════════════════════════════════
1. EFFIZIENZMARKTHYPOTHESE
   Wenn Returns signifikante Autokorrelation hätten, gäbe es
   triviale Trading-Strategien. Faktum: ACF der Returns ist meist
   schwach → ARIMA findet kaum Struktur. ARIMA(0,0,0) (= konstantes
   Mittel) ist oft kompetitiv. Das ist KEIN Misserfolg, sondern
   empirische Bestätigung der EMH.

2. ABER: VOLATILITÄT IST PROGNOSTIZIERBAR
   Der Engle ARCH-Test zeigt klare Vola-Cluster.
   → ARIMA prognostiziert Mittelwert, nicht Varianz.
   → GARCH-Modelle sind die natürliche Erweiterung.
   (Ggf. als Ausblick in der Final-Präsentation erwähnen!)

3. FAT TAILS
   Jarque-Bera verwirft Normalität fast immer bei Finance-Daten.
   → Konfidenzintervalle aus Normalannahme unterschätzen Tail-Risk.
   → Bootstrap-KI oder t-Verteilung wären angemessener.

4. STRUKTURBRÜCHE
   COVID-Crash März 2020 und Ukraine-Krieg Februar 2022 sind
   sichtbar. Klassisches ARIMA assumiert Konstanz der Parameter
   → Markov-Switching oder Regime-Modelle wären robuster.
═══════════════════════════════════════════════════════════════════
""")

# %% 11) Speichern für Pipeline ──────────────────────────────────────────────
import joblib
joblib.dump(best, f"reports/best_arima_{ASSET}.pkl")
print(f"✅ Modell gespeichert unter reports/best_arima_{ASSET}.pkl")
