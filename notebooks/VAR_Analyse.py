"""
VAR-Modell: Bitcoin, DAX, Gold (2016–2026)
==========================================
Ausführen aus dem Projektordner:
    python notebooks/VAR_Analyse.py
oder direkt aus dem notebooks/-Verzeichnis:
    cd notebooks && python VAR_Analyse.py
"""

# ── Warnungen komplett unterdrücken ──────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")
import logging
logging.captureWarnings(True)

# ── Standardbibliotheken ──────────────────────────────────────────────────────
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # kein Display nötig → keine GUI-Warnungen
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

# ── statsmodels ──────────────────────────────────────────────────────────────
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR
from statsmodels.stats.diagnostic import acorr_ljungbox

# ── Pfade ────────────────────────────────────────────────────────────────────
# Skript läuft sowohl aus dem Repo-Root als auch aus notebooks/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(BASE_DIR) == "notebooks":
    ROOT_DIR = os.path.dirname(BASE_DIR)
else:
    ROOT_DIR = BASE_DIR

DATA_DIR    = os.path.join(ROOT_DIR, "data", "raw")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Plot-Stil ────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["font.size"]      = 11
COLORS = {"Bitcoin": "#F7931A", "DAX": "#1f77b4", "Gold": "#FFD700"}

# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATEN LADEN & VORBEREITEN
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("VAR-MODELL  —  BITCOIN · DAX · GOLD  (2016–2026)")
print("=" * 70)

def load_series(filename: str, col_name: str) -> pd.Series:
    """CSV einlesen (skiprows=1 wegen Ticker-Zeile), Index auf Datum setzen."""
    path = os.path.join(DATA_DIR, filename)
    df   = pd.read_csv(path, skiprows=1, header=0)
    df.columns = ["Date", col_name]
    df         = df.dropna(subset=["Date", col_name])
    df["Date"] = pd.to_datetime(df["Date"])
    df         = df.set_index("Date").sort_index()
    df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
    return df[col_name].dropna()

btc  = load_series("bitcoin_kurs_2016_2026.csv", "Bitcoin")
dax  = load_series("dax_kurs_2016_2026.csv",     "DAX")
gold = load_series("goldpreis_2016_2026.csv",     "Gold")

# Gemeinsamen Datumsschnitt bilden (nur Handelstage, an denen ALLE drei Werte vorliegen)
df_raw = pd.concat([btc, dax, gold], axis=1).dropna()
df_raw.index.freq = None   # explizite Frequenz vermeiden → keine Warnung

print(f"\n  Zeitraum  : {df_raw.index[0].date()} → {df_raw.index[-1].date()}")
print(f"  Beobachtungen: {len(df_raw)}")
print(f"\n  Erste Zeilen:\n{df_raw.head(3)}\n")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. LOG-RETURNS  (VAR benötigt stationäre Reihen)
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("LOG-RETURNS (Transformation für Stationarität)")
print("=" * 70)

df_ret = np.log(df_raw).diff().dropna()
df_ret.columns = ["Bitcoin_ret", "DAX_ret", "Gold_ret"]

print("\n  Deskriptive Statistik der Log-Returns:")
print(df_ret.describe().round(6).to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# 3. STATIONARITÄTSTESTS (ADF)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STATIONARITÄTSTESTS (ADF)")
print("=" * 70)

def adf_test(series: pd.Series, name: str) -> bool:
    result = adfuller(series.dropna(), autolag="AIC")
    stat, pval = result[0], result[1]
    stationaer = pval <= 0.05
    symbol = "✅ Stationär" if stationaer else "❌ Nicht stationär"
    print(f"  {name:<25}  Stat={stat:>10.4f}  p={pval:.4f}  → {symbol}")
    return stationaer

print("\n  Niveaupreise:")
for col in df_raw.columns:
    adf_test(df_raw[col], col)

print("\n  Log-Returns:")
for col in df_ret.columns:
    adf_test(df_ret[col], col)

print("\n  → Log-Returns sind I(0) → geeignet für VAR-Modellierung.\n")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. EDA-VISUALISIERUNG
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle("VAR-Analyse: Bitcoin · DAX · Gold  —  Explorative Übersicht",
             fontsize=14, fontweight="bold")

assets = [("Bitcoin", "Bitcoin_ret", "Bitcoin (USD)"),
          ("DAX",     "DAX_ret",     "DAX-ETF (USD)"),
          ("Gold",    "Gold_ret",    "Gold (USD/oz)")]

for i, (asset, ret_col, ylabel) in enumerate(assets):
    color = COLORS[asset]
    # Preisverlauf
    axes[i, 0].plot(df_raw.index, df_raw[asset], color=color, linewidth=0.9, label=asset)
    axes[i, 0].set_title(f"{asset} — Preisverlauf", fontweight="bold")
    axes[i, 0].set_ylabel(ylabel)
    axes[i, 0].fill_between(df_raw.index, df_raw[asset],
                             df_raw[asset].min(), alpha=0.1, color=color)
    # Log-Returns
    axes[i, 1].plot(df_ret.index, df_ret[ret_col], color=color,
                    linewidth=0.5, alpha=0.8)
    axes[i, 1].axhline(0, color="black", linewidth=0.7, linestyle="--")
    axes[i, 1].axhline(df_ret[ret_col].mean() + 2 * df_ret[ret_col].std(),
                       color="red", linewidth=0.7, linestyle=":", label="±2σ")
    axes[i, 1].axhline(df_ret[ret_col].mean() - 2 * df_ret[ret_col].std(),
                       color="red", linewidth=0.7, linestyle=":")
    axes[i, 1].set_title(f"{asset} — Log-Returns", fontweight="bold")
    axes[i, 1].legend(fontsize=8)

plt.tight_layout()
eda_path = os.path.join(REPORTS_DIR, "var_eda.png")
plt.savefig(eda_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Grafik gespeichert: {eda_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. KORRELATIONSANALYSE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("KORRELATIONSANALYSE DER LOG-RETURNS")
print("=" * 70)

corr = df_ret.corr()
corr.index   = ["Bitcoin", "DAX", "Gold"]
corr.columns = ["Bitcoin", "DAX", "Gold"]
print(f"\n{corr.round(4).to_string()}")

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdYlGn",
            center=0, vmin=-1, vmax=1,
            linewidths=0.5, ax=ax, annot_kws={"size": 12})
ax.set_title("Korrelationsmatrix der Log-Returns", fontweight="bold")
plt.tight_layout()
corr_path = os.path.join(REPORTS_DIR, "var_korrelation.png")
plt.savefig(corr_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Grafik gespeichert: {corr_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. GRANGER-KAUSALITÄTSTESTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GRANGER-KAUSALITÄTSTESTS (Lag=5)")
print("=" * 70)

col_labels = {"Bitcoin_ret": "Bitcoin", "DAX_ret": "DAX", "Gold_ret": "Gold"}
pairs = [
    ("Bitcoin_ret", "DAX_ret"),
    ("Bitcoin_ret", "Gold_ret"),
    ("DAX_ret",     "Bitcoin_ret"),
    ("DAX_ret",     "Gold_ret"),
    ("Gold_ret",    "Bitcoin_ret"),
    ("Gold_ret",    "DAX_ret"),
]
print(f"\n  {'Ursache → Wirkung':<30}  p-Wert   Ergebnis")
print("  " + "-" * 55)
for cause, effect in pairs:
    gc_data = df_ret[[effect, cause]].dropna()
    gc_res  = grangercausalitytests(gc_data, maxlag=5, verbose=False)
    # p-Wert des F-Tests bei Lag 1 (konservativster Test)
    p = gc_res[1][0]["ssr_ftest"][1]
    label = f"  {col_labels[cause]} → {col_labels[effect]}"
    result = "✅ Granger-kausal" if p < 0.05 else "  kein Effekt"
    print(f"  {label:<30}  p={p:.4f}  {result}")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. VAR-LAG-SELEKTION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("VAR LAG-SELEKTION (AIC/BIC/HQIC/FPE)")
print("=" * 70)

# Train-Test-Split: letzte 60 Beobachtungen als Test
TEST_SIZE  = 60
train_ret  = df_ret.iloc[:-TEST_SIZE]
test_ret   = df_ret.iloc[-TEST_SIZE:]

var_selector = VAR(train_ret)
lag_results  = var_selector.select_order(maxlags=15)
print(f"\n{lag_results.summary()}")

# Besten Lag nach AIC
best_lag = int(lag_results.aic)
# Sicherheitsnetz: min 1, max 10
best_lag = max(1, min(best_lag, 10))
print(f"\n  → Gewählter Lag (AIC): {best_lag}")

# ═══════════════════════════════════════════════════════════════════════════════
# 8. VAR-MODELL SCHÄTZEN
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"VAR({best_lag}) MODELL — SCHÄTZUNG AUF TRAININGSDATEN")
print("=" * 70)

var_model  = VAR(train_ret)
var_fitted = var_model.fit(maxlags=best_lag, ic=None, trend="c")

print(f"\n{var_fitted.summary()}")

# ═══════════════════════════════════════════════════════════════════════════════
# 9. RESIDUENDIAGNOSTIK
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("RESIDUENDIAGNOSTIK")
print("=" * 70)

residuals_df = var_fitted.resid.copy()
residuals_df.columns = ["Bitcoin_res", "DAX_res", "Gold_res"]
residuals_df = residuals_df.dropna()

# Durbin-Watson (grob)
from statsmodels.stats.stattools import durbin_watson
dw_stats = durbin_watson(residuals_df.values)
for col, dw in zip(["Bitcoin", "DAX", "Gold"], dw_stats):
    flag = "✅ ok" if 1.5 < dw < 2.5 else "⚠️  prüfen"
    print(f"  Durbin-Watson {col:<10}: {dw:.4f}  {flag}")

# Ljung-Box auf jede Residuenreihe
print()
for col in residuals_df.columns:
    series = residuals_df[col].dropna().values
    lb = acorr_ljungbox(series, lags=[10], return_df=True)
    p  = lb["lb_pvalue"].values[0]
    flag = "✅ Weißes Rauschen" if p > 0.05 else "⚠️  Autokorrelation"
    asset = col.replace("_res", "")
    print(f"  Ljung-Box (Lag 10) {asset:<10}: p={p:.4f}  → {flag}")

# Residuen-Grafik
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
fig.suptitle(f"VAR({best_lag}) — Residuendiagnostik",
             fontsize=14, fontweight="bold")

res_labels = [("Bitcoin_res", "Bitcoin"), ("DAX_res", "DAX"), ("Gold_res", "Gold")]
for i, (col, label) in enumerate(res_labels):
    color = list(COLORS.values())[i]
    res   = residuals_df[col]

    # Zeitverlauf
    axes[i, 0].plot(res.index, res, color=color, linewidth=0.6, alpha=0.8)
    axes[i, 0].axhline(0, color="black", linewidth=0.7, linestyle="--")
    axes[i, 0].set_title(f"{label} — Residuenverlauf")
    axes[i, 0].set_ylabel("Residuum")

    # Histogramm + Normalverteilung
    axes[i, 1].hist(res, bins=60, density=True, color=color, alpha=0.7, edgecolor="none")
    x = np.linspace(res.min(), res.max(), 300)
    axes[i, 1].plot(x, stats.norm.pdf(x, res.mean(), res.std()),
                    color="red", linewidth=2, label="N(μ,σ²)")
    axes[i, 1].set_title(f"{label} — Residuenverteilung")
    axes[i, 1].legend(fontsize=8)

    # Q-Q-Plot
    stats.probplot(res, dist="norm", plot=axes[i, 2])
    axes[i, 2].set_title(f"{label} — Q-Q-Plot")

plt.tight_layout()
resid_path = os.path.join(REPORTS_DIR, "var_residuals.png")
plt.savefig(resid_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Grafik gespeichert: {resid_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# 10. BACKTESTING (60-Tage Out-of-Sample)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BACKTESTING — 60 HANDELSTAGE OUT-OF-SAMPLE")
print("=" * 70)

# Forecast auf Log-Return-Basis
lag_input  = train_ret.values[-best_lag:]
fc_obj     = var_fitted.forecast(y=lag_input, steps=TEST_SIZE)
fc_ret_df  = pd.DataFrame(fc_obj, index=test_ret.index,
                           columns=["Bitcoin_ret", "DAX_ret", "Gold_ret"])

# Preisprognose aus Log-Returns rekonstruieren
last_prices = df_raw.iloc[-(TEST_SIZE + 1)].values   # letzter bekannter Preis
fc_prices   = last_prices * np.exp(fc_ret_df.values.cumsum(axis=0))
fc_prices_df = pd.DataFrame(fc_prices, index=test_ret.index,
                              columns=["Bitcoin", "DAX", "Gold"])

test_prices = df_raw.iloc[-TEST_SIZE:]

# Metriken auf Preisniveau
metrics = {}
print(f"\n  {'Asset':<10}  {'MAE':>12}  {'RMSE':>12}  {'MAPE':>8}")
print("  " + "-" * 50)
for col in ["Bitcoin", "DAX", "Gold"]:
    actual    = test_prices[col].values
    predicted = fc_prices_df[col].values
    mae  = np.mean(np.abs(predicted - actual))
    rmse = np.sqrt(np.mean((predicted - actual) ** 2))
    mape = np.mean(np.abs((predicted - actual) / actual)) * 100
    metrics[col] = {"MAE": mae, "RMSE": rmse, "MAPE": mape}
    unit = "USD" if col != "DAX" else "USD"
    print(f"  {col:<10}  {mae:>12,.2f}  {rmse:>12,.2f}  {mape:>7.3f}%")

# Backtesting-Grafik
fig, axes = plt.subplots(3, 1, figsize=(14, 14))
fig.suptitle(f"VAR({best_lag}) — Backtesting (letzte 60 Handelstage)",
             fontsize=14, fontweight="bold")

for i, (col, label) in enumerate([("Bitcoin", "Bitcoin (USD)"),
                                   ("DAX",     "DAX-ETF (USD)"),
                                   ("Gold",    "Gold (USD/oz)")]):
    color     = COLORS[col]
    show_from = df_raw.index[-180]   # letzte 180 Tage anzeigen

    axes[i].plot(df_raw.loc[show_from:df_raw.index[-(TEST_SIZE + 1)]].index,
                 df_raw.loc[show_from:df_raw.index[-(TEST_SIZE + 1)], col],
                 color=color, linewidth=1.2, label="Trainingsdaten")
    axes[i].plot(test_prices.index, test_prices[col],
                 color="black", linewidth=1.3, label="Tatsächlicher Kurs")
    axes[i].plot(fc_prices_df.index, fc_prices_df[col],
                 color="steelblue", linewidth=1.5, linestyle="--",
                 label=f"VAR-Prognose (MAPE={metrics[col]['MAPE']:.2f}%)")
    axes[i].axvline(test_prices.index[0], color="red", linewidth=1,
                    linestyle=":", alpha=0.8, label="Prognosebeginn")
    axes[i].set_title(f"{col} — Backtest", fontweight="bold")
    axes[i].set_ylabel(label)
    axes[i].legend(fontsize=9, loc="upper left")
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
bt_path = os.path.join(REPORTS_DIR, "var_backtest.png")
plt.savefig(bt_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Grafik gespeichert: {bt_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# 11. FINALES MODELL + 10-TAGE-PROGNOSE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FINALES VAR-MODELL (Gesamtdaten) + 10-TAGE-PROGNOSE")
print("=" * 70)

var_final        = VAR(df_ret)
var_final_fitted = var_final.fit(maxlags=best_lag, ic=None, trend="c")

lag_input_final = df_ret.values[-best_lag:]
fc10_ret        = var_final_fitted.forecast(y=lag_input_final, steps=10)
fc10_ret_df     = pd.DataFrame(fc10_ret,
                                columns=["Bitcoin_ret", "DAX_ret", "Gold_ret"])

last_known      = df_raw.iloc[-1].values
fc10_prices     = last_known * np.exp(fc10_ret_df.values.cumsum(axis=0))
fc10_dates      = pd.bdate_range(df_raw.index[-1] + pd.Timedelta(days=1), periods=10)
fc10_df         = pd.DataFrame(fc10_prices, index=fc10_dates,
                                 columns=["Bitcoin", "DAX", "Gold"])

print("\n  10-Tage-Prognose (Preise):")
print(f"\n  {'Tag':<4} {'Datum':<12}  {'Bitcoin (USD)':>15}  {'DAX (USD)':>12}  {'Gold (USD/oz)':>15}")
print("  " + "=" * 65)
for i, (date, row) in enumerate(fc10_df.iterrows(), 1):
    print(f"  {i:<4} {date.strftime('%Y-%m-%d'):<12}  "
          f"{row['Bitcoin']:>15,.2f}  {row['DAX']:>12,.4f}  {row['Gold']:>15,.2f}")

print("\n  Letzter bekannter Preis:")
print(f"    Bitcoin : ${df_raw['Bitcoin'].iloc[-1]:,.2f}")
print(f"    DAX     : ${df_raw['DAX'].iloc[-1]:,.4f}")
print(f"    Gold    : ${df_raw['Gold'].iloc[-1]:,.2f}")

# Prognose-Grafik
fig, axes = plt.subplots(3, 1, figsize=(14, 14))
fig.suptitle(f"VAR({best_lag}) — 10-Tage-Prognose  (Finales Modell)",
             fontsize=14, fontweight="bold")

for i, (col, label) in enumerate([("Bitcoin", "Bitcoin (USD)"),
                                   ("DAX",     "DAX-ETF (USD)"),
                                   ("Gold",    "Gold (USD/oz)")]):
    color     = COLORS[col]
    show_from = df_raw.index[-120]

    axes[i].plot(df_raw.loc[show_from:].index,
                 df_raw.loc[show_from:, col],
                 color=color, linewidth=1.3, label="Historische Daten")
    axes[i].plot(fc10_df.index, fc10_df[col],
                 color="steelblue", linewidth=2.0,
                 marker="o", markersize=5, label="10-Tage-Prognose")
    axes[i].axvline(df_raw.index[-1], color="purple", linewidth=1,
                    linestyle=":", alpha=0.8, label="Prognosebeginn")
    axes[i].set_title(f"{col} — 10-Tage-Prognose", fontweight="bold")
    axes[i].set_ylabel(label)
    axes[i].legend(fontsize=9)
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
fc_path = os.path.join(REPORTS_DIR, "var_forecast_10d.png")
plt.savefig(fc_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Grafik gespeichert: {fc_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# 12. IMPULS-ANTWORT-FUNKTIONEN (IRF)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("IMPULS-ANTWORT-FUNKTIONEN (IRF)")
print("=" * 70)

irf = var_final_fitted.irf(periods=20)

fig = irf.plot(orth=True, impulse=None, response=None,
               figsize=(16, 12))
fig.suptitle(f"VAR({best_lag}) — Orthogonalisierte IRF (20 Perioden)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
irf_path = os.path.join(REPORTS_DIR, "var_irf.png")
fig.savefig(irf_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Grafik gespeichert: {irf_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# 13. FORECAST ERROR VARIANCE DECOMPOSITION (FEVD)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FORECAST ERROR VARIANCE DECOMPOSITION (FEVD)")
print("=" * 70)

fevd = var_final_fitted.fevd(periods=20)
print(f"\n{fevd.summary()}")

fig = fevd.plot(figsize=(14, 8))
plt.suptitle(f"VAR({best_lag}) — FEVD (20 Perioden)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
fevd_path = os.path.join(REPORTS_DIR, "var_fevd.png")
plt.savefig(fevd_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Grafik gespeichert: {fevd_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# 14. ERGEBNISZUSAMMENFASSUNG
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ERGEBNISZUSAMMENFASSUNG")
print("=" * 70)

print(f"""
  VAR-Modell     : VAR({best_lag})
  Trainingsdaten : {df_ret.index[0].date()} → {df_ret.index[-best_lag - TEST_SIZE].date()}  ({len(df_ret) - TEST_SIZE} Beobachtungen)
  Testdaten      : {test_ret.index[0].date()} → {test_ret.index[-1].date()}  ({TEST_SIZE} Beobachtungen)
  Variablen      : Bitcoin Log-Returns, DAX Log-Returns, Gold Log-Returns

  Backtesting-Metriken (60-Tage Out-of-Sample):
  ┌──────────────┬────────────────┬────────────────┬──────────┐
  │ Asset        │      MAE       │      RMSE      │   MAPE   │
  ├──────────────┼────────────────┼────────────────┼──────────┤
  │ Bitcoin      │ ${metrics['Bitcoin']['MAE']:>12,.2f} │ ${metrics['Bitcoin']['RMSE']:>12,.2f} │ {metrics['Bitcoin']['MAPE']:>6.3f}%  │
  │ DAX          │ ${metrics['DAX']['MAE']:>12,.4f} │ ${metrics['DAX']['RMSE']:>12,.4f} │ {metrics['DAX']['MAPE']:>6.3f}%  │
  │ Gold         │ ${metrics['Gold']['MAE']:>12,.2f} │ ${metrics['Gold']['RMSE']:>12,.2f} │ {metrics['Gold']['MAPE']:>6.3f}%  │
  └──────────────┴────────────────┴────────────────┴──────────┘

  Gespeicherte Grafiken:
    {eda_path}
    {corr_path}
    {resid_path}
    {bt_path}
    {fc_path}
    {irf_path}
    {fevd_path}
""")

print("=" * 70)
print("ANALYSE ABGESCHLOSSEN")
print("=" * 70)
