# Asset Forecasting — DAX, S&P 500 & Bitcoin

> **Kursprojekt** „Vertiefung Business Analytics" · THWS · SoSe 2026
> **Prof.** Dr. Christian Menden

[![CI](https://github.com/<USER>/zeitreihenanalyse-thws/actions/workflows/ci.yml/badge.svg)](https://github.com/<USER>/zeitreihenanalyse-thws/actions)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Forschungsfragen

1. **Lassen sich monatliche Asset-Returns mit ARIMA-Modellen prognostizieren** — oder bestätigt sich die schwache Form der Effizienzmarkthypothese empirisch?
2. **Welches Modell schneidet bei welchem Asset am besten ab** (DAX, S&P 500, Bitcoin)?
3. **Bestehen Lead-Lag-Beziehungen** zwischen den Märkten (Granger-Kausalität, VAR)?
4. Wie gut funktionieren klassische Modelle bei **Strukturbrüchen** (COVID-Crash, Ukraine-Krieg, Krypto-Winter)?

## Team & Zeitreihen

| Person | Asset | Ticker | Frequenz | Periode |
|---|---|---|---|---|
| **Nico Hirsch** | DAX | `^GDAXI` | monatlich | 2015–2026 |
| **David Grünwald** | S&P 500 | `^GSPC` | monatlich | 2015–2026 |
| **Antonio Sicaja** | Bitcoin | `BTC-USD` | monatlich | 2015–2026 |

Alle Daten via [yfinance](https://github.com/ranaroussi/yfinance), open-end von Yahoo Finance.

## Hauptergebnisse

| Asset | Bestes Modell | RMSE (Return) | MAPE | MASE |
|---|---|---|---|---|
| DAX     | _ARIMA(p,0,q)_ | _xxx_ | _xx %_ | _x.xx_ |
| S&P 500 | _xxx_         | _xxx_ | _xx %_ | _x.xx_ |
| Bitcoin | _xxx_         | _xxx_ | _xx %_ | _x.xx_ |

Vollständige Tabellen: [`reports/results_cv.csv`](reports/results_cv.csv)

## Schlüsselerkenntnisse (Vorschau)

- **Returns sind annähernd White Noise** → ARIMA findet kaum Struktur. Das ist *die empirische Bestätigung* der schwachen EMH, nicht ein Modellversagen.
- **Volatilität ist hingegen klar geclustert** (Engle-ARCH-Test signifikant) → GARCH-Erweiterung ist angemessen.
- **Granger-Kausalität:** S&P 500 führt den DAX (US-Markt → europäischer Markt um einen Monat).
- **Bitcoin korreliert seit 2020 stärker mit Aktien** → entzaubert das „Diversifier"-Narrativ.

## Quickstart

```bash
git clone git@github.com:<USER>/zeitreihenanalyse-thws.git
cd zeitreihenanalyse-thws

python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Daten laden + cachen
python -m src.data.load

# Alle Analysen + Forecasts reproduzieren
python run.py

# Interaktives Dashboard starten
python dashboard.py                 # → http://localhost:8050

# Notebooks öffnen
jupyter lab
```

## Projektstruktur

```
zeitreihenanalyse-thws/
├── data/
│   └── raw/                       # CSV-Cache von yfinance (gitignored)
├── notebooks/
│   ├── 01_eda_<name>.ipynb        # Explorative Analyse pro Asset
│   ├── 02_box_jenkins_<name>.ipynb # Univariate ARIMA pro Asset
│   └── 03_multivariate_pipeline.ipynb # Pipeline + VAR
├── src/
│   ├── data/load.py               # yfinance-Loader (cached)
│   ├── features/transform.py      # Returns, Box-Cox
│   ├── models/
│   │   ├── pipeline.py            # Auto-Selection
│   │   └── multivariate.py        # VAR + Granger
│   ├── evaluation/
│   │   ├── metrics.py             # MAE, RMSE, MAPE, sMAPE, MASE
│   │   └── dm_test.py             # Diebold-Mariano-Test
│   └── visualization/plots.py
├── reports/
│   ├── figures/                   # Alle Plots
│   ├── results_holdout.csv
│   ├── results_cv.csv
│   ├── best_models.csv
│   └── granger.csv
├── tests/                         # pytest
├── dashboard.py                   # Plotly-Dash App
├── run.py                         # End-to-End Reproduktion
├── requirements.txt
├── .gitignore
├── .pre-commit-config.yaml
├── CONTRIBUTING.md
└── README.md
```

## Methodik

**Univariat (Box-Jenkins) — pro Asset:**
1. Stationaritätsprüfung mit ADF + KPSS (auf Preisen → I(1) → Returns betrachten)
2. ACF/PACF auf Log-Returns + auf quadrierten Returns (ARCH-Effekte sichtbar machen)
3. Modellselektion: manuelles Grid + `auto_arima` als Cross-Check
4. Residuendiagnose: Ljung-Box, Q-Q, Jarque-Bera, Engle-ARCH-Test
5. 10-Schritt-Forecast Returns + rück-integrierter Preispfad mit 95%-KI

**Multivariat / Pipeline:**
- Modellfamilien: ARIMA, SARIMA, ETS, Naive, SeasonalNaive — auf Returns
- Evaluation: Holdout + Rolling-Origin Cross-Validation
- Metriken: MAE, RMSE, MAPE, sMAPE, **MASE** (skaleninvariant)
- Modellvergleich statistisch via **Diebold-Mariano-Test**
- **VAR-Modell** über alle drei Returns + **Granger-Kausalitätsmatrix**

## Tech Stack

Python 3.11 · pandas · numpy · statsmodels · pmdarima · scikit-learn · scipy · matplotlib · seaborn · plotly · dash · yfinance · pytest · ruff · black

## 📜 Lizenz

MIT — siehe [LICENSE](LICENSE)

