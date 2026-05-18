# Asset Forecasting — DAX, Gold & Bitcoin

> **Kursprojekt** „Vertiefung Business Analytics" · THWS · SoSe 2026  
> **Prof.** Dr. Christian Menden

[![CI](https://github.com/Neco0607/Zeitreihenanalyse/actions/workflows/ci.yml/badge.svg)](https://github.com/Neco0607/Zeitreihenanalyse/actions)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Projektbeschreibung

Dieses Projekt analysiert drei unterschiedliche Finanzzeitreihen — den deutschen Aktienindex **DAX**, den Rohstoff **Gold** und die Kryptowährung **Bitcoin** — mithilfe statistischer Zeitreihenmodelle.

Ziel ist es, mit der **Box-Jenkins-Methode (ARIMA)** geeignete Modelle zu finden, die die zugrundeliegenden datengenerierenden Prozesse abbilden und eine **10-Perioden-Prognose** mit Konfidenzintervallen erstellen.

Dabei werden die drei Assets sowohl **univariat** (jedes Asset einzeln) als auch **multivariat** (alle zusammen) analysiert und verschiedene Modelle anhand geeigneter Metriken (MAE, RMSE, MAPE, MASE) verglichen.

---

## Team & Zeitreihen

| Person | Asset | Ticker | Frequenz | Periode |
|---|---|---|---|---|
| **Nico Hirsch** | DAX | `^GDAXI` | täglich | 2016–2026 |
| **Antonio Sicaja** | Gold | `GC=F` | täglich | 2016–2026 |
| **David Grünwald** | Bitcoin | `BTC-USD` | täglich | 2016–2026 |

Alle Daten via [yfinance](https://github.com/ranaroussi/yfinance), automatisch gecacht in `data/raw/`.

---

## Hauptergebnisse

| Asset | Bestes Modell | MAE | RMSE | MAPE | MASE |
|---|---|---|---|---|---|
| **DAX** | **ARIMA(3,1,2)** | **2.10** | **2.55** | **4.872 %** | **8.896** |
| **Gold** | **ARIMA(0,1,2)** | **211.49** | **244.51** | **4.421 %** | **14.942** |
| **Bitcoin** | **ARIMA(3,1,2)** | **3,489.75** | **3,968.05** | **4.88 %** | **5.403** |

> Vollständige Tabellen: [`reports/results_cv.csv`](reports/results_cv.csv)  
> Detaillierte Gold-Analyse: [`docs/gold_analyse.md`](docs/gold_analyse.md)

---

## Quickstart

```bash
git clone git@github.com:Neco0607/Zeitreihenanalyse.git
cd Zeitreihenanalyse

python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Notebooks öffnen
jupyter lab
```

> **Hinweis:** Alle Notebooks liegen in `notebooks/`. Die Daten werden beim ersten Ausführen automatisch via yfinance heruntergeladen und in `data/raw/` gecacht.

---

## Projektstruktur

```
Zeitreihenanalyse/
├── data/
│   ├── interim/
│   ├── processed/
│   └── raw/                            # CSV-Rohdaten (gitignored)
├── notebooks/
│   ├── Bitcoin_Zeitreihe.ipynb         # Box-Jenkins Analyse Bitcoin (David Grünwald)
│   ├── DAX_Zeitreihe.ipynb             # Box-Jenkins Analyse DAX (Nico Hirsch)
│   ├── Gold_Zeitreihe.ipynb            # Box-Jenkins Analyse Gold (Antonio Sicaja)
│   └── VAR_Analyse.ipynb               # Multivariate Analyse — alle drei Assets (Nico Hirsch)
├── reports/
│   ├── results_cv.csv                  # Modellvergleich alle Assets
│   └── results_var.csv                 # VAR-Modell Ergebnisse
├── requirements.txt
├── .gitignore
├── CONTRIBUTING.md
└── README.md
```

---

## Methodik — Box-Jenkins (ARIMA)

Die Analyse folgt dem klassischen **Box-Jenkins-Verfahren** in vier Schritten:

1. **Identifikation** — Stationaritätsprüfung via ADF- & KPSS-Test, visuelle ACF/PACF-Analyse zur Bestimmung von p, d, q
2. **Schätzung** — Gittersuche über ARIMA(p, d, q) mit Modellwahl nach AIC
4. **Diagnostik** — Residualanalyse: Ljung-Box-Test, Jarque-Bera-Test, Q-Q-Plot
5. **Prognose** — 10-Perioden-Forecast mit 95%-Konfidenzintervall, Backtesting auf 60 Handelstagen

---

## Tech Stack

`Python 3.11` · `pandas` · `numpy` · `statsmodels` · `pmdarima` · `scikit-learn` · `scipy` · `matplotlib` · `seaborn` · `plotly` · `dash` · `yfinance` · `pytest` · `ruff` · `black`
