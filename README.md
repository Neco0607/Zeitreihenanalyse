# Zeitreihenanalyse – THWS Business Analytics

Gruppenarbeit im Rahmen der Veranstaltung **Vertiefung Business Analytics**  
bei Prof. Dr. Christian Menden – Technische Hochschule Würzburg-Schweinfurt

## Gruppe
| Person | Zeitreihe | Ticker |
|--------|-----------|--------|
| Antonio | Apple Aktie | AAPL |
| Person 2 | Microsoft Aktie | MSFT |
| Person 3 | Google Aktie | GOOGL |

## Projektstruktur
Zeitreihenanalyse/
├── README.md
├── .gitignore
├── requirements.txt
├── data/
│   ├── raw/          # Rohdaten (CSV je Ticker)
│   └── processed/    # Kombinierter Datensatz
├── notebooks/
│   ├── univariate_analysis.py    # Teil 2: ARIMA je Zeitreihe
│   └── multivariate_analysis.py  # Teil 3: Modellvergleich & Prognose
├── models/
├── results/
│   └── plots/        # Alle Grafiken
└── docs/

## Installation

```bash
git clone git@github.com:Neco0607/Zeitreihenanalyse.git
cd Zeitreihenanalyse
pip install -r requirements.txt
```

## Ausführung

**Teil 2 – Univariate Analyse:**
```bash
python notebooks/univariate_analysis.py
```

**Teil 3 – Multivariate Analyse:**
```bash
python notebooks/multivariate_analysis.py
```

## Methodik

### Teil 2: Univariate Zeitreihenanalyse (ARIMA)
- ADF-Test zur Bestimmung der Integrationsordnung
- Differenzierung zur Herstellung schwacher Stationarität
- ACF & PACF Analyse zur Modellidentifikation
- ARIMA Modellselektion via AIC Grid-Search
- Residuenanalyse (Ljung-Box Test)
- 10-Perioden-Prognose mit 95% Konfidenzintervall

### Teil 3: Multivariate Analyse
- Loop über alle 3 Zeitreihen × 5 ARIMA-Spezifikationen
- Evaluationsmetriken: MAE, RMSE, MAPE
- Modellvergleich als Heatmap
- Prognose mit bestem Modell für alle Zeitreihen

## Ergebnisse
Alle Plots werden in `results/plots/` gespeichert.  
Der Modellvergleich liegt in `results/model_comparison.csv`.
