# 📊 Zeitreihenanalyse THWS — Master-Projektplan

> **Kurs:** Vertiefung Business Analytics · **Prof:** Dr. Christian Menden
> **Gruppe:** 3 Personen · **Punkte:** 30 (5+5+15)
> **Story:** „Asset Forecasting im Spannungsfeld der Effizienzmarkthypothese — DAX, S&P 500 und Bitcoin"

## Warum diese Story den Prof beeindruckt

Aktien-Returns sind **klassisches Lehrbuch-Beispiel für die Effizienzmarkthypothese (EMH)**. Was das so wertvoll macht:

- **Du kannst ehrlich diskutieren:** „Unser ARIMA findet kaum Struktur — und genau das *ist* das Ergebnis. Empirische EMH-Bestätigung."
- **Volatilität ist prognostizierbar** (ARCH-Effekte) → eleganter Aufhänger für GARCH als Ausblick
- **Strukturbrüche** (COVID 2020, Ukraine 2022, Krypto-Winter 2022) → ehrliche Diskussion zu Modell-Limitationen
- **Multivariate Story:** S&P führt DAX (Lead-Lag), Bitcoin entkorreliert sich von Aktien (oder doch nicht?) → echte Granger-Kausalitäts-Spannung

## 👥 Rollenverteilung

| Rolle | Person | Asset | Verantwortung |
|---|---|---|---|
| **Analytics-Lead / Repo-Owner** | **Nico** | **DAX** | Repo-Maintenance, Pipeline-Code, VAR-Modul, Multivariat |
| **DevOps / Pipeline-Lead** | Person B | **S&P 500** | CI/Reproduzierbarkeit, Tests, Cross-Validation |
| **Communication-Lead** | Person C | **Bitcoin** | Visualisierung, Plotly-Dashboard, Präsentation |

> Jeder macht **seine eigene** Box-Jenkins-Analyse — die Rollen verteilen die Querschnittsaufgaben.

---

## 🗓 Termin 1 — Montag, 4.5.26 (HEUTE)

### Ziel der Präsentation 1 (5 Punkte, 10–15 Min)
„Wir haben ein professionelles Setup, klare Datenbasis, durchdachte Methodik."

### ⏰ Aufgaben pro Person (heute, ca. 3–4 Stunden)

#### Nico (Analytics-Lead, DAX)
1. **GitHub Repo `zeitreihenanalyse-thws` anlegen** (privat, B & C als Collaborator)
2. Lokal klonen, **Verzeichnisstruktur** committen (Starter-ZIP nutzen!)
3. **Initial-Files** committen: README, .gitignore, requirements.txt, CONTRIBUTING.md
4. **Branch `develop`** anlegen, `main` per Branch-Protection schützen (Settings → Branches)
5. DAX-Daten via `src/data/load.py` ziehen, in `data/raw/` cachen
6. Notebook `01_eda_nico.ipynb` mit erster EDA → ins Repo
7. Zusammen mit C die Slides Termin 1 zusammenstellen

#### Person B (DevOps-Lead, S&P 500)
1. S&P 500-Daten via gleichem Loader ziehen
2. Notebook `01_eda_<name>.ipynb` mit erster EDA
3. **`pre-commit-config.yaml`** lokal einrichten + testen (`pre-commit run --all-files`)
4. **Branch-Strategie + Commit-Convention** im CONTRIBUTING.md prüfen, ggf. ergänzen

#### Person C (Communication-Lead, Bitcoin)
1. Bitcoin-Daten via gleichem Loader ziehen
2. Notebook `01_eda_<name>.ipynb` mit erster EDA
3. **Slide-Deck Termin 1** in PowerPoint/Keynote bauen (Vorlage unten)
4. **README** ausformulieren (Story, Tech-Stack, Quickstart)

### 🛠 Lösung Termin 1

#### Datenladen (alle Personen, identische Funktion)

```python
# In jedem Notebook am Anfang
from src.data.load import load_asset, load_returns

# Person 1 (Nico)
prices = load_asset("DAX", start="2015-01-01", freq="ME")

# Person 2
prices = load_asset("SP500", start="2015-01-01", freq="ME")

# Person 3
prices = load_asset("Bitcoin", start="2015-01-01", freq="ME")

returns = load_returns(prices, log=True)
```

#### EDA-Template (`01_eda_<name>.ipynb`)

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set_style("whitegrid")

# Übersicht
print(prices.describe().round(2))
print(f"Periode: {prices.index.min().date()} – {prices.index.max().date()}")
print(f"n = {len(prices)} Monate")

fig, ax = plt.subplots(2, 2, figsize=(14, 8))
prices.plot(ax=ax[0,0], title=f"{prices.name} — Schlusskurs")
returns.plot(ax=ax[0,1], title="Log-Returns"); ax[0,1].axhline(0, color="grey")
returns.abs().plot(ax=ax[1,0], title="|Returns| (Vola-Cluster)")
sns.boxplot(x=returns.index.month, y=returns.values, ax=ax[1,1])
ax[1,1].set_title("Monatlicher Saisonalitäts-Check")
plt.tight_layout(); plt.savefig(f"reports/figures/01_eda_{prices.name}.png")
plt.show()
```

**Beobachtungen pro Asset (für Slide-Stichpunkte):**

| Asset | Wesentliche Beobachtungen |
|---|---|
| **DAX** | Trend nach oben mit klaren Crashs (COVID 03/20, Ukraine 02/22). Vola-Cluster sichtbar. Returns ~0% Mittel, ~5% σ. |
| **S&P 500** | Ähnlich DAX, aber gleichmäßigerer Aufwärtstrend. Geringere Vola. „Goldener Standard" der Aktienindizes. |
| **Bitcoin** | Extreme Strukturen: 2017-Boom, 2018-Crash, 2021-Boom, 2022-Crash, Recovery 2023+. Vola um Faktor 3-4 höher als Aktien. |

#### Slide-Deck Termin 1 (Vorlage, 7 Folien)

**Folie 1 — Titel**
- „Asset Forecasting im Spannungsfeld der Effizienzmarkthypothese"
- Untertitel: „Box-Jenkins, ARIMA und automatisierte Modellselektion für DAX, S&P 500 und Bitcoin"
- Team + Datum + Repo-Link (QR-Code!)

**Folie 2 — Motivation**
- 1 Bild: Zeitreihen-Plot aller drei Assets normalisiert (alle = 100 zu Beginn)
- 3 Bullets:
  - „Können wir Märkte schlagen? Was sagen klassische Zeitreihenmodelle?"
  - „Drei sehr verschiedene Asset-Klassen → kontrastierender Vergleich"
  - „Methodische Frage: Wann scheitern ARIMA-Modelle und warum?"

**Folie 3 — Datenbasis**
| Person | Asset | Ticker | Frequenz | Periode |
|---|---|---|---|---|
| Nico | DAX | ^GDAXI | monatlich | 2015–2026 |
| B | S&P 500 | ^GSPC | monatlich | 2015–2026 |
| C | Bitcoin | BTC-USD | monatlich | 2015–2026 |

**Folie 4 — Erste Beobachtungen**
- Je 1 Plot pro Asset (3 nebeneinander) — normalisierte Preise + Returns
- Kurze Statistik-Tabelle: μ, σ, Skew, Kurtosis pro Asset
- 1 Bullet: „Bitcoin σ ≈ 3× DAX σ"

**Folie 5 — Methodik**
- Linker Kasten „Univariat (Box-Jenkins)":
  ADF/KPSS → Differenzieren (Returns) → ACF/PACF → Modellselektion → Diagnose → Forecast
- Rechter Kasten „Multivariat":
  Auto-Pipeline mit ARIMA/SARIMA/ETS/Naive + Metriken + Bestmodell + VAR + Granger

**Folie 6 — Repo & Workflow**
- Screenshot der Repo-Struktur
- Branching-Diagramm: main ← develop ← feature/*
- 1 Bullet: „Conventional Commits, PRs mit Review, pre-commit + GitHub Actions CI"

**Folie 7 — Zeitplan**
- 4.5. ✅ Setup, EDA
- 11.5. → ARIMA pro Asset fertig
- 18.5. → Pipeline + VAR + Final
- „Nächste Schritte: Stationaritätstests, ARIMA-Grid, Diagnose"

---

## 🗓 Termin 2 — Montag, 11.5.26

### Ziel der Präsentation 2 (5 Punkte, 10–15 Min)
„Wir beherrschen Box-Jenkins handwerklich sauber für jede Reihe — und reflektieren ehrlich, was funktioniert."

### ⏰ Aufgaben pro Person (Mo–So, ~6h pro Person)

#### Jeder — Box-Jenkins für eigenes Asset
Notebook `02_box_jenkins_<name>.ipynb` mit allen 7 Schritten der Aufgabe:
1. Integrationsordnung (ADF + KPSS auf Preisen → I(1))
2. Transformation: Log-Returns als analysierte Reihe
3. ACF/PACF auf Returns + auf Returns² (ARCH-Diskussion!)
4. Modellgrid + auto_arima
5. Residuendiagnose (inkl. Engle-ARCH-Test)
6. t-Statistiken
7. 10-Schritt-Forecast (Returns + rück-integrierter Preispfad)

> **Vorlage:** `notebooks/02_box_jenkins_dax.py` ist bereits im Starter-ZIP. Person B und C kopieren das File, ändern `ASSET = "SP500"` bzw. `ASSET = "Bitcoin"` — fertig.

#### Querschnittsaufgaben

| Person | Zusatzaufgabe |
|---|---|
| **Nico** | `src/models/univariate.py` als wiederverwendbaren Wrapper anlegen, Code-Review für B & C |
| **Person B** | `src/data/load.py` finalisieren (Caching robust), `tests/test_pipeline.py` mit ersten Smoke-Tests, GitHub Action `.github/workflows/ci.yml` |
| **Person C** | Slides Termin 2 bauen, Repo-Doku verbessern (Docstrings checken) |

### 🛠 Lösung Termin 2

Komplettes lauffähiges Notebook ist im Starter-ZIP unter `notebooks/02_box_jenkins_dax.py`. Kernschritte hier verkürzt:

#### Schritt 1+2 — Stationarität & Transformation

```python
from statsmodels.tsa.stattools import adfuller, kpss
from pmdarima.arima.utils import ndiffs

# Auf Preisen: typischerweise nicht stationär
print(adfuller(prices)[1])      # > 0.05 → I(1)
print(adfuller(np.log(prices))[1])  # immer noch > 0.05
print(adfuller(returns)[1])     # << 0.05 → stationär ✅

# d=1 für log(prices), d=0 für returns
```

> **Profi-Move für Slides:** ADF und KPSS *gemeinsam* interpretieren. Tabelle zeigen mit ADF + KPSS pro Stufe (Niveau / Log / Diff). Bei Aktien typisch: Niveau nicht stationär, Returns stationär.

#### Schritt 3 — ACF / PACF

```python
fig, ax = plt.subplots(2, 2, figsize=(14, 8))
plot_acf(returns, lags=36, ax=ax[0,0])      # meist alles innerhalb KI
plot_pacf(returns, lags=36, ax=ax[0,1])
plot_acf(returns**2, lags=36, ax=ax[1,0])   # SIGNIFIKANT! Vola-Cluster
plot_pacf(returns**2, lags=36, ax=ax[1,1])
```

**Faustregeln (Slide-Tabelle!):**
| Reines Modell | ACF | PACF |
|---|---|---|
| AR(p) | exponentiell abklingend | bricht nach Lag p ab |
| MA(q) | bricht nach Lag q ab | exponentiell abklingend |
| ARMA(p,q) | beide klingen ab | beide klingen ab |
| **White Noise** | **alles im KI** | **alles im KI** |

> Wenn deine ACF/PACF nahezu White Noise zeigen — **das ist die Geschichte**: empirische EMH-Bestätigung.

#### Schritt 4 — Modellselektion

```python
from itertools import product
from statsmodels.tsa.arima.model import ARIMA

results = []
for p, q in product(range(0, 4), range(0, 4)):
    try:
        m = ARIMA(returns, order=(p, 0, q)).fit()
        results.append({"order":(p,0,q), "AIC":m.aic, "BIC":m.bic})
    except Exception: pass
grid = pd.DataFrame(results).sort_values("AIC").head()
```

#### Schritt 5 — Residuendiagnose mit ARCH-Test

```python
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

resid = best.resid
print(acorr_ljungbox(resid, lags=[10,20,30]))      # p>0.05 ist gut
print(scs.jarque_bera(resid))                      # meist verworfen → Fat Tails
print(het_arch(resid, nlags=12))                   # p<0.05 → ARCH-Effekte ⇒ GARCH-Ausblick!
```

#### Schritt 6 + 7 — t-Stats + Forecast

```python
print(pd.DataFrame({"coef":best.params, "t":best.tvalues, "p":best.pvalues}))

fc = best.get_forecast(steps=10)
mean_r, ci_r = fc.predicted_mean, fc.conf_int()

# Rück-Integration zu Preisen
last_price = prices.iloc[-1]
price_path = last_price * np.exp(mean_r.cumsum())
ci_lower   = last_price * np.exp(ci_r.iloc[:,0].cumsum())
ci_upper   = last_price * np.exp(ci_r.iloc[:,1].cumsum())
```

#### Slide-Deck Termin 2 (Vorlage, 8 Folien)

1. **Titel + Recap** — was beim letzten Mal stand
2. **Workflow** — Box-Jenkins als Flow-Diagramm
3. **Stationarität** — Tabelle ADF/KPSS für Niveau/Log/Returns aller 3 Assets
4. **ACF/PACF** — 3 Assets × 2 Plots = 6 Mini-Plots, je 1 Satz Interpretation
5. **„Plot-Twist"-Folie:** ACF auf Returns² zeigt klare Struktur → ARCH-Effekte → **Volatilität ist prognostizierbar, Returns nicht**
6. **Modellselektion** — Grid-Top-5 pro Asset, auto_arima-Wahl markiert
7. **Diagnose & Forecast** — Bestmodell-Diagnose + 10-Schritt-Forecast pro Asset
8. **Lessons Learned** — 3 Bullets:
   - „Returns sind nahezu White Noise → EMH bestätigt"
   - „ARCH-Effekte signifikant → GARCH wäre angemessen"
   - „Strukturbrüche (COVID, Ukraine) verletzen Konstanz-Annahme"

---

## 🗓 Termin 3 — Montag, 18.5.26 (FINAL, 30 Min, 15 Punkte)

### Ziel
„Wir liefern eine reproduzierbare End-to-End-Pipeline mit beeindruckender wissenschaftlicher Tiefe und einem reflektierten Schlusswort."

### ⏰ Aufgaben pro Person (Mo 12.5. bis So 17.5., ~10h pro Person)

#### Nico (Analytics-Lead)
1. **`src/models/pipeline.py`** finalisieren (steht schon im Starter-ZIP, nur testen + tunen)
2. **`src/evaluation/metrics.py`** finalisieren (steht schon im Starter-ZIP)
3. **`src/models/multivariate.py`** implementieren (VAR + Granger-Matrix)
4. **Diebold-Mariano-Test** in `src/evaluation/dm_test.py`
5. Notebook `03_multivariate_pipeline.ipynb` mit Endergebnissen

#### Person B (DevOps)
1. **GitHub Actions CI** finalisieren (`ci.yml` testet + lintet bei jedem PR)
2. **`run.py`** Master-Skript: ein Aufruf reproduziert alle Ergebnisse
3. **`tests/`** ausbauen (mind. 5 Tests)
4. **Cross-Validation** mit `TimeSeriesSplit` (steht schon im Starter)
5. requirements.txt einfrieren (Versionen pinnen)

#### Person C (Communication)
1. **Plotly-Dashboard** `dashboard.py` (Dropdown Asset → Forecast + Metriken-Tabelle)
2. **README** finalisieren mit Ergebnissen + Plots
3. **30-Min-Final-Präsentation** als pptx
4. **eLearning-Upload** koordinieren
5. **Sprechzettel** für 30-Min-Vortrag (10 Min pro Person)

### 🛠 Lösung Termin 3

#### `src/models/multivariate.py` — VAR + Granger
```python
"""Multivariates Vektor-Autoregressives Modell + Granger-Kausalität."""
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests

def fit_var(df: pd.DataFrame, maxlags: int = 12, ic: str = "aic"):
    """df: DataFrame mit alignten, schwach-stationären Reihen (Returns)."""
    model = VAR(df)
    selection = model.select_order(maxlags=maxlags)
    fitted = model.fit(maxlags=selection.aic, ic=ic)
    return fitted, selection

def granger_matrix(df: pd.DataFrame, maxlag: int = 4) -> pd.DataFrame:
    """p-Werte der paarweisen Granger-Kausalitätstests.

    Niedriges p in Zelle [cause, effect] → cause Granger-kausal für effect.
    """
    cols = df.columns
    out = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for cause in cols:
        for effect in cols:
            if cause == effect:
                out.loc[cause, effect] = float("nan")
            else:
                test = grangercausalitytests(
                    df[[effect, cause]], maxlag=maxlag, verbose=False
                )
                # Bestes p über alle Lags (oder Lag mit niedrigstem p)
                pvals = [test[lag][0]["ssr_ftest"][1]
                         for lag in range(1, maxlag+1)]
                out.loc[cause, effect] = min(pvals)
    return out
```

#### `src/evaluation/dm_test.py` — Diebold-Mariano-Test
```python
"""Diebold-Mariano-Test zum statistischen Modellvergleich.

H0: beide Modelle haben gleiche prognostische Genauigkeit.
"""
import numpy as np
from scipy.stats import t

def diebold_mariano(y, p1, p2, h: int = 1, loss: str = "MSE"):
    y, p1, p2 = map(np.asarray, (y, p1, p2))
    e1, e2 = y - p1, y - p2
    if loss == "MSE":  d = e1**2 - e2**2
    else:              d = np.abs(e1) - np.abs(e2)
    n = len(d)
    d_mean = d.mean()
    # HAC-konsistente Varianz (Newey-West-light)
    gamma = lambda k: ((d[k:] - d_mean) * (d[:n-k] - d_mean)).sum() / n
    var = gamma(0) + 2 * sum(gamma(k) for k in range(1, h))
    dm  = d_mean / np.sqrt(var / n)
    pval = 2 * (1 - t.cdf(abs(dm), df=n-1))
    interp = ("Modell 1 besser" if dm < 0 else "Modell 2 besser") \
             if pval < 0.05 else "kein signifikanter Unterschied"
    return {"DM": dm, "p_value": pval, "interpretation": interp, "n": n}
```

#### Plotly-Dashboard `dashboard.py`
```python
"""Interaktives Asset-Forecasting-Dashboard. Start: python dashboard.py"""
import dash, plotly.graph_objects as go
from dash import dcc, html, Input, Output
from src.data.load import load_all_assets, load_returns
from src.models.pipeline import (evaluate_holdout, evaluate_cv,
                                 select_best, MODEL_REGISTRY)

assets = load_all_assets()
returns_dict = {k: load_returns(v) for k, v in assets.items()}
results = evaluate_holdout(returns_dict, horizon=12, m=12)

app = dash.Dash(__name__); app.title = "Asset Forecasting THWS"

app.layout = html.Div([
    html.H1("📈 Asset Forecasting Dashboard"),
    html.Div([
        html.Label("Asset wählen:"),
        dcc.Dropdown(id="asset", value=list(assets)[0],
                     options=[{"label": k, "value": k} for k in assets])
    ], style={"width":"40%"}),
    dcc.Graph(id="price-plot"),
    dcc.Graph(id="forecast-plot"),
    html.H3("Modellvergleich (RMSE↓ ist besser)"),
    dcc.Graph(id="metric-bar"),
])

@app.callback(Output("price-plot","figure"),
              Output("forecast-plot","figure"),
              Output("metric-bar","figure"),
              Input("asset","value"))
def update(name):
    prices = assets[name]; r = returns_dict[name]

    # Preisplot
    fp = go.Figure()
    fp.add_scatter(x=prices.index, y=prices.values, name="Preis",
                   line={"color":"#264653","width":2})
    fp.update_layout(title=f"{name} — Schlusskurs",
                     template="plotly_white", height=350)

    # Forecast Returns
    train, test = r.iloc[:-12], r.iloc[-12:]
    ff = go.Figure()
    ff.add_scatter(x=r.index, y=r.values, name="Returns",
                   line={"color":"#333","width":1})
    for mname, factory in MODEL_REGISTRY.items():
        try:
            pred = factory(12).fit(train).forecast(12)
            ff.add_scatter(x=test.index, y=pred, mode="lines",
                           name=mname, line={"dash":"dot"})
        except Exception: pass
    ff.update_layout(title=f"Returns-Forecast — {name}",
                     template="plotly_white", height=400)

    # Metriken-Bar
    sub = results[results.series == name].sort_values("RMSE")
    fb = go.Figure(go.Bar(x=sub.model, y=sub.RMSE,
                          text=sub.RMSE.round(4), marker_color="#2A9D8F"))
    fb.update_layout(title="RMSE pro Modell",
                     template="plotly_white", height=350)
    return fp, ff, fb

if __name__ == "__main__":
    app.run(debug=True)
```

#### `run.py` (One-Click-Reproduktion)
```python
"""Reproduziert alle Ergebnisse: python run.py"""
import pandas as pd
from src.data.load import load_all_assets, load_returns
from src.models.pipeline import evaluate_holdout, evaluate_cv, select_best
from src.models.multivariate import fit_var, granger_matrix

if __name__ == "__main__":
    print("⏳ Lade Assets ...")
    assets = load_all_assets()
    returns_dict = {k: load_returns(v) for k, v in assets.items()}

    print("⏳ Holdout-Evaluation ...")
    res_h = evaluate_holdout(returns_dict, horizon=12, m=12)
    res_h.to_csv("reports/results_holdout.csv", index=False)

    print("⏳ Cross-Validation ...")
    res_cv = evaluate_cv(returns_dict, horizon=12, m=12, n_splits=4)
    res_cv.to_csv("reports/results_cv.csv", index=False)

    best = select_best(res_cv, "RMSE")
    print("\n🏆 Beste Modelle pro Asset:"); print(best.to_string(index=False))
    best.to_csv("reports/best_models.csv", index=False)

    print("\n⏳ VAR-Modell + Granger-Kausalität ...")
    df = pd.concat(returns_dict, axis=1).dropna()
    var, sel = fit_var(df)
    print(var.summary())
    granger = granger_matrix(df, maxlag=4)
    granger.to_csv("reports/granger.csv")
    print("\nGranger-p-Werte (Zeile=Cause, Spalte=Effect):"); print(granger.round(4))

    print("\n✅ Fertig. Alle Ergebnisse in reports/")
```

#### GitHub Actions CI (`.github/workflows/ci.yml`)
```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r requirements.txt
      - run: pip install ruff black pytest
      - run: ruff check src/ tests/
      - run: black --check src/ tests/
      - run: pytest -v
```

#### Final-Präsentation (Vorlage, 30 Min, ~16 Folien)

**Block A — Story (5 Min, Person C)**
1. Titel + Team (30 Sek)
2. Motivation: „Sind Märkte effizient?" (1 Schaubild aller 3 Assets)
3. Forschungsfragen (3 Bullets)
4. Datenüberblick (Tabelle 3 Assets)

**Block B — Univariat (10 Min, je 3 Min)**
5. Methodik Box-Jenkins (Workflow-Diagramm)
6. **Nico — DAX:** ADF/KPSS, Bestmodell, Forecast (1 Folie)
7. **Person B — S&P 500:** analog
8. **Person C — Bitcoin:** analog (mit Vergleich Vola-Magnitude!)
9. **Vergleich** der univariaten Modelle (RMSE-Tabelle + Bar-Chart)

**Block C — Multivariat (10 Min, Nico)**
10. Pipeline-Architektur (Code-Diagramm)
11. Metriken (warum MASE? Diskussion 30 Sek)
12. Cross-Validation-Ergebnisse (Boxplots der Folds pro Asset)
13. **Diebold-Mariano:** Welcher Vergleich ist statistisch signifikant?
14. **VAR + Granger-Heatmap:** „S&P → DAX signifikant, Bitcoin entkoppelt"

**Block D — Abschluss (5 Min, Person C)**
15. **Live-Demo Plotly-Dashboard** (60 Sek, Wow-Effekt!)
16. **Reflektierter Schluss:**
    - „Returns kaum prognostizierbar → empirische EMH-Bestätigung"
    - „Aber Volatilität ist prognostizierbar (ARCH-Effekte) → GARCH"
    - „Strukturbrüche limitieren klassische Modelle → Markov-Switching als Ausblick"
17. Repo-Highlights (CI-Badge, „make all", README-Screenshot)
18. Backup-Folien: detaillierte Tabellen, Residuen, Code

> **Wow-Trigger für 15/15 Punkte:**
> 1. Live-Demo Dashboard
> 2. „Returns sind White Noise — und das *ist* unser Ergebnis" als reflektierter Schluss
> 3. Granger-Heatmap mit ökonomischer Interpretation (US führt EU)
> 4. Diebold-Mariano statt nur „kleinerer RMSE"
> 5. CI-Badge + `python run.py` live ausführen

---

## ✅ Abgabe-Checkliste 18.5. EOB

- [ ] Repo `main` aktuell, `python run.py` läuft komplett durch
- [ ] README zeigt Story, Setup, Ergebnis-Tabelle, Plots, CI-Badge
- [ ] `requirements.txt` mit gepinnten Versionen
- [ ] Tests grün in CI
- [ ] `reports/figures/` enthält alle Plots der Präsentation
- [ ] Final-Folien als pptx im Repo unter `reports/`
- [ ] Repo-Link an Prof per Mail (christian.menden@thws.de)
- [ ] eLearning-Upload (Slides + Repo-Link)
