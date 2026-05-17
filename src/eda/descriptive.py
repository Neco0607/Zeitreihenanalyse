"""
Deskriptive Analyse fuer DAX, Gold und Bitcoin.

Liest die bereinigten CSVs aus data/processed/ und erzeugt fuer jedes
Asset folgende Outputs in reports/eda/<asset>/:

  - timeseries.png        Preisverlauf
  - returns.png           Log-Returns ueber die Zeit
  - distribution.png      Histogramm der Returns + Normal-Overlay
  - rolling.png           Rolling Mean & Std (252-Tage-Fenster)
  - acf_pacf.png          ACF & PACF der Returns
  - boxplot_yearly.png    Jaehrliche Boxplots der Returns

Zusaetzlich pro Lauf:
  - reports/eda/summary_stats.csv         Statistik-Tabelle aller Assets
  - reports/eda/stationarity_tests.csv    ADF & KPSS aller Assets

Aufruf vom Repo-Root:
    python -m src.eda.descriptive

Programmatisch:
    from src.eda.descriptive import run_eda_for_asset
    run_eda_for_asset("DAX")
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

# KPSS-Warnings ueber Lookup-Tabellen-Grenzen sind hier harmlos
warnings.filterwarnings("ignore", category=InterpolationWarning)

# ---------------------------------------------------------------------------
# Pfade & Konfiguration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
REPORTS_DIR = REPO_ROOT / "reports" / "eda"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

ASSETS = ["DAX", "Gold", "Bitcoin"]

# Einheitliche matplotlib-Konfiguration
plt.rcParams.update({
    "figure.figsize": (10, 5),
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
# Daten laden
# ---------------------------------------------------------------------------

def load_clean(asset: str) -> pd.Series:
    """Laedt eine bereinigte Reihe aus data/processed/."""
    path = PROCESSED_DIR / f"{asset.lower()}_clean.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Bereinigte Daten fuer {asset} fehlen: {path}\n"
            f"Erst ausfuehren: python -m src.data.clean_data"
        )
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return df.iloc[:, 0].astype(float)


def log_returns(prices: pd.Series) -> pd.Series:
    """Stetige (logarithmische) Renditen."""
    return np.log(prices / prices.shift(1)).dropna()


# ---------------------------------------------------------------------------
# Basisstatistiken
# ---------------------------------------------------------------------------

def basic_stats(prices: pd.Series, returns: pd.Series, asset: str) -> pd.Series:
    """Sammelt Basisstatistiken fuer Preise und Returns in einer Series."""
    return pd.Series({
        "Asset": asset,
        "N": len(prices),
        "Start": prices.index.min().date(),
        "Ende": prices.index.max().date(),
        "Preis_Min": prices.min(),
        "Preis_Max": prices.max(),
        "Preis_Mean": prices.mean(),
        "Preis_Median": prices.median(),
        "Preis_Std": prices.std(),
        "Return_Mean_pct": returns.mean() * 100,
        "Return_Std_pct": returns.std() * 100,
        "Return_Annual_Vol_pct": returns.std() * np.sqrt(252) * 100,
        "Return_Skew": returns.skew(),
        "Return_Kurtosis": returns.kurtosis(),  # Excess Kurtosis (Normal = 0)
        "Return_Min_pct": returns.min() * 100,
        "Return_Max_pct": returns.max() * 100,
    })


# ---------------------------------------------------------------------------
# Stationaritaetstests
# ---------------------------------------------------------------------------

def stationarity_tests(series: pd.Series, name: str) -> pd.Series:
    """
    ADF und KPSS auf einer Reihe.

    ADF:  H0 = Unit Root vorhanden (nicht-stationaer). p < 0.05 -> stationaer.
    KPSS: H0 = stationaer.                                p < 0.05 -> nicht-stationaer.

    Beide ergaenzen sich: idealerweise ADF p < 0.05 UND KPSS p > 0.05.
    """
    adf_stat, adf_p, *_ = adfuller(series.dropna(), autolag="AIC")
    kpss_stat, kpss_p, *_ = kpss(series.dropna(), regression="c", nlags="auto")

    return pd.Series({
        "Reihe": name,
        "ADF_Statistik": adf_stat,
        "ADF_p_Wert": adf_p,
        "ADF_stationaer_5%": adf_p < 0.05,
        "KPSS_Statistik": kpss_stat,
        "KPSS_p_Wert": kpss_p,
        "KPSS_stationaer_5%": kpss_p > 0.05,
    })


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_timeseries(prices: pd.Series, asset: str, outdir: Path) -> None:
    fig, ax = plt.subplots()
    ax.plot(prices.index, prices.values, color=ASSET_COLORS[asset], lw=0.8)
    ax.set_title(f"{asset} - Preisverlauf {prices.index.min().year}-{prices.index.max().year}")
    ax.set_xlabel("Datum")
    ax.set_ylabel("Preis")
    fig.tight_layout()
    fig.savefig(outdir / "timeseries.png")
    plt.close(fig)


def plot_returns(returns: pd.Series, asset: str, outdir: Path) -> None:
    fig, ax = plt.subplots()
    ax.plot(returns.index, returns.values * 100, color=ASSET_COLORS[asset], lw=0.5)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title(f"{asset} - Tägliche Log-Returns")
    ax.set_xlabel("Datum")
    ax.set_ylabel("Return [%]")
    fig.tight_layout()
    fig.savefig(outdir / "returns.png")
    plt.close(fig)


def plot_distribution(returns: pd.Series, asset: str, outdir: Path) -> None:
    """Histogramm + Normal-Overlay + QQ-Plot nebeneinander."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Links: Histogramm mit Normal-Overlay
    r_pct = returns.values * 100
    axes[0].hist(r_pct, bins=80, density=True, color=ASSET_COLORS[asset],
                 alpha=0.7, edgecolor="white")
    x = np.linspace(r_pct.min(), r_pct.max(), 200)
    axes[0].plot(x, stats.norm.pdf(x, r_pct.mean(), r_pct.std()),
                 "k--", lw=1.5, label="Normalverteilung")
    axes[0].set_title(f"{asset} - Verteilung der Returns")
    axes[0].set_xlabel("Return [%]")
    axes[0].set_ylabel("Dichte")
    axes[0].legend()

    # Rechts: QQ-Plot gegen Normal
    stats.probplot(returns.values, dist="norm", plot=axes[1])
    axes[1].set_title(f"{asset} - QQ-Plot vs. Normalverteilung")
    axes[1].get_lines()[0].set_markerfacecolor(ASSET_COLORS[asset])
    axes[1].get_lines()[0].set_markeredgecolor(ASSET_COLORS[asset])
    axes[1].get_lines()[0].set_markersize(3)

    fig.tight_layout()
    fig.savefig(outdir / "distribution.png")
    plt.close(fig)


def plot_rolling(prices: pd.Series, returns: pd.Series, asset: str, outdir: Path) -> None:
    """Rolling Mean (Preis) + Rolling Std (Returns) - Volatilitaets-Cluster."""
    window = 252  # ~1 Handelsjahr

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # Preis + Rolling Mean
    axes[0].plot(prices.index, prices.values,
                 color=ASSET_COLORS[asset], lw=0.6, alpha=0.5, label="Preis")
    axes[0].plot(prices.index, prices.rolling(window).mean(),
                 color="black", lw=1.2, label=f"Rolling Mean ({window}T)")
    axes[0].set_title(f"{asset} - Preis und {window}-Tage-Rolling-Mean")
    axes[0].set_ylabel("Preis")
    axes[0].legend(loc="upper left")

    # Rolling Std der Returns (annualisiert)
    rolling_vol = returns.rolling(window).std() * np.sqrt(252) * 100
    axes[1].plot(rolling_vol.index, rolling_vol.values,
                 color=ASSET_COLORS[asset], lw=1.0)
    axes[1].set_title(f"{asset} - Annualisierte Volatilität ({window}-Tage-Rolling-Std)")
    axes[1].set_xlabel("Datum")
    axes[1].set_ylabel("Vol [%]")

    fig.tight_layout()
    fig.savefig(outdir / "rolling.png")
    plt.close(fig)


def plot_acf_pacf(returns: pd.Series, asset: str, outdir: Path, lags: int = 40) -> None:
    """ACF und PACF der Returns - zentral fuer ARIMA-Modellordnung."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    plot_acf(returns, lags=lags, ax=axes[0], color=ASSET_COLORS[asset])
    axes[0].set_title(f"{asset} - ACF der Log-Returns")
    plot_pacf(returns, lags=lags, ax=axes[1], color=ASSET_COLORS[asset], method="ywm")
    axes[1].set_title(f"{asset} - PACF der Log-Returns")
    fig.tight_layout()
    fig.savefig(outdir / "acf_pacf.png")
    plt.close(fig)


def plot_boxplot_yearly(returns: pd.Series, asset: str, outdir: Path) -> None:
    """Jaehrliche Boxplots zeigen Verteilungsverschiebungen ueber die Zeit."""
    df = returns.to_frame(name="Return")
    df["Jahr"] = df.index.year

    fig, ax = plt.subplots()
    df.boxplot(column="Return", by="Jahr", ax=ax,
               showfliers=True, patch_artist=True,
               boxprops=dict(facecolor=ASSET_COLORS[asset], alpha=0.6))
    ax.set_title(f"{asset} - Jährliche Verteilung der Returns")
    ax.set_xlabel("Jahr")
    ax.set_ylabel("Log-Return")
    plt.suptitle("")  # Default-Suptitle von pandas weg
    fig.tight_layout()
    fig.savefig(outdir / "boxplot_yearly.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pipeline pro Asset
# ---------------------------------------------------------------------------

def run_eda_for_asset(asset: str) -> tuple[pd.Series, pd.DataFrame]:
    """
    Fuehrt die komplette deskriptive Analyse fuer ein Asset durch.

    Returns
    -------
    stats_row : Series mit Basisstatistiken
    tests_df  : DataFrame mit Stationaritaetstests (Levels & Returns)
    """
    print(f"\n=== {asset} ===")

    outdir = REPORTS_DIR / asset.lower()
    outdir.mkdir(parents=True, exist_ok=True)

    prices = load_clean(asset)
    returns = log_returns(prices)

    # Stats
    stats_row = basic_stats(prices, returns, asset)
    print(f"  N = {stats_row['N']:>5}  |  "
          f"Mean Return = {stats_row['Return_Mean_pct']:+.3f}%  |  "
          f"Annual Vol = {stats_row['Return_Annual_Vol_pct']:.2f}%  |  "
          f"Kurtosis = {stats_row['Return_Kurtosis']:.2f}")

    # Stationaritaetstests auf Levels und Returns
    test_levels = stationarity_tests(prices, f"{asset}_Levels")
    test_returns = stationarity_tests(returns, f"{asset}_Returns")
    tests_df = pd.DataFrame([test_levels, test_returns])
    print(f"  ADF auf Levels:  p = {test_levels['ADF_p_Wert']:.4f}  "
          f"-> {'stationaer' if test_levels['ADF_stationaer_5%'] else 'NICHT stationaer'}")
    print(f"  ADF auf Returns: p = {test_returns['ADF_p_Wert']:.4f}  "
          f"-> {'stationaer' if test_returns['ADF_stationaer_5%'] else 'NICHT stationaer'}")

    # Plots
    plot_timeseries(prices, asset, outdir)
    plot_returns(returns, asset, outdir)
    plot_distribution(returns, asset, outdir)
    plot_rolling(prices, returns, asset, outdir)
    plot_acf_pacf(returns, asset, outdir)
    plot_boxplot_yearly(returns, asset, outdir)
    print(f"  6 Plots geschrieben nach {outdir.relative_to(REPO_ROOT)}/")

    return stats_row, tests_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"EDA  |  Lese aus:     {PROCESSED_DIR}")
    print(f"     |  Schreibe nach: {REPORTS_DIR}")

    all_stats = []
    all_tests = []

    for asset in ASSETS:
        stats_row, tests_df = run_eda_for_asset(asset)
        all_stats.append(stats_row)
        all_tests.append(tests_df)

    # Zentrale Tabellen
    summary = pd.DataFrame(all_stats).set_index("Asset")
    summary_path = REPORTS_DIR / "summary_stats.csv"
    summary.to_csv(summary_path)

    tests = pd.concat(all_tests, ignore_index=True)
    tests_path = REPORTS_DIR / "stationarity_tests.csv"
    tests.to_csv(tests_path, index=False)

    print(f"\n{'-'*78}")
    print(f"Zusammenfassung -> {summary_path.relative_to(REPO_ROOT)}")
    print(f"Tests          -> {tests_path.relative_to(REPO_ROOT)}")
    print(f"\n{summary.round(3).T}")
    print(f"\nFertig.")


if __name__ == "__main__":
    main()
