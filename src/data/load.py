"""
yfinance-Loader fuer DAX, Gold und Bitcoin (2016-2026).

Schreibt eine bereinigte Close-Preis-Spalte je Asset nach data/raw/.

Aufruf vom Repo-Root:
    python -m src.data.load

Programmatisch:
    from src.data.load import load_all
    load_all()
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Pfade - automatisch vom Repo-Root aus aufgeloest
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Asset-Konfiguration
#
# Die Ticker entsprechen den im README dokumentierten Symbolen.
# Falls die alten CSV-Werte gewuenscht sind, hier umstellen auf:
#   DAX:  "EWG"   (iShares MSCI Germany ETF, USD)
#   Gold: "GC=F"  (Gold Futures, USD/oz)
# ---------------------------------------------------------------------------

ASSETS: dict[str, dict[str, str]] = {
    "DAX": {
        "ticker": "^GDAXI",
        "filename": "dax_kurs_2016_2026.csv",
        "column": "DAX_Kurs",
    },
    "Gold": {
        "ticker": "^XAU",
        "filename": "goldpreis_2016_2026.csv",
        "column": "Goldpreis",
    },
    "Bitcoin": {
        "ticker": "BTC-USD",
        "filename": "bitcoin_kurs_2016_2026.csv",
        "column": "Bitcoin_Kurs",
    },
}

START = "2016-01-01"
END = "2026-05-04"  # exclusive end-date in yfinance


# ---------------------------------------------------------------------------
# Funktionen
# ---------------------------------------------------------------------------

def fetch_one(ticker: str, start: str = START, end: str = END) -> pd.Series:
    """
    Laedt einen einzelnen Ticker und gibt eine Close-Series mit
    DatetimeIndex zurueck. Robust gegen den yfinance-MultiIndex.
    """
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if df is None or df.empty:
        raise RuntimeError(
            f"yfinance hat keine Daten fuer Ticker '{ticker}' geliefert. "
            f"Pruefe Internet-Verbindung oder Ticker-Symbol."
        )

    # Seit yfinance 0.2.x kommt ein MultiIndex auf den Spalten zurueck
    # (z.B. ('Close', '^GDAXI')). Wir flachen ihn auf die erste Ebene.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Close" not in df.columns:
        raise RuntimeError(
            f"yfinance-Antwort fuer '{ticker}' enthaelt keine 'Close'-Spalte. "
            f"Verfuegbar: {list(df.columns)}"
        )

    close = df["Close"].astype(float)
    close.index.name = "Datum"
    return close


def load_all(force: bool = True) -> dict[str, Path]:
    """
    Laedt alle drei Assets und schreibt sie nach data/raw/.

    Parameters
    ----------
    force : bool
        True (default) -> immer neu laden und ueberschreiben.
        False          -> ueberspringen falls Datei bereits existiert.

    Returns
    -------
    dict[str, Path]
        Mapping Asset-Name -> geschriebener Pfad.
    """
    written: dict[str, Path] = {}

    for asset, cfg in ASSETS.items():
        out_path = RAW_DIR / cfg["filename"]

        if out_path.exists() and not force:
            print(f"  [skip]  {asset:8s} bereits vorhanden -> {out_path.name}")
            written[asset] = out_path
            continue

        print(f"  [load]  {asset:8s} via {cfg['ticker']:10s} ...", end=" ", flush=True)
        s = fetch_one(cfg["ticker"])
        s.name = cfg["column"]
        s.to_csv(out_path)
        print(
            f"{len(s):4d} Zeilen | "
            f"{s.index.min().date()} bis {s.index.max().date()} "
            f"-> {out_path.name}"
        )
        written[asset] = out_path

    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"yfinance-Loader  |  Schreibe nach: {RAW_DIR}")
    print("-" * 78)
    load_all(force=True)
    print("-" * 78)
    print("Fertig.")
