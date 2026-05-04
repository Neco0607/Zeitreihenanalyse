"""Datenloader für DAX, S&P 500 und Bitcoin via yfinance.

Lädt monatliche Schlusskurse und cached lokal als CSV in data/raw/.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import yfinance as yf

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Ticker-Mapping: Anzeigename → yfinance-Symbol
TICKERS = {
    "DAX":      "^GDAXI",
    "SP500":    "^GSPC",
    "Bitcoin":  "BTC-USD",
}


def load_asset(
    name: str,
    start: str = "2015-01-01",
    end: str | None = None,
    freq: str = "ME",
    use_cache: bool = True,
) -> pd.Series:
    """Lädt einen Asset-Schlusskurs in gewünschter Frequenz.

    Args:
        name:   Schlüssel aus TICKERS (z.B. "DAX")
        start:  Startdatum (ISO)
        end:    Enddatum oder None (=heute)
        freq:   Pandas-Frequenz ("ME"=Monatsende, "W"=Woche, "D"=Tag)
        use_cache: Wenn True, lokal zwischenspeichern
    """
    if name not in TICKERS:
        raise ValueError(f"Unbekannter Asset-Name {name!r}. "
                         f"Verfügbar: {list(TICKERS)}")
    ticker = TICKERS[name]
    cache = RAW_DIR / f"{name.lower()}_{freq}.csv"

    if use_cache and cache.exists():
        s = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
        s.name = name
        return s

    df = yf.download(ticker, start=start, end=end, auto_adjust=True,
                     progress=False)
    if df.empty:
        raise RuntimeError(f"yfinance lieferte keine Daten für {ticker}")

    close = df["Close"].squeeze()                # Series
    s = close.resample(freq).last().dropna()
    s.name = name

    if use_cache:
        s.to_csv(cache)
    return s


def load_all_assets(
    start: str = "2015-01-01",
    end: str | None = None,
    freq: str = "ME",
) -> dict[str, pd.Series]:
    """Lädt alle drei Reihen in identischer Frequenz."""
    return {name: load_asset(name, start, end, freq) for name in TICKERS}


def load_returns(prices: pd.Series, log: bool = True) -> pd.Series:
    """Berechnet (Log-)Returns aus einer Preisreihe."""
    if log:
        r = (prices / prices.shift(1)).apply("log")
    else:
        r = prices.pct_change()
    return r.dropna().rename(f"{prices.name}_return")


if __name__ == "__main__":
    series = load_all_assets()
    for name, s in series.items():
        print(f"{name:8s} n={len(s):3d}  "
              f"{s.index.min().date()} → {s.index.max().date()}  "
              f"letzter Wert: {s.iloc[-1]:,.2f}")
