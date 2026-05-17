"""
Bereinigt die rohen yfinance-CSVs fuer DAX, Gold und Bitcoin.
Jede Reihe wird einzeln verarbeitet und nach data/processed/ geschrieben.

Aufruf vom Repo-Root:
    python -m src.data.clean_data

Programmatisch:
    from src.data.clean_data import clean_all
    clean_all()

Falls die Rohdaten in data/raw/ fehlen, wird der yfinance-Loader
automatisch aufgerufen - das Skript funktioniert also direkt nach
einem frischen `git clone`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.load import ASSETS, RAW_DIR, load_all

# ---------------------------------------------------------------------------
# Pfade
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Bereinigung einer einzelnen Reihe
# ---------------------------------------------------------------------------

def clean_single(filepath: Path, asset_name: str) -> pd.Series:
    """
    Bereinigt eine einzelne yfinance-CSV.

    Akzeptiert beide Formate:
      a) Sauberes Format vom neuen Loader (nur Header + Daten)
      b) Altes Format mit Ticker-Muellzeile direkt unter dem Header
         (z.B. ',DAX' / ',BTC-USD' / ',GC=F')

    Bereinigungsschritte:
      1. CSV laden, Muellzeile (falls vorhanden) wegwerfen
      2. Datum als echten DatetimeIndex parsen
      3. Preisspalte zu float casten
      4. Nach Datum sortieren
      5. Duplikate auf dem Index entfernen
      6. NaN-Zeilen droppen
    """
    df = pd.read_csv(filepath)

    # --- Heuristik fuer altes Format mit Muellzeile ---
    # Wenn die erste Datenzeile in Spalte 0 kein Datum-String ist,
    # haengt darunter die Ticker-Zeile. -> nochmal mit skiprows=[1] laden.
    first_value = df.iloc[0, 0] if len(df) > 0 else None
    is_date_like = isinstance(first_value, str) and "-" in first_value
    if not is_date_like:
        df = pd.read_csv(filepath, skiprows=[1])

    # --- Spalten standardisieren ---
    # Spalte 0 = Datum, Spalte 1 = Preis (welcher Spaltenname auch immer drinsteht)
    if df.shape[1] < 2:
        raise ValueError(
            f"CSV {filepath.name} hat zu wenig Spalten: erwartet (Datum, Preis), "
            f"gefunden {list(df.columns)}"
        )
    df = df.iloc[:, :2]
    df.columns = ["Date", asset_name]

    # --- Datum parsen ---
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # --- Index, Sortierung, Duplikate ---
    df = df.set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # --- Preis numerisch ---
    df[asset_name] = pd.to_numeric(df[asset_name], errors="coerce")
    df = df.dropna(subset=[asset_name])

    return df[asset_name]


# ---------------------------------------------------------------------------
# Bereinigung aller Reihen (jeweils einzeln gespeichert)
# ---------------------------------------------------------------------------

def clean_all(auto_load: bool = True) -> dict[str, Path]:
    """
    Bereinigt alle drei Assets und speichert jede Reihe einzeln.

    Parameters
    ----------
    auto_load : bool
        True (default) -> falls Rohdaten in data/raw/ fehlen,
                          wird automatisch yfinance aufgerufen.
        False          -> bricht ab falls Daten fehlen.

    Returns
    -------
    dict[str, Path]
        Mapping Asset -> Pfad der bereinigten CSV.
    """
    # --- Pruefen ob alle Rohdaten vorhanden sind ---
    missing = [
        cfg["filename"]
        for cfg in ASSETS.values()
        if not (RAW_DIR / cfg["filename"]).exists()
    ]

    if missing:
        if auto_load:
            print("Rohdaten fehlen - lade automatisch von yfinance:")
            load_all(force=False)
            print()
        else:
            raise FileNotFoundError(
                f"Folgende Rohdaten fehlen in {RAW_DIR}:\n  - "
                + "\n  - ".join(missing)
                + "\nAusfuehren: python -m src.data.load"
            )

    # --- Jede Reihe bereinigen und einzeln speichern ---
    written: dict[str, Path] = {}

    for asset, cfg in ASSETS.items():
        in_path = RAW_DIR / cfg["filename"]
        out_path = PROCESSED_DIR / f"{asset.lower()}_clean.csv"

        s = clean_single(in_path, asset)
        s.to_csv(out_path)

        print(
            f"{asset:8s} | {len(s):5d} Zeilen | "
            f"{s.index.min().date()} bis {s.index.max().date()} | "
            f"NaN: {s.isna().sum():>3d} | "
            f"-> {out_path.relative_to(REPO_ROOT)}"
        )
        written[asset] = out_path

    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Cleaning  |  Lese aus:     {RAW_DIR}")
    print(f"          |  Schreibe nach: {PROCESSED_DIR}")
    print("-" * 78)
    clean_all(auto_load=True)
    print("-" * 78)
    print("Fertig.")
