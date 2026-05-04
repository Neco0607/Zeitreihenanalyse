# Zusammenarbeit — Team-Regeln

## Branching-Strategie (GitHub Flow, vereinfacht)

| Branch | Zweck |
|---|---|
| `main` | Stabiler Stand, immer lauffähig — geschützt, nur via PR |
| `develop` | Integrationsbranch für Features |
| `feature/<initialen>-<thema>` | Persönliche Arbeit, z.B. `feature/nh-arima-strom` |

**Regeln:**
- Direktes Pushen auf `main` ist verboten (Branch Protection aktiv)
- Jedes Feature → eigener Branch von `develop`
- Bei Fertigstellung Pull Request → mind. 1 Reviewer → Merge in `develop`
- `develop` → `main` nur bei stabilen Meilensteinen (vor Präsentationen)

## Commit-Konvention (Conventional Commits)

```
<type>(<scope>): <kurze Beschreibung>

[optional: Body mit längerer Erklärung]
```

**Erlaubte Types:**
- `feat`     — neues Feature
- `fix`      — Bugfix
- `docs`     — nur Dokumentation
- `refactor` — Code-Umbau ohne Verhaltensänderung
- `test`     — Tests hinzugefügt/geändert
- `chore`    — Build, Dependencies, Konfig
- `style`    — Formatierung (kein Logik-Change)

**Beispiele:**
```
feat(pipeline): SARIMA-Wrapper mit auto_arima-Backend
fix(loader): SMARD-CSV mit deutschem Zahlenformat parsen
docs(readme): Quickstart mit Conda-Variante ergänzt
test(metrics): MASE bei naive=0 abgesichert
```

## Pull Requests

- **Titel** im Conventional-Format
- **Beschreibung** mit:
  - Was wurde geändert?
  - Warum?
  - Wie getestet? (Screenshots/Outputs willkommen)
- **Mind. 1 Reviewer** aus dem Team
- **CI muss grün sein** vor Merge
- Reviewer-Kommentare adressieren, dann mergen

## Code-Style

- **PEP 8** mit max. 100 Zeichen pro Zeile
- **Docstrings** (Google-Style) für jede öffentliche Funktion
- **Type Hints** wo es Klarheit schafft
- `black` + `ruff` automatisch via pre-commit

### Setup pre-commit (einmalig pro Person)
```bash
pip install pre-commit
pre-commit install
```

## Notebook-Regeln

- Vor Commit: **„Restart Kernel & Run All"** — muss durchlaufen
- **Outputs DRIN lassen** — der Prof will Ergebnisse sehen
- Klare **Markdown-Sections** zwischen Code-Blöcken
- Keine privaten Pfade oder API-Keys committen

## Daten-Regeln

- **Keine Files >50 MB** ins Repo
- Rohdaten in `data/raw/` — gitignored, dafür Quelle in README dokumentieren
- Aufbereitete Daten in `data/processed/` (klein genug) gerne ins Repo
- Datenquellen-Links + Lizenz-Info in `data/SOURCES.md`

## Vor jeder Präsentation

- [ ] `develop` → `main` mergen
- [ ] `python run.py` läuft komplett durch
- [ ] CI ist grün
- [ ] Plots in `reports/figures/` aktuell
- [ ] README zeigt aktuelle Ergebnisse
