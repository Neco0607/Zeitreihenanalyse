# ✈ SkyLens – Global Aviation Intelligence

> A production-grade data analytics project built on the OpenFlights dataset.
> Three-layer Medallion Architecture · DuckDB · Interactive Storytelling Dashboard

---

## What is this?

SkyLens transforms raw aviation data (67K+ routes, 8K+ airports, 6K+ airlines)
into an interactive analytics experience. It's not a toy project — it's designed
as a real portfolio piece demonstrating data engineering, analytics engineering,
and product design skills.

**Live Stack:** Python · DuckDB · Plotly Dash · Haversine Geospatial Analysis

---

## Architecture

```
CSV Sources ──▶ Bronze (Raw) ──▶ Silver (Clean) ──▶ Gold (KPIs) ──▶ Dashboard
                  │                   │                   │
                  │  \\N → NULL       │  Haversine        │  Hub Scores
                  │  Type casting    │  Region mapping   │  Airline Rankings
                  │  Quality flags   │  Star Schema      │  Network Flows
                  │                   │  Distance class   │  Country Stats
```

**Why Medallion?** Each layer has a single responsibility. Bronze preserves
lineage. Silver enables analysis. Gold powers dashboards. Changes propagate
cleanly.

**Why DuckDB?** Zero infrastructure. Columnar OLAP. Embedded in process.
Sub-second queries on 67K routes without a server.

→ Detailed architecture: [`docs/architecture.md`](docs/architecture.md)

---

## Quick Start

```bash
# 1. Clone & install
git clone <repo-url> && cd openflights-analytics
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Place OpenFlights CSVs in data/raw/
#    (airlines.csv, airports_extended.csv, routes.csv, countries.csv, planes.csv)

# 3. Run pipeline (Bronze → Silver → Gold)
python -m pipelines.run_pipeline

# 4. Launch dashboard
python dashboard/app.py
# → Open http://localhost:8050
```

---

## Project Structure

```
openflights-analytics/
├── data/
│   ├── raw/                  # OpenFlights CSVs (not tracked in git)
│   └── processed/            # DuckDB database (generated)
├── pipelines/
│   ├── bronze.py             # Raw ingestion + quality flags
│   ├── silver.py             # Cleaning, Haversine, star schema
│   ├── gold.py               # KPI tables, aggregations
│   └── run_pipeline.py       # Orchestrator: B → S → G
├── dashboard/
│   ├── app.py                # Plotly Dash application
│   └── assets/
│       └── style.css         # Apple-inspired dark theme
├── docs/
│   ├── architecture.md       # Technical architecture
│   └── analysis.md           # Deep analysis & findings
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Example Queries

```sql
-- Top 10 airports by hub score
SELECT name, city, country, hub_score, total_connections
FROM gold.kpi_airport_hubs
ORDER BY hub_score DESC LIMIT 10;

-- Airlines with highest international route percentage
SELECT airline_name, route_count, intl_pct
FROM gold.kpi_top_airlines
WHERE route_count > 50
ORDER BY intl_pct DESC LIMIT 15;

-- Inter-region traffic flows
SELECT src_region, dst_region, route_count, avg_distance_km
FROM gold.network_region_flows
WHERE src_region != dst_region
ORDER BY route_count DESC LIMIT 20;

-- Distance distribution breakdown
SELECT distance_class, route_type, route_count, avg_km
FROM gold.agg_distance_distribution;

-- Africa vs Europe connectivity gap
SELECT region, total_routes, airlines, airports, countries
FROM gold.agg_region_stats
WHERE region IN ('Europe', 'Africa');
```

---

## Dashboard Storytelling Structure

The dashboard follows a **narrative arc**, not just a collection of charts:

### Chapter 1: Global Overview
→ KPI cards + globe route map. First impression: scale & density.

### Chapter 2: Key Insights
→ Distance classification donut, region bar chart, country treemap.
Question answered: *Where does aviation concentrate?*

### Chapter 3: Deep Dive
→ Airline rankings (filterable by region), hub scatter analysis.
Question answered: *Who dominates, and why?*

### Chapter 4: Hidden Patterns
→ Sankey inter-region flows + insight cards with auto-generated statistics.
Question answered: *What's the structure behind the numbers?*

---

## Design Philosophy

**Apple-inspired:** Dark mode, Inter typography, generous whitespace,
subtle borders, gradient accents, glassmorphism header.

**Principles:**
- Every chart earns its place (no decoration)
- Insight cards with auto-generated text (data-driven storytelling)
- Hover reveals detail, layout reveals structure
- Color encodes meaning: blue = volume, orange = ratio, green = growth

---

## Key Findings

| Finding | Detail |
|---------|--------|
| Hub-and-Spoke dominance | Top 10 airports hold disproportionate network share |
| Power Law distribution | Airport connections follow scale-free network pattern |
| Aviation Divide | Europe has 10–15× more routes than Africa despite smaller population |
| Regional game | >50% of all routes are short-haul (<1,500 km) |
| Middle East as connector | Gulf hubs bridge Asia–Europe–Africa despite small population base |
| Airline oligopolies | A handful of carriers dominate each regional market |

→ Full analysis: [`docs/analysis.md`](docs/analysis.md)

---

## Assets

The dashboard uses CSS-only styling (no external images required). If you want
to add a custom logo or screenshots:

| Asset | Size | Format | Location |
|-------|------|--------|----------|
| Logo | 64×64 px | PNG/SVG | `dashboard/assets/logo.svg` |
| Screenshot (Hero) | 1200×675 px | PNG | `docs/screenshot-hero.png` |
| Screenshot (Detail) | 1200×675 px | PNG | `docs/screenshot-detail.png` |
| Social Preview | 1280×640 px | PNG | `docs/social-preview.png` |

---

## Tech Stack & Versions

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | 3.10+ | Runtime |
| DuckDB | ≥ 0.10 | OLAP engine |
| Plotly | ≥ 5.22 | Chart library |
| Dash | ≥ 2.17 | Web framework |
| Pandas | ≥ 2.2 | Data transport |

---

## Learnings

1. **DuckDB is powerful for embedded analytics.** No server, no config,
   columnar performance. The entire pipeline runs in <7 seconds.
2. **Medallion Architecture pays off.** Even at this scale, the separation
   of concerns (raw → clean → aggregated) made debugging and iteration
   dramatically faster.
3. **Storytelling > Features.** A dashboard with a narrative arc is more
   impactful than one with more charts but no structure.
4. **Haversine works.** For route-level distance estimation, the spherical
   approximation is accurate within ~0.3% vs. Vincenty — more than
   sufficient for classification.
5. **Data quality matters.** Bronze quality flags caught ~400 invalid
   routes (missing airport IDs) and 1 invalid airline early.

---

*Built by Wlad · THWS Business Analytics · 2025*
