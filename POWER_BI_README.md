# Power BI Descriptive Analysis — Sales Pipeline Intelligence

This folder pair (`CRM Recovered.pbip` + `SQL Scripts/`) is the **descriptive half** of the CRM Sales Analytics project: an interactive Power BI report that shows *what happened* across the sales pipeline — revenue, win rates, funnel health, agent and product performance — for the leadership team at **SPOTA**, a mid-sized B2B technology & manufacturing company.

> The **predictive half** (XGBoost + FastAPI deal-outcome scoring) lives in [`python_codes/`](python_codes/README.md). Both halves are fed from the same SQL Server database, `CRM_Sales_Opportunity`.

---

## 1. The business problem

From `Problem statements.docx`. SPOTA sells across retail, technology, medical and industrial sectors in multiple global regions. Despite overall revenue growth, management **can't reliably see how the business is actually performing**:

| Challenge | What it means |
|---|---|
| **Fragmented performance visibility** | Sales data is scattered across regions, product lines and agents — no single view of performance. |
| **Sales funnel blind spots** | No data-driven view of *where* deals stall, *why* they are lost, and which stages lose the most. |
| **Untapped potential** | Top products, sectors and agents aren't consistently identified or rewarded → poor resource allocation. |
| **Reactive decision-making** | No forecast or risk view of the active pipeline → decisions are made after the fact, not before. |

**The need:** one integrated analytics solution that identifies the most profitable products/sectors/regions, benchmarks sales-team performance, and supports data-driven resourcing decisions.

### The solution (in one sentence)

A cleaned SQL Server data model, a Power BI semantic model with ~45 DAX measures, and a three-section interactive dashboard that answers the descriptive business questions — while the predictive half of this project extends it with deal-outcome risk scoring.

---

## 2. What was done, and why

### Step 1 — Source data (`datasets/`)

Seven raw CSVs: `sales_pipeline.csv` (623 KB, the fact table), `accounts.csv`, `products.csv`, `sales_teams.csv`, `Agent Photos.csv`, `Flag url.csv` (country-flag image URLs for the map) and `data_dictionary.csv`.

### Step 2 — SQL cleaning (`SQL Scripts/`)

Raw exports are never analysis-ready. Two scripts load and clean the data in SQL Server (database `CRM_Sales_Opportunity`), and every rule exists because of a specific problem:

**`Sales_pipeline cleaning.sql`** (the fact table):

| Action | Why |
|---|---|
| New `is_active` bit column, `= 1` for Engaging/Lead/Prospecting | Separates *open* pipeline from *closed* outcomes. The dashboard counts active deals without mixing them into won/lost stats — and the ML half uses the same flag to pick the deals to score. |
| `close_value IS NULL → 0` for non-Won deals | A NULL here means "no revenue realised yet", not "unknown" — leaving it NULL would break revenue sums. |
| `close_date IS NULL → GETDATE()` for closed deals | Won/Lost deals with no close date would vanish from every month-over-month time comparison. |
| `account IS NULL → 'Unknown'` | `account` is a join key; NULLs would silently drop those rows from every relationship in the model. |
| Consistent `Deal_stage` casing | DAX string comparisons and slicer labels are case-sensitive; `won`/`Won`/`WON` would fragment every measure. |
| Deduplicate `opportunity_id`, keeping the earliest `engage_date` | Prevents double-counted opportunities in every count-based measure. |
| Audit query by `deal_stage` | Sanity check on remaining NULLs after cleaning. |

**`account cleaning.sql`** (the account dimension):

| Action | Why |
|---|---|
| Consistent `sector` casing | Same reasoning as deal_stage — clean slicers and groupings. |
| Drop `subsidiary_of` | ~80% NULL; an almost-empty column adds noise and model size for no analytic value. |
| New `account_tier` (Micro / Small / Medium / Large_Enterprise by employee count) | Directly answers the business question *"Does client firmographic data affect deal success?"* — it gives the dashboard a size-based slice of the customer base. |

### Step 3 — Semantic model (`CRM Recovered.SemanticModel/`)

Power Query imports the four tables straight from SQL Server (`CRM_Sales_Opportunity{[Schema="dbo", Item=…]}`). A classic star-ish schema:

```
                    ┌──────────────┐
                    │  DateTable   │  (calculated: CALENDAR over close_date range)
                    └──────┬───────┘
                           │ 1:*
┌───────────┐   1:*   ┌────┴───────────┐   *:1   ┌──────────────┐
│ accounts  │─────────┤ sales_pipeline ├─────────┤   products   │
└───────────┘         │   (fact)       │         └──────────────┘
                      └────┬───────────┘
                     *:1   │
                    ┌──────┴───────┐
                    │ sales_teams  │
                    └──────────────┘
```

- **`DateTable`** — a calculated `CALENDAR(MIN(close_date), MAX(close_date))` table marks the model as a proper date dimension so `DATEADD`/`PREVIOUSMONTH` time-intelligence works.
- **`Flag url`** — country-flag image URLs joined **both directions** to `accounts` so the map shows each location's flag.
- Measure tables (`deals measures`, `revenue measures`, `Measures (2)`, `Agent Rankings`) hold the DAX, keeping measures off the fact table.
- `Filters` / `Slicer Table` — helper tables that power the dynamic filter pane (see below).

### Step 4 — DAX measures (~45, in 4 dedicated tables)

**Core KPIs** (`Measures (2)`, `deals measures`, `revenue measures`):

| Measure | Definition |
|---|---|
| `opp_count` | Opportunities excluding Prospecting (the top of funnel is tracked separately as `prospecting_count`) |
| `deal_won` / `deal_lost` / `Engaging_count` / `Active_deal` | Pipeline counts by stage |
| `success_rate` | `deal_won / (deal_won + deal_lost)` — win rate over *closed* deals |
| `conversion_rates` | `deal_won / opp_count` — win rate over *all* real opportunities |
| `total_revenue_closed` | `SUM(close_value)` |
| `avg_deal_size` / `avg_won_deal_size` | Revenue per opportunity / per won deal |
| `avg_deal_time` | Days from `engage_date` to `close_date` |
| `Avg_opp_per_month`, `Avg_rev_per_month` | Monthly-normalised averages via `AVERAGEX(VALUES(DateTable[month_num]), …)` |

**Month-over-month intelligence** — the dashboard's signature pattern. For every KPI there is a trio of measures: a `vs Last Month` percentage with ▲/▼ arrows (`UNICHAR(11165/11167)` + `DATEADD(DateTable[Date], -1, MONTH)`), a `CF Last Month` measure returning Green/Red for conditional formatting, and a `TABLE_CF` variant returning translucent hex colours for table-row shading:

```dax
'vs Last Month %' =
VAR CM = [total_revenue_closed]
VAR PM_sales = CALCULATE([total_revenue_closed], DATEADD(DateTable[Date], -1, MONTH))
VAR perc = DIVIDE(CM - PM_sales, PM_sales)
RETURN SWITCH(TRUE(),
    perc > 0, UNICHAR(11165) & " " & FORMAT(perc, "0.0%"),
    perc < 0, UNICHAR(11167) & " " & FORMAT(perc * -1, "0.0%"),
    FORMAT(perc, "0.0%"))
```

**Agent rankings** (`Agent Rankings`):

| Measure | What it does |
|---|---|
| `Current_Rank_won` / `Current_Rank_success_rate` | `RANKX(ALL(sales_pipeline[sales_agent]), …, DESC, Dense)` — ranks all agents on one axis |
| `Previous_Rank` + `Rank_Chng` | Rank last month via `PREVIOUSMONTH`, then ▲/▼ movement — the leaderboard shows *momentum*, not just position |
| `Qrt Agent_by_conv_rate` / `…_by_succ_rate` / `…_by_rev_rate` | Quarterly awards: builds a `CALCULATETABLE` of agents for the current quarter, ranks them, and `CONCATENATEX`-returns the top performer per metric |

**Experience touches:** `WelcomeMsg` greets the signed-in manager by time of day (`Good Morning/Afternoon/Evening, <manager>…`), and `Max_rev`/`Min_rev` flag the best/worst revenue months for the trend chart.

### Step 5 — Report design (`CRM Recovered.Report/`)

Three pages, ~150 visuals total:

| Page | Contents |
|---|---|
| **Dashboard** (main) | ~136 visuals: 19+ KPI cards with MoM deltas, revenue trend (area/line charts), opportunities by sector & product (bar charts), **win rate by location on a shape map** (uses `Img/` country shapefiles + flag images), agent leaderboard table (`Agent perf_Table`: total/won/lost/active deals, deal size, rank movement, conditional-format shaded rows), pipeline detail table, donut + 100%-stacked breakdowns |
| **Info** | In-report documentation: what each dashboard section answers, and the design principles |
| **Tooltip** | Custom hover tooltip for the charts |

Design decisions (documented on the Info page itself):

- **Every visual answers one specific question** — no decoration charts.
- **Colour is semantic**: green = success, red = loss, blue = neutral/active — applied *by measure* (`CF Last Month` variants), not hand-painted.
- **40 action buttons + bookmarks** drive a menu-style navigation (Home, Table views, Filter pane toggle, Info, GitHub/LinkedIn links) instead of page tabs.
- **A slide-out filter pane** with a DAX-driven `Filter count` badge, an advanced slicer, and slicers for manager, location, sector, account tier, product series and month.
- **Clean, hierarchical layout** — KPIs top, drivers middle, detail tables bottom.

---

## 3. Business questions this dashboard answers

Mapped to the project mandate (`Problem statements.docx`):

| Mandate question | Where it's answered |
|---|---|
| How do the Retail sector and US region drive revenue vs. other segments? | Sector bar chart + location map with revenue and flags |
| Which agents have the highest win rates, and can their success be documented for replication? | `Agent perf_Table` + ranking measures (`Current_Rank_success_rate`, `Rank_Chng`, quarterly top-agent cards) |
| How have Win Rate, Opportunities and avg Deal Size trended, and what does the shift in deal value mean? | KPI cards' `vs Last Month` trio + revenue trend chart + `Max_rev`/`Min_rev` |
| What is the overall win rate and opportunity-volume growth? | `success_rate` / `conversion_rates` cards with MoM arrows |
| Average deal duration by product/sector — does time-active correlate with success? | `avg_deal_time` + product/sector breakdowns |
| Which sectors or account tiers bring the highest average deal value? | `account_tier` and sector slicers + `avg_won_deal_size` |
| *(Predictive)* Can active deals get a risk score for proactive intervention? | Handled by the other half of this project → [`python_codes/`](python_codes/README.md) |

---

## 4. Opening the report

1. Install **Power BI Desktop** (PBIP projects require a recent version).
2. Open **`CRM Recovered.pbip`** — it loads the report and semantic model as separate source-controlled items.
3. Refreshing visuals requires the `CRM_Sales_Opportunity` SQL Server database (local instance `ELICE99\SQLEXPRESS` by default). Recreate it by running the scripts in `SQL Scripts/` against the CSVs in `datasets/` if you don't have it.
4. The PBIP format means every change (DAX, layout, model) is plain text — this whole report is reviewable in git.

---

*End-to-end overview, including the ML predictive half: [root README](README.md). Predictive API details: [`python_codes/README.md`](python_codes/README.md).*
