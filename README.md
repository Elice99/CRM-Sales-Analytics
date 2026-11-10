# CRM Sales Analytics — End-to-End Project

One project, two halves that share the same data:

| Part | What it does | Where |
|---|---|---|
| 📊 **Descriptive** | Interactive Power BI report on the CRM sales pipeline (pipeline health, agent & product performance, geography) — [read the analysis write-up](POWER_BI_README.md) | `CRM Recovered.pbip` → `CRM Recovered.Report/` + `CRM Recovered.SemanticModel/` |
| 🔮 **Predictive** | XGBoost model + FastAPI that predicts whether an active deal will be **WON** or **LOST** | `python_codes/` (see its own [README](python_codes/README.md)) |

Both are fed from the same source data in `datasets/`, cleaned with the SQL scripts in `SQL Scripts/` and loaded into a SQL Server database (`CRM_Sales_Opportunity`).

## Project structure

```
├── CRM Recovered.pbip            # Power BI project entry point (open this)
├── CRM Recovered.Report/         # Report definition (source-controlled PBIP format)
├── CRM Recovered.SemanticModel/  # Data model definition
├── POWER_BI_README.md            # Descriptive-analysis write-up: problem, SQL cleaning, DAX, dashboard design
├── datasets/                     # Raw CSVs: sales_pipeline, accounts, products, sales_teams…
├── SQL Scripts/                  # Data cleaning scripts (SQL Server)
├── Img/                          # Map shapefiles and flag images used by the report
├── bg/                           # Presentation assets
├── python_codes/                 # ML training, testing and FastAPI prediction service
│   ├── models.py                 #   Trains the model (needs the SQL Server DB)
│   ├── test.py                   #   Batch-scores active deals and exports a report
│   ├── main.py                   #   FastAPI service (single + batch predictions)
│   └── sales_pipeline_model.pkl  #   Trained model + feature artifacts
└── Problem statements.docx       # Business problem definition
```

## Quick start

**Descriptive part (Power BI Desktop):** open `CRM Recovered.pbip`. Refreshing the semantic model requires the SQL Server database loaded from `datasets/` via the scripts in `SQL Scripts/`.

**Predictive part (ML API):**

```bash
cd python_codes
pipenv install
pipenv run uvicorn main:app --host 0.0.0.0 --port 9696
# Swagger UI: http://localhost:9696/docs
```

To retrain the model from scratch (requires the `CRM_Sales_Opportunity` SQL Server database; set the server name with the `DB_SERVER` environment variable):

```bash
pipenv run python models.py   # writes sales_pipeline_model.pkl + active_deals_predictions.csv
pipenv run python test.py     # batch-scores active deals, exports results
```

## Model at a glance

- XGBoost classifier over 28 engineered features (temporal patterns, deal velocity and interaction features; hand-rolled win-rate features were tested and dropped)
- Honest test scores (2026-11-10 retrain after a data-leakage fix): **ROC-AUC 0.6085, accuracy 62.8%** - the earlier 0.6482 was inflated because test-set statistics were built from the test set's own targets

## Repo hygiene

Generated files (logs, prediction CSVs, `__pycache__`, `.vs/`, Office `~$` temp files) are ignored via `.gitignore` — don't commit them.
