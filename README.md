# 🔄 Sales Analytics Solution — Python ETL & Power BI

## 📋 Project Overview

An end-to-end data analytics solution featuring a Python ETL pipeline built on the **Medallion Architecture** (Bronze → Silver → Gold), star schema dimensional modeling, and a Power BI dashboard for interactive sales analysis and forecasting.

**Tech Stack:** Python (pandas), JSON, CSV, Power BI (DAX), Star Schema

---

## 🎯 Objective

Build a production-grade data pipeline that:
1. **Extracts** raw sales and forecast data from JSON sources
2. **Transforms** and cleanses the data through a staged pipeline
3. **Models** it into a star schema (dimensional model)
4. **Visualizes** insights via an interactive Power BI dashboard

---

## 🏗️ Data Architecture — Medallion Pattern

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   🥉 BRONZE      │────▶│   🥈 SILVER      │────▶│   🥇 GOLD        │
│   Raw Data       │     │   Cleaned Data   │     │   Star Schema   │
│   (JSON)         │     │   (CSV)          │     │   (CSV)         │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
                                                  ┌──────────────┐
                                                  │  Power BI    │
                                                  │  Dashboard   │
                                                  └──────────────┘
```

### Stage Details

| Stage | Input | Transformations | Output |
|-------|-------|-----------------|--------|
| **Bronze** | `Sales.json`, `forecast.json` | Raw load with validation | In-memory DataFrames |
| **Silver** | Bronze DataFrames | Date parsing, null handling, type casting | Cleaned CSVs |
| **Gold** | Silver DataFrames | Dimensional modeling, surrogate key generation, star schema split | `fact_sales.csv`, `dim_product.csv`, `dim_customer.csv`, `dim_geo.csv` |

---

## 📊 Star Schema Data Model

![Data Model](Data%20Model.png)

### Fact Table
- **fact_sales** — `ProductKey`, `CustomerKey`, `GeoKey`, `OrderDate`, `Quantity`, `Net Price`

### Dimension Tables

| Dimension | Key Fields |
|-----------|------------|
| **dim_product** | ProductKey, Product Name, Brand, Color, Subcategory, Category |
| **dim_customer** | CustomerKey, Customer Code, Name, Education, Occupation |
| **dim_geo** | GeoKey (generated), City, State, CountryRegion, Continent |

---

## ⚙️ ETL Pipeline (`ETL.py`)

### Key Features
- **Config-driven**: All paths and filenames externalized to `config.json`
- **Auto-directory creation**: `setup_environment()` ensures output folders exist
- **Validation at every stage**: Row counts logged at Bronze, Silver, and Gold
- **Data integrity check**: Post-merge assertion to catch row loss during joins
- **Medallion architecture**: Clean separation of raw → cleaned → modeled data

---

## 📸 Screenshots

> 📷 *Add screenshots of the Orion Dashboard main page and the Power BI data model view.*

---

## 📂 Project Structure

```
Sales-Analytics-Solution-main/
├── Data Model.png          # Star schema diagram
├── Documentation.docx      # Project documentation
├── Orion Dashboard.pbix    # Power BI report
└── Root/
    ├── ETL.py              # Python ETL pipeline (Bronze → Silver → Gold)
    ├── config.json         # Pipeline configuration
    └── data/
        ├── raw/            # Bronze: Sales.json, forecast.json
        ├── processed/      # Silver: cleaned CSVs
        └── final/          # Gold: Star schema CSVs
```

---

## 🎓 Skills Demonstrated

| Category | Details |
|----------|---------|
| **Data Engineering** | Medallion architecture (Bronze/Silver/Gold), config-driven pipelines |
| **Data Modeling** | Star schema design, surrogate key generation, dimensional modeling |
| **Python (pandas)** | JSON ingestion, null handling, type casting, merge operations |
| **Data Validation** | Row count assertions, integrity checks at every pipeline stage |
| **Power BI** | Interactive dashboard, DAX calculations, data model relationships |

---

## 👤 Author

**Amgad Mohamed Abdelghfar**  
📧 amgadabdelghfar3@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/amgadabdelghfar/)
