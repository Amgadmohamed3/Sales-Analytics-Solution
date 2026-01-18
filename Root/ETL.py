import pandas as pd
import json
from pathlib import Path

# 1. Get the Project Root directory
ROOT_DIR = Path(__file__).resolve().parent

def load_config():
    config_path = ROOT_DIR / 'config.json'
    with open(config_path, 'r') as f:
        return json.load(f)

def setup_environment(config):
    for path_str in config['paths'].values():
        Path(ROOT_DIR / path_str).mkdir(parents=True, exist_ok=True)

def extract_from_raw(config):
    """STAGE: BRONZE -> Loading original data"""
    raw_path = ROOT_DIR / config['paths']['raw_dir']
    
    with open(raw_path / config['files']['sales'], 'r', encoding='utf-8') as f:
        df_sales = pd.DataFrame(json.load(f))
    
    with open(raw_path / config['files']['forecast'], 'r', encoding='utf-8') as f:
        df_forecast = pd.DataFrame(json.load(f))
        
    # VALIDATION
    print(f"\n[BRONZE VALIDATION]")
    print(f"-> Raw Sales Loaded: {len(df_sales)} rows")
    print(f"-> Raw Forecast Loaded: {len(df_forecast)} rows")
    
    return df_sales, df_forecast

def transform_to_silver(df_sales, df_forecast, config):
    """STAGE: SILVER -> Cleaning, Standardizing, and Saving to Processed"""
    processed_out = ROOT_DIR / config['paths']['processed_dir']
    initial_count = len(df_sales)

    # Cleaning
    df_sales['OrderDate'] = pd.to_datetime(df_sales['OrderDate'])
    df_sales['Name'] = df_sales['Name'].fillna('Unknown')
    df_sales['Education'] = df_sales['Education'].fillna('Not Provided')
    df_sales['Occupation'] = df_sales['Occupation'].fillna('Not Provided')
    df_forecast['Year'] = df_forecast['Year'].astype(int)

    # Save Silver Layer
    df_sales.to_csv(processed_out / 'cleaned_sales.csv', index=False)
    df_forecast.to_csv(processed_out / 'cleaned_forecast.csv', index=False)
    
    # VALIDATION
    print(f"\n[SILVER VALIDATION]")
    print(f"-> Cleaned Sales: {len(df_sales)} rows (Dropped: {initial_count - len(df_sales)})")
    
    return df_sales, df_forecast

def model_to_gold(df_sales, df_forecast, config):
    """STAGE: GOLD -> Dimensional Modeling (Star Schema) and Saving to Final"""
    final_out = ROOT_DIR / config['paths']['final_dir']
    input_count = len(df_sales)

    # Create Dimensions
    dim_product = df_sales[['ProductKey', 'Product Name', 'Brand', 'Color', 'Subcategory', 'Category']].drop_duplicates()
    dim_customer = df_sales[['CustomerKey', 'Customer Code', 'Name', 'Education', 'Occupation']].drop_duplicates()
    
    dim_geo = df_sales[['City', 'State', 'CountryRegion', 'Continent']].drop_duplicates().copy()
    dim_geo['GeoKey'] = range(1, len(dim_geo) + 1)
    
    # Create Fact Sales (Validation point: Check if the merge drops rows)
    fact_sales = df_sales.merge(dim_geo, on=['City', 'State', 'CountryRegion', 'Continent'])
    fact_sales = fact_sales[['ProductKey', 'CustomerKey', 'GeoKey', 'OrderDate', 'Quantity', 'Net Price']]

    # Save Gold Layer
    fact_sales.to_csv(final_out / 'fact_sales.csv', index=False)
    df_forecast.to_csv(final_out / 'fact_forecast.csv', index=False)
    dim_product.to_csv(final_out / 'dim_product.csv', index=False)
    dim_customer.to_csv(final_out / 'dim_customer.csv', index=False)
    dim_geo.to_csv(final_out / 'dim_geo.csv', index=False)

    # VALIDATION
    print(f"\n[GOLD VALIDATION]")
    print(f"-> Dim_Product: {len(dim_product)} unique items")
    print(f"-> Dim_Customer: {len(dim_customer)} unique customers")
    print(f"-> Fact_Sales: {len(fact_sales)} rows")
    
    if len(fact_sales) != input_count:
        print(f"!! WARNING: {input_count - len(fact_sales)} rows lost during modeling merge!")
    else:
        print("-> Data Integrity Check: Passed (Sales row count maintained)")

def run_etl():
    print("Starting ETL Pipeline...")
    config = load_config()
    setup_environment(config)
    
    # Stage 1: Bronze
    raw_sales, raw_forecast = extract_from_raw(config)
    
    # Stage 2: Silver
    silver_sales, silver_forecast = transform_to_silver(raw_sales, raw_forecast, config)
    
    # Stage 3: Gold
    model_to_gold(silver_sales, silver_forecast, config)

    print("\n" + "="*30)
    print("ETL SUCCESSFUL")
    print("="*30)

if __name__ == "__main__":
    run_etl()