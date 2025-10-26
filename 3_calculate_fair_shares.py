import pandas as pd
import numpy as np
import os
import warnings
try:
    # This script requires pmdarima
    # To install, run: pip install pmdarima
    import pmdarima as pm
except ImportError:
    print("--- ERROR ---")
    print("Module 'pmdarima' not found. This script requires it.")
    print("Please install it by running: pip install pmdarima")
    print("---------------")
    exit()

# Sheet name for historical overshoot data
HISTORICAL_OVERSHOOT_SHEET_NAME = "Biophysical_Historical"

def create_full_projection_analysis(
    historical_file="all_countries_long_format.xlsx",
    historical_sheet="all_countries_data",
    overshoot_data_file="SocialShortfallAndEcologicalOvershoot_SupplementaryData.xlsx",
    overshoot_hist_sheet=HISTORICAL_OVERSHOOT_SHEET_NAME,
    overshoot_bau_sheet="Biophysical_BAU_middle",
    output_file="full_fair_share_projection_2050.xlsx"
):
    """
    Loads historical emissions (1990-2023) and overshoot data (hist 1990-2015 + BAU 2016-2050).
    1. Projects overshoot 2024-2050 based on training data 1990-2023 (hist+BAU).
    2. Projects consumption_co2 2024-2050 based on training data 1990-2023.
    3. Merges everything to calculate fair shares through 2050.

    Creates two analysis sheets:
    1. 'implied_fair_share_hist': Based on original 1990-2015 historical overshoot.
    2. 'full_projection_fair_share_2050': Based on projections for both datasets to 2050.
    """

    warnings.filterwarnings("ignore")

    # --- 1: Load Input Files ---
    print(f"\n--- Starting Full Projection Analysis ---")
    print(f"Loading historical emissions from: {historical_file} (Sheet: {historical_sheet})")
    try:
        df_historical_emissions = pd.read_excel(historical_file, sheet_name=historical_sheet)
        df_historical_emissions['iso_code'] = df_historical_emissions['iso_code'].astype(str).str.upper().str.strip()
        print("...success.")
    except Exception as e:
        print(f"--- ERROR loading {historical_file}: {e}")
        return

    print(f"Loading HISTORICAL overshoot (1990-2015) from: {overshoot_data_file} (Sheet: {overshoot_hist_sheet})")
    try:
        df_overshoot_hist_raw = pd.read_excel(overshoot_data_file, sheet_name=overshoot_hist_sheet)
        print("...success.")
    except Exception as e:
        print(f"--- ERROR loading HISTORICAL overshoot sheet '{overshoot_hist_sheet}': {e}")
        return

    print(f"Loading BAU overshoot (2016+) from: {overshoot_data_file} (Sheet: {overshoot_bau_sheet})")
    try:
        df_overshoot_bau_raw = pd.read_excel(overshoot_data_file, sheet_name=overshoot_bau_sheet)
        print("...success.")
    except Exception as e:
        print(f"--- ERROR loading BAU overshoot sheet '{overshoot_bau_sheet}': {e}")
        return

    # --- 2: Prepare and Combine Overshoot Data ---
    print("Preparing and combining overshoot data...")

    def prepare_overshoot_df(df_raw, data_type):
        """Helper function to clean and filter overshoot data."""
        print(f"  Preparing {data_type} data...")
        try:
            df = df_raw.rename(columns={
                'date': 'year',
                'CO2 Emissions': 'overshoot_estimate',
                'iso3c': 'iso_code'
            })
            if 'indicator' in df.columns:
                 df = df[df['indicator'] == 'CO2 emissions'].copy()
                 if df.empty: raise ValueError("'CO2 emissions' not found in 'indicator' column.")
                 print("    Filtered for 'CO2 emissions' indicator.")
            expected_cols = ['country', 'iso_code', 'year', 'overshoot_estimate']
            if not all(col in df.columns for col in expected_cols):
                 raise KeyError(f"Missing expected columns. Found: {df.columns.tolist()}")
            df = df[expected_cols].dropna(subset=['overshoot_estimate'])
            df['year'] = df['year'].astype(int)
            df['iso_code'] = df['iso_code'].astype(str).str.upper().str.strip()
            print(f"    Cleaned {data_type} data successfully.")
            return df
        except Exception as e:
            print(f"--- ERROR preparing {data_type} data: {e}")
            return None

    df_overshoot_hist = prepare_overshoot_df(df_overshoot_hist_raw, "Historical")
    df_overshoot_bau = prepare_overshoot_df(df_overshoot_bau_raw, "BAU")
    if df_overshoot_hist is None or df_overshoot_bau is None: return

    hist_start_year = 1990
    hist_end_year = 2015
    df_overshoot_hist_filtered = df_overshoot_hist[(df_overshoot_hist['year'] >= hist_start_year) & (df_overshoot_hist['year'] <= hist_end_year)]

    bau_train_start_year = 2016
    bau_train_end_year = 2023
    df_overshoot_bau_train_part = df_overshoot_bau[(df_overshoot_bau['year'] >= bau_train_start_year) & (df_overshoot_bau['year'] <= bau_train_end_year)]

    df_overshoot_train = pd.concat([df_overshoot_hist_filtered, df_overshoot_bau_train_part]).sort_values(by=['iso_code', 'year'])
    print(f"Created overshoot training dataset (1990-{bau_train_end_year}) with {len(df_overshoot_train)} rows.")

    # --- 3: Create Sheet 1 (Based on ORIGINAL Historical Overshoot 1990-2015) ---
    print("Creating Sheet 1: 'implied_fair_share_hist'")
    df_merged_hist = pd.merge(df_historical_emissions, df_overshoot_hist_filtered[['iso_code', 'year', 'overshoot_estimate']], on=['iso_code', 'year'], how='inner')
    if len(df_merged_hist) == 0: print("\n--- WARNING ---: Merge for Sheet 1 resulted in 0 rows.")
    else: print(f"...Sheet 1 will contain {len(df_merged_hist)} rows.")
    df_merged_hist['implied_fair_share'] = df_merged_hist.get('consumption_co2', pd.Series(dtype='float')) / df_merged_hist.get('overshoot_estimate', pd.Series(dtype='float'))
    df_merged_hist['implied_fair_share'] = df_merged_hist['implied_fair_share'].replace([np.inf, -np.inf], np.nan)
    output_cols_hist = ['country', 'iso_code', 'year', 'consumption_co2', 'overshoot_estimate', 'implied_fair_share']
    df_sheet1 = df_merged_hist.reindex(columns=output_cols_hist)

    # --- 4: Project BOTH Datasets (2024-2050) ---
    print("Projecting BOTH overshoot and consumption_co2 to 2050...")
    
    projection_start_year = bau_train_end_year + 1 # 2024
    projection_end_year = 2050
    n_periods = projection_end_year - bau_train_end_year # 2050 - 2023 = 27 periods

    iso_codes = df_historical_emissions['iso_code'].unique() # Use all ISOs from emissions data
    
    full_overshoot_list = [df_overshoot_train] # Start with 1990-2023 overshoot data
    full_consumption_list = [df_historical_emissions[['iso_code', 'year', 'consumption_co2']].dropna(subset=['consumption_co2'])] # Start with 1990-2023 consumption data

    for i, code in enumerate(iso_codes):
        print(f"  ({i+1}/{len(iso_codes)}) Projecting for {code}...")
        
        # --- Project Overshoot ---
        ts_overshoot_train = df_overshoot_train[df_overshoot_train['iso_code'] == code].set_index('year')['overshoot_estimate']
        if len(ts_overshoot_train) >= 10:
            try:
                overshoot_model = pm.auto_arima(ts_overshoot_train, start_p=1, start_q=1, test='kpss', max_p=3, max_q=3, m=1, d=None, seasonal=False, suppress_warnings=True, stepwise=True, information_criterion='aicc')
                overshoot_forecast = overshoot_model.predict(n_periods=n_periods)
                overshoot_proj_df = pd.DataFrame({'iso_code': code, 'year': range(projection_start_year, projection_end_year + 1), 'overshoot_estimate': overshoot_forecast})
                full_overshoot_list.append(overshoot_proj_df)
            except Exception as e: print(f"    - ERROR projecting overshoot for {code}: {e}")
        else: print(f"    - Skipping overshoot projection for {code} (not enough data: {len(ts_overshoot_train)})")

        # --- Project Consumption ---
        ts_consumption_train = df_historical_emissions[(df_historical_emissions['iso_code'] == code) & (df_historical_emissions['year'] <= bau_train_end_year)].set_index('year')['consumption_co2'].dropna()
        if len(ts_consumption_train) >= 10:
            try:
                consumption_model = pm.auto_arima(ts_consumption_train, start_p=1, start_q=1, test='kpss', max_p=3, max_q=3, m=1, d=None, seasonal=False, suppress_warnings=True, stepwise=True, information_criterion='aicc')
                consumption_forecast = consumption_model.predict(n_periods=n_periods)
                consumption_proj_df = pd.DataFrame({'iso_code': code, 'year': range(projection_start_year, projection_end_year + 1), 'consumption_co2': consumption_forecast})
                full_consumption_list.append(consumption_proj_df)
            except Exception as e: print(f"    - ERROR projecting consumption for {code}: {e}")
        else: print(f"    - Skipping consumption projection for {code} (not enough data: {len(ts_consumption_train)})")

    # --- 5: Combine Full Datasets and Calculate Fair Share for Sheet 2 ---
    print("...projections complete. Combining full datasets and calculating fair shares for Sheet 2.")

    df_full_overshoot = pd.concat(full_overshoot_list).sort_values(by=['iso_code', 'year'])
    df_full_consumption = pd.concat(full_consumption_list).sort_values(by=['iso_code', 'year'])

    # Merge full consumption (hist+proj) with full overshoot (hist+proj)
    # Use an outer merge to keep all years from both, then clean up
    df_merged_full = pd.merge(
        df_full_consumption,
        df_full_overshoot[['iso_code', 'year', 'overshoot_estimate']],
        on=['iso_code', 'year'],
        how='outer' # Keep all rows from both projections
    )

    if len(df_merged_full) == 0:
        print("\n--- WARNING ---: Final merge for Sheet 2 resulted in 0 rows.")
    else:
        print(f"...Sheet 2 will contain {len(df_merged_full)} rows (before potential drops).")

    df_merged_full['full_fair_share'] = df_merged_full['consumption_co2'] / df_merged_full['overshoot_estimate']
    df_merged_full['full_fair_share'] = df_merged_full['full_fair_share'].replace([np.inf, -np.inf], np.nan)

    # Add country names back by merging with the historical data's mapping
    country_map = df_historical_emissions[['iso_code', 'country']].drop_duplicates()
    df_merged_full = pd.merge(df_merged_full, country_map, on='iso_code', how='left')
    # Fill any potentially missing country names if new ISOs appeared somehow
    df_merged_full['country'] = df_merged_full.groupby('iso_code')['country'].ffill().bfill()


    output_cols_full = ['country', 'iso_code', 'year', 'consumption_co2', 'overshoot_estimate', 'full_fair_share']
    df_sheet2 = df_merged_full.reindex(columns=output_cols_full).sort_values(by=['iso_code', 'year'])
    # Optional: Drop rows where fair share couldn't be calculated if desired
    # df_sheet2 = df_sheet2.dropna(subset=['full_fair_share'])

    # --- 6: Save Output File ---
    output_file_name = "full_fair_share_projection_2050.xlsx"
    print(f"Saving final analysis to '{output_file_name}' with two sheets.")
    try:
        with pd.ExcelWriter(output_file_name, engine='openpyxl') as writer:
            df_sheet1.to_excel(
                writer, sheet_name='implied_fair_share_hist', index=False
            )
            df_sheet2.to_excel(
                writer, sheet_name='full_projection_fair_share_2050', index=False
            )

        print(f"\n--- SUCCESS ---")
        print(f"Analysis complete. File saved to:")
        print(f"{os.path.abspath(output_file_name)}")
        print("---------------")

    except Exception as e:
        print(f"\n--- ERROR ---: Could not save Excel file: {e}")

# --- Execute Plan ---
if __name__ == "__main__":
    create_full_projection_analysis()