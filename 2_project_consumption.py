import pandas as pd
import pmdarima as pm
import os
import warnings

def project_consumption_co2_standardized(
    input_file="all_countries_long_format.xlsx", 
    output_file="country_co2_projections_standardized.xlsx"
):
    """
    Ingests long-format data, uses a fixed historical period (1990-2022) 
    to train an Auto-ARIMA model, and projects a standardized 
    BAU trend from 2023-2050 for all countries.
    
    Saves two sheets to Excel:
    1. A summary of cumulative totals and 2050 values.
    2. A full time series (1990-2050) in long format.
    """
    
    print(f"Loading data from {input_file}...")
    try:
        df_full_historical = pd.read_excel(input_file)
    except FileNotFoundError:
        print(f"ERROR: Input file not found: {input_file}")
        print("Please ensure this script is in the same folder as the data file.")
        return
    except Exception as e:
        print(f"ERROR: Could not read Excel file. {e}")
        return

    # --- Define Standardized Year Ranges ---
    historical_end_year = 2022
    projection_start_year = 2023
    projection_end_year = 2050
    n_periods = projection_end_year - historical_end_year # 2050 - 2022 = 28 periods

    print(f"Standardizing analysis: History = 1990-{historical_end_year}, Projection = {projection_start_year}-{projection_end_year}")

    # Suppress warnings from model fitting
    warnings.filterwarnings("ignore")

    countries = df_full_historical['country'].unique()
    
    # List to store summary data (for Sheet 1)
    summary_results = []
    # List to store all time series data (for Sheet 2)
    full_timeseries_list = []
    
    # For 66% prediction intervals, alpha = 1.0 - 0.66 = 0.34
    prediction_alpha = 0.34 

    print(f"Found {len(countries)} countries. Starting projections...")

    for i, country in enumerate(countries):
        
        # --- 1. Prepare Historical Data ---
        country_iso = df_full_historical[df_full_historical['country'] == country]['iso_code'].iloc[0]
        
        # Filter for this country's data UP TO the fixed historical end year
        ts_data_train = df_full_historical[
            (df_full_historical['country'] == country) &
            (df_full_historical['year'] <= historical_end_year)
        ][['year', 'consumption_co2']].set_index('year').dropna()
        
        if len(ts_data_train) < 10:
            print(f"  ({i+1}/{len(countries)}) Skipping {country} (not enough historical data by {historical_end_year})")
            continue

        print(f"  ({i+1}/{len(countries)}) Processing {country}...")
        
        # --- 2. Store Historical Data for Full Time Series (Sheet 2) ---
        historical_df = pd.DataFrame({
            'country': country,
            'iso_code': country_iso,
            'year': ts_data_train.index,
            'type': 'Historical',
            'forecast_mean': ts_data_train['consumption_co2'].values,
            'forecast_lower_66PI': ts_data_train['consumption_co2'].values,
            'forecast_upper_66PI': ts_data_train['consumption_co2'].values
        })
        full_timeseries_list.append(historical_df)

        try:
            # --- 3. Fit Auto-ARIMA Model ---
            # Model is trained *only* on data up to 2022
            auto_model = pm.auto_arima(
                ts_data_train['consumption_co2'],
                start_p=1, start_q=1,
                test='kpss', max_p=3, max_q=3, m=1,
                d=None, seasonal=False,
                suppress_warnings=True, stepwise=True,
                information_criterion='aicc'
            )

            # --- 4. Generate Projections (2023-2050) ---
            forecast_mean, conf_int = auto_model.predict(
                n_periods=n_periods, # Fixed 28 periods (2023-2050)
                return_conf_int=True,
                alpha=prediction_alpha
            )
            
            # --- 5. Store Projected Data for Full Time Series (Sheet 2) ---
            forecast_years = range(projection_start_year, projection_end_year + 1)
            projected_df = pd.DataFrame({
                'country': country,
                'iso_code': country_iso,
                'year': forecast_years,
                'type': 'Projected',
                'forecast_mean': forecast_mean,
                'forecast_lower_66PI': conf_int[:, 0], # Lower bound
                'forecast_upper_66PI': conf_int[:, 1]  # Upper bound
            })
            full_timeseries_list.append(projected_df)
            
            # --- 6. Calculate Summary Statistics (Sheet 1) ---
            cumul_1990_2022 = ts_data_train['consumption_co2'].sum()
            projected_cumul_2023_2050 = forecast_mean.sum()
            
            summary_results.append({
                'country': country,
                'iso_code': country_iso,
                'model_used': auto_model.order,
                'model_aicc': auto_model.aicc(),
                'cumulative_1990_2022': cumul_1990_2022,
                'projected_cumulative_2023_2050': projected_cumul_2023_2050,
                'total_cumulative_1990_2050': cumul_1990_2022 + projected_cumul_2023_2050,
                'forecast_2050_mean': forecast_mean.iloc[-1],
                'forecast_2050_lower_66PI': conf_int[-1, 0],
                'forecast_2050_upper_66PI': conf_int[-1, 1]
            })

        except Exception as e:
            # Handle cases where model fitting fails
            print(f"  - ERROR processing {country}: {e}")

    # --- 7. Prepare Final DataFrames and Save to Excel ---
    print("\nAll projections complete. Saving results to Excel...")
    
    # Create the Summary DataFrame (Sheet 1)
    summary_df = pd.DataFrame(summary_results)
    summary_df = summary_df.sort_values(by='total_cumulative_1990_2050', ascending=False)
    
    # Create the Full Time Series DataFrame (Sheet 2)
    full_timeseries_df = pd.concat(full_timeseries_list)
    cols_order = [
        'country', 'iso_code', 'year', 'type', 
        'forecast_mean', 'forecast_lower_66PI', 'forecast_upper_66PI'
    ]
    full_timeseries_df = full_timeseries_df[cols_order].sort_values(by=['country', 'year'])

    # Use pd.ExcelWriter to save both DataFrames to different sheets
    try:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            summary_df.to_excel(
                writer, sheet_name='summary_projections', index=False
            )
            full_timeseries_df.to_excel(
                writer, sheet_name='full_timeseries_1990_2050', index=False
            )
        
        print(f"\nSuccess! Standardized report generated.")
        print(f"File saved to: {os.path.abspath(output_file)}")
        
    except Exception as e:
        print(f"\nERROR: Could not save Excel file. {e}")
        print("Please ensure the file is not already open.")

# --- START OF SCRIPT ---
if __name__ == "__main__":
    project_consumption_co2_standardized(
        # Make sure your input file (with 2023 data) is named this:
        input_file="all_countries_long_format.xlsx", 
        # This will be the new output file:
        output_file="country_co2_projections_standardized.xlsx"
    )