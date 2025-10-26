import pandas as pd
import os

def create_long_format_report(filename="all_countries_long_format.xlsx"):
    """
    Fetches the full OWID CO2 data, filters for countries only,
    and saves the data in a long-format Excel file.
    """
    
    # URL for the complete OWID CO2 and GHG emissions dataset
    url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
    
    # Define the columns we want to keep in our final output.
    # This includes identifiers and the key metrics.
    columns_to_keep = [
        'country',
        'iso_code',
        'year',
        'co2',
        'consumption_co2',
        'co2_per_capita',
        'consumption_co2_per_capita',
        'gdp',
        'population'
    ]
    
    print(f"Downloading full dataset from {url}...")
    
    try:
        # Load the entire dataset
        df = pd.read_csv(url)
        
        # --- Filter the data ---
        
        # 1. Filter for countries only:
        #    Aggregates (World, continents, income groups) have a blank 'iso_code'.
        #    We drop rows where 'iso_code' is NaN to keep only countries.
        df_countries = df.dropna(subset=['iso_code']).copy()
        
        # 2. Filter for relevant years (consumption data mostly starts in 1990)
        df_countries = df_countries[df_countries['year'] >= 1990]

        # 3. Keep only the columns we defined
        df_long_format = df_countries[columns_to_keep]
        
        # --- Save to Excel ---
        print(f"Saving data to {filename}...")
        
        # Save the final long-format DataFrame to one Excel sheet
        # index=False prevents pandas from writing the row numbers
        df_long_format.to_excel(filename, sheet_name='all_countries_data', index=False)
                
        print("\nSuccess! Report generated.")
        print(f"File saved to: {os.path.abspath(filename)}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please check your internet connection and file permissions.")

# --- START OF SCRIPT ---

# Run the function to create the report
create_long_format_report()