# Carbon Fair Share Analysis Pipeline

This project analyzes carbon emissions and calculates "fair share" metrics by comparing countries' consumption-based CO2 emissions against ecological overshoot estimates, with projections extending to 2050.

## Overview

The analysis pipeline consists of four sequential scripts that:
1. Download historical emissions data from Our World in Data (OWID)
2. Project consumption-based CO2 emissions to 2050 using ARIMA models
3. Calculate fair share metrics by comparing emissions against overshoot estimates
4. Generate comprehensive visualizations of the results

## Scripts (Run in Order)

### 1. `1_download_emissions.py`
Downloads the complete OWID CO2 and GHG emissions dataset, filters for countries only (1990+), and saves data in long format.

**Output:** `all_countries_long_format.xlsx`

### 2. `2_project_consumption.py`
Uses Auto-ARIMA models to project consumption-based CO2 emissions from 2023 to 2050 for all countries based on historical data (1990-2022).

**Input:** `all_countries_long_format.xlsx`
**Output:** `country_co2_projections_standardized.xlsx`

### 3. `3_calculate_fair_shares.py`
Combines historical and projected data to calculate fair share metrics through 2050. Projects both overshoot estimates and consumption emissions using ARIMA models.

**Inputs:**
- `all_countries_long_format.xlsx`
- `SocialShortfallAndEcologicalOvershoot_SupplementaryData.xlsx`

**Output:** `full_fair_share_projection_2050.xlsx`

### 4. `4_visualize_results.py`
Creates comprehensive visualizations including:
- Top overshooting/undershooting countries
- Time series for major economies
- Distribution analysis
- CO2 consumption trends
- Overshoot estimate patterns

**Input:** `full_fair_share_projection_2050.xlsx`
**Output:** 5 PNG visualization files

## Installation

1. Clone this repository
2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the scripts in sequential order:

```bash
python 1_download_emissions.py
python 2_project_consumption.py
python 3_calculate_fair_shares.py
python 4_visualize_results.py
```

## Requirements

- Python 3.8+
- pandas
- openpyxl
- pmdarima
- matplotlib
- seaborn
- numpy

See `requirements.txt` for specific versions.

## Data Sources

- **OWID CO2 Data:** https://github.com/owid/co2-data
- **Ecological Overshoot Data:** `SocialShortfallAndEcologicalOvershoot_SupplementaryData.xlsx`

## Methodology

The analysis uses Auto-ARIMA time series models to:
- Project future consumption-based CO2 emissions (2023-2050)
- Project ecological overshoot estimates (2024-2050)
- Calculate fair share ratios (consumption_co2 / overshoot_estimate)

A ratio > 1 indicates a country is overshooting its fair share, while < 1 indicates undershooting.

## Output Visualizations

1. `overshoot_comparison_2050.png` - Top 15 overshooting and undershooting countries
2. `overshoot_timeseries_2050.png` - Time series trends for major economies
3. `overshoot_distribution_2050.png` - Distribution of fair share ratios
4. `co2_consumption_trends_2050.png` - CO2 consumption patterns over time
5. `overshoot_estimate_trends_2050.png` - Ecological overshoot patterns

## Archive

- `archive_implied_fair_shares_v1.py` - Earlier version of the fair share calculation script (superseded by script 3)

## License

[Add your license here]

## Contact

[Add your contact information here]
