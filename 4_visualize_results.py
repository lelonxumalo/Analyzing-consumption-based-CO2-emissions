import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Define the transition year between historical and forecasted data
FORECAST_START_YEAR = 2024

# Load data
print("Loading data...")
df = pd.read_excel('full_fair_share_projection_2050.xlsx', sheet_name='full_projection_fair_share_2050')

# Remove rows with missing fair share values
df = df.dropna(subset=['full_fair_share'])

# Calculate overshoot/undershoot metrics
# If fair_share > 1, country is overshooting (using more than fair share)
# If fair_share < 1, country is undershooting (using less than fair share)
df['overshoot_ratio'] = df['overshoot_estimate']
df['is_overshooting'] = df['overshoot_estimate'] > 1

# Mark historical vs forecasted data
df['is_forecast'] = df['year'] >= FORECAST_START_YEAR

# Get latest year data for comparison
latest_year = df['year'].max()
df_latest = df[df['year'] == latest_year].copy()

# Get last historical year for comparison
last_historical_year = FORECAST_START_YEAR - 1
df_historical_latest = df[df['year'] == last_historical_year].copy()

print(f"Data loaded: {len(df)} rows, {df['country'].nunique()} countries, years {df['year'].min()}-{df['year'].max()}")
print(f"Historical data: {df['year'].min()}-{last_historical_year}")
print(f"Forecasted data: {FORECAST_START_YEAR}-{latest_year}")

# Figure 1: Top 15 Overshooting and Undershooting Countries (Projected 2050)
print("\nCreating Figure 1: Top overshooting and undershooting countries...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Top 15 overshooting countries
top_overshoot = df_latest.nlargest(15, 'overshoot_ratio')[['country', 'overshoot_ratio']].sort_values('overshoot_ratio')
ax1.barh(range(len(top_overshoot)), top_overshoot['overshoot_ratio'], color='#e74c3c', alpha=0.8)
ax1.set_yticks(range(len(top_overshoot)))
ax1.set_yticklabels(top_overshoot['country'])
ax1.set_xlabel('Fair Share Ratio', fontsize=12)
ax1.set_title(f'Top 15 Overshooting Countries (Projected {latest_year})', fontsize=14, fontweight='bold')
ax1.axvline(x=1, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax1.text(1.05, len(top_overshoot)-1, 'Fair Share Threshold', rotation=90, va='top', fontsize=9, alpha=0.7)

# Top 15 undershooting countries (lowest ratios)
top_undershoot = df_latest.nsmallest(15, 'overshoot_ratio')[['country', 'overshoot_ratio']].sort_values('overshoot_ratio', ascending=False)
ax2.barh(range(len(top_undershoot)), top_undershoot['overshoot_ratio'], color='#3498db', alpha=0.8)
ax2.set_yticks(range(len(top_undershoot)))
ax2.set_yticklabels(top_undershoot['country'])
ax2.set_xlabel('Fair Share Ratio', fontsize=12)
ax2.set_title(f'Top 15 Undershooting Countries (Projected {latest_year})', fontsize=14, fontweight='bold')
ax2.axvline(x=1, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax2.text(1.05, len(top_undershoot)-1, 'Fair Share Threshold', rotation=90, va='top', fontsize=9, alpha=0.7)

plt.tight_layout()
plt.savefig('overshoot_comparison_2050.png', dpi=300, bbox_inches='tight')
print("  Saved: overshoot_comparison_2050.png")

# Figure 2: Time Series for Selected Major Economies (Historical + Forecast)
print("\nCreating Figure 2: Time series for major economies...")
major_economies = ['United States', 'China', 'India', 'Germany', 'United Kingdom',
                   'Japan', 'France', 'Brazil', 'Russia', 'South Africa']
df_major = df[df['country'].isin(major_economies)]

fig, ax = plt.subplots(figsize=(14, 8))
for country in major_economies:
    country_data = df_major[df_major['country'] == country].sort_values('year')
    if len(country_data) > 0:
        # Split into historical and forecast
        historical = country_data[country_data['year'] < FORECAST_START_YEAR]
        forecast = country_data[country_data['year'] >= FORECAST_START_YEAR]

        # Plot historical data with solid line
        if len(historical) > 0:
            ax.plot(historical['year'], historical['overshoot_ratio'],
                    marker='o', linewidth=2, markersize=3, label=country)

        # Plot forecast data with dashed line (same color as historical)
        if len(forecast) > 0 and len(historical) > 0:
            # Connect last historical point to forecast
            transition_data = pd.concat([historical.iloc[-1:], forecast])
            ax.plot(transition_data['year'], transition_data['overshoot_ratio'],
                    linestyle='--', linewidth=2, alpha=0.7)
        elif len(forecast) > 0:
            ax.plot(forecast['year'], forecast['overshoot_ratio'],
                    linestyle='--', linewidth=2, markersize=3, label=country, alpha=0.7)

# Add vertical line at forecast start
ax.axvline(x=FORECAST_START_YEAR, color='gray', linestyle=':', linewidth=2, alpha=0.5)
ax.text(FORECAST_START_YEAR, ax.get_ylim()[1]*0.95, 'Forecast Start',
        ha='center', va='top', fontsize=10, alpha=0.7,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

ax.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.6, label='Fair Share Threshold')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Fair Share Ratio', fontsize=12)
ax.set_title(f'Carbon Fair Share Trends: Major Economies ({df["year"].min()}-{latest_year})',
             fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('overshoot_timeseries_2050.png', dpi=300, bbox_inches='tight')
print("  Saved: overshoot_timeseries_2050.png")

# Figure 3: Distribution of Countries Above and Below Fair Share
print("\nCreating Figure 3: Distribution of overshoot/undershoot...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Histogram of fair share ratios
ax1.hist(df_latest['overshoot_ratio'], bins=30, color='#9b59b6', alpha=0.7, edgecolor='black')
ax1.axvline(x=1, color='red', linestyle='--', linewidth=2, label='Fair Share Threshold')
ax1.set_xlabel('Fair Share Ratio', fontsize=12)
ax1.set_ylabel('Number of Countries', fontsize=12)
ax1.set_title(f'Distribution of Fair Share Ratios (Projected {latest_year})', fontsize=14, fontweight='bold')
ax1.legend()
ax1.set_xlim(0, max(10, df_latest['overshoot_ratio'].quantile(0.95)))

# Count of countries over time with forecast indication
year_counts = df.groupby('year')['is_overshooting'].agg(['sum', 'count']).reset_index()
year_counts['undershooting'] = year_counts['count'] - year_counts['sum']
year_counts.rename(columns={'sum': 'overshooting'}, inplace=True)

ax2.fill_between(year_counts['year'], 0, year_counts['overshooting'],
                 color='#e74c3c', alpha=0.6, label='Overshooting')
ax2.fill_between(year_counts['year'], year_counts['overshooting'],
                 year_counts['overshooting'] + year_counts['undershooting'],
                 color='#3498db', alpha=0.6, label='Undershooting')

# Add vertical line at forecast start
ax2.axvline(x=FORECAST_START_YEAR, color='gray', linestyle=':', linewidth=2, alpha=0.5)
ax2.text(FORECAST_START_YEAR, ax2.get_ylim()[1]*0.95, 'Forecast Start',
        ha='center', va='top', fontsize=10, alpha=0.7,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Number of Countries', fontsize=12)
ax2.set_title(f'Countries Above vs Below Fair Share ({df["year"].min()}-{latest_year})',
              fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('overshoot_distribution_2050.png', dpi=300, bbox_inches='tight')
print("  Saved: overshoot_distribution_2050.png")

# Figure 4: CO2 Consumption Patterns (Historical + Forecast)
print("\nCreating Figure 4: CO2 consumption patterns for major economies...")
fig, ax = plt.subplots(figsize=(14, 8))

for country in major_economies:
    country_data = df_major[df_major['country'] == country].sort_values('year')
    if len(country_data) > 0 and country_data['consumption_co2'].notna().any():
        # Split into historical and forecast
        historical = country_data[country_data['year'] < FORECAST_START_YEAR]
        forecast = country_data[country_data['year'] >= FORECAST_START_YEAR]

        # Plot historical data with solid line
        if len(historical) > 0 and historical['consumption_co2'].notna().any():
            ax.plot(historical['year'], historical['consumption_co2'],
                    marker='o', linewidth=2, markersize=3, label=country)

        # Plot forecast data with dashed line
        if len(forecast) > 0 and len(historical) > 0 and forecast['consumption_co2'].notna().any():
            transition_data = pd.concat([historical.iloc[-1:], forecast])
            ax.plot(transition_data['year'], transition_data['consumption_co2'],
                    linestyle='--', linewidth=2, alpha=0.7)

# Add vertical line at forecast start
ax.axvline(x=FORECAST_START_YEAR, color='gray', linestyle=':', linewidth=2, alpha=0.5)
ax.text(FORECAST_START_YEAR, ax.get_ylim()[1]*0.95, 'Forecast Start',
        ha='center', va='top', fontsize=10, alpha=0.7,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Consumption-Based CO2 Emissions (Million Tonnes)', fontsize=12)
ax.set_title(f'CO2 Consumption Patterns: Major Economies ({df["year"].min()}-{latest_year})',
             fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('co2_consumption_trends_2050.png', dpi=300, bbox_inches='tight')
print("  Saved: co2_consumption_trends_2050.png")

# Figure 5: Overshoot Estimate Patterns (Historical + Forecast)
print("\nCreating Figure 5: Overshoot estimate patterns for major economies...")
fig, ax = plt.subplots(figsize=(14, 8))

for country in major_economies:
    country_data = df_major[df_major['country'] == country].sort_values('year')
    if len(country_data) > 0 and country_data['overshoot_estimate'].notna().any():
        # Split into historical and forecast
        historical = country_data[country_data['year'] < FORECAST_START_YEAR]
        forecast = country_data[country_data['year'] >= FORECAST_START_YEAR]

        # Plot historical data with solid line
        if len(historical) > 0 and historical['overshoot_estimate'].notna().any():
            ax.plot(historical['year'], historical['overshoot_estimate'],
                    marker='o', linewidth=2, markersize=3, label=country)

        # Plot forecast data with dashed line
        if len(forecast) > 0 and len(historical) > 0 and forecast['overshoot_estimate'].notna().any():
            transition_data = pd.concat([historical.iloc[-1:], forecast])
            ax.plot(transition_data['year'], transition_data['overshoot_estimate'],
                    linestyle='--', linewidth=2, alpha=0.7)

# Add vertical line at forecast start
ax.axvline(x=FORECAST_START_YEAR, color='gray', linestyle=':', linewidth=2, alpha=0.5)
ax.text(FORECAST_START_YEAR, ax.get_ylim()[1]*0.95, 'Forecast Start',
        ha='center', va='top', fontsize=10, alpha=0.7,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Ecological Overshoot Estimate (Million Tonnes CO2)', fontsize=12)
ax.set_title(f'Ecological Overshoot Patterns: Major Economies ({df["year"].min()}-{latest_year})',
             fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('overshoot_estimate_trends_2050.png', dpi=300, bbox_inches='tight')
print("  Saved: overshoot_estimate_trends_2050.png")

# Generate summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

print(f"\n--- HISTORICAL (Last Year: {last_historical_year}) ---")
print(f"Total Countries: {len(df_historical_latest)}")
if len(df_historical_latest) > 0:
    print(f"Overshooting Countries: {df_historical_latest['is_overshooting'].sum()} ({df_historical_latest['is_overshooting'].sum()/len(df_historical_latest)*100:.1f}%)")
    print(f"Undershooting Countries: {(~df_historical_latest['is_overshooting']).sum()} ({(~df_historical_latest['is_overshooting']).sum()/len(df_historical_latest)*100:.1f}%)")

print(f"\n--- PROJECTED (Year: {latest_year}) ---")
print(f"Total Countries: {len(df_latest)}")
print(f"Overshooting Countries: {df_latest['is_overshooting'].sum()} ({df_latest['is_overshooting'].sum()/len(df_latest)*100:.1f}%)")
print(f"Undershooting Countries: {(~df_latest['is_overshooting']).sum()} ({(~df_latest['is_overshooting']).sum()/len(df_latest)*100:.1f}%)")

print(f"\nTop 5 Overshooting Countries (Projected {latest_year}):")
for idx, row in df_latest.nlargest(5, 'overshoot_ratio').iterrows():
    print(f"  {row['country']}: {row['overshoot_ratio']:.2f}x fair share")

print(f"\nTop 5 Undershooting Countries (Projected {latest_year}):")
for idx, row in df_latest.nsmallest(5, 'overshoot_ratio').iterrows():
    print(f"  {row['country']}: {row['overshoot_ratio']:.2f}x fair share")

# Calculate change trends for major economies
print("\n--- PROJECTED CHANGES (2023 vs 2050) ---")
df_2023 = df[df['year'] == 2023]
df_2050 = df[df['year'] == 2050]
for country in major_economies:
    data_2023 = df_2023[df_2023['country'] == country]
    data_2050 = df_2050[df_2050['country'] == country]

    if len(data_2023) > 0 and len(data_2050) > 0:
        ratio_2023 = data_2023['overshoot_ratio'].values[0]
        ratio_2050 = data_2050['overshoot_ratio'].values[0]
        change = ratio_2050 - ratio_2023
        pct_change = (change / ratio_2023) * 100 if ratio_2023 != 0 else 0
        direction = "↑" if change > 0 else "↓"
        print(f"  {country}: {ratio_2023:.2f} → {ratio_2050:.2f} ({direction}{abs(pct_change):.1f}%)")

print("\n" + "="*60)
print("Visualizations complete! Generated 5 PNG files:")
print("  1. overshoot_comparison_2050.png")
print("  2. overshoot_timeseries_2050.png")
print("  3. overshoot_distribution_2050.png")
print("  4. co2_consumption_trends_2050.png")
print("  5. overshoot_estimate_trends_2050.png")
print("="*60)
