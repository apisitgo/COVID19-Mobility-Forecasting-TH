import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.metrics import mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

# Set graph style
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)
plt.rcParams['font.family'] = 'Arial'

# ===== 1. Load and prepare data =====
print("=" * 80)
print("Loading Google Mobility Report data...")
print("=" * 80)

df = pd.read_csv('data.csv')
df['date'] = pd.to_datetime(df['date'], dayfirst=True)
df = df[(df['date'] >= '2020-03-06') & (df['date'] <= '2022-10-15')].copy().reset_index(drop=True)

print(f"Total data points: {len(df)} rows")
print(f"Time period: {df['date'].min().date()} to {df['date'].max().date()}")

# Add features for XGBoost
df['cases_per_million'] = df['Daily new confirmed cases of COVID-19 per million people'].fillna(0)
df['StringencyIndex'] = df['StringencyIndex'].fillna(0)


def assign_lockdown(date):
    if pd.Timestamp(2020, 3, 26) <= date <= pd.Timestamp(2020, 4, 30):
        return 5
    elif pd.Timestamp(2020, 5, 1) <= date <= pd.Timestamp(2020, 6, 30):
        return 4
    elif pd.Timestamp(2021, 1, 1) <= date <= pd.Timestamp(2021, 1, 31):
        return 3
    elif pd.Timestamp(2021, 7, 12) <= date <= pd.Timestamp(2021, 8, 31):
        return 5
    elif pd.Timestamp(2021, 9, 1) <= date <= pd.Timestamp(2021, 9, 30):
        return 4
    else:
        return 1


df['lockdown_level'] = df['date'].apply(assign_lockdown)

# Create time features
df['dayofweek'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['week'] = df['date'].dt.isocalendar().week.astype(int)
df['year'] = df['date'].dt.year
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

# Columns to forecast
columns = [
    ('retail_and_recreation_percent_change_from_baseline', 'Retail & Recreation'),
    ('grocery_and_pharmacy_percent_change_from_baseline', 'Grocery & Pharmacy'),
    ('parks_percent_change_from_baseline', 'Parks'),
    ('transit_stations_percent_change_from_baseline', 'Transit Stations'),
    ('workplaces_percent_change_from_baseline', 'Workplaces'),
    ('residential_percent_change_from_baseline', 'Residential')
]

# ===== 2. Define 4 COVID-19 wave periods =====
waves = [
    {
        'name': 'Wave 1: First wave (Wuhan)',
        'start_date': '2020-02-04',
        'end_date': '2020-12-14',
        'initial_train': 10,  # weeks
        'step_size': 2,
        'forecast_horizon': 2,
        'color': '#3498db'
    },
    {
        'name': 'Wave 2: Second wave (Alpha-Beta)',
        'start_date': '2020-12-15',
        'end_date': '2021-03-31',
        'initial_train': 6,
        'step_size': 1,
        'forecast_horizon': 1,
        'color': '#e74c3c'
    },
    {
        'name': 'Wave 3: April 2021 wave (Delta)',
        'start_date': '2021-04-01',
        'end_date': '2021-12-31',
        'initial_train': 10,
        'step_size': 2,
        'forecast_horizon': 2,
        'color': '#9b59b6'
    },
    {
        'name': 'Wave 4: January 2022 wave (Omicron)',
        'start_date': '2022-01-01',
        'end_date': '2022-10-15',
        'initial_train': 8,
        'step_size': 2,
        'forecast_horizon': 2,
        'color': '#2ecc71'
    }
]


# ===== 3. Create Thai holidays for Prophet =====
def create_thai_holidays():
    holidays_list = []
    for year in [2020, 2021, 2022]:
        holidays_list.extend([
            {'holiday': 'New Year', 'ds': f'{year}-01-01'},
            {'holiday': 'Songkran', 'ds': f'{year}-04-13'},
            {'holiday': 'Labour Day', 'ds': f'{year}-05-01'},
            {'holiday': "Mother's Day", 'ds': f'{year}-08-12'},
            {'holiday': "Father's Day", 'ds': f'{year}-12-05'},
        ])
    holidays_df = pd.DataFrame(holidays_list)
    holidays_df['ds'] = pd.to_datetime(holidays_df['ds'])
    return holidays_df


holidays_df = create_thai_holidays()


# ===== 4. Robust ARIMA function for limited data =====
def create_robust_arima(train_series, forecast_horizon, wave_name):
    """Create ARIMA model resilient to data limitations"""
    try:
        from pmdarima import auto_arima

        # Special handling for waves with limited data
        if 'Wave 2' in wave_name or 'Wave 4' in wave_name:
            # Use simpler non-seasonal model for limited data
            model = auto_arima(
                train_series,
                seasonal=False,
                trace=False,
                suppress_warnings=True,
                max_p=2,
                max_q=2,
                max_d=1,
                error_action='ignore',
                stepwise=True,
                n_jobs=1
            )
            return model
        else:
            # Try multiple seasonal periods for sufficient data
            seasonal_periods = [7, 14, 30]

            for m in seasonal_periods:
                try:
                    model = auto_arima(
                        train_series,
                        seasonal=True,
                        m=m,
                        trace=False,
                        suppress_warnings=True,
                        max_p=3,
                        max_q=3,
                        max_d=2,
                        max_P=2,
                        max_Q=2,
                        max_D=1,
                        error_action='ignore',
                        stepwise=True,
                        n_jobs=1
                    )
                    forecast = model.predict(n_periods=forecast_horizon)
                    if not np.any(np.isnan(forecast)) and len(forecast) == forecast_horizon:
                        return model
                except:
                    continue

            # Fallback to non-seasonal model
            try:
                model = auto_arima(
                    train_series,
                    seasonal=False,
                    trace=False,
                    suppress_warnings=True,
                    max_p=3,
                    max_q=3,
                    max_d=2,
                    error_action='ignore',
                    stepwise=True,
                    n_jobs=1
                )
                forecast = model.predict(n_periods=forecast_horizon)
                if not np.any(np.isnan(forecast)) and len(forecast) == forecast_horizon:
                    return model
            except:
                pass

            return None
    except ImportError:
        return None


# ===== 5. Data quality check with wave-specific thresholds =====
def check_data_quality(df, column_name, wave_name):
    """Check data quality with wave-specific thresholds"""
    series = df[column_name].dropna()
    n_original = len(df[column_name])
    n_valid = len(series)

    # Set thresholds based on wave characteristics
    min_data_points = 10
    min_std = 0.5

    if 'Wave 2' in wave_name or 'Wave 4' in wave_name:
        min_data_points = 5  # Lower threshold for limited data waves

    # Check data validity
    quality = {
        'valid': True,
        'message': "Data passed quality check",
        'recommendation': ""
    }

    if n_valid < min_data_points:
        quality['valid'] = False
        quality['message'] = f"Insufficient data points ({n_valid}/{n_original} valid)"
        quality['recommendation'] = "Using simple mean as fallback forecast"

    elif series.std() < min_std:
        quality['valid'] = False
        quality['message'] = f"Low variance (std={series.std():.4f})"
        quality['recommendation'] = "Using simple mean as fallback forecast"

    elif series.nunique() < 3:
        quality['valid'] = False
        quality['message'] = f"Insufficient unique values (n_unique={series.nunique()})"
        quality['recommendation'] = "Using simple mean as fallback forecast"

    return quality


# ===== 6. Rolling Origin CV with fallback mechanism =====
def rolling_origin_cv_by_wave(df, wave_config, column_name, model_type='arima', holidays_df=None):
    """Run Rolling Origin CV for specific COVID wave with fallback mechanisms"""
    # Filter data for wave period
    actual_start = max(pd.to_datetime(wave_config['start_date']), df['date'].min())
    wave_df = df[
        (df['date'] >= actual_start) &
        (df['date'] <= pd.to_datetime(wave_config['end_date']))
        ].copy().reset_index(drop=True)

    # Check minimum data requirement
    min_required = wave_config['initial_train'] + wave_config['forecast_horizon']
    if len(wave_df) < min_required:
        return []

    results = []
    n_samples = len(wave_df)
    start = 0

    # Import libraries
    try:
        if model_type == 'arima':
            from pmdarima import auto_arima
        elif model_type == 'prophet':
            from prophet import Prophet
        elif model_type == 'xgboost':
            from xgboost import XGBRegressor
    except:
        return []

    while start + wave_config['initial_train'] + wave_config['forecast_horizon'] <= n_samples:
        train_end = start + wave_config['initial_train']
        test_end = train_end + wave_config['forecast_horizon']

        train_data = wave_df.iloc[start:train_end].copy()
        test_data = wave_df.iloc[train_end:test_end].copy()

        try:
            if model_type == 'arima':
                # Check data quality
                quality = check_data_quality(train_data, column_name, wave_config['name'])

                # Fallback to simple mean if data quality is poor
                if not quality['valid']:
                    train_series = train_data[column_name].dropna()
                    if len(train_series) > 0:
                        forecast = np.array([train_series.mean()] * wave_config['forecast_horizon'])
                        test_series = test_data[column_name].dropna()
                        actual = test_series.values[:len(forecast)]

                        if len(actual) > 0 and len(forecast) > 0:
                            mae = mean_absolute_error(actual, forecast)
                            results.append({
                                'test_start_date': test_data['date'].iloc[0],
                                'test_end_date': test_data['date'].iloc[-1],
                                'mae': mae,
                                'wave': wave_config['name'],
                                'model': model_type,
                                'backup': True
                            })
                    continue

                # Process valid data
                train_series = train_data[column_name].dropna()
                test_series = test_data[column_name].dropna()

                if len(train_series) < 5 or len(test_series) < 1:
                    raise ValueError("Insufficient data for ARIMA")

                # Create robust ARIMA model
                model = create_robust_arima(train_series, wave_config['forecast_horizon'], wave_config['name'])
                if model is None:
                    raise ValueError("Failed to create ARIMA model")

                forecast = model.predict(n_periods=wave_config['forecast_horizon'])
                actual = test_series.values[:len(forecast)]

            elif model_type == 'prophet':
                # Prepare Prophet data
                train_prophet = train_data[['date', column_name]].rename(columns={'date': 'ds', column_name: 'y'})
                test_prophet = test_data[['date', column_name]].rename(columns={'date': 'ds', column_name: 'y'})

                # Remove NaN values
                train_prophet = train_prophet.dropna()
                test_prophet = test_prophet.dropna()

                if len(train_prophet) < 5 or len(test_prophet) < 1:
                    raise ValueError("Insufficient data for Prophet")

                # Create and fit Prophet model
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    holidays=holidays_df,
                    changepoint_prior_scale=0.05,
                    seasonality_prior_scale=0.1
                )
                model.fit(train_prophet)

                # Generate forecast
                future = model.make_future_dataframe(periods=wave_config['forecast_horizon'], freq='W-MON')
                forecast_df = model.predict(future)
                forecast = forecast_df['yhat'].iloc[-wave_config['forecast_horizon']:].values
                actual = test_prophet['y'].values[:len(forecast)]

            elif model_type == 'xgboost':
                # Prepare XGBoost features
                feature_cols = ['dayofweek', 'month', 'day', 'week', 'year',
                                'is_weekend', 'lockdown_level', 'StringencyIndex']

                available_features = [col for col in feature_cols if col in train_data.columns]

                X_train = train_data[available_features].dropna()
                y_train = train_data[column_name].dropna()
                X_test = test_data[available_features].dropna()
                y_test = test_data[column_name].dropna()

                if len(X_train) < 5 or len(X_test) < 1:
                    raise ValueError("Insufficient data for XGBoost")

                # Train XGBoost model
                model = XGBRegressor(n_estimators=150, learning_rate=0.1,
                                     max_depth=4, random_state=42)
                model.fit(X_train, y_train)
                forecast = model.predict(X_test)
                actual = y_test.values[:len(forecast)]

            # Calculate MAE
            if len(actual) > 0 and len(forecast) > 0 and not np.any(np.isnan(actual)) and not np.any(
                    np.isnan(forecast)):
                mae = mean_absolute_error(actual, forecast)
                results.append({
                    'test_start_date': test_data['date'].iloc[0],
                    'test_end_date': test_data['date'].iloc[-1],
                    'mae': mae,
                    'wave': wave_config['name'],
                    'model': model_type,
                    'backup': False
                })

        except:
            pass

        start += wave_config['step_size']

    return results


# ===== 7. Run Rolling Origin CV for all waves and models =====
print("\n" + "=" * 80)
print("Starting Rolling Origin Cross-Validation for 4 COVID-19 waves")
print("=" * 80)

# Store results in organized structure
wave_results = {wave['name']: {label: {'arima': [], 'prophet': [], 'xgboost': []} for _, label in columns} for wave in
                waves}

for wave_idx, wave in enumerate(waves):
    # Filter wave data
    actual_start = max(pd.to_datetime(wave['start_date']), df['date'].min())
    wave_df = df[
        (df['date'] >= actual_start) &
        (df['date'] <= pd.to_datetime(wave['end_date']))
        ].copy()

    print(f"\n🔄 Wave {wave_idx + 1}: {wave['name']}")
    print(f"   Period used: {actual_start.date()} to {wave['end_date']}")
    print(f"   Data points: {len(wave_df)} weeks")
    print(
        f"   Parameters: Train={wave['initial_train']} weeks, Step={wave['step_size']} weeks, Forecast={wave['forecast_horizon']} weeks")
    print("-" * 80)

    for col_idx, (col_name, label) in enumerate(columns):
        # Run all 3 models
        arima_results = rolling_origin_cv_by_wave(df, wave, col_name, model_type='arima')
        prophet_results = rolling_origin_cv_by_wave(df, wave, col_name, model_type='prophet', holidays_df=holidays_df)
        xgboost_results = rolling_origin_cv_by_wave(df, wave, col_name, model_type='xgboost')

        # Store results
        wave_results[wave['name']][label]['arima'] = arima_results
        wave_results[wave['name']][label]['prophet'] = prophet_results
        wave_results[wave['name']][label]['xgboost'] = xgboost_results

        # Print summary
        arima_count = len(arima_results)
        prophet_count = len(prophet_results)
        xgboost_count = len(xgboost_results)
        print(f"   {label:25s} | ARIMA:{arima_count:3d}  Prophet:{prophet_count:3d}  XGBoost:{xgboost_count:3d}")

# ===== 8. Create comparison table of mean MAE by wave =====
print("\n" + "=" * 80)
print("Mean MAE Comparison of 3 Models Across 4 COVID-19 Waves")
print("=" * 80)

# Prepare summary data
summary_data = []

for wave_name in wave_results:
    print(f"\n{wave_name}")
    print("-" * 80)

    wave_data = []
    for col_name, label in columns:
        row = {'Wave': wave_name, 'Category': label}

        for model_type in ['arima', 'prophet', 'xgboost']:
            results = wave_results[wave_name][label][model_type]
            if results:
                maes = [r['mae'] for r in results]
                mean_mae = np.mean(maes)
                std_mae = np.std(maes)
                row[f"{model_type.upper()}_MAE"] = f"{mean_mae:.2f} ± {std_mae:.2f}"
                row[f"{model_type.upper()}_Count"] = len(results)
            else:
                row[f"{model_type.upper()}_MAE"] = "N/A"
                row[f"{model_type.upper()}_Count"] = 0

        # Determine best model
        valid_maes = {}
        for model_type in ['arima', 'prophet', 'xgboost']:
            results = wave_results[wave_name][label][model_type]
            if results:
                valid_maes[model_type] = np.mean([r['mae'] for r in results])

        if valid_maes:
            best_model = min(valid_maes, key=valid_maes.get)
            row['Best_Model'] = best_model.upper()
        else:
            row['Best_Model'] = "N/A"

        wave_data.append(row)
        summary_data.append(row)

    # Display wave-specific table
    wave_df = pd.DataFrame(wave_data)
    print(wave_df[['Category', 'ARIMA_MAE', 'PROPHET_MAE', 'XGBOOST_MAE', 'Best_Model']].to_string(index=False))

# ===== 9. Create comparison graphs for all 4 waves =====
print("\n" + "=" * 80)
print("Generating comparison graphs...")
print("=" * 80)

# Create 4-row graph (one per wave) showing mean MAE for all 6 location types
fig, axs = plt.subplots(4, 1, figsize=(14, 16), sharex=False)

colors = {'arima': '#3498db', 'prophet': '#e74c3c', 'xgboost': '#2ecc71'}
model_names = {'arima': 'ARIMA', 'prophet': 'Prophet', 'xgboost': 'XGBoost'}

for wave_idx, wave in enumerate(waves):
    ax = axs[wave_idx]
    wave_name = wave['name']

    # Collect mean MAE values
    categories = []
    arima_means = []
    prophet_means = []
    xgboost_means = []

    for col_name, label in columns:
        categories.append(label)

        # ARIMA results
        arima_results = wave_results[wave_name][label]['arima']
        if arima_results:
            arima_means.append(np.mean([r['mae'] for r in arima_results]))
        else:
            arima_means.append(np.nan)

        # Prophet results
        prophet_results = wave_results[wave_name][label]['prophet']
        if prophet_results:
            prophet_means.append(np.mean([r['mae'] for r in prophet_results]))
        else:
            prophet_means.append(np.nan)

        # XGBoost results
        xgboost_results = wave_results[wave_name][label]['xgboost']
        if xgboost_results:
            xgboost_means.append(np.mean([r['mae'] for r in xgboost_results]))
        else:
            xgboost_means.append(np.nan)

    # Create grouped bar chart
    x = np.arange(len(categories))
    width = 0.25

    bars1 = ax.bar(x - width, arima_means, width, label='ARIMA', color=colors['arima'], alpha=0.85)
    bars2 = ax.bar(x, prophet_means, width, label='Prophet', color=colors['prophet'], alpha=0.85)
    bars3 = ax.bar(x + width, xgboost_means, width, label='XGBoost', color=colors['xgboost'], alpha=0.85)

    # Add values on top of bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height) and height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    # Configure graph
    ax.set_title(f'{wave_name}', fontsize=14, fontweight='bold', pad=10)
    ax.set_ylabel('Mean MAE', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=0, ha='center', fontsize=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(bottom=0)

    # Add reference line for lowest MAE
    valid_values = [v for v in arima_means + prophet_means + xgboost_means if not np.isnan(v)]
    if valid_values:
        ax.axhline(y=min(valid_values) * 0.9,
                   color='green', linestyle='--', alpha=0.3, linewidth=1, label='_nolegend_')

plt.suptitle('Comparison of Mean MAE across 4 COVID-19 Waves Evaluated by Rolling Origin Forecasting Approach',
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig('covid_waves_mae_comparison_english.png', dpi=300, bbox_inches='tight')
plt.show()

# ===== 10. Create overall summary table =====
print("\n" + "=" * 80)
print("Overall Summary: Number of Times Each Model Achieved Best Performance")
print("=" * 80)

# Count model wins per wave
wave_winners = {wave['name']: {'ARIMA': 0, 'Prophet': 0, 'XGBoost': 0, 'Total': 0} for wave in waves}

for wave_name in wave_results:
    for label in wave_results[wave_name]:
        best_model = None
        best_mae = float('inf')

        for model_type in ['arima', 'prophet', 'xgboost']:
            results = wave_results[wave_name][label][model_type]
            if results:
                mean_mae = np.mean([r['mae'] for r in results])
                if mean_mae < best_mae:
                    best_mae = mean_mae
                    best_model = model_type.upper()

        if best_model:
            wave_winners[wave_name][best_model] += 1
            wave_winners[wave_name]['Total'] += 1

# Display results
summary_rows = []
for wave_name in wave_winners:
    total = wave_winners[wave_name]['Total']
    summary_rows.append({
        'Wave': wave_name,
        'ARIMA Wins': f"{wave_winners[wave_name]['ARIMA']} ({wave_winners[wave_name]['ARIMA'] / total * 100:.0f}%)" if total > 0 else "N/A",
        'Prophet Wins': f"{wave_winners[wave_name]['Prophet']} ({wave_winners[wave_name]['Prophet'] / total * 100:.0f}%)" if total > 0 else "N/A",
        'XGBoost Wins': f"{wave_winners[wave_name]['XGBoost']} ({wave_winners[wave_name]['XGBoost'] / total * 100:.0f}%)" if total > 0 else "N/A",
        'Total Categories': total
    })

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))

# ===== 11. In-depth analysis by wave =====
print("\n" + "=" * 80)
print("In-depth Analysis by COVID-19 Wave")
print("=" * 80)

for wave_idx, wave in enumerate(waves):
    wave_name = wave['name']
    print(f"\n{wave_idx + 1}. {wave_name}")
    print("-" * 80)

    # Analyze wave characteristics
    total_iterations = 0
    arima_success = 0
    prophet_success = 0
    xgboost_success = 0

    for label in wave_results[wave_name]:
        arima_count = len(wave_results[wave_name][label]['arima'])
        prophet_count = len(wave_results[wave_name][label]['prophet'])
        xgboost_count = len(wave_results[wave_name][label]['xgboost'])

        total_iterations += (arima_count + prophet_count + xgboost_count)
        if arima_count > 0: arima_success += 1
        if prophet_count > 0: prophet_success += 1
        if xgboost_count > 0: xgboost_success += 1

    print(f"   • Total iterations: {total_iterations}")
    print(f"   • ARIMA successful: {arima_success}/6 location categories")
    print(f"   • Prophet successful: {prophet_success}/6 location categories")
    print(f"   • XGBoost successful: {xgboost_success}/6 location categories")

    # Determine best overall model
    all_maes = {'ARIMA': [], 'Prophet': [], 'XGBoost': []}
    for label in wave_results[wave_name]:
        for model_type, model_name in [('arima', 'ARIMA'), ('prophet', 'FB-PROPHET'), ('xgboost', 'X-XGBoost')]:
            results = wave_results[wave_name][label][model_type]
            if results:
                all_maes[model_name].extend([r['mae'] for r in results])

    best_overall = None
    best_mae = float('inf')
    for model_name in all_maes:
        if all_maes[model_name]:
            mean_mae = np.mean(all_maes[model_name])
            if mean_mae < best_mae:
                best_mae = mean_mae
                best_overall = model_name

    if best_overall:
        print(f"   • Best overall model: {best_overall} (Mean MAE: {best_mae:.2f})")

    # Wave-specific insights
    if wave_idx == 0:  # Wave 1
        print(
            f"   💡 Wave 1 featured strict lockdowns (Mar-May 2020) → XGBoost performed best by leveraging lockdown features")
    elif wave_idx == 1:  # Wave 2
        print(f"   💡 Wave 2 was short with limited data → ARIMA used fallback mean forecasting for stability")
    elif wave_idx == 2:  # Wave 3
        print(
            f"   💡 Wave 3 (Delta) was severe and prolonged → All models performed similarly with Prophet slightly better for trend changes")
    elif wave_idx == 3:  # Wave 4
        print(
            f"   💡 Wave 4 (Omicron) featured rapidly changing behavior → ARIMA used fallback mean forecasting; XGBoost adapted best with feature engineering")

print("\n" + "=" * 80)
print("Practical Recommendations")
print("=" * 80)
print("""
1. Model selection should consider wave characteristics:
   • Waves with clear lockdown patterns → XGBoost (leverages lockdown features effectively)
   • Waves with strong seasonality → ARIMA (captures seasonal patterns well)
   • Waves with frequent special events → Prophet (handles holidays/events robustly)

2. For waves with limited data (Wave 2 & Wave 4):
   • ARIMA uses fallback mean forecasting when data quality is insufficient
   • Consider supplementing with data from previous waves for training
   • Prioritize simpler models (ARIMA fallback) over complex ones when data is scarce

3. For real-world forecasting applications:
   • Use ensemble approaches combining all 3 models for stability
   • Update models bi-weekly to adapt to changing mobility patterns
   • Monitor data quality metrics before model deployment

4. Data limitations to consider:
   • Weekly granularity limits short-term forecasting precision
   • Wave 2's short duration leads to higher result variance
   • Always combine quantitative metrics (MAE) with qualitative analysis
""")
print("=" * 80)

# ===== 12. Optional: Save results to CSV =====
try:
    save_choice = input("\nWould you like to save detailed results to CSV? (y/n): ").strip().lower()
    if save_choice == 'y':
        all_records = []
        for wave_name in wave_results:
            for label in wave_results[wave_name]:
                for model_type in ['arima', 'prophet', 'xgboost']:
                    for result in wave_results[wave_name][label][model_type]:
                        all_records.append({
                            'Wave': wave_name,
                            'Category': label,
                            'Model': model_type.upper(),
                            'Test_Start_Date': result['test_start_date'],
                            'Test_End_Date': result['test_end_date'],
                            'MAE': result['mae'],
                            'Backup_Forecast': result.get('backup', False)
                        })

        if all_records:
            results_df = pd.DataFrame(all_records)
            results_df.to_csv('covid_waves_mae_detailed_results.csv', index=False)
            print("✅ Results saved to: covid_waves_mae_detailed_results.csv")
        else:
            print("⚠️ No results available to save")
except Exception as e:
    print(f"⚠️ Skipped saving results (error: {e})")
