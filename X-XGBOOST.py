# นำเข้าชุดไลบรารีที่จำเป็น
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
from datetime import datetime
import holidays

# ตั้งค่ารูปแบบกราฟ
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=0.9)
plt.rcParams['font.family'] = 'sans-serif'

# โหลดข้อมูล
df = pd.read_csv('data.csv')
df['date'] = pd.to_datetime(df['date'], dayfirst=True)
df = df[(df['date'] >= '2020-03-06') & (df['date'] <= '2022-10-15')]

# ===== เพิ่มฟีเจอร์ผู้ติดเชื้อ =====
df['cases_per_million'] = df['Daily new confirmed cases of COVID-19 per million people'].fillna(0)

# ===== เพิ่มฟีเจอร์ StringencyIndex =====
df['StringencyIndex'] = df['StringencyIndex'].fillna(0)


# ===== สร้างฟังก์ชัน assign_lockdown =====
def assign_lockdown(date):
    if datetime(2020, 3, 26) <= date <= datetime(2020, 4, 30):
        return 5
    elif datetime(2020, 5, 1) <= date <= datetime(2020, 6, 30):
        return 4
    elif datetime(2021, 1, 1) <= date <= datetime(2021, 1, 31):
        return 3
    elif datetime(2021, 7, 12) <= date <= datetime(2021, 8, 31):
        return 5
    elif datetime(2021, 9, 1) <= date <= datetime(2021, 9, 30):
        return 4
    else:
        return 1


df['lockdown_level'] = df['date'].apply(assign_lockdown)

# รายชื่อคอลัมน์ mobility
columns = [
    ('retail_and_recreation_percent_change_from_baseline', 'Retail & Recreation'),
    ('grocery_and_pharmacy_percent_change_from_baseline', 'Grocery & Pharmacy'),
    ('parks_percent_change_from_baseline', 'Parks'),
    ('transit_stations_percent_change_from_baseline', 'Transit Stations'),
    ('workplaces_percent_change_from_baseline', 'Workplaces'),
    ('residential_percent_change_from_baseline', 'Residential')
]

mobility_cols = [col for col, _ in columns]

# ===== เพิ่ม Global Mobility Average =====
df['avg_all_mobility'] = df[mobility_cols].mean(axis=1)

# ===== เพิ่ม Public Holidays (ปี 2020–2022) =====
th_holidays = holidays.TH(years=[2020, 2021, 2022])
df['is_holiday'] = df['date'].apply(lambda x: x in th_holidays).astype(int)


# ===== สร้างฟีเจอร์ด้านเวลา =====
def create_features(df):
    df = df.copy()
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['week'] = df['date'].dt.isocalendar().week.astype(int)
    df['year'] = df['date'].dt.year
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    return df


df = create_features(df)

# ===== แบ่ง train/test =====
train_df = df.iloc[:123]
test_df = df.iloc[122:]


# ===== ฟังก์ชันเทรนและเก็บผลลัพธ์ =====
def train_and_evaluate(train_df, test_df, column_name, label):
    features = [
        'dayofweek', 'month', 'day', 'week', 'year', 'is_weekend',
        'lockdown_level', 'StringencyIndex',
        'is_holiday',
        'avg_all_mobility'
    ]

    X_train = train_df[features]
    y_train = train_df[column_name]
    X_test = test_df[features]
    y_test = test_df[column_name]
    dates = test_df['date']

    model = XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # คำนวณ metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred) if not np.any(y_test == 0) else np.nan

    # แสดงผลใน console - แก้ไขการจัดรูปแบบเพื่อหลีกเลี่ยงข้อผิดพลาด
    mape_str = f"{mape:.4f}" if not np.isnan(mape) else "N/A"
    print(f"\n--- {label} ---")
    print(f"{'Date':<12} {'Actual':>10} {'Forecast':>12} {'Error':>10}")
    print("-" * 45)
    for date, actual, forecast_val in zip(dates, y_test, y_pred):
        error = actual - forecast_val
        print(f"{pd.to_datetime(date).date()} {actual:10.2f} {forecast_val:12.2f} {error:10.2f}")
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape_str}")

    # ดึงทั้ง 3 ประเภทของ feature importance
    booster = model.get_booster()
    weight = booster.get_score(importance_type='weight')
    gain = booster.get_score(importance_type='gain')
    cover = booster.get_score(importance_type='cover')

    # normalize to percentages
    total_weight = sum(weight.values()) or 1
    total_gain = sum(gain.values()) or 1
    total_cover = sum(cover.values()) or 1

    weight_pct = [weight.get(f, 0) / total_weight * 100 for f in features]
    gain_pct = [gain.get(f, 0) / total_gain * 100 for f in features]
    cover_pct = [cover.get(f, 0) / total_cover * 100 for f in features]

    importance_dict = {
        'feature': features,
        'weight_pct': weight_pct,
        'gain_pct': gain_pct,
        'cover_pct': cover_pct,
        'category': label
    }

    return y_pred, importance_dict, model


# ===== สร้างกราฟ mobility forecast =====
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
all_importance_data = []

for i, (column_name, label) in enumerate(columns):
    row = i // 3
    col = i % 3
    y_pred, importance_dict, _ = train_and_evaluate(train_df, test_df, column_name, label)
    all_importance_data.append(importance_dict)

    # พล็อตกราฟเดิม
    ax = axs[row, col]
    sns.lineplot(x=train_df['date'], y=train_df[column_name], label=f'Actual {label} (Train)', ax=ax, color='#50C878')
    sns.lineplot(x=test_df['date'], y=test_df[column_name], label=f'Actual {label} (Test)', ax=ax, color='blue')
    sns.lineplot(x=test_df['date'], y=y_pred, label=f'Forecast {label}', ax=ax, color='red')

    ax.set_xlabel('Date')
    ax.set_ylabel(f'{label} % Change')
    ax.set_title(f'{label} Mobility Forecast (X-XGBoost)')
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('mobility_forecasts.png', dpi=300, bbox_inches='tight')
plt.show()

# ===== รวมข้อมูล importance ทั้งหมดเป็น DataFrame =====
importance_df = pd.DataFrame()
for data in all_importance_data:
    df_temp = pd.DataFrame({
        'Feature': data['feature'],
        'Category': data['category'],
        'Weight (%)': data['weight_pct'],
        'Gain (%)': data['gain_pct'],
        'Cover (%)': data['cover_pct']
    })
    importance_df = pd.concat([importance_df, df_temp], ignore_index=True)

#===== สร้างกราฟรวมทั้ง 3 ประเภทในรูปเดียว เรียงแนวตั้ง =====
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 14), sharex=False)

# เตรียมข้อมูลสำหรับกราฟ
categories = [label for _, label in columns]
feature_order = importance_df.groupby('Feature')['Gain (%)'].mean().sort_values(ascending=False).index.tolist()
x = np.arange(len(feature_order))
width = 0.12  # ความกว้างของแต่ละแท่ง

# สีสำหรับแต่ละหมวดหมู่ (6 สี)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# คำนวณค่าสูงสุดสำหรับแต่ละเมตริกเพื่อปรับสเกลกราฟให้เหมาะสม
max_weight = importance_df['Weight (%)'].max() * 1.15  # เพิ่ม padding 15%
max_gain = importance_df['Gain (%)'].max() * 1.15
max_cover = importance_df['Cover (%)'].max() * 1.15

# Subplot 1: Weight (บนสุด)
for i, category in enumerate(categories):
    cat_data = importance_df[importance_df['Category'] == category]
    values = [cat_data[cat_data['Feature'] == f]['Weight (%)'].values[0] if f in cat_data['Feature'].values else 0
              for f in feature_order]
    ax1.bar(x + i * width, values, width, label=category, color=colors[i])
ax1.set_ylabel('Weight (%)', fontsize=12, fontweight='bold')
ax1.set_title('Weight: Frequency of Feature Usage in Trees', fontsize=14, fontweight='bold', pad=10)
ax1.set_xticks(x + width * (len(categories) - 1) / 2)
ax1.set_xticklabels(feature_order, rotation=0, ha='center', fontsize=10)  # ✅ แก้เป็น center
ax1.legend(title='Mobility Category', fontsize=9, title_fontsize=10, loc='upper right', ncol=2)
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(0, max_weight)  # ✅ ปรับสเกลตามข้อมูลจริง

# Subplot 2: Gain (ตรงกลาง - เน้นเป็นพิเศษ)
for i, category in enumerate(categories):
    cat_data = importance_df[importance_df['Category'] == category]
    values = [cat_data[cat_data['Feature'] == f]['Gain (%)'].values[0] if f in cat_data['Feature'].values else 0
              for f in feature_order]
    ax2.bar(x + i * width, values, width, label=category, color=colors[i])
ax2.set_ylabel('Gain (%)', fontsize=12, fontweight='bold')
ax2.set_title('Gain: Contribution to Loss Reduction', fontsize=14, fontweight='bold',
            pad=10)
ax2.set_xticks(x + width * (len(categories) - 1) / 2)
ax2.set_xticklabels(feature_order, rotation=0, ha='center', fontsize=10)  # ✅ แก้เป็น center
ax2.legend(title='Mobility Category', fontsize=9, title_fontsize=10, loc='upper right', ncol=2)
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim(0, max_gain)  # ✅ ปรับสเกลตามข้อมูลจริง

# Subplot 3: Cover (ล่างสุด)
for i, category in enumerate(categories):
    cat_data = importance_df[importance_df['Category'] == category]
    values = [cat_data[cat_data['Feature'] == f]['Cover (%)'].values[0] if f in cat_data['Feature'].values else 0
              for f in feature_order]
    ax3.bar(x + i * width, values, width, label=category, color=colors[i])
ax3.set_xlabel('Feature', fontsize=13, fontweight='bold', labelpad=10)
ax3.set_ylabel('Cover (%)', fontsize=12, fontweight='bold')
ax3.set_title('Cover: Relative Number of Samples Affected', fontsize=14, fontweight='bold', pad=10)
ax3.set_xticks(x + width * (len(categories) - 1) / 2)
ax3.set_xticklabels(feature_order, rotation=0, ha='center', fontsize=10)  # ✅ แก้เป็น center
ax3.legend(title='Mobility Category', fontsize=9, title_fontsize=10, loc='upper right', ncol=2)
ax3.grid(axis='y', alpha=0.3)
ax3.set_ylim(0, max_cover)  # ✅ ปรับสเกลตามข้อมูลจริง

plt.suptitle('Feature Importance Across All Mobility Categories\n[Weight,Gain and Cover]',
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(h_pad=3.0)
plt.savefig('feature_importance_vertical_grouped_bar.png', dpi=300, bbox_inches='tight')
plt.show()
# ===== ส่งออกตารางสรุปแบบละเอียด =====
importance_summary = importance_df.groupby('Feature').agg({
    'Weight (%)': ['mean', 'std'],
    'Gain (%)': ['mean', 'std'],
    'Cover (%)': ['mean', 'std']
}).round(2).sort_values(('Gain (%)', 'mean'), ascending=False)

print("\n" + "=" * 90)
print("DETAILED FEATURE IMPORTANCE SUMMARY (Mean ± Std across all 6 mobility categories)")
print("=" * 90)
print(importance_summary.to_string())
print("=" * 90)

# บันทึกเป็น CSV
importance_df.to_csv('feature_importance_detailed.csv', index=False)
importance_summary.to_csv('feature_importance_summary.csv', index=True)

print("\n✅ Feature importance tables saved as:")
print("   - feature_importance_detailed.csv (raw Weight/Gain/Cover per category)")
print("   - feature_importance_summary.csv (averaged rankings by Gain)")
print("✅ Unified vertical grouped bar chart saved as: feature_importance_vertical_grouped_bar.png")
