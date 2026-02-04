# นำเข้าห้องสมุดที่จำเป็นสำหรับการจัดการข้อมูล, การวิเคราะห์, และการพล็อตกราฟ
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from pmdarima import auto_arima  # สำหรับการสร้างโมเดล ARIMA อัตโนมัติ

# ปิด warning ที่ไม่จำเป็นเพื่อไม่ให้แสดงผลขณะรัน
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# แสดงข้อความในคอนโซลว่าเป็นข้อมูลของประเทศไทย
print('Thailand')

# ตั้งค่ารูปแบบของ seaborn สำหรับกราฟให้มีความสวยงาม
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=0.9)

# โหลดชุดข้อมูล Google Mobility Report
df = pd.read_csv('data.csv')

# แปลงคอลัมน์วันที่ให้เป็น datetime object เพื่อใช้ในการวิเคราะห์ช่วงเวลา
df['date'] = pd.to_datetime(df['date'], dayfirst=True)

# กรองข้อมูลให้อยู่ในช่วงวันที่ 6 มี.ค. 2020 ถึง 15 ต.ค. 2022
df = df[(df['date'] >= '2020-03-06') & (df['date'] <= '2022-10-15')]

# 🔹 (Optional) รวมข้อมูลผู้ติดเชื้อ หากมีไฟล์ (ในที่นี้ถูกคอมเมนต์ไว้)
# cases_df = pd.read_csv('/Users/aritath/PyCharmMiscProject/covid_cases.csv')
# cases_df['date'] = pd.to_datetime(cases_df['date'])
# df = df.merge(cases_df[['date', 'cases']], o='date', how='left')

# แบ่งข้อมูลออกเป็น training และ testing โดยใช้ข้อมูล 123 แถวแรกเป็น training
train_df = df.iloc[:123]
test_df = df.iloc[122:]  # เริ่มจากแถว 122 เพื่อให้มีจุดต่อเนื่องจาก train

# รายชื่อคอลัมน์ที่ต้องการทำการพยากรณ์ พร้อมชื่อสำหรับแสดงผล
columns = [
    ('retail_and_recreation_percent_change_from_baseline', 'Retail & Recreation'),
    ('grocery_and_pharmacy_percent_change_from_baseline', 'Grocery & Pharmacy'),
    ('parks_percent_change_from_baseline', 'Parks'),
    ('transit_stations_percent_change_from_baseline', 'Transit Stations'),
    ('workplaces_percent_change_from_baseline', 'Workplaces'),
    ('residential_percent_change_from_baseline', 'Residential')
]

# นำเข้าฟังก์ชันประเมินผลเพิ่มเติม
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_log_error, mean_absolute_percentage_error

# ฟังก์ชันที่ใช้สร้างโมเดล ARIMA, ทำนายค่า และวาดกราฟเปรียบเทียบผล
def plot_forecast(train_df, test_df, column_name, label, ax):
    # แยกข้อมูล series ออกเป็น training และ testing ตามคอลัมน์ที่สนใจ
    train_series = train_df[column_name]
    test_series = test_df[column_name]

    # สร้างโมเดล Auto ARIMA แบบ seasonal โดยมีความถี่ 7 วัน (weekly seasonality)
    model = auto_arima(train_series, seasonal=True, m=7, trace=False, suppress_warnings=True)

    # ทำนายค่าของช่วง test จากโมเดล
    forecast = model.predict(n_periods=len(test_series))

    # คำนวณค่าประเมินผล MAE และ RMSE
    mae = mean_absolute_error(test_series, forecast)
    rmse = np.sqrt(mean_squared_error(test_series, forecast))

    # คำนวณค่า MAPE ถ้าไม่มีค่า 0 ในชุดข้อมูลจริง
    if np.any(test_series == 0):
        mape = np.nan
    else:
        mape = mean_absolute_percentage_error(test_series, forecast)

    # แสดงค่าการประเมินผลในคอนโซล
    print(f"\n--- {label} ---")
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.4f}")

    # วาดกราฟแสดงค่า Actual และ Forecast
    sns.lineplot(x=train_df['date'], y=train_series, label=f'Actual {label} (Train)', ax=ax, color='#50C878')
    sns.lineplot(x=test_df['date'], y=test_series, label=f'Actual {label} (Test)', ax=ax, color='blue')
    sns.lineplot(x=test_df['date'], y=forecast, label=f'Forecast {label}', ax=ax, color='red')

    # ปรับค่าการแสดงผลของแกน X
    ax.set_xlabel('Date')
    ax.set_ylabel(f'{label} % Change')
    ax.set_title(f'{label} Mobility Forecast (ARIMA)')

    # ตั้งค่ารูปแบบของวันที่ในแกน X
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))
    ax.tick_params(axis='x', rotation=45)

    # คำสั่งสำหรับแสดงผลประเมินในกราฟ (ปัจจุบันถูกคอมเมนต์ไว้)
    # ax.text(0.95, 0.25,
    #         f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nMAPE: {mape:.2f}",
    #         ha='right', va='bottom', transform=ax.transAxes, fontsize=8,
    #         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

# สร้างกริดขนาด 2 แถว 3 คอลัมน์ สำหรับวาดกราฟทั้ง 6 ชุด
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

# ลูปแต่ละตัวแปร mobility แล้ววาดกราฟด้วยฟังก์ชัน plot_forecast
for i, (column_name, label) in enumerate(columns):
    row = i // 3  # แถวที่กราฟอยู่
    col = i % 3   # คอลัมน์ที่กราฟอยู่
    plot_forecast(train_df, test_df, column_name, label, axs[row, col])

# จัด layout ให้พอดี และแสดงผลกราฟทั้งหมด
plt.tight_layout()
plt.show()
