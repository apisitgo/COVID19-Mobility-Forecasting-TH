# นำเข้าไลบรารีที่จำเป็นสำหรับการประมวลผลข้อมูลและการพยากรณ์
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from prophet import Prophet
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# แสดงข้อความ “Thailand” เพื่อบอกตำแหน่งหรือบริบทของข้อมูล
print('Thailand')

# ตั้งค่ารูปแบบของกราฟด้วย seaborn
sns.set_style("whitegrid")  # ใช้พื้นหลังแบบตารางสีขาว
sns.set_context("notebook", font_scale=0.9)  # ปรับขนาดตัวอักษรให้เหมาะสมกับโน้ตบุ๊ก

# โหลดข้อมูลจากไฟล์ CSV
df = pd.read_csv('data.csv')

# แปลงคอลัมน์วันที่ให้อยู่ในรูปแบบ datetime และระบุว่าฟอร์แมตเป็นวันก่อนเดือน
df['date'] = pd.to_datetime(df['date'], dayfirst=True)

# คัดกรองช่วงข้อมูลที่ต้องการวิเคราะห์ (ระหว่าง 6 มี.ค. 2020 ถึง 15 ต.ค. 2022)
df = df[(df['date'] >= '2020-03-06') & (df['date'] <= '2022-10-15')]

# กำหนดชื่อคอลัมน์ที่จะนำมาใช้ พร้อม label ที่จะใช้แสดงผลในกราฟ
columns = [
    ('retail_and_recreation_percent_change_from_baseline', 'Retail & Recreation'),
    ('grocery_and_pharmacy_percent_change_from_baseline', 'Grocery & Pharmacy'),
    ('parks_percent_change_from_baseline', 'Parks'),
    ('transit_stations_percent_change_from_baseline', 'Transit Stations'),
    ('workplaces_percent_change_from_baseline', 'Workplaces'),
    ('residential_percent_change_from_baseline', 'Residential')
]

# สร้าง DataFrame ของวันหยุดราชการในประเทศไทย (ปี 2020-2022)
# ซึ่งจะใช้เพื่อแจ้งให้ Prophet ทราบว่าอาจมีผลกระทบต่อพฤติกรรมของประชากร
holidays = pd.DataFrame([
    # ปี 2020
    {'holiday': 'New Year', 'ds': '2020-01-01', 'lower_window': -1, 'upper_window': 1},
    {'holiday': 'Makha Bucha', 'ds': '2020-02-08', 'lower_window': 0, 'upper_window': 0},
    {'holiday': 'Chakri Memorial Day', 'ds': '2020-04-06', 'lower_window': 0, 'upper_window': 0},
    {'holiday': 'Songkran', 'ds': '2020-04-13', 'lower_window': 0, 'upper_window': 2},
    {'holiday': 'Labour Day', 'ds': '2020-05-01', 'lower_window': 0, 'upper_window': 0},
    {'holiday': 'Coronation Day', 'ds': '2020-05-04', 'lower_window': 0, 'upper_window': 0},
    {'holiday': 'Visakha Bucha', 'ds': '2020-05-06', 'lower_window': 0, 'upper_window': 0},
    {'holiday': "Queen Suthida's Birthday", 'ds': '2020-06-03', 'lower_window': 0, 'upper_window': 0},
    {'holiday': "King Vajiralongkorn's Birthday", 'ds': '2020-07-28', 'lower_window': 0, 'upper_window': 0},
    {'holiday': "Mother's Day", 'ds': '2020-08-12', 'lower_window': 0, 'upper_window': 0},
    {'holiday': 'King Bhumibol Memorial Day', 'ds': '2020-10-13', 'lower_window': 0, 'upper_window': 0},
    {'holiday': 'Chulalongkorn Day', 'ds': '2020-10-23', 'lower_window': 0, 'upper_window': 0},
    {'holiday': "Father's Day", 'ds': '2020-12-05', 'lower_window': 0, 'upper_window': 0},
    {'holiday': 'Constitution Day', 'ds': '2020-12-10', 'lower_window': 0, 'upper_window': 0},
    {'holiday': 'New Year', 'ds': '2020-12-31', 'lower_window': 0, 'upper_window': 1},  # 31 ธ.ค. ต่อเนื่องปีใหม่

    # ปี 2021
    {'holiday': 'New Year', 'ds': '2021-01-01', 'lower_window': -1, 'upper_window': 1},
    {'holiday': 'Makha Bucha', 'ds': '2021-02-26', 'lower_window': 0, 'upper_window': 0},
    {'holiday': 'Chakri Memorial Day', 'ds': '2021-04-06', 'lower_window': 0, 'upper_window': 0},
    {'holiday': 'Songkran', 'ds': '2021-04-13', 'lower_window': 0, 'upper_window': 2},
    {'holiday': 'Labour Day', 'ds': '2021-05-01', 'lower_window': 0, 'upper_window': 0},
    {'holiday': 'Coronation Day', 'ds': '2021-05-04', 'lower_window': 0, 'upper_window': 0},
    {'holiday': 'Visakha Bucha', 'ds': '2021-05-26', 'lower_window': 0, 'upper_window': 0},
    {'holiday': "Queen Suthida's Birthday", 'ds': '2021-06-03', 'lower_window': 0, 'upper_window': 0},
    {'holiday': "King Vajiralongkorn's Birthday", 'ds': '2021-07-28', 'lower_window': 0, 'upper_window': 0},
    {'holiday': "Mother's Day", 'ds': '2021-08-12', 'lower_window': 0, 'upper_window': 0},
    {'holiday': 'King Bhumibol Memorial Day', 'ds': '2021-10-13', 'lower_window': 0, 'upper_window': 0},
    {'holiday': 'Chulalongkorn Day', 'ds': '2021-10-23', 'lower_window': 0, 'upper_window': 0},
    {'holiday': "Father's Day", 'ds': '2021-12-05', 'lower_window': 0, 'upper_window': 0},
    {'holiday': 'Constitution Day', 'ds': '2021-12-10', 'lower_window': 0, 'upper_window': 0},
    {'holiday': 'New Year', 'ds': '2021-12-31', 'lower_window': 0, 'upper_window': 1},

    # ปี 2022
    {'holiday': 'New Year', 'ds': '2022-01-01', 'lower_window': -1, 'upper_window': 1},
    {'holiday': 'Makha Bucha', 'ds': '2022-02-16', 'lower_window': 0, 'upper_window': 0},
    {'holiday': 'Chakri Memorial Day', 'ds': '2022-04-06', 'lower_window': 0, 'upper_window': 0},
    {'holiday': 'Songkran', 'ds': '2022-04-13', 'lower_window': 0, 'upper_window': 2},
    {'holiday': 'Labour Day', 'ds': '2022-05-01', 'lower_window': 0, 'upper_window': 0},
    {'holiday': 'Coronation Day', 'ds': '2022-05-04', 'lower_window': 0, 'upper_window': 0},
    {'holiday': 'Visakha Bucha', 'ds': '2022-05-15', 'lower_window': 0, 'upper_window': 0},
    {'holiday': "Queen Suthida's Birthday", 'ds': '2022-06-03', 'lower_window': 0, 'upper_window': 0},
    {'holiday': "King Vajiralongkorn's Birthday", 'ds': '2022-07-28', 'lower_window': 0, 'upper_window': 0},
    {'holiday': "Mother's Day", 'ds': '2022-08-12', 'lower_window': 0, 'upper_window': 0},
    {'holiday': 'King Bhumibol Memorial Day', 'ds': '2022-10-13', 'lower_window': 0, 'upper_window': 0},
])

# แปลงวันที่ในตารางวันหยุดให้อยู่ในรูปแบบ datetime
holidays['ds'] = pd.to_datetime(holidays['ds'])

# แบ่งข้อมูลออกเป็นชุด train และ test
train_df = df.iloc[:123]  # ใช้ 123 แถวแรกในการ train
test_df = df.iloc[122:]   # ใช้ข้อมูลที่เหลือในการทดสอบ (ทับซ้อนเล็กน้อย)

# ฟังก์ชันสำหรับการพยากรณ์และวาดกราฟของแต่ละตัวแปร
def plot_forecast(train_df, test_df, column_name, label, ax):
    # นำเข้า metric สำหรับวัดผล
    from sklearn.metrics import r2_score, explained_variance_score, mean_squared_log_error, mean_absolute_percentage_error

    # เตรียมข้อมูลให้อยู่ในรูปแบบที่ Prophet ต้องการ: คอลัมน์ 'ds' (วันที่) และ 'y' (ค่าที่จะพยากรณ์)
    train_df_prophet = train_df[['date', column_name]].rename(columns={'date': 'ds', column_name: 'y'})
    test_df_prophet = test_df[['date', column_name]].rename(columns={'date': 'ds', column_name: 'y'})

    # กำหนดค่าเริ่มต้นของโมเดล Prophet พร้อมเพิ่ม seasonality และวันหยุด
    model = Prophet(
        changepoint_prior_scale=0.01,
        seasonality_prior_scale=0.01,
        holidays_prior_scale=13.0,
        growth='linear',
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        holidays=holidays
    )

    # เพิ่มคลื่นความถี่ช่วงต่าง ๆ เพื่อจับ pattern ที่อาจเกิดขึ้นเป็นระยะ ๆ
    model.add_seasonality(name='wave_1', period=314, fourier_order=5, prior_scale=20.0)
    model.add_seasonality(name='wave_2', period=106, fourier_order=5, prior_scale=20.0)
    model.add_seasonality(name='wave_3', period=274, fourier_order=5, prior_scale=20.0)
    model.add_seasonality(name='wave_4', period=254, fourier_order=5, prior_scale=20.0)

    # เพิ่ม seasonality สำหรับช่วง lockdown ซึ่งอาจส่งผลต่อพฤติกรรมการเคลื่อนที่ของประชากร
    model.add_seasonality(name='Lockdown1', period=35, fourier_order=3, prior_scale=10.0)
    model.add_seasonality(name='Lockdown2', period=60, fourier_order=3, prior_scale=10.0)
    model.add_seasonality(name='Lockdown3', period=30, fourier_order=3, prior_scale=10.0)
    model.add_seasonality(name='Lockdown4', period=50, fourier_order=3, prior_scale=10.0)
    model.add_seasonality(name='Lockdown5', period=29, fourier_order=3, prior_scale=10.0)

    # Train โมเดลด้วยข้อมูลที่เตรียมไว้
    model.fit(train_df_prophet)

    # กำหนดจำนวนวันในอนาคตที่จะพยากรณ์ โดยอิงจากช่วง test
    periods = (test_df_prophet['ds'].max() - train_df_prophet['ds'].max()).days
    future = model.make_future_dataframe(periods=periods)
    future['cap'] = 100  # กำหนดค่าสูงสุด (ไม่ใช้งานจริงแต่ต้องมีเพื่อหลีกเลี่ยง error ในบาง config)

    # พยากรณ์อนาคต
    forecast = model.predict(future)

    # คัดเฉพาะค่าพยากรณ์ที่ตรงกับวันที่ของ test set
    forecast_test = forecast[forecast['ds'].isin(test_df_prophet['ds'])]

    # ตรวจสอบว่า test set และ forecast มีขนาดเท่ากัน
    if len(test_df_prophet) != len(forecast_test):
        raise ValueError(f"Length mismatch: actual={len(test_df_prophet)}, forecast={len(forecast_test)}")

    # สร้างตัวแปรสำหรับเก็บค่าจริงและค่าที่พยากรณ์ได้
    actual_values = test_df_prophet['y'].values
    forecast_values = forecast_test['yhat'].values
    dates = test_df_prophet['ds'].values

    # คำนวณค่า MAE, RMSE และ MAPE สำหรับประเมินประสิทธิภาพโมเดล
    mae = mean_absolute_error(actual_values, forecast_values)
    rmse = np.sqrt(mean_squared_error(actual_values, forecast_values))
    if np.any(actual_values == 0):
        mape = np.nan  # หลีกเลี่ยงการหารด้วยศูนย์
    else:
        mape = mean_absolute_percentage_error(actual_values, forecast_values)

    # พิมพ์ค่าที่ได้ออกมาทาง console
    print(f"\n--- {label} --- ")
    print(f"{'Date':<12} {'Actual':>10} {'Forecast':>12} {'Error':>10}")
    print("-" * 45)
    for date, actual, forecast_val in zip(dates, actual_values, forecast_values):
        error = actual - forecast_val
        print(f"{pd.to_datetime(date).date()} {actual:10.2f} {forecast_val:12.2f} {error:10.2f}")

    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.4f}")

    # วาดกราฟเปรียบเทียบค่า Actual และ Forecast
    sns.lineplot(x=train_df['date'], y=train_df[column_name], label=f'Actual {label} (Train)', ax=ax, color='#50C878')
    sns.lineplot(x=test_df['date'], y=test_df[column_name], label=f'Actual {label} (Test)', ax=ax, color='blue')
    sns.lineplot(x=forecast_test['ds'], y=forecast_test['yhat'], label=f'Forecast {label}', ax=ax, color='red')

    # ตั้งค่ารูปแบบกราฟ
    ax.set_xlabel('Date')
    ax.set_ylabel(f'{label} % Change')
    ax.set_title(f'{label} Mobility Forecast (FB-PROPHET)')

    # กำหนดรูปแบบของแกน X ให้แสดงเดือนทุก 4 เดือน
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))
    ax.tick_params(axis='x', rotation=45)

# สร้างกรอบรูปกราฟแบบ 2 แถว 3 คอลัมน์
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

# วาดกราฟสำหรับตัวแปรทั้ง 6 ตัว
for i, (column_name, label) in enumerate(columns):
    row = i // 3
    col = i % 3
    plot_forecast(train_df, test_df, column_name, label, axs[row, col])

# ปรับ layout ให้กราฟไม่ซ้อนกัน
plt.tight_layout()
plt.show()