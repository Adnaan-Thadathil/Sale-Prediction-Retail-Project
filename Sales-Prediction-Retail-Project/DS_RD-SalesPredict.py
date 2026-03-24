import subprocess
import sys

required_packages = [
    "pandas",
    "numpy",
    "matplotlib",
    "statsmodels",
    "scikit-learn"
]

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"{package} not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", package])

print("All packages are installed.\n")

# 1. Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.metrics import mean_absolute_error, mean_squared_error

# 2. load data
df = pd.read_csv("Data/Online-Retail.csv", encoding='ISO-8859-1')

# Convert date column
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# 3. Data Cleaning

# Remove missing Customer IDs
df = df.dropna(subset=['CustomerID'])

# Remove cancelled orders
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]

# Remove negative or zero quantities
df = df[df['Quantity'] > 0]

# Create total price column
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# 4. Create Time Series
df.set_index('InvoiceDate', inplace=True)

daily_sales = df['TotalPrice'].resample('D').sum()

# Fill missing dates
daily_sales = daily_sales.fillna(0)

# 5. Visualiation
plt.figure(figsize=(12,6))
plt.plot(daily_sales)
plt.title("Daily Sales")
plt.show()

# 6. Check Stationarity
def adf_test(series):
    result = adfuller(series)
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])

adf_test(daily_sales)

# =========================
# 7. DIFFERENCING (if needed)
# =========================
daily_sales_diff = daily_sales.diff().dropna()

adf_test(daily_sales_diff)

#8. ACF and PACF
plot_acf(daily_sales_diff)
plot_pacf(daily_sales_diff)
plt.show()

#9. Train-Test Split
train = daily_sales[:'2011-10-31']
test = daily_sales['2011-11-01':]

# 10. ARIMA Model
model = ARIMA(train, order=(1,1,1))
model_fit = model.fit()

print(model_fit.summary())

# 11. Forecasting
forecast = model_fit.forecast(steps=len(test))

# 12. Evaluation
mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))

print("MAE:", mae)
print("RMSE:", rmse)

# 13. Plotting Forecast vs Actual

# a) Forecast vs Actual
plt.figure(figsize=(12,6))
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.plot(test.index, forecast, label='Forecast', color='red')
plt.title("ARIMA Forecast vs Actual Sales")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.legend()
plt.show()

# b) Residuals (errors)
residuals = test - forecast
plt.figure(figsize=(12,5))
plt.plot(test.index, residuals, color='purple')
plt.axhline(y=0, color='black', linestyle='--')
plt.title("Residuals (Actual - Forecast)")
plt.xlabel("Date")
plt.ylabel("Residuals")
plt.show()

# c) Histogram of Residuals
plt.figure(figsize=(8,5))
plt.hist(residuals, bins=30, color='orange', edgecolor='black')
plt.title("Residual Distribution")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.show()

# 14. Seasonal Decomposition
decomposition = seasonal_decompose(daily_sales, model='additive')
decomposition.plot()
plt.show()