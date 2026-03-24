# Sale-Prediction-Retail-Project

Sale Prediction For Retail
Sales Forecasting using ARIMA
Overview
	• Predict daily revenue using time series modeling 
Dataset
	• Online Retail Dataset 
Steps
	• Data cleaning 
	• EDA 
	• Stationarity testing 
	• ARIMA modeling 
	• Evaluation 
Results
	• MAE: X 
	• RMSE: X 

Summary of results.
After cleaning and aggregating the online retail dataset, we applied ARIMA(1,1,1) to forecast daily sales. The original series was non-stationary (ADF p = 0.78), so differencing was applied to achieve stationarity (ADF p ≈ 0). The AR(1) and MA(1) coefficients indicate that yesterday’s sales and previous forecast errors significantly affect today’s sales. Model accuracy metrics show MAE ≈ 21,866 and RMSE ≈ 32,530, meaning the forecasts are reasonably accurate given the scale of daily sales. These results suggest predictable short-term trends, which can inform inventory and sales planning. Limitations include inability to capture external events such as promotions or holidays.
