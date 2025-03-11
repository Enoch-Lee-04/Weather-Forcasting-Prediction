import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

data = pd.read_csv("DailyDelhiClimateTrain.csv")
print(data.head())

print(data.describe())

print(data.info())

# plotting average temperature over the years
avg_temp_figure = px.line(data, x = "date",
                y = "meantemp",
                title = "Mean Temperature in Delhi Over the Years")
avg_temp_figure.show()

# plotting average humidity over the years
avg_humidity_figure = px.line(data, x = "date",
                            y = "humidity",
                            title = "Average Humidity in Delhi Over the Years")
avg_humidity_figure.show()

# plotting average wind speed over the years
avg_wind_speed_figure = px.line(data, x = "date",
                                y = "wind_speed",
                                title = "Average Wind Speed in Delhi Over the Years")
avg_wind_speed_figure.show()

# # plotting relationship between average temperature and average humidity
# temp_humidity = px.scatter(data_frame = data, x = "humidity",
#                                 y = "meantemp", size = "meantemp",
#                                 trendline = "ols",
#                                 title = "Relationship between Average Temperature and Humidity")
# temp_humidity.show()

# change the data type to datetime
data['date'] = pd.to_datetime(data['date'], format = '%Y-%m-%d')
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
print(data.head())

# plotting temperature change in Delhi over the years
plt.style.use('fivethirtyeight')
plt.figure(figsize = (15, 10))
plt.title("Temperature Change in Delhi Over the Years")
sns.lineplot(data = data, x = 'month', y = 'meantemp', hue = 'year')
plt.show()

# converting the data to Prophet format
forecast_data = data.rename(columns = {'date': 'ds', 'meantemp': 'y'})

# importing Prophet
model = Prophet()
model.fit(forecast_data)
forecasts = model.make_future_dataframe(periods = 365)
predictions = model.predict(forecasts)
fig = plot_plotly(model, predictions)
fig.show()
