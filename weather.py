import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

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
ave_humidity_figure = px.line(data, x = "date",
                            y = "humidity",
                            title = "Average Humidity in Delhi Over the Years")
ave_humidity_figure.show()

# plotting average wind speed over the years
avg_wind_speed_figure = px.line(data, x = "date",
                                y = "wind_speed",
                                title = "Average Wind Speed in Delhi Over the Years")
avg_wind_speed_figure.show()

# plotting relationship between average temperature and average humidity
temp_humidity = px.scatter(data_frame = data, x = "humidity",
                                y = "meantemp", size = "meantemp",
                                trendline = "ols",
                                title = "Relationship between Average Temperature and Humidity")
temp_humidity.show()

