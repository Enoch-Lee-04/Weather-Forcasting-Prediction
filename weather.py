import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.diagnostics import cross_validation, performance_metrics
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load and explore the data
data = pd.read_csv("ATX 2015-01-01 to 2025-03-09.csv")

# Replace 'T' (trace) values with 0.01 in precipitation data
data['precip'] = data['precip'].replace('T', '0.01')
data['precip'] = pd.to_numeric(data['precip'])

print("Dataset Preview:")
print(data.head())

print("\nDataset Statistics:")
print(data.describe())

print("\nDataset Information:")
print(data.info())

# Data preprocessing
# Convert date to datetime format
data['date'] = pd.to_datetime(data['date'], format='%m/%d/%Y')
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day_of_week'] = data['date'].dt.dayofweek
data['day_of_year'] = data['date'].dt.dayofyear

# Check for missing values
print("\nMissing values in the dataset:")
print(data.isnull().sum())

# Fill missing values if any
data = data.fillna(method='ffill')

# Visualize temperature trends
print("\nGenerating temperature trend visualization...")
avg_temp_figure = px.line(data, x="date", y="avg_temp",
                title="Mean Temperature in Austin TX Over the Years")
avg_temp_figure.show()

# Visualize seasonal patterns
plt.style.use('fivethirtyeight')
plt.figure(figsize=(15, 10))
plt.title("Temperature Change in Austin TX Over the Years")
sns.lineplot(data=data, x='month', y='avg_temp', hue='year')
plt.show()

# Create monthly and yearly aggregations for analysis
monthly_data = data.groupby(['year', 'month']).agg({
    'avg_temp': 'mean',
    'max_temp': 'mean',
    'min_temp': 'mean',
    'precip': 'sum'
}).reset_index()

print("\nMonthly aggregated data:")
print(monthly_data.head())

# Visualize monthly temperature patterns
plt.figure(figsize=(15, 8))
sns.boxplot(data=data, x='month', y='avg_temp')
plt.title('Monthly Temperature Distribution in Austin TX')
plt.xlabel('Month')
plt.ylabel('Average Temperature')
plt.show()

# Prepare data for Prophet
# Add additional features for better forecasting
forecast_data = data.rename(columns={'date': 'ds', 'avg_temp': 'y'})

# Create a more advanced Prophet model
print("\nTraining Prophet model with enhanced parameters...")
model = Prophet(
    yearly_seasonality=20,  # More flexible yearly seasonality
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='multiplicative',  # Better for temperature data
    changepoint_prior_scale=0.05,  # Flexibility for trend changes
    seasonality_prior_scale=10.0,  # Stronger seasonality
)

# Add precipitation as a regressor if it helps prediction
model.add_regressor('precip')

# Add max and min temperatures as additional regressors
forecast_data['max_temp'] = data['max_temp']
forecast_data['min_temp'] = data['min_temp']
model.add_regressor('max_temp')
model.add_regressor('min_temp')

# Fit the model
model.fit(forecast_data)

# Cross-validation to evaluate model performance
print("\nPerforming cross-validation...")
df_cv = cross_validation(model, initial='730 days', period='180 days', horizon='365 days')
df_p = performance_metrics(df_cv)
print("\nModel performance metrics:")
print(df_p)

# Visualize cross-validation results
plt.figure(figsize=(10, 6))
plt.plot(df_p['horizon'], df_p['mae'], 'b-')
plt.xlabel('Forecast Horizon (days)')
plt.ylabel('Mean Absolute Error')
plt.title('Forecast Error by Horizon')
plt.grid(True)
plt.show()

# Make future predictions
print("\nGenerating forecasts...")
future = model.make_future_dataframe(periods=730)

# Add regressor values for future predictions
# For simplicity, we'll use historical averages for each day of the year
day_of_year_avg = data.groupby('day_of_year').agg({
    'precip': 'mean',
    'max_temp': 'mean',
    'min_temp': 'mean'
}).reset_index()

# Map the averages to future dataframe
future['day_of_year'] = pd.to_datetime(future['ds']).dt.dayofyear
future = future.merge(day_of_year_avg, on='day_of_year', how='left')

# Make predictions
forecast = model.predict(future)

# Evaluate model on historical data
historical_forecast = forecast[forecast['ds'].isin(data['date'])]
historical_data = data[data['date'].isin(historical_forecast['ds'])]

y_true = historical_data['avg_temp'].values
y_pred = historical_forecast['yhat'].values

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print(f"\nModel Evaluation Metrics on Historical Data:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Plot the forecast
print("\nPlotting forecast results...")
fig = plot_plotly(model, forecast)
fig.update_layout(title='Temperature Forecast for Austin TX')
fig.show()

# Plot forecast components to understand different factors
components_fig = plot_components_plotly(model, forecast)
components_fig.show()

# Create a more detailed forecast visualization
forecast_subset = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(365)
actual_data = data[['date', 'avg_temp']].rename(columns={'date': 'ds', 'avg_temp': 'y'})

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=forecast_subset['ds'], 
    y=forecast_subset['yhat'],
    mode='lines',
    name='Forecast',
    line=dict(color='blue')
))
fig.add_trace(go.Scatter(
    x=forecast_subset['ds'], 
    y=forecast_subset['yhat_upper'],
    mode='lines',
    name='Upper Bound',
    line=dict(width=0),
    showlegend=False
))
fig.add_trace(go.Scatter(
    x=forecast_subset['ds'], 
    y=forecast_subset['yhat_lower'],
    mode='lines',
    name='Lower Bound',
    line=dict(width=0),
    fill='tonexty',
    fillcolor='rgba(0, 0, 255, 0.2)',
    showlegend=False
))

# Add actual data points where available
actual_subset = actual_data[actual_data['ds'].isin(forecast_subset['ds'])]
if not actual_subset.empty:
    fig.add_trace(go.Scatter(
        x=actual_subset['ds'], 
        y=actual_subset['y'],
        mode='markers',
        name='Actual Values',
        marker=dict(color='red', size=5)
    ))

fig.update_layout(
    title='One Year Temperature Forecast for Austin TX',
    xaxis_title='Date',
    yaxis_title='Temperature',
    legend_title='Legend',
    height=600,
    width=1000
)
fig.show()

print("\nForecasting complete!")
