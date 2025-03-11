# Weather Forecasting Prediction

A machine learning project for analyzing and forecasting weather patterns in Austin, Texas using historical weather data and the Prophet forecasting model.

## Project Overview

This project analyzes historical weather data for Austin, Texas from 2015 to 2025 (including some forecasted data) to:
- Visualize temperature trends over time
- Identify seasonal patterns in temperature data
- Create accurate forecasts using an enhanced Prophet model with multiple predictors
- Evaluate forecast accuracy using cross-validation

## Dataset

The project uses weather data for Austin, Texas with the following features:
- Date
- Maximum temperature
- Minimum temperature
- Average temperature
- Precipitation (with special handling for trace amounts marked as 'T')

The main dataset file is `ATX 2015-01-01 to 2025-03-09.csv`, which contains daily weather records.

## Features

- **Advanced Data Preprocessing**: 
  - Handles trace precipitation values
  - Extracts temporal features (day of week, day of year)
  - Performs missing value detection and imputation

- **Comprehensive Data Visualization**: 
  - Interactive temperature trend plots using Plotly
  - Seasonal pattern visualization with Seaborn
  - Monthly temperature distribution analysis
  - Cross-validation performance plots

- **Enhanced Forecasting Model**:
  - Optimized Prophet configuration with:
    - Flexible yearly seasonality (20 components)
    - Weekly seasonality modeling
    - Multiplicative seasonality mode
    - Tuned changepoint and seasonality parameters
  - Multiple predictor integration:
    - Precipitation as a regressor
    - Maximum and minimum temperatures as additional predictors

- **Model Evaluation**:
  - Cross-validation with configurable parameters
  - Multiple performance metrics (MAE, RMSE, R²)
  - Forecast uncertainty visualization
  - Detailed component analysis

## Technologies Used

- **Python**: Primary programming language
- **Pandas & NumPy**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Statistical visualization
- **Plotly**: Interactive data visualization
- **Prophet**: Advanced time series forecasting
- **scikit-learn**: Model evaluation metrics

## Getting Started

### Prerequisites

To run this project, you'll need Python installed along with the following libraries:
```
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.3.0
prophet>=1.0.0
scikit-learn>=1.0.0
```

### Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/Weather-Forcasting-Prediction.git
cd Weather-Forcasting-Prediction
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

### Usage

Run the main script to perform the analysis and generate forecasts:
```
python weather.py
```

The script will:
1. Load and preprocess the weather data, including handling trace precipitation values
2. Display comprehensive dataset statistics and information
3. Generate interactive visualizations of temperature trends
4. Create and train an enhanced Prophet model with multiple predictors
5. Perform cross-validation and display performance metrics
6. Generate detailed forecast visualizations with uncertainty bounds

## Results

The project produces several sophisticated visualizations:
- Interactive line plots showing temperature trends over time
- Seasonal pattern analysis with yearly comparisons
- Monthly temperature distribution boxplots
- Cross-validation performance analysis
- Detailed forecast plots with confidence intervals
- Component-wise analysis of different factors affecting temperature

## Model Performance

The forecasting model's performance is evaluated using:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Cross-validation metrics across different forecast horizons

## Future Improvements

Potential enhancements for this project:
- Incorporate additional weather parameters (humidity, wind speed, pressure)
- Experiment with different seasonality configurations
- Implement ensemble forecasting methods
- Add interactive parameter tuning capabilities
- Create a web dashboard for real-time forecast updates
- Add extreme weather event prediction capabilities

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Weather data sourced from historical weather records for Austin, Texas
- Facebook's Prophet team for developing the forecasting library
- The open-source community for the various Python libraries used