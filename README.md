# Weather Forecasting Prediction

A machine learning project for analyzing and forecasting weather patterns in Austin, Texas using historical weather data and the Prophet forecasting model.

## Project Overview

This project analyzes historical weather data for Austin, Texas from 2015 to 2025 (including some forecasted data) to:
- Visualize temperature trends over time
- Identify seasonal patterns in temperature data
- Create forecasts for future temperature trends using Facebook's Prophet model

## Dataset

The project uses weather data for Austin, Texas with the following features:
- Date
- Maximum temperature
- Minimum temperature
- Average temperature
- Precipitation

The main dataset file is `ATX 2015-01-01 to 2025-03-09.csv`, which contains daily weather records.

## Features

- **Data Visualization**: Uses Plotly Express and Seaborn to create interactive visualizations of temperature trends
- **Time Series Analysis**: Analyzes temperature patterns by year and month
- **Forecasting**: Implements Facebook's Prophet model to predict future temperature trends
- **Interactive Plots**: Generates interactive plots for exploring the data and forecast results

## Technologies Used

- **Python**: Primary programming language
- **Pandas & NumPy**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Static data visualization
- **Plotly Express**: Interactive data visualization
- **Prophet**: Time series forecasting model developed by Facebook

## Getting Started

### Prerequisites

To run this project, you'll need Python installed along with the following libraries:
```
pandas
numpy
matplotlib
seaborn
plotly
prophet
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

This will:
1. Load and preprocess the weather data
2. Display basic statistics and information about the dataset
3. Generate visualizations of temperature trends
4. Create and train a Prophet forecasting model
5. Display forecast results with interactive plots

## Results

The project produces several visualizations:
- Line plots showing average temperature trends over time
- Monthly temperature patterns across different years
- Prophet forecast plots showing predicted temperature trends with uncertainty intervals

## Future Improvements

Potential enhancements for this project:
- Include additional weather parameters (humidity, wind speed) in the analysis
- Implement more advanced forecasting models for comparison
- Add geographical visualization of weather patterns
- Create a web interface for interactive exploration of the data and forecasts

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Weather data sourced from historical weather records for Austin, Texas
- Facebook's Prophet team for developing the forecasting library 