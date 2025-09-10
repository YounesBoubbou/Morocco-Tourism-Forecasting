# Morocco-Tourism-Forecasting
An end-to-end time series analysis and forecasting pipeline for Morocco's tourism industry, featuring real-time data collection, advanced modeling, and interactive visualizations.


📊 Project Overview
This project analyzes 26 years of Morocco tourism data (1995-2020) to identify patterns, seasonal trends, and forecast future arrivals using multiple time series models. The pipeline demonstrates real-world data science applications including API integration, database management, statistical modeling, and interactive visualization.
Key Findings

Average Growth: 3.8% yearly tourism growth
Peak Season: April (887K average arrivals)
Seasonality: 2x difference between peak and low seasons
COVID Impact: -78.1% drop in 2020 (captured in real data)

🚀 Features

Automated Data Collection: Fetches real tourism data from World Bank API
Time Series Analysis: Seasonal decomposition, trend analysis, and pattern recognition
Multiple Forecasting Models: ARIMA and Exponential Smoothing with performance comparison
Interactive Visualizations: Plotly-powered dashboards with drill-down capabilities
Database Integration: SQLite storage for efficient data management
Comprehensive Reporting: Automated summary generation with key insights

🛠️ Technology Stack

Data Processing: pandas, numpy
Forecasting: statsmodels, scikit-learn
Visualization: plotly, matplotlib, seaborn
Database: SQLite
API Integration: requests (World Bank API)

📁 Project Structure

morocco-tourism-forecasting/
├── data/
│   └── tourism.db              # SQLite database
├── scripts/
│   ├── tourism_forecaster.py   # Main analysis engine
│   ├── tourism_visualizer.py   # Visualization module
│   └── project_runner.py       # One-click executor
├── visualizations/
│   ├── tourism_timeseries.html # Interactive time series
│   ├── tourism_seasonal.html   # Seasonal analysis
│   └── tourism_forecasts.html  # Model comparisons
├── reports/
│   └── tourism_summary.txt     # Analysis summary
├── requirements.txt
└── README.md


🚀 Quick Start
Installation 

1. Clone the repository
   git clone https://github.com/yourusername/morocco-tourism-forecasting.git
   cd morocco-tourism-forecasting

2. Set up Python environment (Python version used: 3.12)
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies
   pip install -r requirements.txt

4. Running the Analysis
   python3 project_runner.py


📈 Sample Results
Tourism Growth Over Time

1995-2007: Steady growth period (avg 4.2% yearly)
2008: Financial crisis impact (-12% drop)
2009-2019: Recovery and expansion (avg 5.1% growth)
2020: COVID-19 disruption (-78.1% decline)

Seasonal Patterns

Peak Months: March-May (spring season)
Low Months: December-February (winter)
Secondary Peak: October-November (autumn)

🔍 Key Insights

Seasonality: Morocco tourism shows clear seasonal patterns with 2x variation
Growth Resilience: Despite economic shocks, long-term growth trend remains positive
Weather Correlation: Peak seasons align with optimal weather conditions
Recovery Potential: Historical data suggests strong post-crisis recovery patterns

🛣️ Future Enhancements

 Advanced Models: Prophet, LSTM, ensemble methods
 External Variables: Weather, economic indicators, events
 Real-time Updates: Automated data pipeline with recent data
 Regional Analysis: City-level and regional breakdowns
 Scenario Planning: What-if analysis for different recovery scenarios
