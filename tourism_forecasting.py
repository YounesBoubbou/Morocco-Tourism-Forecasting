import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import requests
import sqlite3
import os
import warnings
from typing import Dict, List, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# For forecasting
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

class MoroccoTourismForecaster:
    def __init__(self, db_path: str = "data/tourism.db"):
        self.db_path = db_path
        self.data = None
        self.models = {}
        self.forecasts = {}
        
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for tourism data"""
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".", exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tourism_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                year INTEGER NOT NULL,
                month INTEGER,
                date TEXT,
                arrivals INTEGER,
                source TEXT,
                data_type TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
        print(f"Database initialized at {self.db_path}")
    
    def fetch_worldbank_data(self) -> pd.DataFrame:
        """Fetch Morocco tourism data from World Bank API"""
        print("Fetching tourism data from World Bank...")
        
        # World Bank API endpoint for Morocco tourism arrivals
        # ST.INT.ARVL = International tourism, number of arrivals
        url = "https://api.worldbank.org/v2/country/MAR/indicator/ST.INT.ARVL"
        params = {
            "format": "json",
            "date": "1995:2023",
            "per_page": 1000
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if len(data) > 1 and data[1]:
                df = pd.DataFrame(data[1])
                df = df[['date', 'value']].copy()
                df.columns = ['year', 'arrivals']
                df['year'] = df['year'].astype(int)
                df['arrivals'] = pd.to_numeric(df['arrivals'], errors='coerce')
                df = df.dropna().sort_values('year')
                df['source'] = 'World Bank'
                df['data_type'] = 'annual'
                
                print(f"✓ Fetched {len(df)} years of data from World Bank")
                return df
            
        except Exception as e:
            print(f"Error fetching World Bank data: {e}")
        
        return pd.DataFrame()
    
    def generate_synthetic_monthly_data(self, annual_df: pd.DataFrame) -> pd.DataFrame:
        """Generate realistic monthly tourism data from annual data"""
        print("Generating monthly breakdown from annual data...")
        
        # Morocco tourism seasonality (based on typical patterns)
        # Peak: March-May (spring), October-November (autumn)
        # Low: December-February (winter), July-August (very hot)
        monthly_weights = {
            1: 0.06,   # January - low (winter)
            2: 0.07,   # February - low (winter)
            3: 0.11,   # March - high (spring starts)
            4: 0.12,   # April - high (perfect weather)
            5: 0.11,   # May - high (spring)
            6: 0.09,   # June - moderate (getting hot)
            7: 0.08,   # July - low (very hot)
            8: 0.08,   # August - low (very hot)
            9: 0.09,   # September - moderate (cooling)
            10: 0.10,  # October - high (perfect autumn)
            11: 0.09,  # November - moderate (autumn)
            12: 0.06   # December - low (winter)
        }
        
        monthly_data = []
        
        for _, row in annual_df.iterrows():
            year = row['year']
            annual_arrivals = row['arrivals']
            
            for month in range(1, 13):
                # Base monthly arrivals using seasonal weights
                base_arrivals = annual_arrivals * monthly_weights[month]
                
                # Add some random variation (±15%)
                variation = np.random.normal(1.0, 0.15)
                monthly_arrivals = int(base_arrivals * variation)
                
                # Ensure positive values
                monthly_arrivals = max(monthly_arrivals, 1000)
                
                monthly_data.append({
                    'year': year,
                    'month': month,
                    'date': f"{year}-{month:02d}",
                    'arrivals': monthly_arrivals,
                    'source': 'Generated',
                    'data_type': 'monthly'
                })
        
        monthly_df = pd.DataFrame(monthly_data)
        print(f"✓ Generated {len(monthly_df)} months of data")
        
        return monthly_df
    
    def save_data(self, df: pd.DataFrame):
        """Save tourism data to database"""
        conn = sqlite3.connect(self.db_path)
        
        # Clear existing data
        conn.execute("DELETE FROM tourism_data")
        
        # Insert new data
        df.to_sql('tourism_data', conn, if_exists='append', index=False)
        
        conn.commit()
        conn.close()
        print(f"✓ Saved {len(df)} records to database")
    
    def load_data(self) -> pd.DataFrame:
        """Load tourism data from database"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT year, month, date, arrivals, source, data_type
            FROM tourism_data 
            WHERE data_type = 'monthly'
            ORDER BY year, month
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        
        self.data = df
        print(f"✓ Loaded {len(df)} monthly records")
        return df
    
    def analyze_patterns(self):
        """Analyze tourism patterns and seasonality"""
        if self.data is None or self.data.empty:
            print("No data available for analysis")
            return
        
        print("Analyzing tourism patterns...")
        
        # Basic statistics
        print("\n=== Tourism Statistics ===")
        print(f"Data range: {self.data.index.min().strftime('%Y-%m')} to {self.data.index.max().strftime('%Y-%m')}")
        print(f"Average monthly arrivals: {self.data['arrivals'].mean():,.0f}")
        print(f"Peak month arrivals: {self.data['arrivals'].max():,.0f}")
        print(f"Lowest month arrivals: {self.data['arrivals'].min():,.0f}")
        
        # Seasonal decomposition
        decomposition = seasonal_decompose(self.data['arrivals'], model='multiplicative', period=12)
        
        # Yearly trends
        yearly_totals = self.data.groupby('year')['arrivals'].sum()
        growth_rates = yearly_totals.pct_change().dropna()
        
        print(f"\n=== Growth Analysis ===")
        print(f"Average yearly growth: {growth_rates.mean():.1%}")
        print(f"Best year growth: {growth_rates.max():.1%} in {growth_rates.idxmax()}")
        print(f"Worst year growth: {growth_rates.min():.1%} in {growth_rates.idxmin()}")
        
        # Monthly seasonality
        monthly_avg = self.data.groupby(self.data.index.month)['arrivals'].mean()
        peak_month = monthly_avg.idxmax()
        low_month = monthly_avg.idxmin()
        
        month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                      7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        
        print(f"\n=== Seasonal Patterns ===")
        print(f"Peak season: {month_names[peak_month]} ({monthly_avg[peak_month]:,.0f} avg)")
        print(f"Low season: {month_names[low_month]} ({monthly_avg[low_month]:,.0f} avg)")
        print(f"Seasonality ratio: {monthly_avg[peak_month]/monthly_avg[low_month]:.1f}x")
        
        return decomposition, yearly_totals, monthly_avg
    
    def prepare_forecasting_data(self, test_months: int = 12) -> Tuple[pd.Series, pd.Series]:
        """Prepare data for forecasting models"""
        if self.data is None or self.data.empty:
            raise ValueError("No data loaded")
        
        ts = self.data['arrivals']
        
        # Split into train/test
        train_size = len(ts) - test_months
        train_data = ts[:train_size]
        test_data = ts[train_size:]
        
        print(f"Training data: {len(train_data)} months")
        print(f"Test data: {len(test_data)} months")
        
        return train_data, test_data
    
    def fit_arima_model(self, train_data: pd.Series, order: Tuple[int, int, int] = (2, 1, 2)) -> ARIMA:
        """Fit ARIMA model"""
        print(f"Fitting ARIMA{order} model...")
        
        try:
            model = ARIMA(train_data, order=order)
            fitted_model = model.fit()
            
            print(f"✓ ARIMA model fitted - AIC: {fitted_model.aic:.2f}")
            return fitted_model
            
        except Exception as e:
            print(f"Error fitting ARIMA model: {e}")
            return None
    
    def fit_exponential_smoothing(self, train_data: pd.Series) -> ExponentialSmoothing:
        """Fit Exponential Smoothing model"""
        print("Fitting Exponential Smoothing model...")
        
        try:
            model = ExponentialSmoothing(
                train_data, 
                trend='add', 
                seasonal='add', 
                seasonal_periods=12
            )
            fitted_model = model.fit()
            
            print("✓ Exponential Smoothing model fitted")
            return fitted_model
            
        except Exception as e:
            print(f"Error fitting Exponential Smoothing model: {e}")
            return None
    
    def generate_forecasts(self, models: Dict, test_periods: int = 12) -> Dict:
        """Generate forecasts from fitted models"""
        forecasts = {}
        
        for name, model in models.items():
            if model is None:
                continue
                
            print(f"Generating {name} forecast...")
            
            try:
                if name == 'ARIMA':
                    forecast = model.forecast(steps=test_periods)
                    conf_int = model.get_forecast(steps=test_periods).conf_int()
                elif name == 'ExponentialSmoothing':
                    forecast = model.forecast(steps=test_periods)
                    conf_int = None
                
                forecasts[name] = {
                    'forecast': forecast,
                    'confidence_intervals': conf_int
                }
                
                print(f"✓ {name} forecast generated")
                
            except Exception as e:
                print(f"Error generating {name} forecast: {e}")
        
        return forecasts
    
    def evaluate_forecasts(self, forecasts: Dict, test_data: pd.Series):
        """Evaluate forecast accuracy"""
        print("\n=== Forecast Evaluation ===")
        
        results = {}
        
        for name, forecast_data in forecasts.items():
            forecast = forecast_data['forecast']
            
            # Calculate metrics
            mae = mean_absolute_error(test_data, forecast)
            rmse = np.sqrt(mean_squared_error(test_data, forecast))
            mape = np.mean(np.abs((test_data - forecast) / test_data)) * 100
            
            results[name] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
            
            print(f"\n{name}:")
            print(f"  MAE: {mae:,.0f}")
            print(f"  RMSE: {rmse:,.0f}")
            print(f"  MAPE: {mape:.1f}%")
        
        return results
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline"""
        print("=== Morocco Tourism Forecasting Analysis ===\n")
        
        # 1. Fetch data
        annual_data = self.fetch_worldbank_data()
        
        if annual_data.empty:
            print("Failed to fetch data. Using sample data...")
            # Create sample data if API fails
            years = list(range(2000, 2024))
            arrivals = [4000000 + i*200000 + np.random.randint(-300000, 400000) for i in range(len(years))]
            annual_data = pd.DataFrame({'year': years, 'arrivals': arrivals, 'source': 'Sample', 'data_type': 'annual'})
        
        # 2. Generate monthly data
        monthly_data = self.generate_synthetic_monthly_data(annual_data)
        
        # 3. Save and load data
        all_data = pd.concat([annual_data, monthly_data], ignore_index=True)
        self.save_data(all_data)
        self.load_data()
        
        # 4. Analyze patterns
        decomposition, yearly_totals, monthly_avg = self.analyze_patterns()
        
        # 5. Prepare forecasting
        train_data, test_data = self.prepare_forecasting_data()
        
        # 6. Fit models
        arima_model = self.fit_arima_model(train_data)
        exp_smooth_model = self.fit_exponential_smoothing(train_data)
        
        models = {
            'ARIMA': arima_model,
            'ExponentialSmoothing': exp_smooth_model
        }
        
        # 7. Generate forecasts
        forecasts = self.generate_forecasts(models, len(test_data))
        
        # 8. Evaluate
        evaluation = self.evaluate_forecasts(forecasts, test_data)
        
        # Store results
        self.models = models
        self.forecasts = forecasts
        
        print(f"\n✓ Analysis complete! Data saved to {self.db_path}")
        
        return {
            'data': self.data,
            'models': models,
            'forecasts': forecasts,
            'evaluation': evaluation,
            'decomposition': decomposition
        }

def main():
    # Initialize forecaster
    forecaster = MoroccoTourismForecaster()
    
    # Run complete analysis
    results = forecaster.run_full_analysis()
    
    print("\n" + "="*60)
    print("Morocco Tourism Forecasting - Analysis Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Check the database: data/tourism.db")
    print("2. Run visualization script to see charts")
    print("3. Experiment with different model parameters")

if __name__ == "__main__":
    main()