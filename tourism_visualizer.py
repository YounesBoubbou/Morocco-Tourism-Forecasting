import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime
import os

class TourismVisualizer:
    def __init__(self, db_path: str = "data/tourism.db"):
        self.db_path = db_path
        self.data = None
        self.load_data()
    
    def load_data(self):
        """Load data from database"""
        if not os.path.exists(self.db_path):
            print(f"Database not found: {self.db_path}")
            return
        
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT year, month, date, arrivals, source, data_type
            FROM tourism_data 
            WHERE data_type = 'monthly'
            ORDER BY year, month
        """
        
        self.data = pd.read_sql_query(query, conn)
        conn.close()
        
        if not self.data.empty:
            self.data['date'] = pd.to_datetime(self.data['date'])
        
        print(f"Loaded {len(self.data)} records for visualization")
    
    def plot_time_series(self, save_html: bool = True):
        """Create interactive time series plot"""
        if self.data is None or self.data.empty:
            print("No data to plot")
            return
        
        fig = go.Figure()
        
        # Main time series
        fig.add_trace(go.Scatter(
            x=self.data['date'],
            y=self.data['arrivals'],
            mode='lines+markers',
            name='Monthly Arrivals',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4),
            hovertemplate='<b>%{x|%Y-%m}</b><br>Arrivals: %{y:,.0f}<extra></extra>'
        ))
        
        # Add trend line
        z = np.polyfit(range(len(self.data)), self.data['arrivals'], 1)
        trend = np.poly1d(z)(range(len(self.data)))
        
        fig.add_trace(go.Scatter(
            x=self.data['date'],
            y=trend,
            mode='lines',
            name='Trend',
            line=dict(color='red', dash='dash', width=2),
            hovertemplate='Trend: %{y:,.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Morocco Tourism Arrivals Over Time',
            xaxis_title='Date',
            yaxis_title='Monthly Arrivals',
            hovermode='x unified',
            template='plotly_white',
            width=1000,
            height=500
        )
        
        if save_html:
            fig.write_html("visualizations/tourism_timeseries.html")
            print("✓ Time series plot saved to visualizations/tourism_timeseries.html")
        
        fig.show()
        return fig
    
    def plot_seasonal_patterns(self, save_html: bool = True):
        """Create seasonal analysis plots"""
        if self.data is None or self.data.empty:
            return
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Monthly Seasonality (Average)',
                'Yearly Totals',
                'Seasonal Boxplot',
                'Heatmap by Year-Month'
            ],
            specs=[[{"type": "xy"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "xy"}]]
        )
        
        # 1. Monthly seasonality
        monthly_avg = self.data.groupby(self.data['date'].dt.month)['arrivals'].mean()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig.add_trace(
            go.Bar(x=month_names, y=monthly_avg.values, name='Avg Monthly',
                   marker_color='lightblue'),
            row=1, col=1
        )
        
        # 2. Yearly totals
        yearly_totals = self.data.groupby('year')['arrivals'].sum()
        fig.add_trace(
            go.Scatter(x=yearly_totals.index, y=yearly_totals.values,
                      mode='lines+markers', name='Yearly Total',
                      line=dict(color='green', width=3)),
            row=1, col=2
        )
        
        # 3. Seasonal boxplot data
        self.data['month_name'] = self.data['date'].dt.strftime('%b')
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for month in month_order:
            month_data = self.data[self.data['month_name'] == month]['arrivals']
            fig.add_trace(
                go.Box(y=month_data, name=month, showlegend=False),
                row=2, col=1
            )
        
        # 4. Heatmap data
        pivot_data = self.data.pivot(index='year', columns='month', values='arrivals')
        
        fig.add_trace(
            go.Heatmap(
                z=pivot_data.values,
                x=month_names,
                y=pivot_data.index,
                colorscale='Blues',
                name='Arrivals',
                showscale=True
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Morocco Tourism: Seasonal Analysis',
            height=800,
            template='plotly_white'
        )
        
        if save_html:
            fig.write_html("visualizations/tourism_seasonal.html")
            print("✓ Seasonal analysis saved to visualizations/tourism_seasonal.html")
        
        fig.show()
        return fig
    
    def plot_forecast_comparison(self, forecaster_obj, save_html: bool = True):
        """Plot forecast results if available"""
        if not hasattr(forecaster_obj, 'forecasts') or not forecaster_obj.forecasts:
            print("No forecasts available to plot")
            return
        
        # Get test data
        test_months = 12
        train_data = self.data['arrivals'][:-test_months]
        test_data = self.data['arrivals'][-test_months:]
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=self.data['date'][:-test_months],
            y=train_data,
            mode='lines',
            name='Historical Data',
            line=dict(color='blue')
        ))
        
        # Actual test data
        fig.add_trace(go.Scatter(
            x=self.data['date'][-test_months:],
            y=test_data,
            mode='lines+markers',
            name='Actual (Test)',
            line=dict(color='black', width=3),
            marker=dict(size=6)
        ))
        
        # Forecasts
        colors = ['red', 'green', 'purple', 'orange']
        for i, (model_name, forecast_data) in enumerate(forecaster_obj.forecasts.items()):
            forecast = forecast_data['forecast']
            
            fig.add_trace(go.Scatter(
                x=self.data['date'][-test_months:],
                y=forecast,
                mode='lines+markers',
                name=f'{model_name} Forecast',
                line=dict(color=colors[i % len(colors)], dash='dash'),
                marker=dict(size=4)
            ))
            
            # Add confidence intervals if available
            if forecast_data.get('confidence_intervals') is not None:
                ci = forecast_data['confidence_intervals']
                fig.add_trace(go.Scatter(
                    x=self.data['date'][-test_months:],
                    y=ci.iloc[:, 1],  # Upper bound
                    fill=None,
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=self.data['date'][-test_months:],
                    y=ci.iloc[:, 0],  # Lower bound
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    name=f'{model_name} CI',
                    fillcolor=f'rgba({255*i//4},{100},{255*(1-i//4)},0.2)'
                ))
        
        fig.update_layout(
            title='Tourism Forecast vs Actual Results',
            xaxis_title='Date',
            yaxis_title='Monthly Arrivals',
            hovermode='x unified',
            template='plotly_white',
            width=1000,
            height=600
        )
        
        if save_html:
            fig.write_html("visualizations/tourism_forecasts.html")
            print("✓ Forecast comparison saved to visualizations/tourism_forecasts.html")
        
        fig.show()
        return fig
    
    def create_dashboard(self, forecaster_obj=None):
        """Create a comprehensive dashboard"""
        print("Creating tourism analysis dashboard...")
        
        # Ensure visualization directory exists
        os.makedirs("visualizations", exist_ok=True)
        
        # Create all visualizations
        self.plot_time_series()
        self.plot_seasonal_patterns()
        
        if forecaster_obj and hasattr(forecaster_obj, 'forecasts'):
            self.plot_forecast_comparison(forecaster_obj)
        
        print("\n✓ Dashboard complete! Check the visualizations/ folder:")
        print("  - tourism_timeseries.html")
        print("  - tourism_seasonal.html")
        if forecaster_obj and hasattr(forecaster_obj, 'forecasts'):
            print("  - tourism_forecasts.html")
    
    def generate_summary_report(self):
        """Generate a text summary report"""
        if self.data is None or self.data.empty:
            return
        
        report = []
        report.append("=" * 60)
        report.append("MOROCCO TOURISM ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append("")
        
        # Basic stats
        report.append("OVERVIEW:")
        report.append(f"• Data period: {self.data['date'].min().strftime('%Y-%m')} to {self.data['date'].max().strftime('%Y-%m')}")
        report.append(f"• Total months: {len(self.data)}")
        report.append(f"• Average monthly arrivals: {self.data['arrivals'].mean():,.0f}")
        report.append(f"• Peak month: {self.data['arrivals'].max():,.0f}")
        report.append(f"• Lowest month: {self.data['arrivals'].min():,.0f}")
        report.append("")
        
        # Yearly analysis
        yearly_totals = self.data.groupby('year')['arrivals'].sum()
        growth_rates = yearly_totals.pct_change().dropna()
        
        report.append("YEARLY TRENDS:")
        report.append(f"• Average yearly growth: {growth_rates.mean():.1%}")
        report.append(f"• Best performing year: {yearly_totals.idxmax()} ({yearly_totals.max():,.0f} arrivals)")
        report.append(f"• Highest growth year: {growth_rates.idxmax()} ({growth_rates.max():.1%})")
        report.append("")
        
        # Seasonal patterns
        monthly_avg = self.data.groupby(self.data['date'].dt.month)['arrivals'].mean()
        peak_month = monthly_avg.idxmax()
        low_month = monthly_avg.idxmin()
        
        month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 
                      5: 'May', 6: 'June', 7: 'July', 8: 'August',
                      9: 'September', 10: 'October', 11: 'November', 12: 'December'}
        
        report.append("SEASONAL PATTERNS:")
        report.append(f"• Peak season: {month_names[peak_month]} ({monthly_avg[peak_month]:,.0f} avg arrivals)")
        report.append(f"• Low season: {month_names[low_month]} ({monthly_avg[low_month]:,.0f} avg arrivals)")
        report.append(f"• Seasonality factor: {monthly_avg[peak_month]/monthly_avg[low_month]:.1f}x difference")
        report.append("")
        
        # Save report
        report_text = "\n".join(report)
        
        os.makedirs("reports", exist_ok=True)
        with open("reports/tourism_summary.txt", "w") as f:
            f.write(report_text)
        
        print(report_text)
        print("✓ Summary report saved to reports/tourism_summary.txt")
        
        return report_text

def main():
    # Create visualizer
    viz = TourismVisualizer()
    
    if viz.data is None or viz.data.empty:
        print("No data found. Please run the forecasting script first!")
        return
    
    # Create dashboard
    viz.create_dashboard()
    
    # Generate summary
    viz.generate_summary_report()
    
    print("\n" + "="*50)
    print("Visualization complete!")
    print("Open the HTML files in your browser to see interactive charts.")