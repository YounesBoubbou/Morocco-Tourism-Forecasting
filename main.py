"""
Morocco Tourism Forecasting Project
Main runner script to execute the complete analysis pipeline
"""

import os
import sys
from pathlib import Path

def setup_project_structure():
    """Create project directory structure"""
    directories = [
        "data",
        "scripts", 
        "notebooks",
        "visualizations",
        "reports",
        "models"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}/")

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly',
        'requests', 'statsmodels', 'sklearn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing packages:", ', '.join(missing_packages))
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("✓ All required packages are installed")
        return True

def run_analysis():
    """Run the complete tourism analysis"""
    print("\n" + "="*60)
    print("MOROCCO TOURISM FORECASTING PROJECT")
    print("="*60)
    
    try:
        # Import our modules
        from tourism_forecasting import MoroccoTourismForecaster
        from tourism_visualizer import TourismVisualizer
        
        print("\n1. Setting up project structure...")
        setup_project_structure()
        
        print("\n2. Checking dependencies...")
        if not check_dependencies():
            return False
        
        print("\n3. Running tourism analysis...")
        forecaster = MoroccoTourismForecaster()
        results = forecaster.run_full_analysis()
        
        print("\n4. Creating visualizations...")
        visualizer = TourismVisualizer()
        visualizer.create_dashboard(forecaster)
        visualizer.generate_summary_report()
        
        print("\n" + "="*60)
        print("✅ PROJECT COMPLETE!")
        print("="*60)
        print("\nFiles created:")
        print("📊 data/tourism.db - SQLite database with tourism data")
        print("📈 visualizations/ - Interactive HTML charts")
        print("📄 reports/tourism_summary.txt - Analysis summary")
        
        print("\nNext steps:")
        print("1. Open HTML files in visualizations/ folder")
        print("2. Read the summary report")
        print("3. Experiment with different model parameters")
        print("4. Try adding more data sources")
        
        return True
        
    except ImportError as e:
        print(f"❌ Error importing modules: {e}")
        print("Make sure all Python files are in the same directory")
        return False
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return False

def show_project_info():
    """Display project information"""
    info = """
🇲🇦 MOROCCO TOURISM FORECASTING PROJECT
    
This project analyzes and forecasts Morocco's tourism patterns using:
• World Bank tourism arrival data
• Time series decomposition 
• ARIMA and Exponential Smoothing models
• Interactive visualizations
• Performance evaluation

FEATURES:
✓ Automated data collection from World Bank API
✓ Seasonal pattern analysis
✓ Multiple forecasting models
✓ Interactive Plotly visualizations  
✓ Model performance comparison
✓ SQLite database storage

MODELS INCLUDED:
• ARIMA - AutoRegressive Integrated Moving Average
• Exponential Smoothing - Triple exponential smoothing
• (Easy to extend with LSTM, Prophet, etc.)

OUTPUTS:
• Time series plots with trends
• Seasonal decomposition analysis
• Forecast vs actual comparisons
• Summary statistics and insights
    """
    print(info)

def main():
    """Main function"""
    if len(sys.argv) > 1:
        if sys.argv[1] == "info":
            show_project_info()
            return
        elif sys.argv[1] == "setup":
            setup_project_structure()
            return
    
    # Run full analysis
    success = run_analysis()
    
    if success:
        print("\n🎉 Analysis completed successfully!")
        print("Check the visualizations/ folder for interactive charts.")
    else:
        print("\n❌ Analysis failed. Check error messages above.")

if __name__ == "__main__":
    main()