"""
Reporting and Visualization Module for Tax Fraud Detection Pipeline

This module handles:
1. Summary statistics generation
2. Data visualization
3. Fraud report generation
4. Dashboard creation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import jinja2
from datetime import datetime

# Import utility functions
from utils import setup_logger, safe_read_csv, safe_save_csv

logger = setup_logger(__name__)

class TaxFraudReporter:
    def __init__(self, predictions_file: str, output_dir: str = "reports"):
        """
        Initialize the reporter with prediction data.
        
        Args:
            predictions_file (str): Path to the CSV with model predictions
            output_dir (str): Directory to save reports and visualizations
        """
        # Convert to Path and validate
        predictions_path = Path(predictions_file)
        if not predictions_path.exists():
            raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
            
        # Read the data
        self.df = safe_read_csv(predictions_path)
        
        # Setup output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style for matplotlib
        plt.style.use('default')
        
    def generate_summary_statistics(self) -> dict:
        """
        Generate summary statistics for the dataset.
        
        Returns:
            dict: Dictionary containing summary statistics
        """
        stats = {
            'total_cases': len(self.df),
            'fraud_cases': self.df['is_fraud'].sum(),
            'normal_cases': len(self.df) - self.df['is_fraud'].sum(),
            'fraud_percentage': (self.df['is_fraud'].sum() / len(self.df)) * 100,
            'avg_risk_score': self.df['risk_score'].mean(),
            'median_income': self.df['income'].median(),
            'total_unpaid_tax': (self.df['declared_tax'] - self.df['paid_tax']).sum(),
            'high_risk_cases': len(self.df[self.df['risk_score'] > 0.8])
        }
        
        return stats
    
    def plot_distributions(self, save_fig: bool = True):
        """
        Create distribution plots for key metrics.
        """
        # Create subplot figure
        fig = plt.figure(figsize=(15, 10))
        
        # Income distribution
        plt.subplot(2, 2, 1)
        sns.histplot(data=self.df, x='income', hue='is_fraud', bins=50)
        plt.title('Income Distribution by Fraud Status')
        plt.xlabel('Income (KES)')
        
        # Tax rate distribution
        plt.subplot(2, 2, 2)
        sns.histplot(data=self.df, x='tax_rate', hue='is_fraud', bins=50)
        plt.title('Tax Rate Distribution by Fraud Status')
        plt.xlabel('Tax Rate')
        
        # Deduction ratio distribution
        plt.subplot(2, 2, 3)
        sns.histplot(data=self.df, x='deduction_ratio', hue='is_fraud', bins=50)
        plt.title('Deduction Ratio Distribution by Fraud Status')
        plt.xlabel('Deduction Ratio')
        
        # Risk score distribution
        plt.subplot(2, 2, 4)
        sns.histplot(data=self.df, x='risk_score', hue='is_fraud', bins=50)
        plt.title('Risk Score Distribution by Fraud Status')
        plt.xlabel('Risk Score')
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(self.output_dir / 'distributions.png')
            logger.info("Saved distribution plots to distributions.png")
    
    def plot_anomaly_comparisons(self, save_fig: bool = True):
        """
        Create boxplots comparing normal vs fraudulent cases.
        """
        metrics = ['income', 'declared_tax', 'paid_tax', 'deductions']
        
        fig = plt.figure(figsize=(15, 10))
        
        for i, metric in enumerate(metrics, 1):
            plt.subplot(2, 2, i)
            sns.boxplot(data=self.df, x='is_fraud', y=metric)
            plt.title(f'{metric.title()} by Fraud Status')
            plt.xlabel('Is Fraud')
            plt.ylabel(metric.title())
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(self.output_dir / 'anomaly_comparisons.png')
            logger.info("Saved anomaly comparison plots to anomaly_comparisons.png")
    
    def create_interactive_dashboard(self):
        """
        Create an interactive HTML dashboard using Plotly.
        """
        # Create a subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Income vs Tax Rate', 'Deductions vs Income',
                          'Risk Score Distribution', 'Top Risk Factors')
        )
        
        # Scatter plot: Income vs Tax Rate
        fig.add_trace(
            go.Scatter(
                x=self.df['income'],
                y=self.df['tax_rate'],
                mode='markers',
                marker=dict(
                    color=self.df['risk_score'],
                    colorscale='Viridis',
                    showscale=True
                ),
                name='Tax Rate'
            ),
            row=1, col=1
        )
        
        # Scatter plot: Deductions vs Income
        fig.add_trace(
            go.Scatter(
                x=self.df['income'],
                y=self.df['deductions'],
                mode='markers',
                marker=dict(
                    color=self.df['risk_score'],
                    colorscale='Viridis',
                    showscale=True
                ),
                name='Deductions'
            ),
            row=1, col=2
        )
        
        # Histogram: Risk Score Distribution
        fig.add_trace(
            go.Histogram(
                x=self.df['risk_score'],
                nbinsx=50,
                name='Risk Scores'
            ),
            row=2, col=1
        )
        
        # Bar chart: Risk Factors
        risk_factors = [
            'zero_tax_high_income',
            'high_deductions',
            'negative_deductions',
            'tax_rate_too_high'
        ]
        risk_counts = [self.df[factor].sum() for factor in risk_factors]
        
        fig.add_trace(
            go.Bar(
                x=risk_factors,
                y=risk_counts,
                name='Risk Factors'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Tax Fraud Analysis Dashboard",
            showlegend=False
        )
        
        # Save to HTML
        fig.write_html(self.output_dir / 'dashboard.html')
        logger.info("Created interactive dashboard at dashboard.html")
    
    def generate_fraud_report(self):
        """
        Generate a detailed fraud analysis report.
        """
        # Get summary statistics
        stats = self.generate_summary_statistics()
        
        # Get top suspicious cases
        top_suspicious = self.df.nlargest(10, 'risk_score')
        
        # Create the report
        report = []
        report.append("Tax Fraud Detection - Analysis Report")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        report.append("1. Summary Statistics")
        report.append("-" * 50)
        report.append(f"Total Cases Analyzed: {stats['total_cases']:,}")
        report.append(f"Potential Fraud Cases: {stats['fraud_cases']:,} ({stats['fraud_percentage']:.1f}%)")
        report.append(f"High Risk Cases (Risk Score > 0.8): {stats['high_risk_cases']:,}")
        report.append(f"Total Unpaid Tax: {stats['total_unpaid_tax']:,.2f} KES\n")
        
        report.append("2. Top 10 Highest Risk Cases")
        report.append("-" * 50)
        for _, case in top_suspicious.iterrows():
            report.append(f"Taxpayer: {case['name']}")
            report.append(f"Risk Score: {case['risk_score']:.2f}")
            report.append(f"Income: {case['income']:,.2f} KES")
            report.append(f"Declared Tax: {case['declared_tax']:,.2f} KES")
            report.append(f"Paid Tax: {case['paid_tax']:,.2f} KES")
            report.append(f"Deductions: {case['deductions']:,.2f} KES\n")
        
        # Save report
        report_path = self.output_dir / f"fraud_report_{datetime.now().strftime('%Y%m%d')}.txt"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Generated fraud report at {report_path}")
        
        # Also save top suspicious cases to CSV
        csv_path = self.output_dir / f"high_risk_cases_{datetime.now().strftime('%Y%m%d')}.csv"
        safe_save_csv(top_suspicious, csv_path)
        logger.info(f"Saved high risk cases to {csv_path}")

def main():
    """
    Main function to generate all reports and visualizations.
    """
    try:
        logger.info("Starting report generation...")
        
        # Get the current file's directory
        current_dir = Path(__file__).parent
        predictions_file = current_dir / 'tax_predictions.csv'
        output_dir = current_dir.parent / 'reports'
        
        # Check if predictions file exists
        if not predictions_file.exists():
            logger.error(f"Predictions file not found at: {predictions_file}")
            logger.error("Please run the pipeline in order:")
            logger.error("1. python data_ingestion.py")
            logger.error("2. python data_processing.py")
            logger.error("3. python anomaly_detection.py")
            return
        
        # Initialize reporter with absolute paths
        reporter = TaxFraudReporter(str(predictions_file.absolute()), str(output_dir))
        
        # Generate all reports and visualizations
        reporter.plot_distributions()
        reporter.plot_anomaly_comparisons()
        reporter.create_interactive_dashboard()
        reporter.generate_fraud_report()
        
        logger.info("Report generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error generating reports: {str(e)}")
        raise

if __name__ == "__main__":
    main()
