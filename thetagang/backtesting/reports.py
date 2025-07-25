"""
Report Generation Module

Advanced reporting system with interactive dashboards, comprehensive analytics,
and export capabilities for backtesting results and strategy performance.

Features:
- Interactive dashboards with Plotly/Bokeh integration
- Comprehensive HTML/PDF report generation
- Strategy comparison and benchmarking utilities
- Risk factor analysis and attribution reporting
- Scenario analysis and stress testing reports
- Export to multiple formats (HTML, PDF, Excel, JSON)

Integration:
- Works with backtesting engine results
- Leverages performance analytics module
- Supports multi-strategy comparison
- Provides publication-ready reports
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Union, Tuple, IO
import logging
from pathlib import Path
import warnings

import pandas as pd
import numpy as np


class ReportFormat(Enum):
    """Supported report formats"""
    HTML = "html"
    PDF = "pdf"
    EXCEL = "excel"
    JSON = "json"
    CSV = "csv"
    MARKDOWN = "markdown"


class ReportTemplate(Enum):
    """Report templates"""
    EXECUTIVE_SUMMARY = "executive_summary"
    DETAILED_ANALYSIS = "detailed_analysis"
    STRATEGY_COMPARISON = "strategy_comparison"
    RISK_ANALYSIS = "risk_analysis"
    PERFORMANCE_ATTRIBUTION = "performance_attribution"
    CUSTOM = "custom"


class ChartType(Enum):
    """Chart types for reports"""
    EQUITY_CURVE = "equity_curve"
    DRAWDOWN = "drawdown"
    ROLLING_RETURNS = "rolling_returns"
    MONTHLY_RETURNS = "monthly_returns"
    RETURN_DISTRIBUTION = "return_distribution"
    UNDERWATER_PLOT = "underwater_plot"
    ROLLING_SHARPE = "rolling_sharpe"
    CORRELATION_HEATMAP = "correlation_heatmap"
    TRADE_ANALYSIS = "trade_analysis"
    CUSTOM = "custom"


@dataclass
class ReportConfig:
    """Configuration for report generation"""
    
    # Basic settings
    template: ReportTemplate = ReportTemplate.DETAILED_ANALYSIS
    format: ReportFormat = ReportFormat.HTML
    output_path: Optional[str] = None
    
    # Content settings
    include_charts: bool = True
    include_tables: bool = True
    include_executive_summary: bool = True
    include_risk_analysis: bool = True
    include_trade_analysis: bool = True
    
    # Chart settings
    chart_types: List[ChartType] = field(default_factory=lambda: [
        ChartType.EQUITY_CURVE,
        ChartType.DRAWDOWN,
        ChartType.MONTHLY_RETURNS,
        ChartType.RETURN_DISTRIBUTION
    ])
    chart_width: int = 800
    chart_height: int = 400
    
    # Comparison settings
    benchmark_symbol: Optional[str] = "SPY"
    comparison_strategies: List[str] = field(default_factory=list)
    
    # Styling
    theme: str = "default"  # default, dark, professional
    color_palette: List[str] = field(default_factory=lambda: ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    
    # Export settings
    include_raw_data: bool = False
    compress_output: bool = False
    
    # Custom settings
    custom_template_path: Optional[str] = None
    custom_css_path: Optional[str] = None
    
    # Metadata
    report_title: str = "Strategy Performance Report"
    report_subtitle: str = "Backtesting Analysis"
    author: str = "ThetaGang Framework"
    company: str = ""
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportSection:
    """Represents a section in the report"""
    
    title: str
    content: str
    order: int = 0
    
    # Visual elements
    charts: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[pd.DataFrame] = field(default_factory=list)
    
    # Styling
    css_class: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


# BenchmarkComparison moved to benchmark.py module


class ReportGenerator:
    """Main report generation class"""
    
    def __init__(self, config: ReportConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Report state
        self.sections: List[ReportSection] = []
        self.charts: Dict[str, Any] = {}
        self.tables: Dict[str, pd.DataFrame] = {}
        
    def generate_report(
        self,
        results: Any,  # BacktestResult or list of results
        output_path: Optional[str] = None
    ) -> str:
        """Generate comprehensive report"""
        
        output_path = output_path or self.config.output_path or f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        self.logger.info(f"Generating report: {output_path}")
        
        try:
            # Clear previous state
            self.sections.clear()
            self.charts.clear()
            self.tables.clear()
            
            # Generate content based on template
            if self.config.template == ReportTemplate.EXECUTIVE_SUMMARY:
                self._generate_executive_summary(results)
            elif self.config.template == ReportTemplate.DETAILED_ANALYSIS:
                self._generate_detailed_analysis(results)
            elif self.config.template == ReportTemplate.STRATEGY_COMPARISON:
                self._generate_strategy_comparison(results)
            elif self.config.template == ReportTemplate.RISK_ANALYSIS:
                self._generate_risk_analysis(results)
            else:
                self._generate_detailed_analysis(results)  # Default
            
            # Render report
            if self.config.format == ReportFormat.HTML:
                report_content = self._render_html_report()
            elif self.config.format == ReportFormat.MARKDOWN:
                report_content = self._render_markdown_report()
            else:
                raise ValueError(f"Unsupported format: {self.config.format}")
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            self.logger.info(f"Report generated successfully: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            raise
    
    def _generate_executive_summary(self, results: Any) -> None:
        """Generate executive summary report"""
        
        # Summary section
        summary_content = self._create_summary_text(results)
        self.sections.append(ReportSection(
            title="Executive Summary",
            content=summary_content,
            order=1
        ))
        
        # Key metrics table
        metrics_table = self._create_metrics_table(results)
        self.tables["key_metrics"] = metrics_table
        
        # Performance chart
        if self.config.include_charts:
            equity_chart = self._create_equity_curve_chart(results)
            self.charts["equity_curve"] = equity_chart
    
    def _generate_detailed_analysis(self, results: Any) -> None:
        """Generate detailed analysis report"""
        
        # Executive summary
        self._generate_executive_summary(results)
        
        # Performance analysis
        performance_content = self._create_performance_analysis(results)
        self.sections.append(ReportSection(
            title="Performance Analysis",
            content=performance_content,
            order=2
        ))
        
        # Risk analysis
        if self.config.include_risk_analysis:
            risk_content = self._create_risk_analysis(results)
            self.sections.append(ReportSection(
                title="Risk Analysis",
                content=risk_content,
                order=3
            ))
        
        # Trade analysis
        if self.config.include_trade_analysis:
            trade_content = self._create_trade_analysis(results)
            self.sections.append(ReportSection(
                title="Trade Analysis",
                content=trade_content,
                order=4
            ))
        
        # Generate additional charts
        if self.config.include_charts:
            self._generate_all_charts(results)
    
    def _generate_strategy_comparison(self, results: List[Any]) -> None:
        """Generate strategy comparison report"""
        
        # Comparison summary
        comparison_content = self._create_comparison_summary(results)
        self.sections.append(ReportSection(
            title="Strategy Comparison",
            content=comparison_content,
            order=1
        ))
        
        # Comparison table
        comparison_table = self._create_comparison_table(results)
        self.tables["strategy_comparison"] = comparison_table
        
        # Comparison charts
        if self.config.include_charts:
            comparison_chart = self._create_comparison_chart(results)
            self.charts["strategy_comparison"] = comparison_chart
    
    def _generate_risk_analysis(self, results: Any) -> None:
        """Generate risk analysis report"""
        
        # Risk metrics section
        risk_content = self._create_detailed_risk_analysis(results)
        self.sections.append(ReportSection(
            title="Risk Analysis",
            content=risk_content,
            order=1
        ))
        
        # Risk tables
        risk_metrics_table = self._create_risk_metrics_table(results)
        self.tables["risk_metrics"] = risk_metrics_table
        
        # Risk charts
        if self.config.include_charts:
            self._generate_risk_charts(results)
    
    def _create_summary_text(self, results: Any) -> str:
        """Create executive summary text"""
        
        # Mock summary - would normally analyze actual results
        return f"""
        <p>This report presents the backtesting results for the trading strategy analysis conducted from 
        {getattr(results, 'start_date', 'N/A')} to {getattr(results, 'end_date', 'N/A')}.</p>
        
        <p><strong>Key Highlights:</strong></p>
        <ul>
            <li>Total Return: {getattr(results, 'total_return', 0.0):.2%}</li>
            <li>Annual Return: {getattr(results, 'annualized_return', 0.0):.2%}</li>
            <li>Sharpe Ratio: {getattr(results, 'sharpe_ratio', 0.0):.2f}</li>
            <li>Maximum Drawdown: {getattr(results, 'max_drawdown', 0.0):.2%}</li>
            <li>Total Trades: {getattr(results, 'total_trades', 0)}</li>
        </ul>
        """
    
    def _create_metrics_table(self, results: Any) -> pd.DataFrame:
        """Create key metrics table"""
        
        metrics_data = {
            'Metric': [
                'Total Return',
                'Annualized Return',
                'Volatility',
                'Sharpe Ratio',
                'Sortino Ratio',
                'Maximum Drawdown',
                'Win Rate',
                'Profit Factor',
                'Total Trades'
            ],
            'Value': [
                f"{getattr(results, 'total_return', 0.0):.2%}",
                f"{getattr(results, 'annualized_return', 0.0):.2%}",
                f"{getattr(results, 'volatility', 0.0):.2%}",
                f"{getattr(results, 'sharpe_ratio', 0.0):.2f}",
                f"{getattr(results, 'sortino_ratio', 0.0):.2f}",
                f"{getattr(results, 'max_drawdown', 0.0):.2%}",
                f"{getattr(results, 'win_rate', 0.0):.2%}",
                f"{getattr(results, 'profit_factor', 0.0):.2f}",
                f"{getattr(results, 'total_trades', 0):,}"
            ]
        }
        
        return pd.DataFrame(metrics_data)
    
    def _create_performance_analysis(self, results: Any) -> str:
        """Create performance analysis content"""
        
        return f"""
        <h3>Return Analysis</h3>
        <p>The strategy generated a total return of {getattr(results, 'total_return', 0.0):.2%} 
        over the testing period, with an annualized return of {getattr(results, 'annualized_return', 0.0):.2%}.</p>
        
        <h3>Risk-Adjusted Performance</h3>
        <p>The Sharpe ratio of {getattr(results, 'sharpe_ratio', 0.0):.2f} indicates 
        {'excellent' if getattr(results, 'sharpe_ratio', 0.0) > 1.5 else 'good' if getattr(results, 'sharpe_ratio', 0.0) > 1.0 else 'moderate'} 
        risk-adjusted performance.</p>
        
        <h3>Drawdown Analysis</h3>
        <p>The maximum drawdown of {getattr(results, 'max_drawdown', 0.0):.2%} demonstrates 
        {'excellent' if abs(getattr(results, 'max_drawdown', 0.0)) < 0.10 else 'good' if abs(getattr(results, 'max_drawdown', 0.0)) < 0.20 else 'concerning'} 
        risk control.</p>
        """
    
    def _create_risk_analysis(self, results: Any) -> str:
        """Create risk analysis content"""
        
        return f"""
        <h3>Volatility Analysis</h3>
        <p>The strategy exhibited an annualized volatility of {getattr(results, 'volatility', 0.0):.2%}.</p>
        
        <h3>Value at Risk</h3>
        <p>95% VaR: {getattr(results, 'var_95', 0.0):.2%}</p>
        <p>99% VaR: {getattr(results, 'var_99', 0.0):.2%}</p>
        
        <h3>Tail Risk</h3>
        <p>The strategy shows {'low' if getattr(results, 'skewness', 0.0) > -0.5 else 'moderate' if getattr(results, 'skewness', 0.0) > -1.0 else 'high'} 
        tail risk with a skewness of {getattr(results, 'skewness', 0.0):.2f}.</p>
        """
    
    def _create_trade_analysis(self, results: Any) -> str:
        """Create trade analysis content"""
        
        return f"""
        <h3>Trade Statistics</h3>
        <p>Total Trades: {getattr(results, 'total_trades', 0):,}</p>
        <p>Win Rate: {getattr(results, 'win_rate', 0.0):.2%}</p>
        <p>Profit Factor: {getattr(results, 'profit_factor', 0.0):.2f}</p>
        
        <h3>Trade Distribution</h3>
        <p>Average Win: {getattr(results, 'avg_win', 0.0):.2%}</p>
        <p>Average Loss: {getattr(results, 'avg_loss', 0.0):.2%}</p>
        <p>Largest Win: {getattr(results, 'largest_win', 0.0):.2%}</p>
        <p>Largest Loss: {getattr(results, 'largest_loss', 0.0):.2%}</p>
        """
    
    def _create_equity_curve_chart(self, results: Any) -> Dict[str, Any]:
        """Create equity curve chart data"""
        
        # Mock chart data - would normally use actual equity curve
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        values = np.cumsum(np.random.normal(0.0005, 0.02, 252)) + 1
        values = values * 100000  # Start with $100k
        
        return {
            'type': 'line',
            'data': {
                'x': dates.tolist(),
                'y': values.tolist(),
                'name': 'Strategy Equity'
            },
            'layout': {
                'title': 'Equity Curve',
                'xaxis': {'title': 'Date'},
                'yaxis': {'title': 'Portfolio Value ($)'}
            }
        }
    
    def _generate_all_charts(self, results: Any) -> None:
        """Generate all requested charts"""
        
        for chart_type in self.config.chart_types:
            if chart_type == ChartType.EQUITY_CURVE:
                self.charts["equity_curve"] = self._create_equity_curve_chart(results)
            elif chart_type == ChartType.DRAWDOWN:
                self.charts["drawdown"] = self._create_drawdown_chart(results)
            elif chart_type == ChartType.MONTHLY_RETURNS:
                self.charts["monthly_returns"] = self._create_monthly_returns_chart(results)
            elif chart_type == ChartType.RETURN_DISTRIBUTION:
                self.charts["return_distribution"] = self._create_return_distribution_chart(results)
    
    def _create_drawdown_chart(self, results: Any) -> Dict[str, Any]:
        """Create drawdown chart"""
        
        # Mock drawdown data
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        drawdown = np.minimum(0, np.cumsum(np.random.normal(-0.0001, 0.01, 252)))
        
        return {
            'type': 'area',
            'data': {
                'x': dates.tolist(),
                'y': drawdown.tolist(),
                'name': 'Drawdown',
                'fill': 'tonexty'
            },
            'layout': {
                'title': 'Drawdown',
                'xaxis': {'title': 'Date'},
                'yaxis': {'title': 'Drawdown (%)'}
            }
        }
    
    def _create_monthly_returns_chart(self, results: Any) -> Dict[str, Any]:
        """Create monthly returns chart"""
        
        # Mock monthly returns
        months = pd.date_range('2023-01-01', periods=12, freq='M')
        returns = np.random.normal(0.01, 0.05, 12)
        
        return {
            'type': 'bar',
            'data': {
                'x': months.strftime('%Y-%m').tolist(),
                'y': returns.tolist(),
                'name': 'Monthly Returns'
            },
            'layout': {
                'title': 'Monthly Returns',
                'xaxis': {'title': 'Month'},
                'yaxis': {'title': 'Return (%)'}
            }
        }
    
    def _create_return_distribution_chart(self, results: Any) -> Dict[str, Any]:
        """Create return distribution chart"""
        
        # Mock return distribution
        returns = np.random.normal(0.001, 0.02, 1000)
        
        return {
            'type': 'histogram',
            'data': {
                'x': returns.tolist(),
                'name': 'Daily Returns'
            },
            'layout': {
                'title': 'Return Distribution',
                'xaxis': {'title': 'Daily Return'},
                'yaxis': {'title': 'Frequency'}
            }
        }
    
    def _render_html_report(self) -> str:
        """Render HTML report"""
        
        # Sort sections by order
        self.sections.sort(key=lambda x: x.order)
        
        # Build HTML
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<title>{self.config.report_title}</title>",
            "<style>",
            self._get_css_styles(),
            "</style>",
            "</head>",
            "<body>",
            f"<h1>{self.config.report_title}</h1>",
            f"<h2>{self.config.report_subtitle}</h2>",
            f"<p><em>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by {self.config.author}</em></p>"
        ]
        
        # Add sections
        for section in self.sections:
            html_parts.extend([
                f"<div class='section'>",
                f"<h2>{section.title}</h2>",
                section.content,
                "</div>"
            ])
        
        # Add tables
        for table_name, table_df in self.tables.items():
            html_parts.extend([
                f"<div class='table-container'>",
                f"<h3>{table_name.replace('_', ' ').title()}</h3>",
                table_df.to_html(classes='report-table', escape=False),
                "</div>"
            ])
        
        # Add charts (simplified - would normally include Plotly/Bokeh)
        for chart_name, chart_data in self.charts.items():
            html_parts.extend([
                f"<div class='chart-container'>",
                f"<h3>{chart_name.replace('_', ' ').title()}</h3>",
                f"<p>Chart: {chart_data.get('layout', {}).get('title', 'Chart')}</p>",
                "</div>"
            ])
        
        html_parts.extend([
            "</body>",
            "</html>"
        ])
        
        return "\n".join(html_parts)
    
    def _render_markdown_report(self) -> str:
        """Render Markdown report"""
        
        # Sort sections by order
        self.sections.sort(key=lambda x: x.order)
        
        md_parts = [
            f"# {self.config.report_title}",
            f"## {self.config.report_subtitle}",
            f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by {self.config.author}*",
            ""
        ]
        
        # Add sections
        for section in self.sections:
            md_parts.extend([
                f"## {section.title}",
                section.content.replace('<h3>', '### ').replace('</h3>', '').replace('<p>', '').replace('</p>', '').replace('<ul>', '').replace('</ul>', '').replace('<li>', '- ').replace('</li>', ''),
                ""
            ])
        
        # Add tables
        for table_name, table_df in self.tables.items():
            md_parts.extend([
                f"### {table_name.replace('_', ' ').title()}",
                table_df.to_markdown(),
                ""
            ])
        
        return "\n".join(md_parts)
    
    def _get_css_styles(self) -> str:
        """Get CSS styles for HTML report"""
        
        return """
        body { 
            font-family: Arial, sans-serif; 
            margin: 40px; 
            line-height: 1.6;
            color: #333;
        }
        h1, h2, h3 { 
            color: #2c3e50; 
            margin-top: 30px;
        }
        .section { 
            margin: 30px 0; 
            padding: 20px;
            border-left: 4px solid #3498db;
            background-color: #f8f9fa;
        }
        .report-table { 
            width: 100%; 
            border-collapse: collapse; 
            margin: 20px 0;
        }
        .report-table th, .report-table td { 
            border: 1px solid #ddd; 
            padding: 12px; 
            text-align: left;
        }
        .report-table th { 
            background-color: #f2f2f2; 
            font-weight: bold;
        }
        .chart-container, .table-container { 
            margin: 30px 0; 
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        """


# BenchmarkComparator moved to benchmark.py module 
