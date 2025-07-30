"""
Visualization Module

Advanced charting and plotting utilities for performance analytics and 
backtesting results. Provides interactive charts and publication-ready
visualizations for strategy analysis.

Features:
- Interactive performance charts (equity curves, drawdowns)
- Risk analysis visualizations (VaR, correlation heatmaps)
- Return distribution analysis and histograms
- Rolling metrics and time series analysis
- Multi-strategy comparison charts
- Export capabilities for reports

Integration:
- Used by reporting modules for chart generation
- Compatible with performance and risk analytics
- Supports multiple chart formats and styles
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Union, Tuple
import logging

import pandas as pd
import numpy as np


class ChartType(Enum):
    """Types of charts for visualization"""
    EQUITY_CURVE = "equity_curve"
    DRAWDOWN = "drawdown"
    ROLLING_RETURNS = "rolling_returns"
    MONTHLY_RETURNS = "monthly_returns"
    RETURN_DISTRIBUTION = "return_distribution"
    UNDERWATER_PLOT = "underwater_plot"
    ROLLING_SHARPE = "rolling_sharpe"
    CORRELATION_HEATMAP = "correlation_heatmap"
    TRADE_ANALYSIS = "trade_analysis"
    RISK_METRICS = "risk_metrics"
    BENCHMARK_COMPARISON = "benchmark_comparison"


class ChartStyle(Enum):
    """Chart styling options"""
    DEFAULT = "default"
    DARK = "dark"
    PROFESSIONAL = "professional"
    MINIMAL = "minimal"
    PUBLICATION = "publication"


@dataclass
class VisualizationConfig:
    """Configuration for chart generation"""
    
    # Chart dimensions
    width: int = 800
    height: int = 400
    
    # Styling
    style: ChartStyle = ChartStyle.DEFAULT
    color_palette: List[str] = field(default_factory=lambda: [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"
    ])
    
    # Chart options
    show_grid: bool = True
    show_legend: bool = True
    interactive: bool = True
    
    # Export options
    export_format: str = "html"  # html, png, svg, pdf
    dpi: int = 300
    
    # Metadata
    title_prefix: str = ""
    subtitle: str = ""
    watermark: str = ""


class ChartGenerator:
    """Main chart generation engine"""
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        self.logger = logging.getLogger(__name__)
    
    def create_equity_curve_chart(
        self,
        equity_data: Union[pd.Series, Dict[str, pd.Series]],
        title: str = "Equity Curve",
        benchmark: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Create equity curve visualization"""
        
        chart_data = {
            'type': 'line',
            'title': f"{self.config.title_prefix}{title}",
            'layout': {
                'width': self.config.width,
                'height': self.config.height,
                'xaxis': {'title': 'Date'},
                'yaxis': {'title': 'Portfolio Value ($)'},
                'showlegend': self.config.show_legend,
                'grid': self.config.show_grid
            },
            'data': []
        }
        
        # Handle single series or multiple series
        if isinstance(equity_data, pd.Series):
            equity_data = {'Strategy': equity_data}
        
        # Add strategy lines
        for i, (name, series) in enumerate(equity_data.items()):
            color = self.config.color_palette[i % len(self.config.color_palette)]
            
            chart_data['data'].append({
                'x': series.index.tolist(),
                'y': series.values.tolist(),
                'name': name,
                'line': {'color': color, 'width': 2},
                'type': 'scatter',
                'mode': 'lines'
            })
        
        # Add benchmark if provided
        if benchmark is not None:
            chart_data['data'].append({
                'x': benchmark.index.tolist(),
                'y': benchmark.values.tolist(),
                'name': 'Benchmark',
                'line': {'color': '#666666', 'width': 1, 'dash': 'dash'},
                'type': 'scatter',
                'mode': 'lines'
            })
        
        return chart_data
    
    def create_drawdown_chart(
        self,
        equity_curve: pd.Series,
        title: str = "Drawdown Analysis"
    ) -> Dict[str, Any]:
        """Create drawdown visualization"""
        
        # Calculate drawdown
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max * 100  # Convert to percentage
        
        chart_data = {
            'type': 'area',
            'title': f"{self.config.title_prefix}{title}",
            'layout': {
                'width': self.config.width,
                'height': self.config.height,
                'xaxis': {'title': 'Date'},
                'yaxis': {'title': 'Drawdown (%)'},
                'showlegend': False,
                'grid': self.config.show_grid
            },
            'data': [{
                'x': drawdown.index.tolist(),
                'y': drawdown.values.tolist(),
                'name': 'Drawdown',
                'fill': 'tonexty',
                'fillcolor': 'rgba(255, 0, 0, 0.3)',
                'line': {'color': '#d62728', 'width': 1},
                'type': 'scatter',
                'mode': 'lines'
            }]
        }
        
        return chart_data
    
    def create_return_distribution_chart(
        self,
        returns: pd.Series,
        title: str = "Return Distribution",
        bins: int = 50
    ) -> Dict[str, Any]:
        """Create return distribution histogram"""
        
        # Calculate histogram
        hist, bin_edges = np.histogram(returns.dropna(), bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        chart_data = {
            'type': 'histogram',
            'title': f"{self.config.title_prefix}{title}",
            'layout': {
                'width': self.config.width,
                'height': self.config.height,
                'xaxis': {'title': 'Daily Return'},
                'yaxis': {'title': 'Frequency'},
                'showlegend': False,
                'grid': self.config.show_grid
            },
            'data': [{
                'x': returns.dropna().tolist(),
                'type': 'histogram',
                'nbinsx': bins,
                'name': 'Returns',
                'marker': {'color': self.config.color_palette[0], 'opacity': 0.7}
            }]
        }
        
        # Add normal distribution overlay
        mean_return = returns.mean()
        std_return = returns.std()
        x_norm = np.linspace(returns.min(), returns.max(), 100)
        y_norm = ((1 / (std_return * np.sqrt(2 * np.pi))) * 
                  np.exp(-0.5 * ((x_norm - mean_return) / std_return) ** 2))
        
        # Scale normal distribution to match histogram
        y_norm_scaled = y_norm * len(returns) * (returns.max() - returns.min()) / bins
        
        chart_data['data'].append({
            'x': x_norm.tolist(),
            'y': y_norm_scaled.tolist(),
            'name': 'Normal Distribution',
            'line': {'color': '#ff7f0e', 'width': 2},
            'type': 'scatter',
            'mode': 'lines'
        })
        
        return chart_data
    
    def create_monthly_returns_heatmap(
        self,
        equity_curve: pd.Series,
        title: str = "Monthly Returns Heatmap"
    ) -> Dict[str, Any]:
        """Create monthly returns heatmap"""
        
        # Calculate monthly returns
        monthly_equity = equity_curve.resample('M').last()
        monthly_returns = monthly_equity.pct_change().dropna() * 100  # Convert to percentage
        
        # Create pivot table for heatmap
        monthly_returns_df = monthly_returns.to_frame('returns')
        monthly_returns_df['year'] = monthly_returns_df.index.year
        monthly_returns_df['month'] = monthly_returns_df.index.month
        
        pivot_data = monthly_returns_df.pivot(index='year', columns='month', values='returns')
        
        # Fill missing months with NaN
        for month in range(1, 13):
            if month not in pivot_data.columns:
                pivot_data[month] = np.nan
        
        pivot_data = pivot_data.sort_index(axis=1)
        
        chart_data = {
            'type': 'heatmap',
            'title': f"{self.config.title_prefix}{title}",
            'layout': {
                'width': self.config.width,
                'height': self.config.height,
                'xaxis': {
                    'title': 'Month',
                    'tickvals': list(range(1, 13)),
                    'ticktext': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                },
                'yaxis': {'title': 'Year'},
                'grid': False
            },
            'data': [{
                'z': pivot_data.values.tolist(),
                'x': list(range(1, 13)),
                'y': pivot_data.index.tolist(),
                'type': 'heatmap',
                'colorscale': [
                    [0, '#d62728'],      # Red for negative
                    [0.5, '#ffffff'],    # White for zero
                    [1, '#2ca02c']       # Green for positive
                ],
                'zmid': 0,
                'colorbar': {'title': 'Return (%)'}
            }]
        }
        
        return chart_data
    
    def create_rolling_metrics_chart(
        self,
        equity_curve: pd.Series,
        metric: str = "sharpe",
        window: int = 252,
        title: str = None
    ) -> Dict[str, Any]:
        """Create rolling metrics visualization"""
        
        returns = equity_curve.pct_change().dropna()
        
        if metric == "sharpe":
            rolling_values = self._calculate_rolling_sharpe(returns, window)
            y_title = "Rolling Sharpe Ratio"
            chart_title = title or "Rolling Sharpe Ratio"
        elif metric == "volatility":
            rolling_values = returns.rolling(window).std() * np.sqrt(252) * 100
            y_title = "Rolling Volatility (%)"
            chart_title = title or "Rolling Volatility"
        elif metric == "returns":
            rolling_values = returns.rolling(window).mean() * 252 * 100
            y_title = "Rolling Returns (%)"
            chart_title = title or "Rolling Returns"
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        chart_data = {
            'type': 'line',
            'title': f"{self.config.title_prefix}{chart_title}",
            'layout': {
                'width': self.config.width,
                'height': self.config.height,
                'xaxis': {'title': 'Date'},
                'yaxis': {'title': y_title},
                'showlegend': False,
                'grid': self.config.show_grid
            },
            'data': [{
                'x': rolling_values.index.tolist(),
                'y': rolling_values.values.tolist(),
                'name': f'Rolling {metric.title()}',
                'line': {'color': self.config.color_palette[0], 'width': 2},
                'type': 'scatter',
                'mode': 'lines'
            }]
        }
        
        return chart_data
    
    def create_correlation_heatmap(
        self,
        returns_data: Dict[str, pd.Series],
        title: str = "Correlation Matrix"
    ) -> Dict[str, Any]:
        """Create correlation heatmap for multiple return series"""
        
        # Align all series and calculate correlation matrix
        returns_df = pd.DataFrame(returns_data)
        correlation_matrix = returns_df.corr()
        
        strategy_names = list(returns_data.keys())
        
        chart_data = {
            'type': 'heatmap',
            'title': f"{self.config.title_prefix}{title}",
            'layout': {
                'width': self.config.width,
                'height': self.config.height,
                'xaxis': {'title': 'Strategy', 'tickangle': 45},
                'yaxis': {'title': 'Strategy'},
                'grid': False
            },
            'data': [{
                'z': correlation_matrix.values.tolist(),
                'x': strategy_names,
                'y': strategy_names,
                'type': 'heatmap',
                'colorscale': 'RdBu',
                'zmid': 0,
                'zmin': -1,
                'zmax': 1,
                'colorbar': {'title': 'Correlation'}
            }]
        }
        
        return chart_data
    
    def create_underwater_plot(
        self,
        equity_curve: pd.Series,
        title: str = "Underwater Plot"
    ) -> Dict[str, Any]:
        """Create underwater (drawdown) plot"""
        
        # Calculate underwater curve (time underwater)
        running_max = equity_curve.expanding().max()
        is_underwater = equity_curve < running_max
        
        # Calculate days underwater
        underwater_periods = []
        current_period = 0
        
        for is_under in is_underwater:
            if is_under:
                current_period += 1
            else:
                current_period = 0
            underwater_periods.append(current_period)
        
        underwater_series = pd.Series(underwater_periods, index=equity_curve.index)
        
        chart_data = {
            'type': 'area',
            'title': f"{self.config.title_prefix}{title}",
            'layout': {
                'width': self.config.width,
                'height': self.config.height,
                'xaxis': {'title': 'Date'},
                'yaxis': {'title': 'Days Underwater'},
                'showlegend': False,
                'grid': self.config.show_grid
            },
            'data': [{
                'x': underwater_series.index.tolist(),
                'y': underwater_series.values.tolist(),
                'name': 'Days Underwater',
                'fill': 'tozeroy',
                'fillcolor': 'rgba(255, 0, 0, 0.3)',
                'line': {'color': '#d62728', 'width': 1},
                'type': 'scatter',
                'mode': 'lines'
            }]
        }
        
        return chart_data
    
    def create_risk_return_scatter(
        self,
        strategies_data: Dict[str, Dict[str, float]],
        title: str = "Risk-Return Scatter"
    ) -> Dict[str, Any]:
        """Create risk-return scatter plot"""
        
        chart_data = {
            'type': 'scatter',
            'title': f"{self.config.title_prefix}{title}",
            'layout': {
                'width': self.config.width,
                'height': self.config.height,
                'xaxis': {'title': 'Volatility (%)'},
                'yaxis': {'title': 'Return (%)'},
                'showlegend': True,
                'grid': self.config.show_grid
            },
            'data': []
        }
        
        for i, (strategy_name, metrics) in enumerate(strategies_data.items()):
            color = self.config.color_palette[i % len(self.config.color_palette)]
            
            chart_data['data'].append({
                'x': [metrics.get('volatility', 0) * 100],
                'y': [metrics.get('return', 0) * 100],
                'name': strategy_name,
                'mode': 'markers+text',
                'text': [strategy_name],
                'textposition': 'top center',
                'marker': {
                    'color': color,
                    'size': 12,
                    'symbol': 'circle'
                },
                'type': 'scatter'
            })
        
        return chart_data
    
    def _calculate_rolling_sharpe(self, returns: pd.Series, window: int, risk_free_rate: float = 0.02) -> pd.Series:
        """Calculate rolling Sharpe ratio"""
        
        excess_returns = returns - (risk_free_rate / 252)
        rolling_mean = excess_returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        
        rolling_sharpe = rolling_mean / rolling_std * np.sqrt(252)
        return rolling_sharpe.fillna(0)
    
    def export_chart(
        self,
        chart_data: Dict[str, Any],
        filename: str,
        format: str = None
    ) -> str:
        """Export chart to file"""
        
        export_format = format or self.config.export_format
        
        # In a real implementation, this would use a plotting library
        # like Plotly, Matplotlib, or Bokeh to generate actual files
        
        output_file = f"{filename}.{export_format}"
        
        # Mock export - would actually generate the file
        self.logger.info(f"Chart exported to {output_file}")
        
        return output_file
    
    def create_dashboard(
        self,
        equity_curve: pd.Series,
        benchmark: Optional[pd.Series] = None,
        trades: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Create comprehensive dashboard with multiple charts"""
        
        dashboard = {}
        
        # Main equity curve
        dashboard['equity_curve'] = self.create_equity_curve_chart(
            equity_curve, "Strategy Performance", benchmark
        )
        
        # Drawdown analysis
        dashboard['drawdown'] = self.create_drawdown_chart(equity_curve)
        
        # Return distribution
        returns = equity_curve.pct_change().dropna()
        dashboard['return_distribution'] = self.create_return_distribution_chart(returns)
        
        # Monthly returns heatmap
        dashboard['monthly_heatmap'] = self.create_monthly_returns_heatmap(equity_curve)
        
        # Rolling Sharpe
        dashboard['rolling_sharpe'] = self.create_rolling_metrics_chart(
            equity_curve, "sharpe", title="Rolling Sharpe Ratio (1Y)"
        )
        
        # Underwater plot
        dashboard['underwater'] = self.create_underwater_plot(equity_curve)
        
        return dashboard 
