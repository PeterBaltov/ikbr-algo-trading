"""
Performance Analytics Package - Phase 4.2

Advanced performance analytics and risk analysis for backtesting results.
Provides institutional-grade metrics, risk analytics, attribution analysis,
and visualization capabilities.

This package implements the 4.2 Performance Analytics component of Phase 4,
providing comprehensive tools for strategy evaluation and reporting.

Integration:
- Works with backtesting engine results
- Supports multi-strategy analysis  
- Compatible with external benchmarks
- Provides visualization-ready outputs
"""

from .performance import PerformanceCalculator, PerformanceMetrics
from .risk import RiskCalculator, RiskMetrics, VaRCalculator
from .attribution import AttributionAnalyzer, BenchmarkComparator, BenchmarkComparison
from .visualization import ChartGenerator, VisualizationConfig

__all__ = [
    # Core performance analytics
    "PerformanceCalculator",
    "PerformanceMetrics", 
    
    # Risk analytics
    "RiskCalculator",
    "RiskMetrics",
    "VaRCalculator",
    
    # Attribution analysis
    "AttributionAnalyzer", 
    "BenchmarkComparator",
    "BenchmarkComparison",
    
    # Visualization
    "ChartGenerator",
    "VisualizationConfig"
] 
