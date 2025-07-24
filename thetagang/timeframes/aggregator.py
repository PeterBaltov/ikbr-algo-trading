"""
Data Aggregator

The DataAggregator provides efficient utilities for aggregating and processing
multi-timeframe datasets, optimizing memory usage and computation performance.

Key features:
- Efficient data aggregation across timeframes
- Memory-optimized operations
- Statistical computations
- Data quality assessment
- Batch processing capabilities
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Union, Callable
import pandas as pd
import numpy as np
import asyncio
import logging
from collections import defaultdict

from ..strategies.enums import TimeFrame
from ..strategies.exceptions import StrategyError


class AggregationMethod(Enum):
    """Methods for data aggregation"""
    MEAN = "mean"
    MEDIAN = "median"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    STD = "std"
    VAR = "var"
    COUNT = "count"
    FIRST = "first"
    LAST = "last"
    OHLC = "ohlc"
    VWAP = "vwap"
    CUSTOM = "custom"


class AggregationScope(Enum):
    """Scope of data aggregation"""
    SYMBOL = "symbol"              # Aggregate per symbol
    TIMEFRAME = "timeframe"        # Aggregate per timeframe
    CROSS_TIMEFRAME = "cross_timeframe"  # Aggregate across timeframes
    GLOBAL = "global"              # Aggregate all data


@dataclass
class AggregationConfig:
    """Configuration for data aggregation"""
    method: AggregationMethod
    scope: AggregationScope
    timeframes: Set[TimeFrame]
    symbols: Set[str]
    columns: Optional[List[str]] = None
    groupby_columns: Optional[List[str]] = None
    custom_function: Optional[Callable[[pd.DataFrame], Any]] = None
    rolling_window: Optional[int] = None
    min_periods: int = 1
    center: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregationResult:
    """Result of data aggregation operation"""
    method: AggregationMethod
    scope: AggregationScope
    aggregated_data: Dict[str, Any]
    source_timeframes: Set[TimeFrame]
    source_symbols: Set[str]
    aggregation_timestamp: datetime
    processing_duration_ms: float
    data_points_processed: int
    warnings: List[str]
    metadata: Dict[str, Any]


class DataAggregationError(StrategyError):
    """Data aggregation related errors"""
    pass


class DataAggregator:
    """
    Multi-Timeframe Data Aggregator
    
    Provides efficient aggregation utilities for processing large datasets
    across multiple timeframes and symbols with optimized memory usage.
    """
    
    def __init__(self, cache_enabled: bool = True):
        self.cache_enabled = cache_enabled
        self.logger = logging.getLogger(__name__)
        
        # Aggregation cache
        self._aggregation_cache: Dict[str, AggregationResult] = {}
        self._cache_ttl_seconds = 600  # 10 minutes
        
        # Performance tracking
        self._aggregation_operations = 0
        self._total_processing_time_ms = 0.0
        self._cache_hits = 0
        self._cache_misses = 0
        
    def aggregate_data(
        self,
        data_dict: Dict[TimeFrame, Dict[str, pd.DataFrame]],
        config: AggregationConfig
    ) -> AggregationResult:
        """
        Aggregate data according to configuration
        
        Args:
            data_dict: Dictionary mapping timeframes to symbol data
            config: Aggregation configuration
            
        Returns:
            AggregationResult with aggregated data and metadata
        """
        start_time = datetime.now()
        
        try:
            # Check cache first
            if self.cache_enabled:
                cache_key = self._generate_cache_key(data_dict, config)
                cached_result = self._get_cached_result(cache_key)
                if cached_result:
                    self._cache_hits += 1
                    return cached_result
                self._cache_misses += 1
            
            # Perform aggregation based on scope
            if config.scope == AggregationScope.SYMBOL:
                result = self._aggregate_by_symbol(data_dict, config, start_time)
            elif config.scope == AggregationScope.TIMEFRAME:
                result = self._aggregate_by_timeframe(data_dict, config, start_time)
            elif config.scope == AggregationScope.CROSS_TIMEFRAME:
                result = self._aggregate_cross_timeframe(data_dict, config, start_time)
            elif config.scope == AggregationScope.GLOBAL:
                result = self._aggregate_global(data_dict, config, start_time)
            else:
                raise DataAggregationError(f"Unsupported aggregation scope: {config.scope}")
            
            # Cache the result
            if self.cache_enabled:
                self._cache_result(cache_key, result)
            
            # Update performance metrics
            self._update_performance_metrics(result.processing_duration_ms)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error aggregating data: {e}")
            raise DataAggregationError(f"Aggregation failed: {e}")
    
    async def aggregate_data_async(
        self,
        data_dict: Dict[TimeFrame, Dict[str, pd.DataFrame]],
        config: AggregationConfig
    ) -> AggregationResult:
        """Asynchronous version of data aggregation"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.aggregate_data, data_dict, config
        )
    
    def aggregate_multiple_configs(
        self,
        data_dict: Dict[TimeFrame, Dict[str, pd.DataFrame]],
        configs: List[AggregationConfig]
    ) -> Dict[str, AggregationResult]:
        """Aggregate data using multiple configurations"""
        results = {}
        
        for i, config in enumerate(configs):
            try:
                config_name = config.metadata.get('name', f'config_{i}')
                results[config_name] = self.aggregate_data(data_dict, config)
            except Exception as e:
                self.logger.error(f"Error aggregating with config {i}: {e}")
        
        return results
    
    async def aggregate_multiple_configs_async(
        self,
        data_dict: Dict[TimeFrame, Dict[str, pd.DataFrame]],
        configs: List[AggregationConfig],
        max_concurrent: int = 5
    ) -> Dict[str, AggregationResult]:
        """Asynchronously aggregate data using multiple configurations"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def aggregate_config(i: int, config: AggregationConfig):
            async with semaphore:
                return await self.aggregate_data_async(data_dict, config)
        
        tasks = [
            aggregate_config(i, config)
            for i, config in enumerate(configs)
        ]
        
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        results = {}
        for i, (config, result) in enumerate(zip(configs, results_list)):
            if isinstance(result, Exception):
                self.logger.error(f"Async aggregation error for config {i}: {result}")
                continue
            
            config_name = config.metadata.get('name', f'config_{i}')
            results[config_name] = result
        
        return results
    
    def calculate_cross_timeframe_correlation(
        self,
        data_dict: Dict[TimeFrame, Dict[str, pd.DataFrame]],
        symbol: str,
        column: str = 'close'
    ) -> pd.DataFrame:
        """Calculate correlation matrix across timeframes for a symbol"""
        try:
            correlation_data = {}
            
            for timeframe, timeframe_data in data_dict.items():
                if symbol in timeframe_data and column in timeframe_data[symbol].columns:
                    data = timeframe_data[symbol][column].dropna()
                    correlation_data[timeframe.value] = data
            
            if len(correlation_data) < 2:
                raise DataAggregationError("Need at least 2 timeframes for correlation analysis")
            
            # Create DataFrame with aligned timestamps
            df = pd.DataFrame(correlation_data)
            
            # Calculate correlation matrix
            correlation_matrix = df.corr()
            
            return correlation_matrix
            
        except Exception as e:
            raise DataAggregationError(f"Error calculating cross-timeframe correlation: {e}")
    
    def calculate_data_quality_metrics(
        self,
        data_dict: Dict[TimeFrame, Dict[str, pd.DataFrame]]
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate data quality metrics across timeframes and symbols"""
        quality_metrics = {}
        
        for timeframe, timeframe_data in data_dict.items():
            timeframe_metrics = {}
            
            for symbol, data in timeframe_data.items():
                symbol_metrics = {
                    'total_records': len(data),
                    'missing_data_pct': (data.isna().sum().sum() / (len(data) * len(data.columns))) * 100,
                    'duplicate_timestamps': data.index.duplicated().sum(),
                    'date_range': {
                        'start': data.index.min() if not data.empty else None,
                        'end': data.index.max() if not data.empty else None
                    },
                    'data_completeness': (1 - data.isna().mean()).mean() * 100,
                    'columns': list(data.columns)
                }
                
                # Check for gaps in time series
                if len(data) > 1:
                    time_diffs = data.index.to_series().diff().dropna()
                    expected_interval = self._estimate_expected_interval(timeframe)
                    
                    if expected_interval:
                        gaps = time_diffs[time_diffs > expected_interval * 2]
                        symbol_metrics['time_gaps'] = len(gaps)
                        symbol_metrics['largest_gap_hours'] = gaps.max().total_seconds() / 3600 if len(gaps) > 0 else 0
                    else:
                        symbol_metrics['time_gaps'] = 0
                        symbol_metrics['largest_gap_hours'] = 0
                
                timeframe_metrics[symbol] = symbol_metrics
            
            quality_metrics[timeframe.value] = timeframe_metrics
        
        return quality_metrics
    
    def create_summary_statistics(
        self,
        data_dict: Dict[TimeFrame, Dict[str, pd.DataFrame]],
        columns: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Create summary statistics for all data"""
        summary_stats = {}
        
        for timeframe, timeframe_data in data_dict.items():
            timeframe_stats = {}
            
            for symbol, data in timeframe_data.items():
                # Select columns to analyze
                analyze_columns = columns or [col for col in data.columns if data[col].dtype in ['float64', 'int64']]
                
                if analyze_columns:
                    stats_data = data[analyze_columns]
                    
                    # Calculate comprehensive statistics
                    stats = pd.DataFrame({
                        'count': stats_data.count(),
                        'mean': stats_data.mean(),
                        'std': stats_data.std(),
                        'min': stats_data.min(),
                        '25%': stats_data.quantile(0.25),
                        '50%': stats_data.quantile(0.50),
                        '75%': stats_data.quantile(0.75),
                        'max': stats_data.max(),
                        'skew': stats_data.skew(),
                        'kurtosis': stats_data.kurtosis()
                    })
                    
                    timeframe_stats[symbol] = stats
            
            summary_stats[timeframe.value] = timeframe_stats
        
        return summary_stats
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get aggregation performance statistics"""
        avg_processing_time = (
            self._total_processing_time_ms / self._aggregation_operations
            if self._aggregation_operations > 0 else 0.0
        )
        
        cache_hit_rate = (
            self._cache_hits / (self._cache_hits + self._cache_misses)
            if (self._cache_hits + self._cache_misses) > 0 else 0.0
        )
        
        return {
            'total_aggregation_operations': self._aggregation_operations,
            'total_processing_time_ms': self._total_processing_time_ms,
            'average_processing_time_ms': avg_processing_time,
            'cache_enabled': self.cache_enabled,
            'cache_hit_rate': cache_hit_rate,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'cache_size': len(self._aggregation_cache)
        }
    
    def clear_cache(self) -> None:
        """Clear the aggregation cache"""
        self._aggregation_cache.clear()
        self.logger.info("Aggregation cache cleared")
    
    # Private methods
    
    def _aggregate_by_symbol(
        self,
        data_dict: Dict[TimeFrame, Dict[str, pd.DataFrame]],
        config: AggregationConfig,
        start_time: datetime
    ) -> AggregationResult:
        """Aggregate data by symbol across timeframes"""
        aggregated_data = {}
        warnings = []
        total_data_points = 0
        source_timeframes = set()
        source_symbols = set()
        
        for symbol in config.symbols:
            symbol_aggregation = {}
            
            for timeframe in config.timeframes:
                if timeframe in data_dict and symbol in data_dict[timeframe]:
                    data = data_dict[timeframe][symbol]
                    
                    if config.columns:
                        data = data[config.columns]
                    
                    # Apply aggregation method
                    aggregated_value = self._apply_aggregation_method(data, config)
                    symbol_aggregation[timeframe.value] = aggregated_value
                    
                    total_data_points += len(data)
                    source_timeframes.add(timeframe)
                    source_symbols.add(symbol)
                else:
                    warnings.append(f"No data found for {symbol} at {timeframe.value}")
            
            if symbol_aggregation:
                aggregated_data[symbol] = symbol_aggregation
        
        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000
        
        return AggregationResult(
            method=config.method,
            scope=config.scope,
            aggregated_data=aggregated_data,
            source_timeframes=source_timeframes,
            source_symbols=source_symbols,
            aggregation_timestamp=end_time,
            processing_duration_ms=duration_ms,
            data_points_processed=total_data_points,
            warnings=warnings,
            metadata=config.metadata
        )
    
    def _aggregate_by_timeframe(
        self,
        data_dict: Dict[TimeFrame, Dict[str, pd.DataFrame]],
        config: AggregationConfig,
        start_time: datetime
    ) -> AggregationResult:
        """Aggregate data by timeframe across symbols"""
        aggregated_data = {}
        warnings = []
        total_data_points = 0
        source_timeframes = set()
        source_symbols = set()
        
        for timeframe in config.timeframes:
            if timeframe not in data_dict:
                warnings.append(f"No data found for timeframe {timeframe.value}")
                continue
            
            timeframe_aggregation = {}
            timeframe_data = data_dict[timeframe]
            
            for symbol in config.symbols:
                if symbol in timeframe_data:
                    data = timeframe_data[symbol]
                    
                    if config.columns:
                        data = data[config.columns]
                    
                    # Apply aggregation method
                    aggregated_value = self._apply_aggregation_method(data, config)
                    timeframe_aggregation[symbol] = aggregated_value
                    
                    total_data_points += len(data)
                    source_symbols.add(symbol)
                else:
                    warnings.append(f"No data found for {symbol} at {timeframe.value}")
            
            if timeframe_aggregation:
                aggregated_data[timeframe.value] = timeframe_aggregation
                source_timeframes.add(timeframe)
        
        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000
        
        return AggregationResult(
            method=config.method,
            scope=config.scope,
            aggregated_data=aggregated_data,
            source_timeframes=source_timeframes,
            source_symbols=source_symbols,
            aggregation_timestamp=end_time,
            processing_duration_ms=duration_ms,
            data_points_processed=total_data_points,
            warnings=warnings,
            metadata=config.metadata
        )
    
    def _aggregate_cross_timeframe(
        self,
        data_dict: Dict[TimeFrame, Dict[str, pd.DataFrame]],
        config: AggregationConfig,
        start_time: datetime
    ) -> AggregationResult:
        """Aggregate data across multiple timeframes"""
        aggregated_data = {}
        warnings = []
        total_data_points = 0
        source_timeframes = set()
        source_symbols = set()
        
        for symbol in config.symbols:
            # Collect data from all timeframes for this symbol
            symbol_data_list = []
            
            for timeframe in config.timeframes:
                if timeframe in data_dict and symbol in data_dict[timeframe]:
                    data = data_dict[timeframe][symbol]
                    
                    if config.columns:
                        data = data[config.columns]
                    
                    # Add timeframe identifier
                    data_with_tf = data.copy()
                    data_with_tf['timeframe'] = timeframe.value
                    symbol_data_list.append(data_with_tf)
                    
                    total_data_points += len(data)
                    source_timeframes.add(timeframe)
                    source_symbols.add(symbol)
            
            if symbol_data_list:
                # Combine data from all timeframes
                combined_data = pd.concat(symbol_data_list, ignore_index=True)
                
                # Apply aggregation method
                aggregated_value = self._apply_aggregation_method(combined_data, config)
                aggregated_data[symbol] = aggregated_value
            else:
                warnings.append(f"No data found for {symbol} across specified timeframes")
        
        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000
        
        return AggregationResult(
            method=config.method,
            scope=config.scope,
            aggregated_data=aggregated_data,
            source_timeframes=source_timeframes,
            source_symbols=source_symbols,
            aggregation_timestamp=end_time,
            processing_duration_ms=duration_ms,
            data_points_processed=total_data_points,
            warnings=warnings,
            metadata=config.metadata
        )
    
    def _aggregate_global(
        self,
        data_dict: Dict[TimeFrame, Dict[str, pd.DataFrame]],
        config: AggregationConfig,
        start_time: datetime
    ) -> AggregationResult:
        """Aggregate all data globally"""
        all_data_list = []
        warnings = []
        total_data_points = 0
        source_timeframes = set()
        source_symbols = set()
        
        for timeframe in config.timeframes:
            if timeframe not in data_dict:
                warnings.append(f"No data found for timeframe {timeframe.value}")
                continue
            
            for symbol in config.symbols:
                if symbol in data_dict[timeframe]:
                    data = data_dict[timeframe][symbol]
                    
                    if config.columns:
                        data = data[config.columns]
                    
                    # Add identifiers
                    data_with_ids = data.copy()
                    data_with_ids['timeframe'] = timeframe.value
                    data_with_ids['symbol'] = symbol
                    all_data_list.append(data_with_ids)
                    
                    total_data_points += len(data)
                    source_timeframes.add(timeframe)
                    source_symbols.add(symbol)
        
        if all_data_list:
            # Combine all data
            combined_data = pd.concat(all_data_list, ignore_index=True)
            
            # Apply aggregation method
            aggregated_value = self._apply_aggregation_method(combined_data, config)
            aggregated_data = {'global': aggregated_value}
        else:
            aggregated_data = {}
            warnings.append("No data found for global aggregation")
        
        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000
        
        return AggregationResult(
            method=config.method,
            scope=config.scope,
            aggregated_data=aggregated_data,
            source_timeframes=source_timeframes,
            source_symbols=source_symbols,
            aggregation_timestamp=end_time,
            processing_duration_ms=duration_ms,
            data_points_processed=total_data_points,
            warnings=warnings,
            metadata=config.metadata
        )
    
    def _apply_aggregation_method(self, data: pd.DataFrame, config: AggregationConfig) -> Any:
        """Apply the specified aggregation method to data"""
        if data.empty:
            return None
        
        # Filter to only numeric columns for statistical operations
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Handle rolling window if specified
        if config.rolling_window:
            if numeric_columns:
                data_for_rolling = data[numeric_columns]
            else:
                data_for_rolling = data
            data_for_rolling = data_for_rolling.rolling(
                window=config.rolling_window,
                min_periods=config.min_periods,
                center=config.center
            )
            # Replace original data with rolling data for numeric columns
            for col in numeric_columns:
                data[col] = data_for_rolling[col]
        
        # For statistical operations, use only numeric columns
        if config.method in [AggregationMethod.MEAN, AggregationMethod.MEDIAN, AggregationMethod.SUM,
                            AggregationMethod.MIN, AggregationMethod.MAX, AggregationMethod.STD, 
                            AggregationMethod.VAR]:
            if not numeric_columns:
                return None
            working_data = data[numeric_columns]
        else:
            working_data = data
        
        # Apply aggregation method
        if config.method == AggregationMethod.MEAN:
            result = working_data.mean()
            return result.to_dict() if len(result) > 1 else result.iloc[0]
        elif config.method == AggregationMethod.MEDIAN:
            result = working_data.median()
            return result.to_dict() if len(result) > 1 else result.iloc[0]
        elif config.method == AggregationMethod.SUM:
            result = working_data.sum()
            return result.to_dict() if len(result) > 1 else result.iloc[0]
        elif config.method == AggregationMethod.MIN:
            result = working_data.min()
            return result.to_dict() if len(result) > 1 else result.iloc[0]
        elif config.method == AggregationMethod.MAX:
            result = working_data.max()
            return result.to_dict() if len(result) > 1 else result.iloc[0]
        elif config.method == AggregationMethod.STD:
            result = working_data.std()
            return result.to_dict() if len(result) > 1 else result.iloc[0]
        elif config.method == AggregationMethod.VAR:
            result = working_data.var()
            return result.to_dict() if len(result) > 1 else result.iloc[0]
        elif config.method == AggregationMethod.COUNT:
            result = working_data.count()
            return result.to_dict() if len(result) > 1 else result.iloc[0]
        elif config.method == AggregationMethod.FIRST:
            return working_data.iloc[0].to_dict() if len(working_data.columns) > 1 else working_data.iloc[0, 0]
        elif config.method == AggregationMethod.LAST:
            return working_data.iloc[-1].to_dict() if len(working_data.columns) > 1 else working_data.iloc[-1, 0]
        elif config.method == AggregationMethod.OHLC:
            return self._calculate_ohlc(working_data)
        elif config.method == AggregationMethod.VWAP:
            return self._calculate_vwap(working_data)
        elif config.method == AggregationMethod.CUSTOM and config.custom_function:
            return config.custom_function(working_data)
        else:
            raise DataAggregationError(f"Unsupported aggregation method: {config.method}")
    
    def _calculate_ohlc(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate OHLC from price data"""
        if 'close' in data.columns:
            price_col = 'close'
        elif 'price' in data.columns:
            price_col = 'price'
        else:
            # Use first numeric column
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise DataAggregationError("No numeric columns found for OHLC calculation")
            price_col = numeric_cols[0]
        
        prices = data[price_col].dropna()
        
        return {
            'open': float(prices.iloc[0]) if len(prices) > 0 else 0.0,
            'high': float(prices.max()) if len(prices) > 0 else 0.0,
            'low': float(prices.min()) if len(prices) > 0 else 0.0,
            'close': float(prices.iloc[-1]) if len(prices) > 0 else 0.0
        }
    
    def _calculate_vwap(self, data: pd.DataFrame) -> float:
        """Calculate Volume Weighted Average Price"""
        if 'volume' not in data.columns:
            raise DataAggregationError("Volume column required for VWAP calculation")
        
        if 'close' in data.columns:
            price_col = 'close'
        elif 'price' in data.columns:
            price_col = 'price'
        else:
            raise DataAggregationError("Price column required for VWAP calculation")
        
        # Filter out zero volume
        valid_data = data[(data['volume'] > 0) & (data[price_col].notna())]
        
        if valid_data.empty:
            return 0.0
        
        total_volume = valid_data['volume'].sum()
        if total_volume == 0:
            return 0.0
        
        vwap = (valid_data[price_col] * valid_data['volume']).sum() / total_volume
        return float(vwap)
    
    def _estimate_expected_interval(self, timeframe: TimeFrame) -> Optional[timedelta]:
        """Estimate expected interval for a timeframe"""
        mapping = {
            TimeFrame.SECOND_1: timedelta(seconds=1),
            TimeFrame.SECOND_5: timedelta(seconds=5),
            TimeFrame.SECOND_15: timedelta(seconds=15),
            TimeFrame.SECOND_30: timedelta(seconds=30),
            TimeFrame.MINUTE_1: timedelta(minutes=1),
            TimeFrame.MINUTE_5: timedelta(minutes=5),
            TimeFrame.MINUTE_15: timedelta(minutes=15),
            TimeFrame.MINUTE_30: timedelta(minutes=30),
            TimeFrame.HOUR_1: timedelta(hours=1),
            TimeFrame.HOUR_4: timedelta(hours=4),
            TimeFrame.DAY_1: timedelta(days=1),
            TimeFrame.WEEK_1: timedelta(weeks=1)
        }
        
        return mapping.get(timeframe)
    
    def _generate_cache_key(
        self,
        data_dict: Dict[TimeFrame, Dict[str, pd.DataFrame]],
        config: AggregationConfig
    ) -> str:
        """Generate cache key for aggregation result"""
        # Create a hash-like key based on data and config
        key_parts = []
        
        for timeframe in sorted(config.timeframes, key=lambda x: x.value):
            if timeframe in data_dict:
                for symbol in sorted(config.symbols):
                    if symbol in data_dict[timeframe]:
                        data = data_dict[timeframe][symbol]
                        key_parts.append(f"{timeframe.value}:{symbol}:{len(data)}:{data.index[0]}:{data.index[-1]}")
        
        config_key = f"{config.method.value}:{config.scope.value}:{':'.join(sorted(config.symbols))}"
        
        return f"agg:{':'.join(key_parts)}:{config_key}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[AggregationResult]:
        """Get cached aggregation result if valid"""
        if cache_key not in self._aggregation_cache:
            return None
        
        result = self._aggregation_cache[cache_key]
        age_seconds = (datetime.now() - result.aggregation_timestamp).total_seconds()
        
        if age_seconds > self._cache_ttl_seconds:
            del self._aggregation_cache[cache_key]
            return None
        
        return result
    
    def _cache_result(self, cache_key: str, result: AggregationResult) -> None:
        """Cache aggregation result"""
        self._aggregation_cache[cache_key] = result
    
    def _update_performance_metrics(self, processing_duration_ms: float) -> None:
        """Update performance tracking metrics"""
        self._aggregation_operations += 1
        self._total_processing_time_ms += processing_duration_ms


# Helper functions

def create_aggregation_config(
    method: AggregationMethod,
    scope: AggregationScope,
    timeframes: List[TimeFrame],
    symbols: List[str],
    **kwargs
) -> AggregationConfig:
    """Helper function to create aggregation configuration"""
    return AggregationConfig(
        method=method,
        scope=scope,
        timeframes=set(timeframes),
        symbols=set(symbols),
        **kwargs
    )


def quick_summary(
    data_dict: Dict[TimeFrame, Dict[str, pd.DataFrame]],
    symbols: List[str]
) -> Dict[str, AggregationResult]:
    """Quick summary statistics for symbols"""
    aggregator = DataAggregator()
    
    configs = [
        create_aggregation_config(
            AggregationMethod.MEAN, AggregationScope.SYMBOL,
            list(data_dict.keys()), symbols, metadata={'name': 'mean'}
        ),
        create_aggregation_config(
            AggregationMethod.STD, AggregationScope.SYMBOL,
            list(data_dict.keys()), symbols, metadata={'name': 'std'}
        ),
        create_aggregation_config(
            AggregationMethod.OHLC, AggregationScope.SYMBOL,
            list(data_dict.keys()), symbols, metadata={'name': 'ohlc'}
        )
    ]
    
    return aggregator.aggregate_multiple_configs(data_dict, configs) 
