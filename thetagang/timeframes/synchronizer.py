"""
Data Synchronizer

The DataSynchronizer aligns market data across different timeframes, ensuring
consistent timing and proper data relationships between different time horizons.

Key features:
- Cross-timeframe data alignment
- Timestamp synchronization
- Data interpolation and resampling
- Lag compensation and latency optimization
- Memory-efficient data handling
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Tuple, Union, cast
import pandas as pd
import numpy as np
import asyncio
import logging
from collections import deque

from ..strategies.enums import TimeFrame
from ..strategies.exceptions import StrategyError


class SyncStatus(Enum):
    """Status of data synchronization"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    STALE = "stale"


class SyncMethod(Enum):
    """Methods for data synchronization"""
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    INTERPOLATE = "interpolate"
    NEAREST = "nearest"
    DROP_NA = "drop_na"
    RESAMPLE = "resample"


@dataclass
class SyncResult:
    """Result of data synchronization operation"""
    status: SyncStatus
    synchronized_data: Dict[TimeFrame, pd.DataFrame]
    reference_timeframe: TimeFrame
    sync_timestamp: datetime
    data_points_aligned: int
    missing_data_count: int
    sync_duration_ms: float
    method_used: SyncMethod
    warnings: List[str]
    metadata: Dict[str, Any]


@dataclass
class SyncConfig:
    """Configuration for data synchronization"""
    reference_timeframe: TimeFrame
    target_timeframes: Set[TimeFrame]
    sync_method: SyncMethod = SyncMethod.FORWARD_FILL
    max_look_ahead_periods: int = 0
    max_look_back_periods: int = 5
    tolerance_seconds: int = 30
    require_complete_alignment: bool = False
    interpolation_method: str = "linear"
    resample_method: str = "last"
    fill_limit: Optional[int] = None
    drop_na_threshold: float = 0.95  # Drop if less than 95% data available


class DataSyncError(StrategyError):
    """Data synchronization related errors"""
    pass


class DataSynchronizer:
    """
    Multi-Timeframe Data Synchronizer
    
    Handles alignment and synchronization of market data across different
    timeframes, ensuring proper temporal relationships and data consistency.
    """
    
    def __init__(self, default_config: Optional[SyncConfig] = None):
        self.default_config = default_config
        self.logger = logging.getLogger(__name__)
        
        # Synchronization cache
        self._sync_cache: Dict[str, SyncResult] = {}
        self._cache_ttl_seconds = 300  # 5 minutes
        
        # Performance tracking
        self._sync_operations = 0
        self._total_sync_time_ms = 0.0
        self._last_sync_time: Optional[datetime] = None
        
    def synchronize_data(
        self,
        data_dict: Dict[TimeFrame, Dict[str, pd.DataFrame]],
        symbol: str,
        config: Optional[SyncConfig] = None
    ) -> SyncResult:
        """
        Synchronize data across multiple timeframes for a specific symbol
        
        Args:
            data_dict: Dictionary mapping timeframes to symbol data
            symbol: The symbol to synchronize data for
            config: Synchronization configuration
            
        Returns:
            SyncResult with synchronized data and metadata
        """
        start_time = datetime.now()
        config = config or self.default_config
        
        if not config:
            raise DataSyncError("No synchronization configuration provided")
        
        try:
            # Extract data for the symbol from each timeframe
            symbol_data = {}
            for timeframe, timeframe_data in data_dict.items():
                if symbol in timeframe_data:
                    symbol_data[timeframe] = timeframe_data[symbol].copy()
            
            if not symbol_data:
                return self._create_failed_result(
                    config, start_time, f"No data found for symbol {symbol}"
                )
            
            # Check cache first
            cache_key = self._generate_cache_key(symbol_data, config)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                return cached_result
            
            # Perform synchronization
            sync_result = self._perform_synchronization(symbol_data, config, start_time)
            
            # Cache the result
            self._cache_result(cache_key, sync_result)
            
            # Update performance metrics
            self._update_performance_metrics(sync_result.sync_duration_ms)
            
            return sync_result
            
        except Exception as e:
            self.logger.error(f"Error synchronizing data for {symbol}: {e}")
            return self._create_failed_result(config, start_time, str(e))
    
    async def synchronize_data_async(
        self,
        data_dict: Dict[TimeFrame, Dict[str, pd.DataFrame]],
        symbol: str,
        config: Optional[SyncConfig] = None
    ) -> SyncResult:
        """Asynchronous version of data synchronization"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.synchronize_data, data_dict, symbol, config
        )
    
    def synchronize_multiple_symbols(
        self,
        data_dict: Dict[TimeFrame, Dict[str, pd.DataFrame]],
        symbols: List[str],
        config: Optional[SyncConfig] = None
    ) -> Dict[str, SyncResult]:
        """Synchronize data for multiple symbols"""
        results = {}
        
        for symbol in symbols:
            try:
                results[symbol] = self.synchronize_data(data_dict, symbol, config)
            except Exception as e:
                self.logger.error(f"Error synchronizing {symbol}: {e}")
                results[symbol] = self._create_failed_result(
                    config or self.default_config, datetime.now(), str(e)
                )
        
        return results
    
    async def synchronize_multiple_symbols_async(
        self,
        data_dict: Dict[TimeFrame, Dict[str, pd.DataFrame]],
        symbols: List[str],
        config: Optional[SyncConfig] = None,
        max_concurrent: int = 10
    ) -> Dict[str, SyncResult]:
        """Asynchronously synchronize data for multiple symbols"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def sync_symbol(symbol: str) -> Tuple[str, SyncResult]:
            async with semaphore:
                result = await self.synchronize_data_async(data_dict, symbol, config)
                return symbol, result
        
        tasks = [sync_symbol(symbol) for symbol in symbols]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        results = {}
        for item in results_list:
            if isinstance(item, Exception):
                self.logger.error(f"Async synchronization error: {item}")
                continue
            if isinstance(item, tuple):
                symbol, result = item
                results[symbol] = result
        
        return results
    
    def align_to_reference_timeframe(
        self,
        data_dict: Dict[TimeFrame, pd.DataFrame],
        reference_timeframe: TimeFrame,
        method: SyncMethod = SyncMethod.FORWARD_FILL
    ) -> Dict[TimeFrame, pd.DataFrame]:
        """Align all timeframes to a reference timeframe's timestamps"""
        if reference_timeframe not in data_dict:
            raise DataSyncError(f"Reference timeframe {reference_timeframe.value} not found in data")
        
        reference_data = data_dict[reference_timeframe]
        reference_index = cast(pd.DatetimeIndex, reference_data.index)
        
        aligned_data = {reference_timeframe: reference_data.copy()}
        
        for timeframe, data in data_dict.items():
            if timeframe == reference_timeframe:
                continue
                
            aligned_data[timeframe] = self._align_single_timeframe(
                data, reference_index, method
            )
        
        return aligned_data
    
    def resample_to_timeframe(
        self,
        data: pd.DataFrame,
        target_timeframe: TimeFrame,
        method: str = "last"
    ) -> pd.DataFrame:
        """Resample data to a different timeframe"""
        try:
            rule = self._timeframe_to_pandas_rule(target_timeframe)
            
            if method == "last":
                resampled_data = data.resample(rule).last()
            elif method == "first":
                resampled_data = data.resample(rule).first()
            elif method == "mean":
                resampled_data = data.resample(rule).mean()
            elif method == "ohlc":
                # For OHLC data
                resampled_data = data.resample(rule).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                })
            else:
                resampled_data = data.resample(rule).last()
            
            # Ensure we return a DataFrame
            if isinstance(resampled_data, pd.Series):
                return resampled_data.to_frame().dropna()
            else:
                return resampled_data.dropna()
            
        except Exception as e:
            raise DataSyncError(f"Error resampling data to {target_timeframe.value}: {e}")
    
    def detect_data_gaps(
        self,
        data: pd.DataFrame,
        timeframe: TimeFrame,
        tolerance_factor: float = 2.0
    ) -> List[Tuple[datetime, datetime]]:
        """Detect gaps in time series data"""
        if data.empty:
            return []
        
        expected_interval = self._timeframe_to_seconds(timeframe)
        tolerance = expected_interval * tolerance_factor
        
        gaps = []
        prev_timestamp = data.index[0]
        
        for timestamp in data.index[1:]:
            gap_seconds = (timestamp - prev_timestamp).total_seconds()
            
            if gap_seconds > tolerance:
                gaps.append((prev_timestamp, timestamp))
            
            prev_timestamp = timestamp
        
        return gaps
    
    def fill_data_gaps(
        self,
        data: pd.DataFrame,
        method: SyncMethod = SyncMethod.FORWARD_FILL,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Fill gaps in time series data"""
        if data.empty:
            return data
        
        filled_data = data.copy()
        
        if method == SyncMethod.FORWARD_FILL:
            filled_data = filled_data.fillna(method='ffill', limit=limit)
        elif method == SyncMethod.BACKWARD_FILL:
            filled_data = filled_data.fillna(method='bfill', limit=limit)
        elif method == SyncMethod.INTERPOLATE:
            filled_data = filled_data.interpolate(method='linear', limit=limit)
        elif method == SyncMethod.DROP_NA:
            filled_data = filled_data.dropna()
        
        return filled_data
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get synchronization performance statistics"""
        avg_sync_time = (
            self._total_sync_time_ms / self._sync_operations 
            if self._sync_operations > 0 else 0.0
        )
        
        return {
            'total_sync_operations': self._sync_operations,
            'total_sync_time_ms': self._total_sync_time_ms,
            'average_sync_time_ms': avg_sync_time,
            'last_sync_time': self._last_sync_time,
            'cache_size': len(self._sync_cache)
        }
    
    def clear_cache(self) -> None:
        """Clear the synchronization cache"""
        self._sync_cache.clear()
        self.logger.info("Synchronization cache cleared")
    
    def cleanup_expired_cache(self) -> int:
        """Remove expired entries from cache"""
        current_time = datetime.now()
        expired_keys = []
        
        for key, result in self._sync_cache.items():
            age_seconds = (current_time - result.sync_timestamp).total_seconds()
            if age_seconds > self._cache_ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._sync_cache[key]
        
        if expired_keys:
            self.logger.debug(f"Removed {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
    
    # Private methods
    
    def _perform_synchronization(
        self,
        symbol_data: Dict[TimeFrame, pd.DataFrame],
        config: SyncConfig,
        start_time: datetime
    ) -> SyncResult:
        """Perform the actual data synchronization"""
        warnings = []
        
        # Validate reference timeframe
        if config.reference_timeframe not in symbol_data:
            raise DataSyncError(f"Reference timeframe {config.reference_timeframe.value} not available")
        
        reference_data = symbol_data[config.reference_timeframe]
        reference_index = cast(pd.DatetimeIndex, reference_data.index)
        
        synchronized_data = {config.reference_timeframe: reference_data.copy()}
        total_aligned_points = len(reference_data)
        missing_data_count = 0
        
        # Synchronize each target timeframe
        for timeframe in config.target_timeframes:
            if timeframe == config.reference_timeframe:
                continue
                
            if timeframe not in symbol_data:
                warnings.append(f"Timeframe {timeframe.value} not available for synchronization")
                continue
            
            target_data = symbol_data[timeframe]
            
            # Align to reference timeframe
            aligned_data = self._align_single_timeframe(
                target_data, reference_index, config.sync_method
            )
            
            # Check data quality
            missing_count = aligned_data.isna().sum().sum()
            missing_data_count += missing_count
            
            if missing_count > 0:
                missing_pct = missing_count / (len(aligned_data) * len(aligned_data.columns))
                if missing_pct > (1 - config.drop_na_threshold):
                    warnings.append(
                        f"High missing data percentage ({missing_pct:.1%}) for {timeframe.value}"
                    )
            
            synchronized_data[timeframe] = aligned_data
        
        # Calculate duration
        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000
        
        # Determine final status
        status = SyncStatus.COMPLETED
        if warnings:
            status = SyncStatus.PARTIAL
        if config.require_complete_alignment and missing_data_count > 0:
            status = SyncStatus.FAILED
        
        return SyncResult(
            status=status,
            synchronized_data=synchronized_data,
            reference_timeframe=config.reference_timeframe,
            sync_timestamp=end_time,
            data_points_aligned=total_aligned_points,
            missing_data_count=missing_data_count,
            sync_duration_ms=duration_ms,
            method_used=config.sync_method,
            warnings=warnings,
            metadata={
                'target_timeframes': list(config.target_timeframes),
                'sync_config': config
            }
        )
    
    def _align_single_timeframe(
        self,
        data: pd.DataFrame,
        reference_index: pd.DatetimeIndex,
        method: SyncMethod
    ) -> pd.DataFrame:
        """Align a single timeframe's data to a reference index"""
        if method == SyncMethod.FORWARD_FILL:
            return data.reindex(reference_index, method='ffill')
        elif method == SyncMethod.BACKWARD_FILL:
            return data.reindex(reference_index, method='bfill')
        elif method == SyncMethod.NEAREST:
            return data.reindex(reference_index, method='nearest')
        elif method == SyncMethod.INTERPOLATE:
            aligned = data.reindex(reference_index)
            return aligned.interpolate(method='linear')
        elif method == SyncMethod.DROP_NA:
            return data.reindex(reference_index)
        else:
            return data.reindex(reference_index, method='ffill')
    
    def _timeframe_to_pandas_rule(self, timeframe: TimeFrame) -> str:
        """Convert TimeFrame enum to pandas resample rule"""
        mapping = {
            TimeFrame.SECOND_1: '1S',
            TimeFrame.SECOND_5: '5S',
            TimeFrame.SECOND_15: '15S',
            TimeFrame.SECOND_30: '30S',
            TimeFrame.MINUTE_1: '1min',
            TimeFrame.MINUTE_5: '5min',
            TimeFrame.MINUTE_15: '15min',
            TimeFrame.MINUTE_30: '30min',
            TimeFrame.HOUR_1: '1h',
            TimeFrame.HOUR_4: '4h',
            TimeFrame.DAY_1: '1D',
            TimeFrame.WEEK_1: '1W'
        }
        
        return mapping.get(timeframe, '1D')
    
    def _timeframe_to_seconds(self, timeframe: TimeFrame) -> int:
        """Convert TimeFrame enum to seconds"""
        mapping = {
            TimeFrame.SECOND_1: 1,
            TimeFrame.SECOND_5: 5,
            TimeFrame.SECOND_15: 15,
            TimeFrame.SECOND_30: 30,
            TimeFrame.MINUTE_1: 60,
            TimeFrame.MINUTE_5: 300,
            TimeFrame.MINUTE_15: 900,
            TimeFrame.MINUTE_30: 1800,
            TimeFrame.HOUR_1: 3600,
            TimeFrame.HOUR_4: 14400,
            TimeFrame.DAY_1: 86400,
            TimeFrame.WEEK_1: 604800
        }
        
        return mapping.get(timeframe, 86400)
    
    def _generate_cache_key(
        self, 
        symbol_data: Dict[TimeFrame, pd.DataFrame], 
        config: SyncConfig
    ) -> str:
        """Generate cache key for synchronization result"""
        # Create a hash-like key based on data shapes and config
        key_parts = []
        
        for timeframe in sorted(symbol_data.keys(), key=lambda x: x.value):
            data = symbol_data[timeframe]
            key_parts.append(f"{timeframe.value}:{len(data)}:{data.index[0]}:{data.index[-1]}")
        
        config_key = f"{config.reference_timeframe.value}:{config.sync_method.value}"
        
        return f"sync:{':'.join(key_parts)}:{config_key}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[SyncResult]:
        """Get cached synchronization result if valid"""
        if cache_key not in self._sync_cache:
            return None
        
        result = self._sync_cache[cache_key]
        age_seconds = (datetime.now() - result.sync_timestamp).total_seconds()
        
        if age_seconds > self._cache_ttl_seconds:
            del self._sync_cache[cache_key]
            return None
        
        return result
    
    def _cache_result(self, cache_key: str, result: SyncResult) -> None:
        """Cache synchronization result"""
        self._sync_cache[cache_key] = result
        
        # Clean up expired entries periodically
        if len(self._sync_cache) % 100 == 0:
            self.cleanup_expired_cache()
    
    def _create_failed_result(
        self, 
        config: Optional[SyncConfig], 
        start_time: datetime, 
        error_message: str
    ) -> SyncResult:
        """Create a failed synchronization result"""
        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000
        
        return SyncResult(
            status=SyncStatus.FAILED,
            synchronized_data={},
            reference_timeframe=config.reference_timeframe if config else TimeFrame.DAY_1,
            sync_timestamp=end_time,
            data_points_aligned=0,
            missing_data_count=0,
            sync_duration_ms=duration_ms,
            method_used=config.sync_method if config else SyncMethod.FORWARD_FILL,
            warnings=[error_message],
            metadata={'error': error_message}
        )
    
    def _update_performance_metrics(self, sync_duration_ms: float) -> None:
        """Update performance tracking metrics"""
        self._sync_operations += 1
        self._total_sync_time_ms += sync_duration_ms
        self._last_sync_time = datetime.now()


# Helper functions

def create_sync_config(
    reference_timeframe: TimeFrame,
    target_timeframes: List[TimeFrame],
    method: SyncMethod = SyncMethod.FORWARD_FILL,
    **kwargs
) -> SyncConfig:
    """Helper function to create synchronization configuration"""
    return SyncConfig(
        reference_timeframe=reference_timeframe,
        target_timeframes=set(target_timeframes),
        sync_method=method,
        **kwargs
    )


def quick_align(
    data_dict: Dict[TimeFrame, pd.DataFrame],
    reference_timeframe: TimeFrame
) -> Dict[TimeFrame, pd.DataFrame]:
    """Quick alignment utility function"""
    synchronizer = DataSynchronizer()
    return synchronizer.align_to_reference_timeframe(data_dict, reference_timeframe) 
