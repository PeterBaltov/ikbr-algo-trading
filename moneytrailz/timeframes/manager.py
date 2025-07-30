"""
Multi-Timeframe Manager

The TimeFrameManager coordinates multi-timeframe operations, providing:
- Centralized timeframe configuration management
- Memory-efficient data handling across timeframes
- Timeframe state tracking and lifecycle management
- Integration with strategy execution engine

This is the core component that orchestrates all multi-timeframe activities.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Callable
import pandas as pd
import asyncio
from collections import defaultdict, deque
import logging

from ..strategies.enums import TimeFrame
from ..strategies.exceptions import StrategyError


class TimeFrameState(Enum):
    """State of a timeframe in the system"""
    INACTIVE = "inactive"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    SYNCING = "syncing"
    ERROR = "error"
    PAUSED = "paused"


@dataclass
class TimeFrameConfig:
    """Configuration for a specific timeframe"""
    timeframe: TimeFrame
    enabled: bool = True
    max_history_periods: int = 1000
    data_retention_hours: int = 24
    sync_frequency_seconds: int = 60
    priority: int = 1  # Higher number = higher priority
    memory_limit_mb: int = 100
    strategies: Set[str] = field(default_factory=set)
    indicators: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TimeFrameMetrics:
    """Performance metrics for a timeframe"""
    last_sync_time: Optional[datetime] = None
    sync_count: int = 0
    sync_errors: int = 0
    avg_sync_duration_ms: float = 0.0
    memory_usage_mb: float = 0.0
    active_strategies: int = 0
    data_points: int = 0
    last_data_timestamp: Optional[datetime] = None


class TimeFrameError(StrategyError):
    """Timeframe-related errors"""
    pass


class TimeFrameManager:
    """
    Multi-Timeframe Coordinator
    
    Manages multiple timeframes simultaneously, handling data synchronization,
    memory management, and strategy coordination across different time horizons.
    """
    
    def __init__(self, max_concurrent_syncs: int = 5):
        self.configs: Dict[TimeFrame, TimeFrameConfig] = {}
        self.states: Dict[TimeFrame, TimeFrameState] = {}
        self.metrics: Dict[TimeFrame, TimeFrameMetrics] = {}
        self.data_cache: Dict[TimeFrame, Dict[str, pd.DataFrame]] = defaultdict(dict)
        self.callbacks: Dict[str, List[Callable[..., Any]]] = defaultdict(list)
        self.max_concurrent_syncs = max_concurrent_syncs
        self._sync_semaphore = asyncio.Semaphore(max_concurrent_syncs)
        self._active_syncs: Set[TimeFrame] = set()
        self.logger = logging.getLogger(__name__)
        
        # Memory management
        self._memory_monitor_enabled = True
        self._cleanup_task: Optional[asyncio.Task[Any]] = None
        
    def register_timeframe(self, config: TimeFrameConfig) -> None:
        """Register a new timeframe with the manager"""
        try:
            self.configs[config.timeframe] = config
            self.states[config.timeframe] = TimeFrameState.INACTIVE
            self.metrics[config.timeframe] = TimeFrameMetrics()
            
            self.logger.info(f"Registered timeframe {config.timeframe.value} with priority {config.priority}")
            
            # Initialize data cache
            if config.timeframe not in self.data_cache:
                self.data_cache[config.timeframe] = {}
                
        except Exception as e:
            raise TimeFrameError(f"Failed to register timeframe {config.timeframe.value}: {e}")
    
    def unregister_timeframe(self, timeframe: TimeFrame) -> bool:
        """Unregister a timeframe and clean up resources"""
        try:
            if timeframe not in self.configs:
                return False
                
            # Clean up resources
            self._cleanup_timeframe(timeframe)
            
            # Remove from all collections
            self.configs.pop(timeframe, None)
            self.states.pop(timeframe, None)
            self.metrics.pop(timeframe, None)
            self.data_cache.pop(timeframe, None)
            
            self.logger.info(f"Unregistered timeframe {timeframe.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error unregistering timeframe {timeframe.value}: {e}")
            return False
    
    def get_active_timeframes(self) -> List[TimeFrame]:
        """Get list of currently active timeframes"""
        return [
            tf for tf, state in self.states.items() 
            if state == TimeFrameState.ACTIVE
        ]
    
    def get_timeframe_state(self, timeframe: TimeFrame) -> Optional[TimeFrameState]:
        """Get current state of a timeframe"""
        return self.states.get(timeframe)
    
    def set_timeframe_state(self, timeframe: TimeFrame, state: TimeFrameState) -> None:
        """Set state of a timeframe"""
        if timeframe in self.configs:
            old_state = self.states.get(timeframe)
            self.states[timeframe] = state
            
            # Trigger state change callbacks
            self._trigger_callbacks('state_change', {
                'timeframe': timeframe,
                'old_state': old_state,
                'new_state': state,
                'timestamp': datetime.now()
            })
            
            self.logger.debug(f"Timeframe {timeframe.value} state: {old_state} -> {state}")
    
    def get_timeframe_config(self, timeframe: TimeFrame) -> Optional[TimeFrameConfig]:
        """Get configuration for a timeframe"""
        return self.configs.get(timeframe)
    
    def update_timeframe_config(self, timeframe: TimeFrame, **updates) -> bool:
        """Update timeframe configuration"""
        try:
            if timeframe not in self.configs:
                return False
                
            config = self.configs[timeframe]
            for key, value in updates.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                    
            self.logger.info(f"Updated timeframe {timeframe.value} config: {updates}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating timeframe config: {e}")
            return False
    
    def get_timeframe_metrics(self, timeframe: TimeFrame) -> Optional[TimeFrameMetrics]:
        """Get performance metrics for a timeframe"""
        return self.metrics.get(timeframe)
    
    def update_metrics(self, timeframe: TimeFrame, **metrics_updates) -> None:
        """Update performance metrics for a timeframe"""
        if timeframe in self.metrics:
            metrics = self.metrics[timeframe]
            for key, value in metrics_updates.items():
                if hasattr(metrics, key):
                    setattr(metrics, key, value)
    
    def store_data(self, timeframe: TimeFrame, symbol: str, data: pd.DataFrame) -> bool:
        """Store data for a timeframe and symbol"""
        try:
            if timeframe not in self.configs:
                return False
                
            config = self.configs[timeframe]
            
            # Apply data retention policy
            if len(data) > config.max_history_periods:
                data = data.tail(config.max_history_periods)
            
            # Store data
            self.data_cache[timeframe][symbol] = data.copy()
            
            # Update metrics
            self.update_metrics(
                timeframe,
                data_points=len(data),
                last_data_timestamp=data.index[-1] if not data.empty else None,
                memory_usage_mb=self._calculate_memory_usage(timeframe)
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing data for {timeframe.value}/{symbol}: {e}")
            return False
    
    def get_data(self, timeframe: TimeFrame, symbol: str) -> Optional[pd.DataFrame]:
        """Retrieve data for a timeframe and symbol"""
        try:
            return self.data_cache.get(timeframe, {}).get(symbol)
        except Exception as e:
            self.logger.error(f"Error retrieving data for {timeframe.value}/{symbol}: {e}")
            return None
    
    def get_latest_data(self, timeframe: TimeFrame, symbol: str, periods: int = 1) -> Optional[pd.DataFrame]:
        """Get the latest N periods of data"""
        data = self.get_data(timeframe, symbol)
        if data is not None and not data.empty:
            return data.tail(periods)
        return None
    
    def clear_data(self, timeframe: TimeFrame, symbol: Optional[str] = None) -> bool:
        """Clear data for a timeframe (and optionally specific symbol)"""
        try:
            if timeframe not in self.data_cache:
                return False
                
            if symbol:
                self.data_cache[timeframe].pop(symbol, None)
            else:
                self.data_cache[timeframe].clear()
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing data: {e}")
            return False
    
    def register_callback(self, event_type: str, callback: Callable[..., Any]) -> None:
        """Register callback for timeframe events"""
        self.callbacks[event_type].append(callback)
    
    def unregister_callback(self, event_type: str, callback: Callable[..., Any]) -> bool:
        """Unregister a callback"""
        try:
            self.callbacks[event_type].remove(callback)
            return True
        except ValueError:
            return False
    
    async def start_memory_monitor(self, interval_seconds: int = 60) -> None:
        """Start background memory monitoring task"""
        if self._cleanup_task and not self._cleanup_task.done():
            return
            
        self._memory_monitor_enabled = True
        self._cleanup_task = asyncio.create_task(self._memory_monitor_loop(interval_seconds))
    
    async def stop_memory_monitor(self) -> None:
        """Stop background memory monitoring"""
        self._memory_monitor_enabled = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
    
    def get_memory_usage(self) -> Dict[TimeFrame, float]:
        """Get memory usage for all timeframes (in MB)"""
        return {
            tf: self._calculate_memory_usage(tf) 
            for tf in self.configs.keys()
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        active_timeframes = self.get_active_timeframes()
        total_memory = sum(self.get_memory_usage().values())
        
        return {
            'total_timeframes': len(self.configs),
            'active_timeframes': len(active_timeframes),
            'active_timeframe_list': [tf.value for tf in active_timeframes],
            'total_memory_mb': total_memory,
            'active_syncs': len(self._active_syncs),
            'max_concurrent_syncs': self.max_concurrent_syncs,
            'memory_monitor_enabled': self._memory_monitor_enabled,
            'timestamp': datetime.now()
        }
    
    def cleanup_all(self) -> None:
        """Clean up all resources"""
        try:
            # Stop memory monitor
            if self._cleanup_task:
                self._cleanup_task.cancel()
            
            # Clean up all timeframes
            for timeframe in list(self.configs.keys()):
                self._cleanup_timeframe(timeframe)
            
            # Clear all collections
            self.configs.clear()
            self.states.clear()
            self.metrics.clear()
            self.data_cache.clear()
            self.callbacks.clear()
            self._active_syncs.clear()
            
            self.logger.info("TimeFrameManager cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    # Private methods
    
    def _cleanup_timeframe(self, timeframe: TimeFrame) -> None:
        """Clean up resources for a specific timeframe"""
        try:
            # Set state to inactive
            self.set_timeframe_state(timeframe, TimeFrameState.INACTIVE)
            
            # Clear data cache
            self.data_cache.pop(timeframe, None)
            
            # Remove from active syncs
            self._active_syncs.discard(timeframe)
            
        except Exception as e:
            self.logger.error(f"Error cleaning up timeframe {timeframe.value}: {e}")
    
    def _calculate_memory_usage(self, timeframe: TimeFrame) -> float:
        """Calculate memory usage for a timeframe in MB"""
        try:
            data_dict = self.data_cache.get(timeframe, {})
            total_size = 0
            
            for symbol, df in data_dict.items():
                if df is not None:
                    # Estimate DataFrame memory usage
                    total_size += df.memory_usage(deep=True).sum()
            
            return total_size / (1024 * 1024)  # Convert to MB
            
        except Exception:
            return 0.0
    
    def _trigger_callbacks(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Trigger callbacks for an event type"""
        try:
            for callback in self.callbacks.get(event_type, []):
                try:
                    callback(event_data)
                except Exception as e:
                    self.logger.error(f"Callback error for {event_type}: {e}")
        except Exception as e:
            self.logger.error(f"Error triggering callbacks: {e}")
    
    async def _memory_monitor_loop(self, interval_seconds: int) -> None:
        """Background memory monitoring loop"""
        while self._memory_monitor_enabled:
            try:
                await asyncio.sleep(interval_seconds)
                
                if not self._memory_monitor_enabled:
                    break
                
                # Check memory usage and clean up if needed
                for timeframe, config in self.configs.items():
                    memory_mb = self._calculate_memory_usage(timeframe)
                    
                    if memory_mb > config.memory_limit_mb:
                        self.logger.warning(
                            f"Timeframe {timeframe.value} exceeds memory limit: "
                            f"{memory_mb:.1f}MB > {config.memory_limit_mb}MB"
                        )
                        
                        # Trigger cleanup
                        await self._cleanup_excessive_memory(timeframe, config)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Memory monitor error: {e}")
    
    async def _cleanup_excessive_memory(self, timeframe: TimeFrame, config: TimeFrameConfig) -> None:
        """Clean up excessive memory usage for a timeframe"""
        try:
            data_dict = self.data_cache.get(timeframe, {})
            
            # Remove oldest data first
            for symbol in list(data_dict.keys()):
                df = data_dict[symbol]
                if df is not None and len(df) > config.max_history_periods // 2:
                    # Keep only most recent half of the data
                    reduced_df = df.tail(config.max_history_periods // 2)
                    self.data_cache[timeframe][symbol] = reduced_df
                    
                    self.logger.info(
                        f"Reduced data for {timeframe.value}/{symbol}: "
                        f"{len(df)} -> {len(reduced_df)} periods"
                    )
            
            # Update metrics
            new_memory = self._calculate_memory_usage(timeframe)
            self.update_metrics(timeframe, memory_usage_mb=new_memory)
            
        except Exception as e:
            self.logger.error(f"Error cleaning up memory for {timeframe.value}: {e}")


# Global timeframe manager instance
_global_manager: Optional[TimeFrameManager] = None


def get_timeframe_manager() -> TimeFrameManager:
    """Get the global timeframe manager instance"""
    global _global_manager
    if _global_manager is None:
        _global_manager = TimeFrameManager()
    return _global_manager


def create_timeframe_config(
    timeframe: TimeFrame,
    enabled: bool = True,
    max_history: int = 1000,
    memory_limit_mb: int = 100,
    priority: int = 1,
    strategies: Optional[Set[str]] = None,
    **kwargs
) -> TimeFrameConfig:
    """Helper function to create timeframe configuration"""
    return TimeFrameConfig(
        timeframe=timeframe,
        enabled=enabled,
        max_history_periods=max_history,
        memory_limit_mb=memory_limit_mb,
        priority=priority,
        strategies=strategies or set(),
        **kwargs
    ) 
