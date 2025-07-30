"""
Multi-Timeframe Architecture Package

This package provides sophisticated multi-timeframe capabilities for the ThetaGang
algorithmic trading system, including data synchronization, execution scheduling,
memory management, and latency optimization.

Components:
- TimeFrameManager: Main coordinator for multi-timeframe operations
- DataSynchronizer: Aligns data across different timeframes
- ExecutionScheduler: Manages strategy execution timing
- DataAggregator: Efficient multi-timeframe data aggregation utilities

Integration:
- Works seamlessly with Phase 1 strategy framework
- Leverages Phase 2 technical analysis engine
- Provides foundation for real-time trading execution
"""

from .manager import TimeFrameManager, TimeFrameConfig, TimeFrameState, get_timeframe_manager, create_timeframe_config
from .synchronizer import DataSynchronizer, SyncResult, SyncStatus, SyncMethod, create_sync_config, quick_align
from .scheduler import ExecutionScheduler, ScheduleConfig, ExecutionWindow, ScheduleType, ExecutionPriority, create_schedule_config, create_execution_window
from .aggregator import DataAggregator, AggregationResult, AggregationMethod, AggregationScope, create_aggregation_config, quick_summary

__all__ = [
    # Core management
    "TimeFrameManager", 
    "TimeFrameConfig", 
    "TimeFrameState",
    "get_timeframe_manager",
    "create_timeframe_config",
    
    # Data synchronization
    "DataSynchronizer", 
    "SyncResult", 
    "SyncStatus",
    "SyncMethod",
    "create_sync_config",
    "quick_align",
    
    # Execution scheduling
    "ExecutionScheduler", 
    "ScheduleConfig", 
    "ExecutionWindow",
    "ScheduleType",
    "ExecutionPriority", 
    "create_schedule_config",
    "create_execution_window",
    
    # Data aggregation
    "DataAggregator", 
    "AggregationResult", 
    "AggregationMethod",
    "AggregationScope",
    "create_aggregation_config",
    "quick_summary"
] 
