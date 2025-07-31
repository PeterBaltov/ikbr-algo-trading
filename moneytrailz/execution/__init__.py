"""
Strategy Execution Engine Package

This package provides comprehensive strategy execution capabilities for the moneytrailz
algorithmic trading system, including coordination, scheduling, pipeline processing,
and multi-strategy management.

Components:
- StrategyExecutionEngine: Main coordinator for strategy execution
- ExecutionScheduler: Strategy timing and frequency management
- DataPipeline: Real-time data processing pipeline
- StrategyCoordinator: Multi-strategy coordination and conflict resolution

Integration:
- Leverages Phase 1 strategy framework
- Uses Phase 2 technical analysis engine
- Coordinates with Phase 3 multi-timeframe architecture
- Provides real-time execution capabilities
"""

from .engine import StrategyExecutionEngine, ExecutionConfig, ExecutionState, ExecutionMode, create_execution_config
from .scheduler import ExecutionScheduler, ScheduleConfig, ExecutionResult
from .pipeline import DataPipeline, PipelineConfig, PipelineStage, create_pipeline_config
from .coordinator import StrategyCoordinator, CoordinationResult, ConflictResolution, create_coordinator

__all__ = [
    # Core execution engine
    "StrategyExecutionEngine",
    "ExecutionConfig", 
    "ExecutionState",
    "ExecutionMode",
    "create_execution_config",
    
    # Scheduling
    "ExecutionScheduler", 
    "ScheduleConfig", 
    "ExecutionResult",
    
    # Data pipeline
    "DataPipeline", 
    "PipelineConfig", 
    "PipelineStage",
    "create_pipeline_config",
    
    # Strategy coordination
    "StrategyCoordinator", 
    "CoordinationResult", 
    "ConflictResolution",
    "create_coordinator"
] 
