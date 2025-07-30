"""
Strategy Execution Engine

The main coordinator for strategy execution, providing real-time strategy
execution capabilities with comprehensive monitoring, error handling, and
performance optimization.

Key features:
- Real-time strategy execution coordination
- Multi-timeframe strategy management
- Performance monitoring and optimization
- Risk management integration
- Error handling and recovery
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Callable, Union
import asyncio
import logging
from collections import defaultdict, deque
import traceback
import pandas as pd

from ..strategies import BaseStrategy, StrategyResult, StrategyContext, get_registry
from ..strategies.enums import StrategySignal, StrategyType, TimeFrame, StrategyStatus
from ..strategies.exceptions import StrategyError
from ..timeframes import TimeFrameManager, DataSynchronizer, ExecutionScheduler
from ..analysis import TechnicalAnalysisEngine


class ExecutionState(Enum):
    """State of the execution engine"""
    STOPPED = "stopped"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"


class ExecutionMode(Enum):
    """Execution mode for strategies"""
    LIVE = "live"                  # Live trading
    PAPER = "paper"               # Paper trading
    SIMULATION = "simulation"     # Historical simulation
    BACKTEST = "backtest"         # Backtesting mode


@dataclass
class ExecutionConfig:
    """Configuration for strategy execution engine"""
    execution_mode: ExecutionMode = ExecutionMode.PAPER
    max_concurrent_strategies: int = 10
    max_execution_time_seconds: int = 300
    enable_risk_management: bool = True
    enable_technical_analysis: bool = True
    enable_multi_timeframe: bool = True
    auto_retry_failed_executions: bool = True
    max_retry_attempts: int = 3
    retry_delay_seconds: int = 30
    performance_monitoring_enabled: bool = True
    execution_timeout_seconds: int = 60
    data_staleness_threshold_seconds: int = 300
    enable_position_sizing: bool = True
    enable_stop_loss: bool = True
    enable_take_profit: bool = True
    log_level: str = "INFO"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionMetrics:
    """Execution performance metrics"""
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    average_execution_time_ms: float = 0.0
    total_execution_time_ms: float = 0.0
    strategies_executed: Set[str] = field(default_factory=set)
    signals_generated: Dict[StrategySignal, int] = field(default_factory=lambda: defaultdict(int))
    last_execution_time: Optional[datetime] = None
    uptime_seconds: float = 0.0
    error_rate: float = 0.0


class ExecutionError(StrategyError):
    """Execution engine related errors"""
    pass


class StrategyExecutionEngine:
    """
    Main Strategy Execution Engine
    
    Coordinates the execution of trading strategies across multiple timeframes,
    providing real-time execution capabilities with comprehensive monitoring
    and error handling.
    """
    
    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.state = ExecutionState.STOPPED
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, config.log_level))
        
        # Core components
        self.strategy_registry = get_registry()
        self.timeframe_manager: Optional[TimeFrameManager] = None
        self.data_synchronizer: Optional[DataSynchronizer] = None
        self.execution_scheduler: Optional[ExecutionScheduler] = None
        self.technical_analysis_engine: Optional[TechnicalAnalysisEngine] = None
        
        # Execution tracking
        self.active_strategies: Dict[str, BaseStrategy] = {}
        self.execution_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self.execution_tasks: Dict[str, asyncio.Task[Any]] = {}
        self.execution_results: deque[StrategyResult] = deque(maxlen=1000)
        
        # Performance metrics
        self.metrics = ExecutionMetrics()
        self.start_time: Optional[datetime] = None
        
        # Control flags
        self._running = False
        self._main_task: Optional[asyncio.Task[Any]] = None
        self._shutdown_event = asyncio.Event()
        
        # Callbacks
        self.execution_callbacks: Dict[str, List[Callable[..., Any]]] = defaultdict(list)
        
    async def initialize(self) -> None:
        """Initialize the execution engine"""
        try:
            self.state = ExecutionState.INITIALIZING
            self.logger.info("Initializing Strategy Execution Engine...")
            
            # Initialize components based on configuration
            if self.config.enable_multi_timeframe:
                from ..timeframes import get_timeframe_manager
                self.timeframe_manager = get_timeframe_manager()
                self.data_synchronizer = DataSynchronizer()
            
            if self.config.enable_technical_analysis:
                self.technical_analysis_engine = TechnicalAnalysisEngine()
                # Create default indicators for main timeframes
                for timeframe in [TimeFrame.MINUTE_1, TimeFrame.MINUTE_5, TimeFrame.HOUR_1, TimeFrame.DAY_1]:
                    self.technical_analysis_engine.create_default_indicators(timeframe)
            
            self.execution_scheduler = ExecutionScheduler(
                max_concurrent_total=self.config.max_concurrent_strategies
            )
            
            # Load active strategies
            await self._load_active_strategies()
            
            self.logger.info("Strategy Execution Engine initialized successfully")
            
        except Exception as e:
            self.state = ExecutionState.ERROR
            self.logger.error(f"Failed to initialize execution engine: {e}")
            raise ExecutionError(f"Initialization failed: {e}")
    
    async def start(self) -> None:
        """Start the execution engine"""
        try:
            if self.state != ExecutionState.STOPPED:
                await self.initialize()
            
            self.state = ExecutionState.RUNNING
            self._running = True
            self.start_time = datetime.now()
            
            # Start execution scheduler
            if self.execution_scheduler:
                await self.execution_scheduler.start_scheduler()
            
            # Start main execution loop
            self._main_task = asyncio.create_task(self._main_execution_loop())
            
            self.logger.info("Strategy Execution Engine started")
            
        except Exception as e:
            self.state = ExecutionState.ERROR
            self.logger.error(f"Failed to start execution engine: {e}")
            raise ExecutionError(f"Start failed: {e}")
    
    async def stop(self) -> None:
        """Stop the execution engine"""
        try:
            self.state = ExecutionState.SHUTTING_DOWN
            self._running = False
            
            # Stop execution scheduler
            if self.execution_scheduler:
                await self.execution_scheduler.stop_scheduler()
            
            # Cancel all active tasks
            for task in self.execution_tasks.values():
                task.cancel()
            
            if self.execution_tasks:
                await asyncio.gather(*self.execution_tasks.values(), return_exceptions=True)
            
            # Cancel main task
            if self._main_task:
                self._main_task.cancel()
                try:
                    await self._main_task
                except asyncio.CancelledError:
                    pass
            
            self.state = ExecutionState.STOPPED
            self.logger.info("Strategy Execution Engine stopped")
            
        except Exception as e:
            self.state = ExecutionState.ERROR
            self.logger.error(f"Error stopping execution engine: {e}")
    
    async def pause(self) -> None:
        """Pause the execution engine"""
        if self.state == ExecutionState.RUNNING:
            self.state = ExecutionState.PAUSING
            # Cancel current executions but don't stop the engine
            for task in list(self.execution_tasks.values()):
                task.cancel()
            self.state = ExecutionState.PAUSED
            self.logger.info("Strategy Execution Engine paused")
    
    async def resume(self) -> None:
        """Resume the execution engine"""
        if self.state == ExecutionState.PAUSED:
            self.state = ExecutionState.RUNNING
            self.logger.info("Strategy Execution Engine resumed")
    
    async def execute_strategy(
        self,
        strategy_name: str,
        symbol: str,
        data: Dict[TimeFrame, pd.DataFrame],
        context: StrategyContext,
        force: bool = False
    ) -> Optional[StrategyResult]:
        """Execute a specific strategy"""
        try:
            if not self._can_execute(strategy_name, force):
                return None
            
            strategy = self.active_strategies.get(strategy_name)
            if not strategy:
                self.logger.warning(f"Strategy {strategy_name} not found in active strategies")
                return None
            
            # Create execution task
            execution_id = f"{strategy_name}_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            task = asyncio.create_task(
                self._execute_strategy_with_monitoring(
                    strategy, symbol, data, context, execution_id
                )
            )
            
            self.execution_tasks[execution_id] = task
            
            # Wait for execution with timeout
            try:
                result = await asyncio.wait_for(
                    task,
                    timeout=self.config.execution_timeout_seconds
                )
                return result
                
            except asyncio.TimeoutError:
                self.logger.warning(f"Strategy execution timeout: {strategy_name}")
                task.cancel()
                return None
                
            finally:
                self.execution_tasks.pop(execution_id, None)
                
        except Exception as e:
            self.logger.error(f"Error executing strategy {strategy_name}: {e}")
            return None
    
    async def execute_all_strategies(
        self,
        market_data: Dict[TimeFrame, Dict[str, pd.DataFrame]],
        context: StrategyContext
    ) -> Dict[str, List[StrategyResult]]:
        """Execute all active strategies"""
        results = {}
        
        try:
            # Execute strategies concurrently with semaphore
            semaphore = asyncio.Semaphore(self.config.max_concurrent_strategies)
            
            async def execute_strategy_for_symbols(strategy_name: str):
                async with semaphore:
                    strategy = self.active_strategies.get(strategy_name)
                    if not strategy:
                        return
                    
                    strategy_results = []
                    
                    for symbol in strategy.symbols:
                        # Prepare data for this symbol
                        symbol_data = {}
                        for timeframe in strategy.get_required_timeframes():
                            if timeframe in market_data and symbol in market_data[timeframe]:
                                symbol_data[timeframe] = market_data[timeframe][symbol]
                        
                        if symbol_data:
                            result = await self.execute_strategy(
                                strategy_name, symbol, symbol_data, context
                            )
                            if result:
                                strategy_results.append(result)
                    
                    if strategy_results:
                        results[strategy_name] = strategy_results
            
            # Execute all strategies
            tasks = [
                execute_strategy_for_symbols(strategy_name)
                for strategy_name in self.active_strategies.keys()
            ]
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            self.logger.error(f"Error executing all strategies: {e}")
        
        return results
    
    def register_strategy(self, strategy: BaseStrategy) -> bool:
        """Register a strategy for execution"""
        try:
            self.active_strategies[strategy.name] = strategy
            self.logger.info(f"Registered strategy: {strategy.name}")
            return True
        except Exception as e:
            self.logger.error(f"Error registering strategy {strategy.name}: {e}")
            return False
    
    def unregister_strategy(self, strategy_name: str) -> bool:
        """Unregister a strategy from execution"""
        try:
            if strategy_name in self.active_strategies:
                del self.active_strategies[strategy_name]
                self.logger.info(f"Unregistered strategy: {strategy_name}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error unregistering strategy {strategy_name}: {e}")
            return False
    
    def get_execution_metrics(self) -> ExecutionMetrics:
        """Get current execution metrics"""
        # Update uptime
        if self.start_time:
            self.metrics.uptime_seconds = (datetime.now() - self.start_time).total_seconds()
        
        # Update error rate
        if self.metrics.total_executions > 0:
            self.metrics.error_rate = self.metrics.failed_executions / self.metrics.total_executions
        
        return self.metrics
    
    def get_active_strategies(self) -> List[str]:
        """Get list of active strategy names"""
        return list(self.active_strategies.keys())
    
    def get_execution_history(self, limit: int = 100) -> List[StrategyResult]:
        """Get recent execution results"""
        return list(self.execution_results)[-limit:]
    
    def register_callback(self, event_type: str, callback: Callable[..., Any]) -> None:
        """Register callback for execution events"""
        self.execution_callbacks[event_type].append(callback)
    
    def unregister_callback(self, event_type: str, callback: Callable[..., Any]) -> bool:
        """Unregister an execution callback"""
        try:
            self.execution_callbacks[event_type].remove(callback)
            return True
        except ValueError:
            return False
    
    # Private methods
    
    async def _main_execution_loop(self) -> None:
        """Main execution loop"""
        while self._running:
            try:
                await asyncio.sleep(1.0)  # Check every second
                
                if self.state != ExecutionState.RUNNING:
                    continue
                
                # Clean up completed tasks
                self._cleanup_completed_tasks()
                
                # Process execution queue if needed
                await self._process_execution_queue()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in main execution loop: {e}")
    
    async def _execute_strategy_with_monitoring(
        self,
        strategy: BaseStrategy,
        symbol: str,
        data: Dict[TimeFrame, pd.DataFrame],
        context: StrategyContext,
        execution_id: str
    ) -> Optional[StrategyResult]:
        """Execute strategy with comprehensive monitoring"""
        start_time = datetime.now()
        
        try:
            # Validate data freshness
            if not self._is_data_fresh(data):
                self.logger.warning(f"Stale data detected for {strategy.name}/{symbol}")
                if not self.config.auto_retry_failed_executions:
                    return None
            
            # Perform technical analysis if enabled
            enhanced_context = context
            if self.config.enable_technical_analysis and self.technical_analysis_engine:
                # Add technical analysis to context
                for timeframe, df in data.items():
                    analysis = self.technical_analysis_engine.analyze(df, symbol)
                    # Enhanced context would include technical analysis results
                    # This would require extending StrategyContext
            
            # Execute the strategy
            result = await strategy.execute(symbol, data, enhanced_context)
            
            # Record execution result
            end_time = datetime.now()
            execution_time_ms = (end_time - start_time).total_seconds() * 1000
            
            # Update metrics
            self._update_execution_metrics(True, execution_time_ms, result.signal)
            self.execution_results.append(result)
            
            # Trigger callbacks
            self._trigger_callbacks('strategy_executed', {
                'strategy_name': strategy.name,
                'symbol': symbol,
                'result': result,
                'execution_time_ms': execution_time_ms,
                'execution_id': execution_id
            })
            
            self.logger.debug(f"Strategy {strategy.name} executed for {symbol}: {result.signal.value}")
            return result
            
        except Exception as e:
            end_time = datetime.now()
            execution_time_ms = (end_time - start_time).total_seconds() * 1000
            
            self._update_execution_metrics(False, execution_time_ms)
            
            self.logger.error(f"Strategy execution failed {strategy.name}/{symbol}: {e}")
            self.logger.debug(traceback.format_exc())
            
            # Trigger error callbacks
            self._trigger_callbacks('strategy_error', {
                'strategy_name': strategy.name,
                'symbol': symbol,
                'error': str(e),
                'execution_time_ms': execution_time_ms,
                'execution_id': execution_id
            })
            
            return None
    
    async def _load_active_strategies(self) -> None:
        """Load active strategies from registry"""
        try:
            strategy_names = self.strategy_registry.list_strategies()
            
            for strategy_name in strategy_names:
                try:
                    # This would require strategy configuration
                    # For now, we'll skip automatic loading
                    pass
                except Exception as e:
                    self.logger.error(f"Error loading strategy {strategy_name}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error loading active strategies: {e}")
    
    def _can_execute(self, strategy_name: str, force: bool = False) -> bool:
        """Check if strategy can be executed"""
        if force:
            return True
            
        if self.state != ExecutionState.RUNNING:
            return False
        
        # Check concurrent execution limits
        active_executions = len([
            task for task in self.execution_tasks.values()
            if not task.done()
        ])
        
        if active_executions >= self.config.max_concurrent_strategies:
            return False
        
        return True
    
    def _is_data_fresh(self, data: Dict[TimeFrame, pd.DataFrame]) -> bool:
        """Check if data is fresh enough for execution"""
        if not self.config.data_staleness_threshold_seconds:
            return True
        
        now = datetime.now()
        threshold = timedelta(seconds=self.config.data_staleness_threshold_seconds)
        
        for timeframe, df in data.items():
            if df.empty:
                continue
            
            last_timestamp = df.index[-1]
            if isinstance(last_timestamp, str):
                last_timestamp = pd.to_datetime(last_timestamp)
            
            if now - last_timestamp > threshold:
                return False
        
        return True
    
    def _cleanup_completed_tasks(self) -> None:
        """Clean up completed execution tasks"""
        completed_tasks = [
            execution_id for execution_id, task in self.execution_tasks.items()
            if task.done()
        ]
        
        for execution_id in completed_tasks:
            self.execution_tasks.pop(execution_id, None)
    
    async def _process_execution_queue(self) -> None:
        """Process pending executions from queue"""
        # This would be used for queued executions
        # Implementation depends on specific queueing requirements
        pass
    
    def _update_execution_metrics(
        self,
        success: bool,
        execution_time_ms: float,
        signal: Optional[StrategySignal] = None
    ) -> None:
        """Update execution performance metrics"""
        self.metrics.total_executions += 1
        self.metrics.total_execution_time_ms += execution_time_ms
        
        if success:
            self.metrics.successful_executions += 1
        else:
            self.metrics.failed_executions += 1
        
        # Update average execution time
        self.metrics.average_execution_time_ms = (
            self.metrics.total_execution_time_ms / self.metrics.total_executions
        )
        
        # Track signal types
        if signal:
            self.metrics.signals_generated[signal] += 1
        
        self.metrics.last_execution_time = datetime.now()
    
    def _trigger_callbacks(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Trigger callbacks for an event type"""
        try:
            for callback in self.execution_callbacks.get(event_type, []):
                try:
                    callback(event_data)
                except Exception as e:
                    self.logger.error(f"Callback error for {event_type}: {e}")
        except Exception as e:
            self.logger.error(f"Error triggering callbacks: {e}")


# Helper functions

def create_execution_config(
    mode: ExecutionMode = ExecutionMode.PAPER,
    max_concurrent: int = 10,
    enable_technical_analysis: bool = True,
    **kwargs
) -> ExecutionConfig:
    """Helper function to create execution configuration"""
    return ExecutionConfig(
        execution_mode=mode,
        max_concurrent_strategies=max_concurrent,
        enable_technical_analysis=enable_technical_analysis,
        **kwargs
    ) 
