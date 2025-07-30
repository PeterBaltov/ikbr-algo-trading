"""
Execution Scheduler

The ExecutionScheduler manages strategy execution timing and frequency management
across different timeframes, ensuring optimal performance and resource usage.

Key features:
- Multi-timeframe execution coordination
- Market hours awareness
- Frequency-based scheduling
- Resource optimization
- Execution conflict resolution
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Callable, Tuple
import asyncio
import logging
from collections import defaultdict, deque
import pytz

from ..strategies.enums import TimeFrame
from ..strategies.exceptions import StrategyError


class ScheduleType(Enum):
    """Types of execution schedules"""
    INTERVAL = "interval"           # Fixed interval execution
    TIME_BASED = "time_based"       # Specific time-based execution
    MARKET_OPEN = "market_open"     # Execute at market open
    MARKET_CLOSE = "market_close"   # Execute at market close
    ON_DATA = "on_data"            # Execute when new data arrives
    CUSTOM = "custom"              # Custom scheduling logic


class ExecutionPriority(Enum):
    """Priority levels for strategy execution"""
    CRITICAL = "critical"          # Must execute immediately
    HIGH = "high"                 # High priority execution
    NORMAL = "normal"             # Standard priority
    LOW = "low"                   # Can be delayed if needed
    BACKGROUND = "background"     # Run when resources available


@dataclass
class ExecutionWindow:
    """Defines when execution can occur"""
    start_time: time
    end_time: time
    timezone: str = "UTC"
    weekdays_only: bool = True
    exclude_holidays: bool = True
    
    def is_within_window(self, timestamp: datetime) -> bool:
        """Check if timestamp is within execution window"""
        if self.weekdays_only and timestamp.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Convert to specified timezone
        tz = pytz.timezone(self.timezone)
        if timestamp.tzinfo is None:
            timestamp = tz.localize(timestamp)
        else:
            timestamp = timestamp.astimezone(tz)
        
        current_time = timestamp.time()
        
        if self.start_time <= self.end_time:
            # Same day window
            return self.start_time <= current_time <= self.end_time
        else:
            # Overnight window (crosses midnight)
            return current_time >= self.start_time or current_time <= self.end_time


@dataclass
class ScheduleConfig:
    """Configuration for strategy execution scheduling"""
    strategy_name: str
    timeframes: Set[TimeFrame]
    schedule_type: ScheduleType
    priority: ExecutionPriority = ExecutionPriority.NORMAL
    interval_seconds: Optional[int] = None
    execution_times: List[time] = field(default_factory=list)
    execution_window: Optional[ExecutionWindow] = None
    max_concurrent_executions: int = 1
    timeout_seconds: int = 300
    retry_attempts: int = 3
    retry_delay_seconds: int = 30
    enabled: bool = True
    dependencies: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Result of a scheduled execution"""
    strategy_name: str
    timeframe: TimeFrame
    execution_id: str
    start_time: datetime
    end_time: datetime
    success: bool
    error_message: Optional[str] = None
    execution_duration_ms: float = 0.0
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SchedulingError(StrategyError):
    """Scheduling related errors"""
    pass


class ExecutionScheduler:
    """
    Multi-Timeframe Execution Scheduler
    
    Manages the timing and coordination of strategy executions across different
    timeframes, ensuring optimal resource usage and conflict resolution.
    """
    
    def __init__(self, max_concurrent_total: int = 20):
        self.configs: Dict[str, ScheduleConfig] = {}
        self.execution_history: deque[ExecutionResult] = deque(maxlen=1000)
        self.pending_executions: Dict[str, asyncio.Task[Any]] = {}
        self.active_executions: Dict[str, Set[str]] = defaultdict(set)  # strategy -> execution_ids
        self.max_concurrent_total = max_concurrent_total
        self.logger = logging.getLogger(__name__)
        
        # Scheduling control
        self._scheduler_task: Optional[asyncio.Task[Any]] = None
        self._scheduler_running = False
        self._scheduler_interval = 1.0  # Check every second
        
        # Performance tracking
        self._total_executions = 0
        self._successful_executions = 0
        self._failed_executions = 0
        self._last_execution_time: Optional[datetime] = None
        
        # Callbacks
        self.execution_callbacks: Dict[str, List[Callable[..., Any]]] = defaultdict(list)
        
    def register_schedule(self, config: ScheduleConfig) -> None:
        """Register a new execution schedule"""
        try:
            self.configs[config.strategy_name] = config
            self.active_executions[config.strategy_name] = set()
            
            self.logger.info(
                f"Registered schedule for {config.strategy_name}: "
                f"{config.schedule_type.value}, priority: {config.priority.value}"
            )
            
        except Exception as e:
            raise SchedulingError(f"Failed to register schedule for {config.strategy_name}: {e}")
    
    def unregister_schedule(self, strategy_name: str) -> bool:
        """Unregister an execution schedule"""
        try:
            if strategy_name not in self.configs:
                return False
            
            # Cancel any pending executions
            self._cancel_strategy_executions(strategy_name)
            
            # Remove from collections
            self.configs.pop(strategy_name, None)
            self.active_executions.pop(strategy_name, None)
            
            self.logger.info(f"Unregistered schedule for {strategy_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error unregistering schedule for {strategy_name}: {e}")
            return False
    
    def update_schedule(self, strategy_name: str, **updates) -> bool:
        """Update an existing schedule configuration"""
        try:
            if strategy_name not in self.configs:
                return False
            
            config = self.configs[strategy_name]
            for key, value in updates.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            self.logger.info(f"Updated schedule for {strategy_name}: {updates}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating schedule for {strategy_name}: {e}")
            return False
    
    async def start_scheduler(self) -> None:
        """Start the execution scheduler"""
        if self._scheduler_running:
            return
        
        self._scheduler_running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        
        self.logger.info("Execution scheduler started")
    
    async def stop_scheduler(self) -> None:
        """Stop the execution scheduler"""
        self._scheduler_running = False
        
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all pending executions
        for task in list(self.pending_executions.values()):
            task.cancel()
        
        await asyncio.gather(*self.pending_executions.values(), return_exceptions=True)
        self.pending_executions.clear()
        
        self.logger.info("Execution scheduler stopped")
    
    async def execute_strategy(
        self,
        strategy_name: str,
        execution_callback: Callable[..., Any],
        timeframe: Optional[TimeFrame] = None,
        force: bool = False
    ) -> Optional[ExecutionResult]:
        """Execute a strategy manually or on-demand"""
        try:
            config = self.configs.get(strategy_name)
            if not config:
                raise SchedulingError(f"No schedule configuration found for {strategy_name}")
            
            if not config.enabled and not force:
                self.logger.warning(f"Strategy {strategy_name} is disabled, skipping execution")
                return None
            
            # Check execution constraints
            if not force and not self._can_execute_strategy(strategy_name):
                self.logger.debug(f"Strategy {strategy_name} cannot execute now (constraints)")
                return None
            
            # Create execution
            execution_id = self._generate_execution_id(strategy_name, timeframe)
            
            # Execute the strategy
            result = await self._execute_strategy_with_result(
                strategy_name, execution_callback, timeframe, execution_id
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing strategy {strategy_name}: {e}")
            return self._create_failed_result(strategy_name, timeframe, str(e))
    
    def get_next_execution_time(self, strategy_name: str) -> Optional[datetime]:
        """Get the next scheduled execution time for a strategy"""
        config = self.configs.get(strategy_name)
        if not config or not config.enabled:
            return None
        
        now = datetime.now()
        
        if config.schedule_type == ScheduleType.INTERVAL and config.interval_seconds:
            # For interval-based scheduling, calculate next execution
            last_execution = self._get_last_execution_time(strategy_name)
            if last_execution:
                next_time = last_execution + timedelta(seconds=config.interval_seconds)
            else:
                next_time = now
            
            return max(next_time, now)
        
        elif config.schedule_type == ScheduleType.TIME_BASED and config.execution_times:
            # Find next execution time from the list
            for exec_time in sorted(config.execution_times):
                next_datetime = datetime.combine(now.date(), exec_time)
                if next_datetime > now:
                    return next_datetime
            
            # If no time today, use first time tomorrow
            next_day = now.date() + timedelta(days=1)
            return datetime.combine(next_day, config.execution_times[0])
        
        return None
    
    def get_execution_history(
        self,
        strategy_name: Optional[str] = None,
        limit: int = 100
    ) -> List[ExecutionResult]:
        """Get execution history for strategies"""
        history = list(self.execution_history)
        
        if strategy_name:
            history = [r for r in history if r.strategy_name == strategy_name]
        
        return history[-limit:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get scheduler performance statistics"""
        success_rate = (
            self._successful_executions / self._total_executions
            if self._total_executions > 0 else 0.0
        )
        
        active_count = sum(len(execs) for execs in self.active_executions.values())
        
        return {
            'total_executions': self._total_executions,
            'successful_executions': self._successful_executions,
            'failed_executions': self._failed_executions,
            'success_rate': success_rate,
            'active_executions': active_count,
            'pending_executions': len(self.pending_executions),
            'registered_strategies': len(self.configs),
            'scheduler_running': self._scheduler_running,
            'last_execution_time': self._last_execution_time
        }
    
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
    
    async def _scheduler_loop(self) -> None:
        """Main scheduler loop"""
        while self._scheduler_running:
            try:
                await asyncio.sleep(self._scheduler_interval)
                
                if not self._scheduler_running:
                    break
                
                # Check for scheduled executions
                await self._check_scheduled_executions()
                
                # Clean up completed executions
                self._cleanup_completed_executions()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Scheduler loop error: {e}")
    
    async def _check_scheduled_executions(self) -> None:
        """Check for strategies that should be executed now"""
        now = datetime.now()
        
        for strategy_name, config in self.configs.items():
            if not config.enabled:
                continue
            
            if not self._can_execute_strategy(strategy_name):
                continue
            
            should_execute = False
            
            # Check different schedule types
            if config.schedule_type == ScheduleType.INTERVAL:
                should_execute = self._should_execute_interval(strategy_name, config, now)
            elif config.schedule_type == ScheduleType.TIME_BASED:
                should_execute = self._should_execute_time_based(strategy_name, config, now)
            
            if should_execute:
                # Schedule execution (callback will be provided by the calling system)
                self.logger.debug(f"Scheduling execution for {strategy_name}")
                # Note: Actual execution callback would be provided by the execution engine
    
    def _should_execute_interval(self, strategy_name: str, config: ScheduleConfig, now: datetime) -> bool:
        """Check if interval-based strategy should execute"""
        if not config.interval_seconds:
            return False
        
        last_execution = self._get_last_execution_time(strategy_name)
        if not last_execution:
            return True  # First execution
        
        elapsed_seconds = (now - last_execution).total_seconds()
        return elapsed_seconds >= config.interval_seconds
    
    def _should_execute_time_based(self, strategy_name: str, config: ScheduleConfig, now: datetime) -> bool:
        """Check if time-based strategy should execute"""
        if not config.execution_times:
            return False
        
        current_time = now.time()
        
        # Check if current time matches any execution time (within 1 minute tolerance)
        for exec_time in config.execution_times:
            time_diff = abs(
                (datetime.combine(now.date(), current_time) - 
                 datetime.combine(now.date(), exec_time)).total_seconds()
            )
            
            if time_diff <= 60:  # 1 minute tolerance
                # Check if we already executed at this time today
                last_execution = self._get_last_execution_time(strategy_name)
                if last_execution and last_execution.date() == now.date():
                    last_time = last_execution.time()
                    if abs((datetime.combine(now.date(), last_time) - 
                           datetime.combine(now.date(), exec_time)).total_seconds()) <= 60:
                        return False  # Already executed
                
                return True
        
        return False
    
    def _can_execute_strategy(self, strategy_name: str) -> bool:
        """Check if strategy can be executed now"""
        config = self.configs.get(strategy_name)
        if not config:
            return False
        
        # Check execution window
        if config.execution_window:
            if not config.execution_window.is_within_window(datetime.now()):
                return False
        
        # Check concurrent execution limits
        active_count = len(self.active_executions.get(strategy_name, set()))
        if active_count >= config.max_concurrent_executions:
            return False
        
        # Check total concurrent executions
        total_active = sum(len(execs) for execs in self.active_executions.values())
        if total_active >= self.max_concurrent_total:
            return False
        
        # Check dependencies
        for dependency in config.dependencies:
            dep_config = self.configs.get(dependency)
            if dep_config and len(self.active_executions.get(dependency, set())) > 0:
                return False  # Dependency is still running
        
        return True
    
    async def _execute_strategy_with_result(
        self,
        strategy_name: str,
        execution_callback: Callable[..., Any],
        timeframe: Optional[TimeFrame],
        execution_id: str
    ) -> ExecutionResult:
        """Execute strategy and return result"""
        start_time = datetime.now()
        config = self.configs[strategy_name]
        
        try:
            # Add to active executions
            self.active_executions[strategy_name].add(execution_id)
            
            # Execute with timeout
            result = await asyncio.wait_for(
                execution_callback(strategy_name, timeframe),
                timeout=config.timeout_seconds
            )
            
            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            execution_result = ExecutionResult(
                strategy_name=strategy_name,
                timeframe=timeframe or TimeFrame.DAY_1,
                execution_id=execution_id,
                start_time=start_time,
                end_time=end_time,
                success=True,
                execution_duration_ms=duration_ms,
                metadata={'result': result}
            )
            
            self._update_performance_stats(True)
            self._record_execution_result(execution_result)
            
            return execution_result
            
        except asyncio.TimeoutError:
            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            execution_result = ExecutionResult(
                strategy_name=strategy_name,
                timeframe=timeframe or TimeFrame.DAY_1,
                execution_id=execution_id,
                start_time=start_time,
                end_time=end_time,
                success=False,
                error_message="Execution timeout",
                execution_duration_ms=duration_ms
            )
            
            self._update_performance_stats(False)
            self._record_execution_result(execution_result)
            
            return execution_result
            
        except Exception as e:
            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            execution_result = ExecutionResult(
                strategy_name=strategy_name,
                timeframe=timeframe or TimeFrame.DAY_1,
                execution_id=execution_id,
                start_time=start_time,
                end_time=end_time,
                success=False,
                error_message=str(e),
                execution_duration_ms=duration_ms
            )
            
            self._update_performance_stats(False)
            self._record_execution_result(execution_result)
            
            return execution_result
            
        finally:
            # Remove from active executions
            self.active_executions[strategy_name].discard(execution_id)
    
    def _generate_execution_id(self, strategy_name: str, timeframe: Optional[TimeFrame]) -> str:
        """Generate unique execution ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        tf_suffix = f"_{timeframe.value}" if timeframe else ""
        return f"{strategy_name}_{timestamp}{tf_suffix}"
    
    def _get_last_execution_time(self, strategy_name: str) -> Optional[datetime]:
        """Get the last execution time for a strategy"""
        for result in reversed(self.execution_history):
            if result.strategy_name == strategy_name and result.success:
                return result.end_time
        return None
    
    def _cancel_strategy_executions(self, strategy_name: str) -> None:
        """Cancel all pending executions for a strategy"""
        tasks_to_cancel = [
            task for exec_id, task in self.pending_executions.items()
            if exec_id.startswith(strategy_name)
        ]
        
        for task in tasks_to_cancel:
            task.cancel()
    
    def _cleanup_completed_executions(self) -> None:
        """Clean up completed execution tasks"""
        completed_tasks = [
            exec_id for exec_id, task in self.pending_executions.items()
            if task.done()
        ]
        
        for exec_id in completed_tasks:
            self.pending_executions.pop(exec_id, None)
    
    def _create_failed_result(
        self,
        strategy_name: str,
        timeframe: Optional[TimeFrame],
        error_message: str
    ) -> ExecutionResult:
        """Create a failed execution result"""
        now = datetime.now()
        
        return ExecutionResult(
            strategy_name=strategy_name,
            timeframe=timeframe or TimeFrame.DAY_1,
            execution_id=self._generate_execution_id(strategy_name, timeframe),
            start_time=now,
            end_time=now,
            success=False,
            error_message=error_message,
            execution_duration_ms=0.0
        )
    
    def _update_performance_stats(self, success: bool) -> None:
        """Update performance tracking statistics"""
        self._total_executions += 1
        if success:
            self._successful_executions += 1
        else:
            self._failed_executions += 1
        self._last_execution_time = datetime.now()
    
    def _record_execution_result(self, result: ExecutionResult) -> None:
        """Record execution result in history"""
        self.execution_history.append(result)
        
        # Trigger callbacks
        self._trigger_callbacks('execution_completed', {
            'result': result,
            'timestamp': result.end_time
        })
    
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

def create_schedule_config(
    strategy_name: str,
    timeframes: List[TimeFrame],
    schedule_type: ScheduleType,
    **kwargs
) -> ScheduleConfig:
    """Helper function to create schedule configuration"""
    return ScheduleConfig(
        strategy_name=strategy_name,
        timeframes=set(timeframes),
        schedule_type=schedule_type,
        **kwargs
    )


def create_execution_window(
    start_hour: int,
    start_minute: int,
    end_hour: int,
    end_minute: int,
    timezone: str = "America/New_York"
) -> ExecutionWindow:
    """Helper function to create execution window"""
    return ExecutionWindow(
        start_time=time(start_hour, start_minute),
        end_time=time(end_hour, end_minute),
        timezone=timezone
    ) 
