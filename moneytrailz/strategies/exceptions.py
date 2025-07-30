"""
Custom exceptions for the MoneyTrailz strategy framework.

This module defines strategy-specific exceptions that provide detailed
error information for debugging and error handling.
"""

from typing import Any, Dict, List, Optional


class StrategyError(Exception):
    """Base exception for all strategy-related errors."""
    
    def __init__(
        self, 
        message: str, 
        strategy_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message)
        self.strategy_name = strategy_name
        self.context = context or {}
        
    def __str__(self) -> str:
        base_msg = super().__str__()
        if self.strategy_name:
            base_msg = f"[{self.strategy_name}] {base_msg}"
        return base_msg


class StrategyConfigError(StrategyError):
    """Raised when strategy configuration is invalid."""
    
    def __init__(
        self, 
        message: str, 
        strategy_name: Optional[str] = None,
        config_field: Optional[str] = None,
        invalid_value: Optional[Any] = None
    ) -> None:
        context = {}
        if config_field:
            context["config_field"] = config_field
        if invalid_value is not None:
            context["invalid_value"] = invalid_value
            
        super().__init__(message, strategy_name, context)
        self.config_field = config_field
        self.invalid_value = invalid_value


class StrategyExecutionError(StrategyError):
    """Raised when strategy execution fails."""
    
    def __init__(
        self, 
        message: str, 
        strategy_name: Optional[str] = None,
        execution_phase: Optional[str] = None,
        original_exception: Optional[Exception] = None
    ) -> None:
        context = {}
        if execution_phase:
            context["execution_phase"] = execution_phase
        if original_exception:
            context["original_exception"] = str(original_exception)
            
        super().__init__(message, strategy_name, context)
        self.execution_phase = execution_phase
        self.original_exception = original_exception


class StrategyDataError(StrategyError):
    """Raised when required market data is unavailable or invalid."""
    
    def __init__(
        self, 
        message: str, 
        strategy_name: Optional[str] = None,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        required_fields: Optional[List[str]] = None
    ) -> None:
        context = {}
        if symbol:
            context["symbol"] = symbol
        if timeframe:
            context["timeframe"] = timeframe
        if required_fields:
            context["required_fields"] = required_fields
            
        super().__init__(message, strategy_name, context)
        self.symbol = symbol
        self.timeframe = timeframe
        self.required_fields = required_fields


class StrategyRegistrationError(StrategyError):
    """Raised when strategy registration fails."""
    pass


class StrategyValidationError(StrategyError):
    """Raised when strategy validation fails."""
    
    def __init__(
        self, 
        message: str, 
        strategy_name: Optional[str] = None,
        validation_errors: Optional[List[str]] = None
    ) -> None:
        context = {}
        if validation_errors:
            context["validation_errors"] = validation_errors
            
        super().__init__(message, strategy_name, context)
        self.validation_errors = validation_errors or []


class StrategyTimeoutError(StrategyError):
    """Raised when strategy execution times out."""
    
    def __init__(
        self, 
        message: str, 
        strategy_name: Optional[str] = None,
        timeout_seconds: Optional[float] = None
    ) -> None:
        context = {}
        if timeout_seconds:
            context["timeout_seconds"] = timeout_seconds
            
        super().__init__(message, strategy_name, context)
        self.timeout_seconds = timeout_seconds 
