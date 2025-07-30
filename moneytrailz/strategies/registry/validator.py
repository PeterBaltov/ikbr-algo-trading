"""
Strategy validator for validating strategy classes and configurations.

This module provides validation functionality to ensure strategy classes
and configurations meet the required standards and interfaces.
"""

import inspect
from typing import Any, Dict, List, Optional, Type

from ..base import BaseStrategy
from ..enums import StrategyType, TimeFrame
from ..exceptions import StrategyValidationError


class StrategyValidator:
    """
    Validates strategy classes and configurations.
    
    The validator ensures that strategy implementations follow the
    required interface and that configurations are valid.
    """
    
    def __init__(self) -> None:
        """Initialize the strategy validator."""
        pass
    
    def validate_strategy_class(self, strategy_class: Type[BaseStrategy]) -> bool:
        """
        Validate that a class properly implements the strategy interface.
        
        Args:
            strategy_class: Strategy class to validate
            
        Returns:
            True if class is valid
            
        Raises:
            StrategyValidationError: If validation fails
        """
        errors = []
        
        # Check if it's a class
        if not inspect.isclass(strategy_class):
            errors.append("Not a class")
        
        # Check if it inherits from BaseStrategy
        if not issubclass(strategy_class, BaseStrategy):
            errors.append("Must inherit from BaseStrategy")
        
        # Check if it's the base class itself
        if strategy_class is BaseStrategy:
            errors.append("Cannot register the base BaseStrategy class")
        
        # Check required abstract methods are implemented
        abstract_methods = self._get_abstract_methods(strategy_class)
        if abstract_methods:
            errors.append(f"Abstract methods not implemented: {abstract_methods}")
        
        # Check for required methods
        required_methods = ['analyze', 'validate_config', 'get_required_timeframes', 'get_required_symbols']
        for method_name in required_methods:
            if not hasattr(strategy_class, method_name):
                errors.append(f"Missing required method: {method_name}")
            else:
                method = getattr(strategy_class, method_name)
                if not callable(method):
                    errors.append(f"Method {method_name} is not callable")
        
        # Check constructor signature
        self._validate_constructor_signature(strategy_class, errors)
        
        if errors:
            raise StrategyValidationError(
                f"Strategy class {strategy_class.__name__} validation failed",
                strategy_name=strategy_class.__name__,
                validation_errors=errors
            )
        
        return True
    
    def validate_strategy_config(
        self, 
        config: Dict[str, Any], 
        strategy_class: Optional[Type[BaseStrategy]] = None
    ) -> bool:
        """
        Validate strategy configuration.
        
        Args:
            config: Configuration dictionary to validate
            strategy_class: Optional strategy class for specific validation
            
        Returns:
            True if configuration is valid
            
        Raises:
            StrategyValidationError: If validation fails
        """
        errors = []
        
        # Check required fields
        required_fields = ['type', 'enabled']
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        # Validate strategy type
        if 'type' in config:
            strategy_type = config['type']
            if isinstance(strategy_type, str):
                try:
                    StrategyType(strategy_type)
                except ValueError:
                    errors.append(f"Invalid strategy type: {strategy_type}")
            elif not isinstance(strategy_type, StrategyType):
                errors.append("Strategy type must be string or StrategyType enum")
        
        # Validate enabled flag
        if 'enabled' in config and not isinstance(config['enabled'], bool):
            errors.append("'enabled' must be a boolean")
        
        # Validate timeframes if present
        if 'timeframes' in config:
            timeframes = config['timeframes']
            if not isinstance(timeframes, list):
                errors.append("'timeframes' must be a list")
            else:
                for tf in timeframes:
                    if isinstance(tf, str):
                        try:
                            TimeFrame(tf)
                        except ValueError:
                            errors.append(f"Invalid timeframe: {tf}")
                    elif not isinstance(tf, TimeFrame):
                        errors.append("Timeframe items must be strings or TimeFrame enums")
        
        # Validate symbols if present
        if 'symbols' in config:
            symbols = config['symbols']
            if not isinstance(symbols, list):
                errors.append("'symbols' must be a list")
            else:
                for symbol in symbols:
                    if not isinstance(symbol, str):
                        errors.append("Symbol items must be strings")
                    elif not symbol:
                        errors.append("Symbols cannot be empty strings")
        
        # Strategy-specific validation
        if strategy_class and hasattr(strategy_class, 'validate_config'):
            try:
                # Create a temporary instance to validate config
                temp_instance = strategy_class.__new__(strategy_class)
                temp_instance.config = config
                temp_instance.validate_config()
            except Exception as e:
                errors.append(f"Strategy-specific validation failed: {e}")
        
        if errors:
            raise StrategyValidationError(
                "Strategy configuration validation failed",
                validation_errors=errors
            )
        
        return True
    
    def validate_strategy_symbols(self, symbols: List[str]) -> bool:
        """
        Validate a list of trading symbols.
        
        Args:
            symbols: List of symbols to validate
            
        Returns:
            True if symbols are valid
            
        Raises:
            StrategyValidationError: If validation fails
        """
        errors = []
        
        if not isinstance(symbols, list):
            errors.append("Symbols must be a list")
        elif not symbols:
            errors.append("Symbols list cannot be empty")
        else:
            for symbol in symbols:
                if not isinstance(symbol, str):
                    errors.append(f"Symbol must be string, got {type(symbol)}")
                elif not symbol or not symbol.strip():
                    errors.append("Symbol cannot be empty or whitespace")
                elif len(symbol) > 10:  # Reasonable symbol length limit
                    errors.append(f"Symbol too long: {symbol}")
                elif not symbol.replace('.', '').replace('-', '').isalnum():
                    errors.append(f"Symbol contains invalid characters: {symbol}")
        
        if errors:
            raise StrategyValidationError(
                "Symbol validation failed",
                validation_errors=errors
            )
        
        return True
    
    def validate_timeframes(self, timeframes: List[TimeFrame]) -> bool:
        """
        Validate a list of timeframes.
        
        Args:
            timeframes: List of timeframes to validate
            
        Returns:
            True if timeframes are valid
            
        Raises:
            StrategyValidationError: If validation fails
        """
        errors = []
        
        if not isinstance(timeframes, list):
            errors.append("Timeframes must be a list")
        elif not timeframes:
            errors.append("Timeframes list cannot be empty")
        else:
            for tf in timeframes:
                if not isinstance(tf, TimeFrame):
                    errors.append(f"Timeframe must be TimeFrame enum, got {type(tf)}")
        
        if errors:
            raise StrategyValidationError(
                "Timeframe validation failed",
                validation_errors=errors
            )
        
        return True
    
    def _get_abstract_methods(self, cls: Type[BaseStrategy]) -> List[str]:
        """
        Get list of abstract methods that are not implemented.
        
        Args:
            cls: Class to check
            
        Returns:
            List of unimplemented abstract method names
        """
        if not hasattr(cls, '__abstractmethods__'):
            return []
        
        return list(cls.__abstractmethods__)
    
    def _validate_constructor_signature(
        self, 
        strategy_class: Type[BaseStrategy], 
        errors: List[str]
    ) -> None:
        """
        Validate that the constructor has the expected signature.
        
        Args:
            strategy_class: Strategy class to validate
            errors: List to append errors to
        """
        try:
            signature = inspect.signature(strategy_class.__init__)
            params = list(signature.parameters.keys())
            
            # Expected parameters (excluding 'self')
            expected_params = ['name', 'strategy_type', 'config', 'symbols', 'timeframes']
            
            # Check if all expected parameters are present
            for param in expected_params:
                if param not in params:
                    errors.append(f"Constructor missing parameter: {param}")
            
        except Exception as e:
            errors.append(f"Could not validate constructor signature: {e}") 
