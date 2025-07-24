"""
Strategy Registry implementation.

This module provides the central registry for managing strategy classes,
including registration, retrieval, and lifecycle management.
"""

import importlib
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Set

from ..base import BaseStrategy
from ..enums import StrategyType
from ..exceptions import StrategyRegistrationError
from .loader import StrategyLoader
from .validator import StrategyValidator


class StrategyRegistry:
    """
    Central registry for managing strategy classes.
    
    The registry handles dynamic loading, registration, and retrieval of
    strategy implementations. It supports both manual registration and
    automatic discovery of strategies.
    """
    
    def __init__(self) -> None:
        """Initialize the strategy registry."""
        self._strategies: Dict[str, Type[BaseStrategy]] = {}
        self._strategy_metadata: Dict[str, Dict[str, Any]] = {}
        self._loader = StrategyLoader()
        self._validator = StrategyValidator()
        
    def register_strategy(
        self,
        strategy_class: Type[BaseStrategy],
        name: Optional[str] = None,
        override: bool = False
    ) -> None:
        """
        Register a strategy class.
        
        Args:
            strategy_class: The strategy class to register
            name: Optional custom name for the strategy
            override: Whether to override existing strategy with same name
            
        Raises:
            StrategyRegistrationError: If registration fails
        """
        # Validate strategy class
        if not self._validator.validate_strategy_class(strategy_class):
            raise StrategyRegistrationError(
                f"Strategy class {strategy_class.__name__} failed validation"
            )
        
        # Determine strategy name
        strategy_name = name or self._get_strategy_name(strategy_class)
        
        # Check for conflicts
        if strategy_name in self._strategies and not override:
            raise StrategyRegistrationError(
                f"Strategy '{strategy_name}' is already registered. "
                f"Use override=True to replace it."
            )
        
        # Register the strategy
        self._strategies[strategy_name] = strategy_class
        self._strategy_metadata[strategy_name] = {
            "class_name": strategy_class.__name__,
            "module": strategy_class.__module__,
            "strategy_type": self._get_strategy_type(strategy_class),
            "description": strategy_class.__doc__ or "",
            "abstract": inspect.isabstract(strategy_class)
        }
        
    def get_strategy(self, name: str) -> Optional[Type[BaseStrategy]]:
        """
        Get a strategy class by name.
        
        Args:
            name: Name of the strategy to retrieve
            
        Returns:
            Strategy class if found, None otherwise
        """
        return self._strategies.get(name)
    
    def unregister_strategy(self, name: str) -> bool:
        """
        Unregister a strategy.
        
        Args:
            name: Name of the strategy to unregister
            
        Returns:
            True if strategy was unregistered, False if not found
        """
        if name in self._strategies:
            del self._strategies[name]
            del self._strategy_metadata[name]
            return True
        return False
    
    def list_strategies(self, strategy_type: Optional[StrategyType] = None) -> List[str]:
        """
        List all registered strategy names.
        
        Args:
            strategy_type: Optional filter by strategy type
            
        Returns:
            List of strategy names
        """
        if strategy_type is None:
            return list(self._strategies.keys())
        
        return [
            name for name, metadata in self._strategy_metadata.items()
            if metadata.get("strategy_type") == strategy_type
        ]
    
    def is_registered(self, name: str) -> bool:
        """
        Check if a strategy is registered.
        
        Args:
            name: Strategy name to check
            
        Returns:
            True if strategy is registered
        """
        return name in self._strategies
    
    def get_strategy_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a strategy.
        
        Args:
            name: Strategy name
            
        Returns:
            Strategy metadata dictionary if found
        """
        if name not in self._strategies:
            return None
        
        metadata = self._strategy_metadata[name].copy()
        metadata["name"] = name
        metadata["registered"] = True
        
        # Add runtime information
        strategy_class = self._strategies[name]
        if hasattr(strategy_class, 'get_default_config'):
            try:
                metadata["default_config"] = strategy_class.get_default_config()
            except Exception:
                metadata["default_config"] = {}
        
        return metadata
    
    def discover_strategies(
        self,
        search_paths: List[Path],
        pattern: str = "*_strategy.py"
    ) -> int:
        """
        Automatically discover and register strategies from filesystem.
        
        Args:
            search_paths: List of directories to search
            pattern: File pattern to match (default: *_strategy.py)
            
        Returns:
            Number of strategies discovered and registered
        """
        discovered_count = 0
        
        for search_path in search_paths:
            strategy_classes = self._loader.discover_strategies(search_path, pattern)
            
            for strategy_class in strategy_classes:
                try:
                    self.register_strategy(strategy_class)
                    discovered_count += 1
                except StrategyRegistrationError as e:
                    # Log error but continue with other strategies
                    print(f"Failed to register {strategy_class.__name__}: {e}")
        
        return discovered_count
    
    def load_strategies_from_config(self, config: Dict[str, Any]) -> int:
        """
        Load strategies based on configuration.
        
        Args:
            config: Configuration dictionary with strategy definitions
            
        Returns:
            Number of strategies loaded and registered
        """
        loaded_count = 0
        
        for strategy_name, strategy_config in config.items():
            if not isinstance(strategy_config, dict):
                continue
            
            module_path = strategy_config.get("module")
            class_name = strategy_config.get("class")
            
            if not module_path or not class_name:
                continue
            
            try:
                strategy_class = self._loader.load_strategy_class(module_path, class_name)
                self.register_strategy(strategy_class, strategy_name)
                loaded_count += 1
            except Exception as e:
                print(f"Failed to load strategy {strategy_name}: {e}")
        
        return loaded_count
    
    def create_strategy_instance(
        self,
        name: str,
        config: Dict[str, Any],
        symbols: List[str],
        **kwargs
    ) -> Optional[BaseStrategy]:
        """
        Create an instance of a registered strategy.
        
        Args:
            name: Strategy name
            config: Strategy configuration
            symbols: List of symbols for the strategy
            **kwargs: Additional arguments for strategy initialization
            
        Returns:
            Strategy instance if successful, None otherwise
        """
        strategy_class = self.get_strategy(name)
        if not strategy_class:
            return None
        
        try:
            # Extract required parameters
            strategy_type = config.get("type", StrategyType.MIXED)
            if isinstance(strategy_type, str):
                strategy_type = StrategyType(strategy_type)
            
            timeframes = config.get("timeframes", ["1d"])
            if isinstance(timeframes, list) and timeframes:
                from ..enums import TimeFrame
                timeframes = [TimeFrame(tf) if isinstance(tf, str) else tf for tf in timeframes]
            
            return strategy_class(
                name=name,
                strategy_type=strategy_type,
                config=config,
                symbols=symbols,
                timeframes=timeframes,
                **kwargs
            )
        except Exception as e:
            print(f"Failed to create strategy instance {name}: {e}")
            return None
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the registry.
        
        Returns:
            Dictionary with registry statistics
        """
        type_counts = {}
        for metadata in self._strategy_metadata.values():
            strategy_type = metadata.get("strategy_type", "unknown")
            type_counts[strategy_type] = type_counts.get(strategy_type, 0) + 1
        
        return {
            "total_strategies": len(self._strategies),
            "by_type": type_counts,
            "strategy_names": list(self._strategies.keys())
        }
    
    def clear_registry(self) -> None:
        """Clear all registered strategies."""
        self._strategies.clear()
        self._strategy_metadata.clear()
    
    def _get_strategy_name(self, strategy_class: Type[BaseStrategy]) -> str:
        """
        Get a strategy name from its class.
        
        Args:
            strategy_class: Strategy class
            
        Returns:
            Strategy name
        """
        # Try to get name from class attribute
        if hasattr(strategy_class, "STRATEGY_NAME"):
            return strategy_class.STRATEGY_NAME
        
        # Convert class name to snake_case
        name = strategy_class.__name__
        if name.endswith("Strategy"):
            name = name[:-8]  # Remove "Strategy" suffix
        
        # Convert CamelCase to snake_case
        import re
        name = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
        
        return name
    
    def _get_strategy_type(self, strategy_class: Type[BaseStrategy]) -> Optional[StrategyType]:
        """
        Determine strategy type from class.
        
        Args:
            strategy_class: Strategy class
            
        Returns:
            Strategy type if determinable
        """
        if hasattr(strategy_class, "STRATEGY_TYPE"):
            return strategy_class.STRATEGY_TYPE
        
        # Try to infer from class name
        class_name = strategy_class.__name__.lower()
        if "option" in class_name or "wheel" in class_name:
            return StrategyType.OPTIONS
        elif "stock" in class_name or "equity" in class_name:
            return StrategyType.STOCKS
        elif "vix" in class_name or "hedge" in class_name:
            return StrategyType.HEDGING
        elif "cash" in class_name:
            return StrategyType.CASH_MANAGEMENT
        else:
            return StrategyType.MIXED


# Global registry instance
_global_registry: Optional[StrategyRegistry] = None


def get_registry() -> StrategyRegistry:
    """
    Get the global strategy registry instance.
    
    Returns:
        Global strategy registry
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = StrategyRegistry()
    return _global_registry


def register_strategy(strategy_class: Type[BaseStrategy], name: Optional[str] = None) -> None:
    """
    Register a strategy with the global registry.
    
    Args:
        strategy_class: Strategy class to register
        name: Optional custom name
    """
    get_registry().register_strategy(strategy_class, name) 
