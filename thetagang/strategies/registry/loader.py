"""
Strategy loader for dynamic strategy discovery and loading.

This module handles the loading of strategy classes from modules and
filesystem discovery of strategy implementations.
"""

import importlib
import importlib.util
import inspect
from pathlib import Path
from typing import List, Type, Optional

from ..base import BaseStrategy
from ..exceptions import StrategyRegistrationError


class StrategyLoader:
    """
    Handles dynamic loading of strategy classes.
    
    The loader can discover strategies from filesystem paths and
    load strategy classes from module paths.
    """
    
    def __init__(self) -> None:
        """Initialize the strategy loader."""
        pass
    
    def load_strategy_class(self, module_path: str, class_name: str) -> Type[BaseStrategy]:
        """
        Load a strategy class from a module path.
        
        Args:
            module_path: Python module path (e.g., 'mypackage.strategies.my_strategy')
            class_name: Name of the strategy class
            
        Returns:
            Strategy class
            
        Raises:
            StrategyRegistrationError: If loading fails
        """
        try:
            module = importlib.import_module(module_path)
            strategy_class = getattr(module, class_name)
            
            if not self._is_strategy_class(strategy_class):
                raise StrategyRegistrationError(
                    f"Class {class_name} in {module_path} is not a valid strategy class"
                )
            
            return strategy_class
            
        except ImportError as e:
            raise StrategyRegistrationError(
                f"Could not import module {module_path}: {e}"
            ) from e
        except AttributeError as e:
            raise StrategyRegistrationError(
                f"Class {class_name} not found in module {module_path}: {e}"
            ) from e
    
    def load_strategy_from_file(self, file_path: Path, class_name: str) -> Type[BaseStrategy]:
        """
        Load a strategy class from a Python file.
        
        Args:
            file_path: Path to Python file containing strategy
            class_name: Name of the strategy class
            
        Returns:
            Strategy class
            
        Raises:
            StrategyRegistrationError: If loading fails
        """
        if not file_path.exists() or not file_path.is_file():
            raise StrategyRegistrationError(f"Strategy file not found: {file_path}")
        
        try:
            spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
            if spec is None or spec.loader is None:
                raise StrategyRegistrationError(f"Could not create module spec for {file_path}")
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            strategy_class = getattr(module, class_name)
            
            if not self._is_strategy_class(strategy_class):
                raise StrategyRegistrationError(
                    f"Class {class_name} in {file_path} is not a valid strategy class"
                )
            
            return strategy_class
            
        except Exception as e:
            raise StrategyRegistrationError(
                f"Failed to load strategy from {file_path}: {e}"
            ) from e
    
    def discover_strategies(
        self, 
        search_path: Path, 
        pattern: str = "*_strategy.py"
    ) -> List[Type[BaseStrategy]]:
        """
        Discover strategy classes in a directory.
        
        Args:
            search_path: Directory to search for strategy files
            pattern: File pattern to match (supports wildcards)
            
        Returns:
            List of discovered strategy classes
        """
        strategies = []
        
        if not search_path.exists() or not search_path.is_dir():
            return strategies
        
        # Find matching files
        strategy_files = list(search_path.glob(pattern))
        
        for file_path in strategy_files:
            try:
                discovered = self._discover_strategies_in_file(file_path)
                strategies.extend(discovered)
            except Exception as e:
                # Log error but continue discovery
                print(f"Error discovering strategies in {file_path}: {e}")
        
        return strategies
    
    def _discover_strategies_in_file(self, file_path: Path) -> List[Type[BaseStrategy]]:
        """
        Discover all strategy classes in a single file.
        
        Args:
            file_path: Path to Python file
            
        Returns:
            List of strategy classes found in the file
        """
        strategies = []
        
        try:
            spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
            if spec is None or spec.loader is None:
                return strategies
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find all classes in the module that are strategy classes
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if self._is_strategy_class(obj) and not inspect.isabstract(obj):
                    strategies.append(obj)
            
        except Exception as e:
            raise StrategyRegistrationError(
                f"Failed to discover strategies in {file_path}: {e}"
            ) from e
        
        return strategies
    
    def _is_strategy_class(self, cls: type) -> bool:
        """
        Check if a class is a valid strategy class.
        
        Args:
            cls: Class to check
            
        Returns:
            True if class is a valid strategy
        """
        return (
            inspect.isclass(cls) and
            issubclass(cls, BaseStrategy) and
            cls is not BaseStrategy
        ) 
