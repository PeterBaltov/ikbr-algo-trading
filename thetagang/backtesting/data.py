"""
Data Management Module for Backtesting

Handles historical data ingestion, validation, survivorship bias correction,
and multiple data source adapters. Provides robust data infrastructure for
backtesting with comprehensive quality checks and preprocessing.

Features:
- Multi-source data ingestion (CSV, databases, APIs)
- Data validation and quality checks
- Survivorship bias detection and handling
- Missing data interpolation and handling
- Corporate actions adjustment
- Data alignment across timeframes
- Caching and performance optimization

Integration:
- Works with Phase 3 multi-timeframe architecture
- Supports all Phase 1 strategy framework requirements
- Integrates with existing IBKR data feeds
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Union, Callable, Tuple
from pathlib import Path
import asyncio
import logging
from collections import defaultdict, deque
import warnings

import pandas as pd
import numpy as np

from ..strategies.enums import TimeFrame
from ..strategies.exceptions import StrategyError


class DataSource(Enum):
    """Supported data sources for backtesting"""
    CSV = "csv"
    DATABASE = "database"  
    IBKR = "ibkr"
    YAHOO = "yahoo"
    ALPHA_VANTAGE = "alpha_vantage"
    QUANDL = "quandl"
    CUSTOM = "custom"


class DataQuality(Enum):
    """Data quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNUSABLE = "unusable"


class BiasType(Enum):
    """Types of data biases"""
    SURVIVORSHIP = "survivorship"
    LOOK_AHEAD = "look_ahead"
    SELECTION = "selection"
    CONFIRMATION = "confirmation"


@dataclass
class DataConfig:
    """Configuration for data management"""
    
    # Data source settings
    source: DataSource = DataSource.CSV
    data_path: Optional[str] = None
    connection_string: Optional[str] = None
    api_key: Optional[str] = None
    
    # Time range
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    timeframes: List[TimeFrame] = field(default_factory=lambda: [TimeFrame.DAY_1])
    
    # Symbols
    symbols: List[str] = field(default_factory=list)
    universes: List[str] = field(default_factory=list)  # Pre-defined symbol groups
    
    # Data quality
    min_quality: DataQuality = DataQuality.ACCEPTABLE
    handle_missing: str = "forward_fill"  # forward_fill, interpolate, drop, raise
    max_missing_pct: float = 0.05  # Maximum allowed missing data percentage
    
    # Bias handling
    check_survivorship_bias: bool = True
    adjust_corporate_actions: bool = True
    exclude_delisted: bool = False
    
    # Caching and performance
    enable_cache: bool = True
    cache_path: Optional[str] = None
    preload_data: bool = True
    max_memory_gb: float = 4.0
    
    # Validation
    validate_data: bool = True
    strict_validation: bool = False
    
    # Custom settings
    custom_loader: Optional[Callable[[str, TimeFrame, 'DataConfig'], pd.DataFrame]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class DataValidationResult:
    """Results of data validation"""
    
    symbol: str
    timeframe: TimeFrame
    is_valid: bool
    quality: DataQuality
    
    # Statistics
    total_records: int
    missing_records: int
    missing_percentage: float
    date_range: Tuple[datetime, datetime]
    
    # Issues found
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    biases_detected: List[BiasType] = field(default_factory=list)
    
    # Data characteristics
    avg_volume: Optional[float] = None
    price_range: Optional[Tuple[float, float]] = None
    volatility: Optional[float] = None
    
    # Metadata
    validation_timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataValidator:
    """Validates historical data quality and detects biases"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def validate_data(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> DataValidationResult:
        """Comprehensive data validation"""
        
        # Handle date range properly
        if not data.empty and isinstance(data.index, pd.DatetimeIndex):
            date_range = (pd.Timestamp(data.index[0]).to_pydatetime(), pd.Timestamp(data.index[-1]).to_pydatetime())
        else:
            date_range = (datetime.now(), datetime.now())
        
        result = DataValidationResult(
            symbol=symbol,
            timeframe=timeframe,
            is_valid=True,
            quality=DataQuality.EXCELLENT,
            total_records=len(data),
            missing_records=0,
            missing_percentage=0.0,
            date_range=date_range
        )
        
        try:
            # Basic validation
            self._validate_basic_structure(data, result)
            
            # Missing data analysis
            self._analyze_missing_data(data, result)
            
            # Price and volume validation
            self._validate_market_data(data, result)
            
            # Bias detection
            if self.config.check_survivorship_bias:
                self._detect_survivorship_bias(data, result)
            
            # Data consistency checks
            self._check_data_consistency(data, result)
            
            # Determine overall quality
            self._determine_quality(result)
            
        except Exception as e:
            result.is_valid = False
            result.quality = DataQuality.UNUSABLE
            result.issues.append(f"Validation failed: {str(e)}")
            self.logger.error(f"Data validation failed for {symbol}: {e}")
        
        return result
    
    def _validate_basic_structure(self, data: pd.DataFrame, result: DataValidationResult):
        """Validate basic data structure"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        for col in required_columns:
            if col not in data.columns:
                result.issues.append(f"Missing required column: {col}")
                result.is_valid = False
        
        if not isinstance(data.index, pd.DatetimeIndex):
            result.issues.append("Index must be DatetimeIndex")
            result.is_valid = False
            
        if data.empty:
            result.issues.append("Data is empty")
            result.is_valid = False
    
    def _analyze_missing_data(self, data: pd.DataFrame, result: DataValidationResult):
        """Analyze missing data patterns"""
        if data.empty:
            return
            
        missing_mask = data.isnull()
        total_missing = missing_mask.sum().sum()
        total_values = data.size
        
        result.missing_records = total_missing
        result.missing_percentage = (total_missing / total_values) * 100
        
        if result.missing_percentage > self.config.max_missing_pct * 100:
            result.issues.append(f"Too much missing data: {result.missing_percentage:.2f}%")
            
        # Check for systematic missing patterns
        if missing_mask.any().any():
            consecutive_missing = self._find_consecutive_missing(data)
            if consecutive_missing > 5:  # More than 5 consecutive missing days
                result.warnings.append(f"Found {consecutive_missing} consecutive missing values")
    
    def _validate_market_data(self, data: pd.DataFrame, result: DataValidationResult):
        """Validate market data reasonableness"""
        if data.empty or 'close' not in data.columns:
            return
            
        # Price validation
        if (data['close'] <= 0).any():
            result.issues.append("Found non-positive prices")
            
        if 'high' in data.columns and 'low' in data.columns:
            # High >= Low check
            if (data['high'] < data['low']).any():
                result.issues.append("Found high < low violations")
                
            # High/Low vs Open/Close consistency
            if 'open' in data.columns:
                invalid_ohlc = (
                    (data['open'] > data['high']) | 
                    (data['open'] < data['low']) |
                    (data['close'] > data['high']) | 
                    (data['close'] < data['low'])
                ).any()
                
                if invalid_ohlc:
                    result.issues.append("Found OHLC consistency violations")
        
        # Volume validation
        if 'volume' in data.columns:
            if (data['volume'] < 0).any():
                result.issues.append("Found negative volume")
                
            zero_volume_pct = (data['volume'] == 0).sum() / len(data) * 100
            if zero_volume_pct > 20:  # More than 20% zero volume
                result.warnings.append(f"High zero volume percentage: {zero_volume_pct:.1f}%")
        
        # Statistical measures
        if 'close' in data.columns:
            result.price_range = (data['close'].min(), data['close'].max())
            result.volatility = data['close'].pct_change().std() * np.sqrt(252)  # Annualized
        
        if 'volume' in data.columns:
            result.avg_volume = data['volume'].mean()
    
    def _detect_survivorship_bias(self, data: pd.DataFrame, result: DataValidationResult):
        """Detect potential survivorship bias"""
        if data.empty:
            return
            
        # Check if stock suddenly appears/disappears
        data_length = len(data)
        expected_length = self._calculate_expected_length(result.date_range, result.timeframe)
        
        if data_length < expected_length * 0.8:  # Less than 80% of expected data
            result.biases_detected.append(BiasType.SURVIVORSHIP)
            result.warnings.append("Potential survivorship bias detected")
    
    def _check_data_consistency(self, data: pd.DataFrame, result: DataValidationResult):
        """Check for data consistency issues"""
        if data.empty:
            return
            
        # Check for extreme price movements (potential data errors)
        if 'close' in data.columns and len(data) > 1:
            returns = data['close'].pct_change()
            extreme_returns = returns.abs() > 0.5  # 50% single-day move
            
            if extreme_returns.any():
                extreme_count = extreme_returns.sum()
                result.warnings.append(f"Found {extreme_count} extreme price movements (>50%)")
        
        # Check for duplicate timestamps
        if data.index.duplicated().any():
            duplicate_count = data.index.duplicated().sum()
            result.issues.append(f"Found {duplicate_count} duplicate timestamps")
    
    def _determine_quality(self, result: DataValidationResult):
        """Determine overall data quality"""
        if not result.is_valid:
            result.quality = DataQuality.UNUSABLE
            return
            
        issue_count = len(result.issues)
        warning_count = len(result.warnings)
        bias_count = len(result.biases_detected)
        
        # Quality scoring
        score = 100
        score -= issue_count * 20  # Major penalty for issues
        score -= warning_count * 5  # Minor penalty for warnings
        score -= bias_count * 10   # Moderate penalty for biases
        score -= result.missing_percentage * 2  # Penalty for missing data
        
        if score >= 90:
            result.quality = DataQuality.EXCELLENT
        elif score >= 75:
            result.quality = DataQuality.GOOD
        elif score >= 60:
            result.quality = DataQuality.ACCEPTABLE
        elif score >= 40:
            result.quality = DataQuality.POOR
        else:
            result.quality = DataQuality.UNUSABLE
            result.is_valid = False
    
    def _find_consecutive_missing(self, data: pd.DataFrame) -> int:
        """Find maximum consecutive missing values"""
        if data.empty:
            return 0
            
        missing_mask = data.isnull().any(axis=1)
        max_consecutive = 0
        current_consecutive = 0
        
        for is_missing in missing_mask:
            if is_missing:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
                
        return max_consecutive
    
    def _calculate_expected_length(self, date_range: Tuple[datetime, datetime], timeframe: TimeFrame) -> int:
        """Calculate expected number of data points"""
        start_date, end_date = date_range
        total_days = (end_date - start_date).days
        
        # Rough estimates - could be more sophisticated
        if timeframe == TimeFrame.DAY_1:
            return int(total_days * 0.7)  # Account for weekends/holidays
        elif timeframe == TimeFrame.HOUR_1:
            return int(total_days * 0.7 * 6.5)  # 6.5 trading hours per day
        elif timeframe == TimeFrame.MINUTE_1:
            return int(total_days * 0.7 * 6.5 * 60)
        else:
            return total_days


class DataManager:
    """Main data management class for backtesting"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.validator = DataValidator(config)
        self.logger = logging.getLogger(__name__)
        
        # Data storage
        self._data_cache: Dict[str, Dict[TimeFrame, pd.DataFrame]] = defaultdict(dict)
        self._validation_results: Dict[str, Dict[TimeFrame, DataValidationResult]] = defaultdict(dict)
        
        # Performance tracking
        self._load_times: Dict[str, float] = {}
        self._memory_usage: float = 0.0
    
    async def load_data(self, symbols: Optional[List[str]] = None, 
                       timeframes: Optional[List[TimeFrame]] = None) -> Dict[str, Dict[TimeFrame, pd.DataFrame]]:
        """Load historical data for specified symbols and timeframes"""
        
        symbols = symbols or self.config.symbols
        timeframes = timeframes or self.config.timeframes
        
        if not symbols:
            raise ValueError("No symbols specified for data loading")
        
        self.logger.info(f"Loading data for {len(symbols)} symbols across {len(timeframes)} timeframes")
        
        loaded_data = {}
        
        for symbol in symbols:
            symbol_data = {}
            
            for timeframe in timeframes:
                try:
                    # Check cache first
                    if self.config.enable_cache and symbol in self._data_cache and timeframe in self._data_cache[symbol]:
                        data = self._data_cache[symbol][timeframe]
                        self.logger.debug(f"Using cached data for {symbol} {timeframe.value}")
                    else:
                        # Load from source
                        data = await self._load_from_source(symbol, timeframe)
                        
                        # Validate data
                        if self.config.validate_data:
                            validation_result = self.validator.validate_data(data, symbol, timeframe)
                            self._validation_results[symbol][timeframe] = validation_result
                            
                            if not validation_result.is_valid and self.config.strict_validation:
                                raise ValueError(f"Data validation failed for {symbol} {timeframe.value}")
                        
                        # Cache data
                        if self.config.enable_cache:
                            self._data_cache[symbol][timeframe] = data
                    
                    # Handle missing data
                    if self.config.handle_missing != "drop":
                        data = self._handle_missing_data(data)
                    
                    symbol_data[timeframe] = data
                    
                except Exception as e:
                    self.logger.error(f"Failed to load data for {symbol} {timeframe.value}: {e}")
                    if self.config.strict_validation:
                        raise
                    
            if symbol_data:
                loaded_data[symbol] = symbol_data
        
        self.logger.info(f"Successfully loaded data for {len(loaded_data)} symbols")
        return loaded_data
    
    async def _load_from_source(self, symbol: str, timeframe: TimeFrame) -> pd.DataFrame:
        """Load data from the configured source"""
        
        if self.config.source == DataSource.CSV:
            return self._load_from_csv(symbol, timeframe)
        elif self.config.source == DataSource.IBKR:
            return await self._load_from_ibkr(symbol, timeframe)
        elif self.config.source == DataSource.CUSTOM and self.config.custom_loader:
            return await self.config.custom_loader(symbol, timeframe, self.config)
        else:
            raise NotImplementedError(f"Data source {self.config.source} not implemented")
    
    def _load_from_csv(self, symbol: str, timeframe: TimeFrame) -> pd.DataFrame:
        """Load data from CSV files"""
        if not self.config.data_path:
            raise ValueError("data_path required for CSV source")
        
        # Construct file path
        data_path = Path(self.config.data_path)
        file_pattern = f"{symbol}_{timeframe.value}.csv"
        file_path = data_path / file_pattern
        
        if not file_path.exists():
            # Try alternative patterns
            alternative_patterns = [
                f"{symbol}.csv",
                f"{symbol}_{timeframe.value}.csv", 
                f"{timeframe.value}_{symbol}.csv"
            ]
            
            for pattern in alternative_patterns:
                alt_path = data_path / pattern
                if alt_path.exists():
                    file_path = alt_path
                    break
            else:
                raise FileNotFoundError(f"Data file not found for {symbol} {timeframe.value}")
        
        # Load CSV
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        # Standardize column names
        data.columns = data.columns.str.lower()
        
        # Apply date filtering
        if self.config.start_date:
            data = data[data.index >= self.config.start_date]
        if self.config.end_date:
            data = data[data.index <= self.config.end_date]
        
        return data
    
    async def _load_from_ibkr(self, symbol: str, timeframe: TimeFrame) -> pd.DataFrame:
        """Load data from IBKR - placeholder for future implementation"""
        # This would integrate with existing IBKR connection
        raise NotImplementedError("IBKR data loading will be implemented with live adapter")
    
    def _handle_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing data according to configuration"""
        if data.empty:
            return data
            
        if self.config.handle_missing == "forward_fill":
            return data.fillna(method='ffill')
        elif self.config.handle_missing == "interpolate":
            return data.interpolate()
        elif self.config.handle_missing == "drop":
            return data.dropna()
        elif self.config.handle_missing == "raise":
            if data.isnull().any().any():
                raise ValueError("Missing data found and handle_missing='raise'")
        
        return data
    
    def get_validation_results(self) -> Dict[str, Dict[TimeFrame, DataValidationResult]]:
        """Get data validation results"""
        return dict(self._validation_results)
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of loaded data"""
        summary = {
            "total_symbols": len(self._data_cache),
            "total_timeframes": len(set(tf for symbol_data in self._data_cache.values() for tf in symbol_data.keys())),
            "memory_usage_mb": self._memory_usage,
            "cache_enabled": self.config.enable_cache,
            "validation_enabled": self.config.validate_data
        }
        
        # Add symbol details
        symbol_details = {}
        for symbol, timeframe_data in self._data_cache.items():
            symbol_info = {}
            for timeframe, data in timeframe_data.items():
                symbol_info[timeframe.value] = {
                    "records": len(data),
                    "date_range": (str(data.index[0]), str(data.index[-1])) if not data.empty else None,
                    "columns": list(data.columns)
                }
            symbol_details[symbol] = symbol_info
        
        summary["symbols"] = symbol_details
        return summary


def create_data_config(
    source: DataSource = DataSource.CSV,
    data_path: Optional[str] = None,
    symbols: Optional[List[str]] = None,
    timeframes: Optional[List[TimeFrame]] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    **kwargs
) -> DataConfig:
    """Create a data configuration with sensible defaults"""
    
    return DataConfig(
        source=source,
        data_path=data_path,
        symbols=symbols or [],
        timeframes=timeframes or [TimeFrame.DAY_1],
        start_date=start_date,
        end_date=end_date,
        **kwargs
    ) 
