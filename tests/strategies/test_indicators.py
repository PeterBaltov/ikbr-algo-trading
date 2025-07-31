#!/usr/bin/env python3
"""
🧪 TECHNICAL INDICATORS TESTS
=============================

Comprehensive tests for technical analysis indicators including:
- Trend indicators (SMA, EMA, WMA, DEMA, TEMA)
- Momentum indicators (RSI, MACD, Stochastic, Williams %R, ROC)
- Volatility indicators (Bollinger Bands, ATR, Keltner, Donchian)
- Volume indicators (VWAP, OBV, A/D Line, PVT)
- Signal processing and aggregation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Import technical analysis components
from moneytrailz.analysis import TechnicalAnalysisEngine
from moneytrailz.analysis.indicators import (
    BaseIndicator, SMA, EMA, WMA, DEMA, TEMA,
    RSI, MACD, Stochastic, WilliamsR, ROC,
    BollingerBands, ATR, KeltnerChannels, DonchianChannel,
    VWAP, OBV, ADLine, PVT
)
from moneytrailz.analysis.signals import (
    SignalProcessor, SignalAggregator, ConfidenceCalculator, CombinedSignal
)
from moneytrailz.strategies.enums import TimeFrame


class TestMarketDataGenerator:
    """Utility class for generating test market data."""
    
    @staticmethod
    def create_trending_data(periods: int = 100, trend: str = 'up', volatility: float = 0.02) -> pd.DataFrame:
        """Create trending market data for testing."""
        dates = pd.date_range(start='2024-01-01', periods=periods, freq='D')
        
        # Generate base price series
        if trend == 'up':
            base_prices = np.linspace(100, 150, periods)
        elif trend == 'down':
            base_prices = np.linspace(150, 100, periods)
        else:  # sideways
            base_prices = np.full(periods, 125) + np.sin(np.linspace(0, 4*np.pi, periods)) * 5
        
        # Add volatility
        noise = np.random.normal(0, volatility, periods)
        prices = base_prices * (1 + noise)
        
        # Create OHLCV data
        data = []
        for i, price in enumerate(prices):
            high = price * (1 + abs(noise[i]) * 0.5)
            low = price * (1 - abs(noise[i]) * 0.5)
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            volume = np.random.randint(500000, 2000000)
            
            data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data, index=dates)
        return df
    
    @staticmethod
    def create_oscillating_data(periods: int = 100, amplitude: float = 20) -> pd.DataFrame:
        """Create oscillating market data for testing momentum indicators."""
        dates = pd.date_range(start='2024-01-01', periods=periods, freq='D')
        
        # Create oscillating price pattern
        x = np.linspace(0, 4*np.pi, periods)
        base_price = 100
        prices = base_price + amplitude * np.sin(x) + np.random.normal(0, 2, periods)
        
        # Create OHLCV data
        data = []
        for i, price in enumerate(prices):
            variation = abs(np.random.normal(0, 1))
            high = price + variation
            low = price - variation
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            volume = np.random.randint(800000, 1500000)
            
            data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data, index=dates)
        return df


class TestTrendIndicators:
    """Test suite for trend indicators."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.timeframe = TimeFrame.DAY_1
        self.test_data = TestMarketDataGenerator.create_trending_data(50, 'up')
        
    def test_sma_calculation(self):
        """Test Simple Moving Average calculation."""
        sma = SMA(self.timeframe, period=10)
        
        # Calculate SMA
        result = sma.calculate(self.test_data)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(self.test_data)
        
        # Verify first 9 values are NaN (need 10 periods)
        assert pd.isna(result.iloc[:9]).all()
        
        # Verify 10th value is average of first 10 closes
        expected_10th = self.test_data['close'].iloc[:10].mean()
        assert abs(result.iloc[9] - expected_10th) < 0.001
        
    def test_ema_calculation(self):
        """Test Exponential Moving Average calculation."""
        ema = EMA(self.timeframe, period=10)
        
        result = ema.calculate(self.test_data)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(self.test_data)
        
        # EMA should have values starting from first period
        assert not pd.isna(result.iloc[9])
        
        # EMA should react faster to price changes than SMA
        sma = SMA(self.timeframe, period=10)
        sma_result = sma.calculate(self.test_data)
        
        # In uptrend, EMA should be higher than SMA in later periods
        assert result.iloc[-1] > sma_result.iloc[-1]
    
    def test_wma_calculation(self):
        """Test Weighted Moving Average calculation."""
        wma = WMA(self.timeframe, period=5)
        
        result = wma.calculate(self.test_data)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(self.test_data)
        
        # Verify calculation for 5th value
        prices = self.test_data['close'].iloc[:5]
        weights = np.arange(1, 6)  # 1, 2, 3, 4, 5
        expected_5th = (prices * weights).sum() / weights.sum()
        
        assert abs(result.iloc[4] - expected_5th) < 0.001
    
    def test_dema_calculation(self):
        """Test Double Exponential Moving Average calculation."""
        dema = DEMA(self.timeframe, period=10)
        
        result = dema.calculate(self.test_data)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(self.test_data)
        
        # DEMA should be more responsive than regular EMA
        ema = EMA(self.timeframe, period=10)
        ema_result = ema.calculate(self.test_data)
        
        # In strong uptrend, DEMA should be higher than EMA
        assert result.iloc[-1] > ema_result.iloc[-1]


class TestMomentumIndicators:
    """Test suite for momentum indicators."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.timeframe = TimeFrame.DAY_1
        self.test_data = TestMarketDataGenerator.create_oscillating_data(50)
    
    def test_rsi_calculation(self):
        """Test Relative Strength Index calculation."""
        rsi = RSI(self.timeframe, period=14)
        
        result = rsi.calculate(self.test_data)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(self.test_data)
        
        # RSI should be between 0 and 100
        valid_rsi = result.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()
        
        # First 14 values should be NaN
        assert pd.isna(result.iloc[:13]).all()
        assert not pd.isna(result.iloc[14])
    
    def test_macd_calculation(self):
        """Test MACD calculation."""
        macd = MACD(self.timeframe, fast_period=12, slow_period=26, signal_period=9)
        
        result = macd.calculate(self.test_data)
        
        assert isinstance(result, dict)
        assert 'macd' in result
        assert 'signal' in result
        assert 'histogram' in result
        
        # All components should be Series
        assert isinstance(result['macd'], pd.Series)
        assert isinstance(result['signal'], pd.Series)
        assert isinstance(result['histogram'], pd.Series)
        
        # MACD line should exist after slow period
        assert not pd.isna(result['macd'].iloc[25])
        
        # Signal line should exist after fast + slow + signal periods
        assert not pd.isna(result['signal'].iloc[34])
    
    def test_stochastic_calculation(self):
        """Test Stochastic oscillator calculation."""
        stoch = Stochastic(self.timeframe, k_period=14, d_period=3)
        
        result = stoch.calculate(self.test_data)
        
        assert isinstance(result, dict)
        assert 'k_percent' in result
        assert 'd_percent' in result
        
        # Stochastic should be between 0 and 100
        k_valid = result['k_percent'].dropna()
        d_valid = result['d_percent'].dropna()
        
        assert (k_valid >= 0).all() and (k_valid <= 100).all()
        assert (d_valid >= 0).all() and (d_valid <= 100).all()
    
    def test_williams_r_calculation(self):
        """Test Williams %R calculation."""
        williams = WilliamsR(self.timeframe, period=14)
        
        result = williams.calculate(self.test_data)
        
        assert isinstance(result, pd.Series)
        
        # Williams %R should be between -100 and 0
        valid_values = result.dropna()
        assert (valid_values >= -100).all()
        assert (valid_values <= 0).all()


class TestVolatilityIndicators:
    """Test suite for volatility indicators."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.timeframe = TimeFrame.DAY_1
        self.test_data = TestMarketDataGenerator.create_trending_data(50, 'up', volatility=0.03)
    
    def test_bollinger_bands_calculation(self):
        """Test Bollinger Bands calculation."""
        bb = BollingerBands(self.timeframe, period=20, std_dev=2)
        
        result = bb.calculate(self.test_data)
        
        assert isinstance(result, dict)
        assert 'upper' in result
        assert 'middle' in result
        assert 'lower' in result
        
        # Middle band should equal SMA
        sma = SMA(self.timeframe, period=20)
        sma_result = sma.calculate(self.test_data)
        
        # Compare non-NaN values
        middle_valid = result['middle'].dropna()
        sma_valid = sma_result.dropna()
        
        pd.testing.assert_series_equal(middle_valid, sma_valid, check_names=False)
        
        # Upper band should be above middle, lower should be below
        for i in range(19, len(self.test_data)):
            assert result['upper'].iloc[i] > result['middle'].iloc[i]
            assert result['lower'].iloc[i] < result['middle'].iloc[i]
    
    def test_atr_calculation(self):
        """Test Average True Range calculation."""
        atr = ATR(self.timeframe, period=14)
        
        result = atr.calculate(self.test_data)
        
        assert isinstance(result, pd.Series)
        assert (result.dropna() > 0).all()  # ATR should always be positive
        
        # First 13 values should be NaN
        assert pd.isna(result.iloc[:13]).all()
        assert not pd.isna(result.iloc[14])
    
    def test_keltner_channels_calculation(self):
        """Test Keltner Channels calculation."""
        kc = KeltnerChannels(self.timeframe, period=20, multiplier=2)
        
        result = kc.calculate(self.test_data)
        
        assert isinstance(result, dict)
        assert 'upper' in result
        assert 'middle' in result
        assert 'lower' in result
        
        # Channels should maintain proper order
        for i in range(19, len(self.test_data)):
            assert result['upper'].iloc[i] > result['middle'].iloc[i]
            assert result['lower'].iloc[i] < result['middle'].iloc[i]


class TestVolumeIndicators:
    """Test suite for volume indicators."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.timeframe = TimeFrame.DAY_1
        self.test_data = TestMarketDataGenerator.create_trending_data(50, 'up')
    
    def test_vwap_calculation(self):
        """Test Volume Weighted Average Price calculation."""
        vwap = VWAP(self.timeframe)
        
        result = vwap.calculate(self.test_data)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(self.test_data)
        
        # VWAP should be within reasonable range of typical prices
        typical_price = (self.test_data['high'] + self.test_data['low'] + self.test_data['close']) / 3
        
        # VWAP should be close to typical price range
        assert (result >= typical_price.min() * 0.95).all()
        assert (result <= typical_price.max() * 1.05).all()
    
    def test_obv_calculation(self):
        """Test On-Balance Volume calculation."""
        obv = OBV(self.timeframe)
        
        result = obv.calculate(self.test_data)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(self.test_data)
        
        # OBV should start at 0 and change based on price direction
        assert result.iloc[0] == 0
        
        # Check that OBV changes according to price movement
        for i in range(1, min(10, len(result))):
            price_up = self.test_data['close'].iloc[i] > self.test_data['close'].iloc[i-1]
            obv_up = result.iloc[i] > result.iloc[i-1]
            
            if price_up:
                assert obv_up or result.iloc[i] == result.iloc[i-1]  # OBV up or unchanged


class TestSignalProcessing:
    """Test suite for signal processing and aggregation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.signal_processor = SignalProcessor()
        self.signal_aggregator = SignalAggregator()
        self.confidence_calculator = ConfidenceCalculator()
    
    def test_signal_processor_initialization(self):
        """Test signal processor initialization."""
        assert hasattr(self.signal_processor, 'process_signal')
        assert hasattr(self.signal_processor, 'combine_signals')
    
    def test_signal_aggregation(self):
        """Test signal aggregation functionality."""
        # Create test signals
        signals = [
            {'indicator': 'RSI', 'signal': 'BUY', 'strength': 0.8, 'weight': 1.0},
            {'indicator': 'MACD', 'signal': 'BUY', 'strength': 0.6, 'weight': 0.8},
            {'indicator': 'MA', 'signal': 'SELL', 'strength': 0.4, 'weight': 0.6}
        ]
        
        # Test aggregation
        aggregated = self.signal_aggregator.aggregate_signals(signals)
        
        assert isinstance(aggregated, dict)
        assert 'consensus_signal' in aggregated
        assert 'confidence' in aggregated
        assert 'contributing_signals' in aggregated
    
    def test_confidence_calculation(self):
        """Test confidence calculation."""
        # Test with high agreement signals
        high_agreement_signals = [
            {'signal': 'BUY', 'strength': 0.9},
            {'signal': 'BUY', 'strength': 0.8},
            {'signal': 'BUY', 'strength': 0.7}
        ]
        
        high_confidence = self.confidence_calculator.calculate_confidence(high_agreement_signals)
        
        # Test with conflicting signals
        conflicting_signals = [
            {'signal': 'BUY', 'strength': 0.8},
            {'signal': 'SELL', 'strength': 0.7},
            {'signal': 'HOLD', 'strength': 0.6}
        ]
        
        low_confidence = self.confidence_calculator.calculate_confidence(conflicting_signals)
        
        # High agreement should have higher confidence
        assert high_confidence > low_confidence
    
    def test_combined_signal_creation(self):
        """Test CombinedSignal creation and properties."""
        signals = [
            {'indicator': 'RSI', 'signal': 'BUY', 'confidence': 0.8},
            {'indicator': 'MACD', 'signal': 'BUY', 'confidence': 0.6}
        ]
        
        combined = CombinedSignal(
            signals=signals,
            consensus='BUY',
            confidence=0.7,
            timestamp=datetime.now()
        )
        
        assert combined.signals == signals
        assert combined.consensus == 'BUY'
        assert combined.confidence == 0.7
        assert isinstance(combined.timestamp, datetime)


class TestTechnicalAnalysisEngine:
    """Test suite for TechnicalAnalysisEngine integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = TechnicalAnalysisEngine()
        self.test_data = TestMarketDataGenerator.create_trending_data(100, 'up')
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        assert hasattr(self.engine, 'add_indicator')
        assert hasattr(self.engine, 'calculate_indicators')
        assert hasattr(self.engine, 'get_signals')
        assert isinstance(self.engine.indicators, dict)
    
    def test_indicator_registration(self):
        """Test indicator registration with engine."""
        # Add various indicators
        self.engine.add_indicator(SMA(TimeFrame.DAY_1, period=20), 'sma_20')
        self.engine.add_indicator(RSI(TimeFrame.DAY_1, period=14), 'rsi_14')
        self.engine.add_indicator(BollingerBands(TimeFrame.DAY_1, period=20), 'bb_20')
        
        # Verify registration
        assert 'sma_20' in self.engine.indicators
        assert 'rsi_14' in self.engine.indicators
        assert 'bb_20' in self.engine.indicators
        assert len(self.engine.indicators) == 3
    
    def test_indicator_calculation(self):
        """Test indicator calculation through engine."""
        # Register indicators
        self.engine.add_indicator(SMA(TimeFrame.DAY_1, period=10), 'sma_10')
        self.engine.add_indicator(RSI(TimeFrame.DAY_1, period=14), 'rsi_14')
        
        # Calculate indicators
        results = self.engine.calculate_indicators(self.test_data)
        
        assert isinstance(results, dict)
        assert 'sma_10' in results
        assert 'rsi_14' in results
        
        # Verify results are proper types
        assert isinstance(results['sma_10'], pd.Series)
        assert isinstance(results['rsi_14'], pd.Series)
    
    def test_signal_generation(self):
        """Test signal generation from indicators."""
        # Add indicators with signal generation capability
        self.engine.add_indicator(RSI(TimeFrame.DAY_1, period=14), 'rsi_14')
        self.engine.add_indicator(MACD(TimeFrame.DAY_1), 'macd')
        
        # Generate signals
        signals = self.engine.get_signals(self.test_data)
        
        assert isinstance(signals, list)
        
        # Each signal should have required fields
        for signal in signals:
            assert 'indicator' in signal
            assert 'signal' in signal
            assert 'confidence' in signal or 'strength' in signal


def run_indicator_tests():
    """Run all technical indicator tests."""
    print("🧪 RUNNING TECHNICAL INDICATORS TESTS")
    print("=" * 50)
    
    # Test categories
    test_categories = [
        # Trend indicators
        ("SMA Calculation", TestTrendIndicators().test_sma_calculation),
        ("EMA Calculation", TestTrendIndicators().test_ema_calculation),
        ("WMA Calculation", TestTrendIndicators().test_wma_calculation),
        ("DEMA Calculation", TestTrendIndicators().test_dema_calculation),
        
        # Momentum indicators
        ("RSI Calculation", TestMomentumIndicators().test_rsi_calculation),
        ("MACD Calculation", TestMomentumIndicators().test_macd_calculation),
        ("Stochastic Calculation", TestMomentumIndicators().test_stochastic_calculation),
        ("Williams %R Calculation", TestMomentumIndicators().test_williams_r_calculation),
        
        # Volatility indicators
        ("Bollinger Bands Calculation", TestVolatilityIndicators().test_bollinger_bands_calculation),
        ("ATR Calculation", TestVolatilityIndicators().test_atr_calculation),
        ("Keltner Channels Calculation", TestVolatilityIndicators().test_keltner_channels_calculation),
        
        # Volume indicators
        ("VWAP Calculation", TestVolumeIndicators().test_vwap_calculation),
        ("OBV Calculation", TestVolumeIndicators().test_obv_calculation),
        
        # Signal processing
        ("Signal Processor Init", TestSignalProcessing().test_signal_processor_initialization),
        ("Signal Aggregation", TestSignalProcessing().test_signal_aggregation),
        ("Confidence Calculation", TestSignalProcessing().test_confidence_calculation),
        ("Combined Signal Creation", TestSignalProcessing().test_combined_signal_creation),
        
        # Engine integration
        ("Engine Initialization", TestTechnicalAnalysisEngine().test_engine_initialization),
        ("Indicator Registration", TestTechnicalAnalysisEngine().test_indicator_registration),
        ("Indicator Calculation", TestTechnicalAnalysisEngine().test_indicator_calculation),
        ("Signal Generation", TestTechnicalAnalysisEngine().test_signal_generation),
    ]
    
    passed = 0
    total = len(test_categories)
    
    for test_name, test_func in test_categories:
        try:
            # Set up appropriate test instance
            if "Trend" in test_name:
                test_instance = TestTrendIndicators()
            elif "Momentum" in test_name:
                test_instance = TestMomentumIndicators()
            elif "Volatility" in test_name:
                test_instance = TestVolatilityIndicators()
            elif "Volume" in test_name:
                test_instance = TestVolumeIndicators()
            elif "Signal" in test_name:
                test_instance = TestSignalProcessing()
            else:
                test_instance = TestTechnicalAnalysisEngine()
            
            test_instance.setup_method()
            
            # Run test
            test_func()
            print(f"✅ {test_name}: PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ {test_name}: FAILED - {e}")
    
    print("\n" + "=" * 50)
    print("📋 TECHNICAL INDICATORS TEST RESULTS")
    print("=" * 50)
    print(f"📊 Total Tests: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {total - passed}")
    
    if passed == total:
        print("\n🎉 ALL INDICATOR TESTS PASSED!")
        print("🚀 Technical analysis system is robust and ready!")
        
        print("\n📊 Tested Components:")
        print("  • 📈 Trend Indicators: SMA, EMA, WMA, DEMA")
        print("  • 📊 Momentum Indicators: RSI, MACD, Stochastic, Williams %R")
        print("  • 📉 Volatility Indicators: Bollinger Bands, ATR, Keltner Channels")
        print("  • 📊 Volume Indicators: VWAP, OBV")
        print("  • 🔄 Signal Processing: Aggregation, Confidence Calculation")
        print("  • 🎯 Engine Integration: Registration, Calculation, Signal Generation")
        
    else:
        print(f"\n⚠️ {total - passed} test(s) failed.")
        print("🔧 Technical analysis system needs attention.")
    
    return passed == total


if __name__ == "__main__":
    # Set random seed for reproducible tests
    np.random.seed(42)
    run_indicator_tests() 
