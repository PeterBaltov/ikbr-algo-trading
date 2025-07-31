#!/usr/bin/env python3
"""
Simple Phase 2 Test - Just verify basic functionality works
"""

def test_basic_imports():
    """Test basic Phase 2 imports"""
    print("🔍 Testing Phase 2 Basic Imports...")
    
    try:
        # Test analysis package import
        from moneytrailz.analysis import TechnicalAnalysisEngine
        print("✅ TechnicalAnalysisEngine imported successfully")
        
        # Test indicator imports
        from moneytrailz.analysis.indicators.base import BaseIndicator, IndicatorResult
        print("✅ Base indicator classes imported successfully")
        
        from moneytrailz.analysis.indicators.trend import SMA, EMA
        print("✅ Trend indicators imported successfully")
        
        from moneytrailz.analysis.indicators.momentum import RSI
        print("✅ Momentum indicators imported successfully")
        
        # Test signal processing
        from moneytrailz.analysis.signals import SignalAggregator
        print("✅ Signal processing imported successfully")
        
        print("\n✅ ALL PHASE 2 IMPORTS SUCCESSFUL!")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_engine_creation():
    """Test creating technical analysis engine"""
    print("\n🚀 Testing Engine Creation...")
    
    try:
        from moneytrailz.analysis import TechnicalAnalysisEngine
        from moneytrailz.strategies.enums import TimeFrame
        
        # Create engine
        engine = TechnicalAnalysisEngine()
        print("✅ TechnicalAnalysisEngine created successfully")
        
        # Try to create default indicators
        engine.create_default_indicators(TimeFrame.DAY_1)
        print("✅ Default indicators created successfully")
        
        # List indicators
        indicators = engine.list_indicators()
        print(f"📊 Registered indicators: {len(indicators)}")
        for indicator in indicators:
            print(f"  • {indicator}")
        
        print("\n✅ ENGINE CREATION SUCCESSFUL!")
        return True
        
    except Exception as e:
        print(f"❌ Engine creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 SIMPLE PHASE 2 TEST")
    print("=" * 40)
    
    tests = [
        test_basic_imports,
        test_engine_creation
    ]
    
    results = []
    for test_func in tests:
        result = test_func()
        results.append(result)
    
    print("\n" + "=" * 40)
    print("📋 RESULTS SUMMARY")
    print("=" * 40)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 PHASE 2 BASIC FUNCTIONALITY WORKING!")
        print("🚀 Technical Analysis Engine is operational!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed") 
