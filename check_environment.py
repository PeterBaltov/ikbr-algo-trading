#!/usr/bin/env python3
"""
Environment verification script for moneytrailz Python 3.13 workspace
"""

import sys
import platform
import subprocess
from pathlib import Path

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"🐍 Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version >= (3, 13):
        print("✅ Python 3.13+ - GOOD")
        return True
    else:
        print("❌ Python 3.13+ required")
        return False

def check_virtual_environment():
    """Check if running in virtual environment"""
    venv_path = Path(".venv")
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    print(f"📁 Virtual Environment: {venv_path.exists()}")
    print(f"🔧 Active Environment: {in_venv}")
    
    if venv_path.exists() and in_venv:
        print("✅ Virtual environment - GOOD")
        return True
    else:
        print("⚠️  Run: source activate_env.sh")
        return False

def check_thetagang_installation():
    """Check if moneytrailz is installed"""
    try:
        import moneytrailz
        print(f"📊 moneytrailz: Installed")
        
        # Try to run moneytrailz --help
        result = subprocess.run([sys.executable, "-m", "moneytrailz.entry", "--help"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ moneytrailz command - GOOD")
            return True
        else:
            print("❌ moneytrailz command failed")
            return False
    except ImportError:
        print("❌ moneytrailz not installed")
        return False
    except subprocess.TimeoutExpired:
        print("⚠️  moneytrailz command timeout")
        return False

def check_dependencies():
    """Check key dependencies"""
    dependencies = [
        'ib_async',
        'pandas', 
        'numpy',
        'rich',
        'pydantic',
        'click'
    ]
    
    print("📦 Dependencies:")
    all_good = True
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"  ✅ {dep}")
        except ImportError:
            print(f"  ❌ {dep}")
            all_good = False
    
    return all_good

def main():
    """Main verification function"""
    print("🔍 moneytrailz Environment Verification")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Virtual Environment", check_virtual_environment), 
        ("moneytrailz Installation", check_thetagang_installation),
        ("Dependencies", check_dependencies)
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n📋 Checking {name}...")
        result = check_func()
        results.append(result)
    
    print("\n" + "=" * 50)
    if all(results):
        print("🎉 ALL CHECKS PASSED!")
        print("🚀 Environment ready for moneytrailz development!")
        print("\nQuick start:")
        print("  source activate_env.sh")
        print("  moneytrailz -c moneytrailz.toml --dry-run")
    else:
        print("❌ Some checks failed")
        print("Fix the issues above and run again")

if __name__ == "__main__":
    main() 
