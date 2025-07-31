#!/bin/bash
# MoneyTrailz Python 3.13 Environment Activation Script
# Usage: source activate_env.sh

echo "🐍 Activating MoneyTrailz Python 3.13 Environment..."
source .venv/bin/activate

echo "✅ Environment activated!"
echo "📊 Python version: $(python --version)"
echo "🚀 MoneyTrailz ready!"
echo ""
echo "Quick commands:"
echo "  moneytrailz --help                           # Show help"
echo "  moneytrailz -c moneytrailz.toml --dry-run     # Test run"
echo "  moneytrailz -c moneytrailz.toml               # Live run"
echo "  deactivate                                # Exit environment"
echo "" 
