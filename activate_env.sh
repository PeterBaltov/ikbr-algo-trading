#!/bin/bash
# ThetaGang Python 3.13 Environment Activation Script
# Usage: source activate_env.sh

echo "🐍 Activating ThetaGang Python 3.13 Environment..."
source .venv/bin/activate

echo "✅ Environment activated!"
echo "📊 Python version: $(python --version)"
echo "🚀 ThetaGang ready!"
echo ""
echo "Quick commands:"
echo "  thetagang --help                           # Show help"
echo "  thetagang -c thetagang.toml --dry-run     # Test run"
echo "  thetagang -c thetagang.toml               # Live run"
echo "  deactivate                                # Exit environment"
echo "" 
