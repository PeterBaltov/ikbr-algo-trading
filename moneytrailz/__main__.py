"""
MoneyTrailz command-line interface entry point.

This module allows MoneyTrailz to be executed as a module with:
    python -m moneytrailz
"""

from .main import cli

if __name__ == "__main__":
    cli() 