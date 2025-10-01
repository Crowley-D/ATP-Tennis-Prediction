"""
Player ID Matching System

A comprehensive solution for unifying player identification across multiple
tennis data sources (Jeff Sackmann's database, Infosys API, Tennis API).
"""

__version__ = "1.0.0"
__author__ = "Tennis Predictor System"

# Core modules
from . import database
from . import matching
from . import integration

__all__ = [
    'database',
    'matching', 
    'integration'
]