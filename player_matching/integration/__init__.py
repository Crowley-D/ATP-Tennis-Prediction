"""
DataFrame processing and batch operations for player matching.
"""

# Import modules as they become available
try:
    from .dataframe_processor import (
        process_players_dataframe,
        process_dataframe_programmatically
    )
    _dataframe_available = True
except ImportError:
    _dataframe_available = False

try:
    from .batch_processing import BatchProcessor
    _batch_available = True
except ImportError:
    _batch_available = False

__all__ = []
if _dataframe_available:
    __all__.extend(['process_players_dataframe', 'process_dataframe_programmatically'])
if _batch_available:
    __all__.extend(['BatchProcessor'])