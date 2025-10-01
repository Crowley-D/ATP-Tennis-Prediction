#!/usr/bin/env python3
"""
Chronological Storage System for Tennis Statistics

This module provides chronologically-ordered storage with lazy sorting
to handle non-chronological match input while maintaining performance
for frequent chronological access patterns.
"""

import pandas as pd
from collections import deque
from datetime import datetime
import json


def generate_chronological_key(match_row):
    """
    Generate chronological sort key: (tourney_date, tourney_id, round_order)
    
    Args:
        match_row: pandas Series or dict with match data including:
            - tourney_date: Tournament date
            - tourney_id: Tournament ID
            - round: Round name (F, SF, QF, etc.)
            
    Returns:
        tuple: (tourney_date, tourney_id, round_order) for chronological sorting
    """
    # Extract full tournament date for proper chronological ordering
    tourney_date = match_row.get('tourney_date')
    if tourney_date is None:
        tourney_date = datetime.now()
    else:
        if isinstance(tourney_date, str):
            try:
                tourney_date = pd.to_datetime(tourney_date)
            except:
                tourney_date = datetime.now()
        # Convert to datetime if it's a pandas Timestamp
        if hasattr(tourney_date, 'to_pydatetime'):
            tourney_date = tourney_date.to_pydatetime()
    
    # Get tournament ID (integer for proper sorting)
    tourney_id = match_row.get('tourney_id', 0)
    try:
        tourney_id = int(tourney_id)
    except (ValueError, TypeError):
        tourney_id = 0
    
    # Convert round to numeric order for sorting
    round_name = match_row.get('round', 'UNK')
    round_order = get_round_order(round_name)
    
    return (tourney_date, tourney_id, round_order)


def get_round_order(round_name):
    """
    Convert round name to numeric order for chronological sorting.
    Higher numbers = earlier rounds (chronologically earlier in tournament).
    Descending order ensures proper chronological progression.
    
    Args:
        round_name (str): Round name (e.g., 'F', 'SF', 'QF', etc.)
        
    Returns:
        int: Numeric order for sorting
    """
    # Base round orders - higher = earlier in tournament chronology (descending)
    round_orders = {
        'ER': 256,       # Early Rounds (Qualifying) - earliest chronologically
        'R128': 128,     # Round of 128
        'R64': 64,       # Round of 64
        'R32': 32,       # Round of 32
        'R16': 16,       # Round of 16
        'QF': 8,         # Quarterfinals
        'SF': 4,         # Semifinals  
        'F': 2,          # Final - latest chronologically
    }
    
    # Handle Round Robin special format: 9XX (3 digits starting with 9)
    if isinstance(round_name, str) and len(round_name) == 3 and round_name.startswith('9'):
        try:
            # Extract rank from last two digits (e.g., "903" -> 3)
            rank = int(round_name[1:3])
            # Use 900 + rank format (e.g., 3rd ranked RR match gets round order 903)
            # Higher rank = later chronologically in RR format
            return 900 + rank  
        except (ValueError, IndexError):
            return 999  # Fallback for malformed RR codes (latest chronologically)
    
    return round_orders.get(round_name, 999)  # Unknown rounds go last


class ChronologicalDeque:
    """
    A deque that maintains chronological order with lazy sorting.
    
    Stores (sort_key, value) tuples and provides efficient access to
    chronologically ordered data while minimizing sorting overhead.
    
    New methods for stats interface:
    - remove_by_sort_key(): Remove entries by chronological key
    - remove_by_match_context(): Remove entries by match context
    - get_entries_by_sort_key(): Debug method to find entries by key
    """
    
    def __init__(self, maxlen=1000):
        """
        Initialize chronological deque.
        
        Args:
            maxlen (int): Maximum length of deque
        """
        self._deque = deque(maxlen=maxlen)
        self._is_sorted = True
        self._maxlen = maxlen
    
    def append(self, sort_key, value):
        """
        Add a new entry with its chronological sort key.
        
        Args:
            sort_key (tuple): Chronological sort key from generate_chronological_key()
            value: Value to store
        """
        # Check if appending in order (common case)
        if self._deque and self._is_sorted:
            last_key = self._deque[-1][0]
            # Use custom comparison for chronological order
            if self._compare_chronological_keys(sort_key, last_key) < 0:
                self._is_sorted = False
        
        self._deque.append((sort_key, value))
        
        # If we hit capacity and need sorting, sort now to maintain oldest entries
        if len(self._deque) == self._maxlen and not self._is_sorted:
            self._sort()
    
    def get_chronological_values(self):
        """
        Get all values in chronological order.
        
        Returns:
            list: Values sorted chronologically
        """
        self._sort_if_needed()
        return [value for sort_key, value in self._deque]
    
    def get_last_k_values(self, k):
        """
        Get the k most recent values in chronological order.
        
        Args:
            k (int): Number of recent values to return
            
        Returns:
            list: Last k values in chronological order
        """
        self._sort_if_needed()
        if k >= len(self._deque):
            return [value for sort_key, value in self._deque]
        else:
            return [value for sort_key, value in list(self._deque)[-k:]]
    
    def get_values_before_key(self, cutoff_key):
        """
        Get all values that occurred before the given chronological key.
        
        Args:
            cutoff_key (tuple or str): Chronological sort key cutoff
            
        Returns:
            list: Values that occurred before cutoff_key
        """
        self._sort_if_needed()
        
        # Parse cutoff_key if it's a string
        parsed_cutoff_key = self._parse_chronological_key(cutoff_key)
        if parsed_cutoff_key is None:
            return []
        
        result = []
        for sort_key, value in self._deque:
            # Parse stored sort_key if it's a string
            parsed_sort_key = self._parse_chronological_key(sort_key)
            if parsed_sort_key is not None:
                # Use chronological comparison logic (same as sorting)
                if self._compare_chronological_keys(parsed_sort_key, parsed_cutoff_key) < 0:
                    result.append(value)
        
        return result
    
    def _parse_chronological_key(self, key):
        """
        Parse a chronological key that may be a string, tuple, or list.
        
        Args:
            key: String, tuple, or list chronological key
            
        Returns:
            tuple or None: Parsed chronological key as (datetime, tourney_id, round_order)
        """
        import pandas as pd
        
        # Handle tuple - already in correct format
        if isinstance(key, tuple):
            return key
            
        # Handle list from JSON deserialization
        elif isinstance(key, list) and len(key) >= 3:
            try:
                # First element might be an ISO date string
                if isinstance(key[0], str):
                    if 'T' in key[0]:  # ISO format like "2025-10-08T00:00:00"
                        date_obj = pd.to_datetime(key[0])
                        return (date_obj.to_pydatetime(), key[1], key[2])
                    else:
                        # Regular date string
                        date_obj = pd.to_datetime(key[0])
                        return (date_obj.to_pydatetime(), key[1], key[2])
                # Handle existing datetime objects
                elif hasattr(key[0], 'year'):  # datetime-like object
                    return (key[0], key[1], key[2])
                else:
                    return tuple(key[:3])
            except:
                return None
                
        # Handle string representations like "(2025-10-08, 5000, 8)"
        elif isinstance(key, str) and key.startswith("("):
            import ast
            
            try:
                # First try direct parsing
                parsed_key = ast.literal_eval(key)
                if isinstance(parsed_key, (tuple, list)) and len(parsed_key) >= 3:
                    # Convert date to datetime if needed
                    if isinstance(parsed_key[0], str):
                        try:
                            date_obj = pd.to_datetime(parsed_key[0])
                            return (date_obj.to_pydatetime(), parsed_key[1], parsed_key[2])
                        except:
                            pass
                    elif isinstance(parsed_key[0], (int, float)):
                        # Handle numeric date formats (e.g., Excel serial dates)
                        try:
                            import datetime
                            # Excel serial date: days since 1900-01-01
                            if parsed_key[0] > 40000:  # Reasonable range for modern dates
                                date_obj = datetime.datetime(1900, 1, 1) + datetime.timedelta(days=parsed_key[0] - 2)
                                return (date_obj, parsed_key[1], parsed_key[2])
                        except:
                            pass
                    return parsed_key
            except (ValueError, SyntaxError):
                # If direct parsing fails, try manual extraction
                try:
                    content = key.strip("()")
                    parts = [part.strip().strip("'\"") for part in content.split(",")]
                    if len(parts) == 3:
                        date_part, tourney_id, round_order = parts
                        try:
                            # Try numeric conversion first
                            date_num = float(date_part)
                            if date_num > 40000:  # Excel serial date
                                import datetime
                                date_obj = datetime.datetime(1900, 1, 1) + datetime.timedelta(days=date_num - 2)
                                return (date_obj, int(tourney_id), int(round_order))
                        except ValueError:
                            # Try date parsing
                            date_obj = pd.to_datetime(date_part)
                            return (date_obj.to_pydatetime(), int(tourney_id), int(round_order))
                except:
                    pass
        
        return None
    
    def _compare_chronological_keys(self, key1, key2):
        """
        Compare two chronological keys using proper sorting logic.
        Returns negative if key1 < key2, 0 if equal, positive if key1 > key2.
        """
        def normalize_key(key):
            """Convert key to sortable format: (date, tourney_id, -round_order)"""
            if isinstance(key, tuple) and len(key) >= 3:
                return (key[0], key[1], -key[2])  # Negate round_order for descending
            return key
        
        norm_key1 = normalize_key(key1)
        norm_key2 = normalize_key(key2)
        
        if norm_key1 < norm_key2:
            return -1
        elif norm_key1 > norm_key2:
            return 1
        else:
            return 0
    
    def __len__(self):
        """Return number of entries."""
        return len(self._deque)
    
    def __bool__(self):
        """Return True if deque is not empty."""
        return bool(self._deque)
    
    def _sort_if_needed(self):
        """Sort the deque if it's not already sorted."""
        if not self._is_sorted:
            self._sort()
    
    def _sort(self):
        """Sort the deque by chronological keys."""
        if not self._deque:
            return
            
        # Convert to list, sort with custom key for descending round_order
        def sort_key(item):
            """Sort by (date, tourney_id, -round_order) for proper chronology."""
            sort_tuple = item[0]  # The chronological key tuple
            if isinstance(sort_tuple, tuple) and len(sort_tuple) >= 3:
                return (sort_tuple[0], sort_tuple[1], -sort_tuple[2])  # Negate round_order for descending
            return sort_tuple
            
        sorted_items = sorted(list(self._deque), key=sort_key)
        self._deque.clear()
        self._deque.extend(sorted_items)
        self._is_sorted = True
    
    def _check_if_sorted(self):
        """
        Check if the deque is actually sorted according to current chronological logic.
        
        Returns:
            bool: True if sorted, False otherwise
        """
        if len(self._deque) <= 1:
            return True
            
        for i in range(len(self._deque) - 1):
            current_key = self._deque[i][0]
            next_key = self._deque[i + 1][0]
            
            # Use the same comparison logic as sorting
            if self._compare_chronological_keys(current_key, next_key) > 0:
                return False
                
        return True
    
    def remove_by_sort_key(self, target_sort_key):
        """
        Remove entries matching specific chronological sort key.
        
        Args:
            target_sort_key (tuple): Chronological key to remove
        
        Returns:
            int: Number of entries removed
        """
        original_length = len(self._deque)
        
        # Filter out entries with matching sort key
        filtered_entries = [
            (sort_key, value) for sort_key, value in self._deque 
            if sort_key != target_sort_key
        ]
        
        # Rebuild deque
        self._deque.clear()
        self._deque.extend(filtered_entries)
        
        removed_count = original_length - len(self._deque)
        return removed_count
    
    def remove_by_match_context(self, match_id, player_ids, sort_key):
        """
        Remove entries related to a specific match context.
        More sophisticated removal for complex scenarios.
        
        Args:
            match_id (str): Match identifier
            player_ids (tuple): (p1_id, p2_id)
            sort_key (tuple): Chronological sort key
        
        Returns:
            int: Number of entries removed
        """
        # For now, delegate to remove_by_sort_key
        # Can be enhanced for more complex removal logic in the future
        return self.remove_by_sort_key(sort_key)
    
    def get_entries_by_sort_key(self, target_sort_key):
        """
        Get all entries matching a specific sort key (for debugging).
        
        Args:
            target_sort_key (tuple): Sort key to find
            
        Returns:
            list: List of (sort_key, value) tuples matching the key
        """
        return [(sort_key, value) for sort_key, value in self._deque 
                if sort_key == target_sort_key]
    
    def to_dict(self):
        """
        Convert to dictionary for serialization.
        
        Returns:
            dict: Serializable representation
        """
        # Convert tuple keys with datetime objects to serializable format
        serializable_entries = []
        for sort_key, value in self._deque:
            # Convert datetime to ISO string for JSON serialization
            if isinstance(sort_key, tuple) and len(sort_key) >= 1:
                serializable_key = list(sort_key)
                if hasattr(sort_key[0], 'isoformat'):  # datetime object
                    serializable_key[0] = sort_key[0].isoformat()
                serializable_entries.append((serializable_key, value))
            else:
                serializable_entries.append((sort_key, value))
        
        return {
            'entries': serializable_entries,
            'maxlen': self._maxlen,
            'is_sorted': self._is_sorted
        }
    
    @classmethod
    def from_dict(cls, data):
        """
        Create ChronologicalDeque from dictionary.
        
        Args:
            data (dict): Dictionary from to_dict()
            
        Returns:
            ChronologicalDeque: Reconstructed deque
        """
        maxlen = data.get('maxlen', 1000)
        instance = cls(maxlen=maxlen)
        
        entries = data.get('entries', [])
        
        # Convert serialized entries back to proper format
        reconstructed_entries = []
        for entry in entries:
            sort_key, value = entry
            
            # Convert list back to tuple and ISO string back to datetime
            if isinstance(sort_key, list) and len(sort_key) >= 1:
                # Try to convert first element from ISO string to datetime
                try:
                    if isinstance(sort_key[0], str):
                        from datetime import datetime
                        sort_key[0] = datetime.fromisoformat(sort_key[0])
                except (ValueError, AttributeError):
                    pass  # Keep as string if conversion fails
                
                sort_key = tuple(sort_key)
            
            reconstructed_entries.append((sort_key, value))
        
        instance._deque.extend(reconstructed_entries)
        # Check if data is actually sorted according to current logic
        instance._is_sorted = instance._check_if_sorted()
        if not instance._is_sorted and reconstructed_entries:
            instance._sort()
        
        return instance


def create_chronological_defaultdict(default_factory=None, maxlen=1000):
    """
    Create a defaultdict that creates ChronologicalDeque instances.
    
    Args:
        default_factory: Factory function for defaultdict (if None, uses ChronologicalDeque)
        maxlen (int): Maximum length for each ChronologicalDeque
        
    Returns:
        defaultdict: Dictionary that creates ChronologicalDeque instances
    """
    from collections import defaultdict
    
    if default_factory is None:
        default_factory = lambda: ChronologicalDeque(maxlen=maxlen)
    
    return defaultdict(default_factory)


def create_nested_chronological_defaultdict(maxlen=1000):
    """
    Create nested defaultdict structure for stats like last_k_matches_stats.
    
    Args:
        maxlen (int): Maximum length for each ChronologicalDeque
        
    Returns:
        defaultdict: Nested structure with ChronologicalDeque leaves
    """
    from collections import defaultdict
    
    return defaultdict(lambda: defaultdict(lambda: ChronologicalDeque(maxlen=maxlen)))


if __name__ == "__main__":
    # Test chronological deque functionality
    
    # Test basic functionality
    deque = ChronologicalDeque(maxlen=5)
    
    # Add some test entries (out of order)
    deque.append((2023, 100, 8), "match1")    # QF 2023
    deque.append((2023, 100, 2), "match3")    # F 2023 (later chronologically)
    deque.append((2023, 100, 4), "match2")    # SF 2023
    
    print("Chronological values:", deque.get_chronological_values())
    print("Last 2 values:", deque.get_last_k_values(2))
    
    # Test serialization
    data = deque.to_dict()
    restored = ChronologicalDeque.from_dict(data)
    print("Restored values:", restored.get_chronological_values())