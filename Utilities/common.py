import numpy as np


def mean(values):
    if not values:
        return np.nan
    return sum(values) / len(values)


def getWinnerLoserIDS(p1_id, p2_id, result):
    if result == 1:
        return p1_id, p2_id
    else:
        return p2_id, p1_id


def get_match_value(match, key, default=0):
    """Helper function to get values from match (pandas Series or dict)"""
    if hasattr(match, 'get'):
        return match.get(key, default)
    else:
        return getattr(match, key, default)


def nan_safe_diff(p1_values, p2_values):
    """
    Calculate difference between two means, handling np.nan according to plan:
    - If P1 has data but P2 is np.nan: diff = P1_value - 0 = P1_value
    - If P1 is np.nan but P2 has data: diff = 0 - P2_value = -P2_value  
    - If both are np.nan: diff = np.nan
    """
    p1_mean = mean(p1_values)
    p2_mean = mean(p2_values)
    
    if np.isnan(p1_mean) and np.isnan(p2_mean):
        return np.nan
    elif np.isnan(p1_mean):
        return -p2_mean
    elif np.isnan(p2_mean):
        return p1_mean
    else:
        return p1_mean - p2_mean


def get_chronological_values(chrono_deque, k, before_time=None):
    """
    Get values from chronological deque, respecting before_time filtering.
    
    Args:
        chrono_deque: ChronologicalDeque instance
        k: Number of recent values to get (ignored if before_time is provided)
        before_time: Optional cutoff key for filtering
        
    Returns:
        list: Values according to filtering criteria
    """
    if before_time is None:
        # Default behavior: get last k values
        return chrono_deque.get_last_k_values(k)
    else:
        # Filter to only matches before cutoff, then take last k
        before_values = chrono_deque.get_values_before_key(before_time)
        if k is None:
            return before_values
        else:
            # Take last k from the filtered values
            return before_values[-k:] if len(before_values) > k else before_values


def get_cumulative_count(chrono_deque, before_time=None):
    """
    Calculate cumulative count from individual events.
    
    Args:
        chrono_deque: ChronologicalDeque containing individual events (each event = +1)
        before_time: Optional cutoff key for filtering
        
    Returns:
        int: Total cumulative count
    """
    if before_time is None:
        events = chrono_deque.get_chronological_values()
    else:
        events = chrono_deque.get_values_before_key(before_time)
    return sum(events)


def get_cumulative_history_stats(chrono_deque, before_time=None):
    """
    Calculate cumulative history stats from individual events.
    
    Args:
        chrono_deque: ChronologicalDeque containing {"wins": int, "matches": int} events
        before_time: Optional cutoff key for filtering
        
    Returns:
        dict: {"wins": total_wins, "matches": total_matches}
    """
    if before_time is None:
        events = chrono_deque.get_chronological_values()
    else:
        events = chrono_deque.get_values_before_key(before_time)
    
    total_wins = sum(event["wins"] for event in events)
    total_matches = sum(event["matches"] for event in events) 
    return {"wins": total_wins, "matches": total_matches}


def deduce_match_date_from_sort_key(sort_key):
    """
    Deduce the actual match date from a chronological sort key.
    
    For tournaments, matches progress through rounds with larger round_order 
    values representing earlier rounds. We increment the tournament date
    by the number of rounds that occurred after this round.
    
    Args:
        sort_key (tuple): (tourney_date, tourney_id, round_order)
        
    Returns:
        datetime: Deduced match date
    """
    from datetime import timedelta
    
    # Handle both string and tuple representations of sort_key
    if isinstance(sort_key, str) and sort_key.startswith("("):
        # Parse string representation like "(03/11/2025, 5000, 2)" or "(45940, 5000, 2)"
        import ast
        import re
        try:
            # First try direct parsing
            sort_key = ast.literal_eval(sort_key)
        except (ValueError, SyntaxError):
            # If that fails, it might contain dates with slashes - extract manually
            try:
                # Extract content between parentheses and split by commas
                content = sort_key.strip("()")
                parts = [part.strip().strip("'\"") for part in content.split(",")]
                if len(parts) == 3:
                    date_part, tourney_id, round_order = parts
                    sort_key = (date_part, int(tourney_id), int(round_order))
                else:
                    print(f"[WARNING] Could not parse sort_key format: {sort_key}")
                    return None
            except (ValueError, IndexError):
                print(f"[WARNING] Could not parse sort_key: {sort_key}")
                return None
    
    tourney_date, tourney_id, round_order = sort_key
    
    # Convert tourney_date to datetime if it's a string
    if isinstance(tourney_date, str):
        import pandas as pd
        try:
            tourney_date = pd.to_datetime(tourney_date)
        except:
            from datetime import datetime
            tourney_date = datetime.now()
    
    # Standard round orders (larger = earlier in tournament)
    round_progression = [2, 4, 8, 16, 32, 64, 128, 256]  # F, SF, QF, R16, R32, R64, R128, ER
    
    # For Round Robin matches (900+ range), treat as day 0
    if round_order >= 900:
        return tourney_date
        
    # Find how many rounds occurred before this one
    rounds_after = 0
    for standard_round in reversed(round_progression):  # Start from latest rounds
        if round_order >= standard_round:
            break
        rounds_after += 1
            
    # Add days based on round progression
    return tourney_date + timedelta(days=rounds_after)


def count_matches_in_date_range(matches_played_deque, cutoff_date, current_date, before_time=None):
    """
    Count matches played within a date range using chronological match data.
    
    Args:
        matches_played_deque: ChronologicalDeque containing match events
        cutoff_date: Start date for counting (matches >= this date are counted)  
        current_date: Current date for reference
        before_time: Optional cutoff key for filtering
        
    Returns:
        int: Number of matches within the date range
    """
    # Get chronological (sort_key, value) pairs directly from deque
    matches_played_deque._sort_if_needed()  # Ensure sorted
    
    match_count = 0
    for sort_key, value in matches_played_deque._deque:
        # Apply before_time filter if specified
        if before_time is not None and sort_key >= before_time:
            break
            
        match_date = deduce_match_date_from_sort_key(sort_key)
        if match_date >= cutoff_date:
            match_count += value  # Each event represents +1 match
            
    return match_count
