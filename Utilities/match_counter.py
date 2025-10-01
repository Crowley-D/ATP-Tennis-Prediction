#!/usr/bin/env python3
"""
Match Counter Utility for Tennis Prediction Pipeline

Provides functionality to count processed matches with flexible filtering options
for tournament, round, and year from the processed matches database.
"""

import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.match_tracking import load_processed_matches


def count_processed_matches_by_filter(tournament_id=None, round_name=None, year=None, processed_dict=None):
    """
    Count processed matches with optional filters for tournament, round, and year.
    
    Args:
        tournament_id (str/int, optional): Tournament ID to filter by
        round_name (str, optional): Round name to filter by (e.g., "F", "SF", "QF", "R16", etc.)
        year (int/str, optional): Year to filter by (4-digit year, e.g., 2025)
        processed_dict (dict, optional): Processed matches dictionary. If None, loads from default file.
        
    Returns:
        int: Number of matches matching the filter criteria
        
    Examples:
        count_processed_matches_by_filter(tournament_id="339", round_name="F", year=2025)
        count_processed_matches_by_filter(tournament_id="339")  # All matches for tournament 339
        count_processed_matches_by_filter(round_name="F")  # All finals
        count_processed_matches_by_filter(year=2025)  # All matches in 2025
    """
    # Load processed matches if not provided
    if processed_dict is None:
        processed_dict = load_processed_matches()
    
    # Convert inputs to match the storage format
    tourney_filter = None
    if tournament_id is not None:
        tourney_filter = f"{int(tournament_id):04d}"
    
    round_filter = None
    if round_name is not None:
        round_codes = {
            'F': '002',      # Final
            'SF': '004',     # Semifinals  
            'QF': '008',     # Quarterfinals
            'R16': '016',    # Round of 16
            'R32': '032',    # Round of 32
            'R64': '064',    # Round of 64
            'R128': '128',   # Round of 128
            'RR': '256',     # Round Robin
            'ER': '256'      # Early Rounds (Qualifying)
        }
        round_filter = round_codes.get(round_name.upper(), None)
        if round_filter is None:
            print(f"Warning: Unknown round name '{round_name}'. Available: {list(round_codes.keys())}")
            return 0
    
    year_filter = None
    if year is not None:
        year_filter = f"{int(year) % 100:02d}"  # Convert to 2-digit format
    
    # Count matches with filters
    count = 0
    for tourney_id, tourney_matches in processed_dict.items():
        # Apply tournament filter
        if tourney_filter and tourney_id != tourney_filter:
            continue
            
        for year_key, year_matches in tourney_matches.items():
            # Apply year filter
            if year_filter and year_key != year_filter:
                continue
                
            for round_code, round_matches in year_matches.items():
                # Apply round filter
                if round_filter and round_code != round_filter:
                    continue
                    
                count += len(round_matches)
    
    return count


def get_match_details_by_filter(tournament_id=None, round_name=None, year=None, processed_dict=None):
    """
    Get detailed information about processed matches with optional filters.
    
    Args:
        tournament_id (str/int, optional): Tournament ID to filter by
        round_name (str, optional): Round name to filter by
        year (int/str, optional): Year to filter by (4-digit year)
        processed_dict (dict, optional): Processed matches dictionary
        
    Returns:
        dict: Dictionary with match details organized by tournament/year/round
    """
    if processed_dict is None:
        processed_dict = load_processed_matches()
    
    # Convert inputs to match the storage format
    tourney_filter = None
    if tournament_id is not None:
        tourney_filter = f"{int(tournament_id):04d}"
    
    round_filter = None
    if round_name is not None:
        round_codes = {
            'F': '002', 'SF': '004', 'QF': '008', 'R16': '016',
            'R32': '032', 'R64': '064', 'R128': '128', 'RR': '256', 'ER': '256'
        }
        round_filter = round_codes.get(round_name.upper())
    
    year_filter = None
    if year is not None:
        year_filter = f"{int(year) % 100:02d}"
    
    # Collect matching matches
    results = {}
    
    for tourney_id, tourney_matches in processed_dict.items():
        if tourney_filter and tourney_id != tourney_filter:
            continue
            
        for year_key, year_matches in tourney_matches.items():
            if year_filter and year_key != year_filter:
                continue
                
            for round_code, round_matches in year_matches.items():
                if round_filter and round_code != round_filter:
                    continue
                
                if len(round_matches) > 0:
                    if tourney_id not in results:
                        results[tourney_id] = {}
                    if year_key not in results[tourney_id]:
                        results[tourney_id][year_key] = {}
                    
                    results[tourney_id][year_key][round_code] = {
                        'match_count': len(round_matches),
                        'matches': round_matches
                    }
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Match Counter Examples:")
    print("=" * 50)
    
    # Count all processed matches
    total = count_processed_matches_by_filter()
    print(f"Total processed matches: {total}")
    
    # Count matches for specific tournament
    tournament_count = count_processed_matches_by_filter(tournament_id="339")
    print(f"Matches for tournament 339: {tournament_count}")
    
    # Count finals matches
    finals_count = count_processed_matches_by_filter(round_name="F")
    print(f"Finals matches: {finals_count}")
    
    # Count matches for specific year
    year_count = count_processed_matches_by_filter(year=2025)
    print(f"Matches in 2025: {year_count}")
    
    # Count specific combination
    specific_count = count_processed_matches_by_filter(tournament_id="339", round_name="F", year=2025)
    print(f"Tournament 339, Finals, 2025: {specific_count}")
    
    # Get detailed breakdown
    details = get_match_details_by_filter(tournament_id="339", year=2025)
    if details:
        print(f"\nDetailed breakdown for tournament 339 in 2025:")
        for tourney_id, tourney_data in details.items():
            for year_key, year_data in tourney_data.items():
                for round_code, round_data in year_data.items():
                    print(f"  Round {round_code}: {round_data['match_count']} matches")