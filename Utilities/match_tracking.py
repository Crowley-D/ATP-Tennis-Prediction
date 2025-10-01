#!/usr/bin/env python3
"""
Match Tracking System for Tennis Prediction Pipeline

This module provides functionality to track processed matches using unique match IDs
to prevent duplicate processing and maintain system state across restarts.
"""

import re
import json
import os
from collections import defaultdict
from datetime import datetime
import pandas as pd


def preprocess_rr_round_codes(matches_df):
    """
    Preprocess Round Robin matches to assign proper round codes based on match_num ranking.
    
    For each tourney_id/tourney_date group with RR matches:
    - Determine sort order based on Final round match_num (ascending if Final match_num=1, else ascending by default)
    - Sort RR matches by match_num according to determined order
    - Assign round codes as 9{rank:02d} where rank is 0-based index
    
    Args:
        matches_df (pd.DataFrame): DataFrame containing match data
        
    Returns:
        pd.DataFrame: DataFrame with updated 'round' column for RR matches
    """
    # Make a copy to avoid modifying original data
    df = matches_df.copy()
    
    # Find all RR matches
    rr_matches = df[df['round'] == 'RR'].copy()
    
    if len(rr_matches) == 0:
        return df
    
    print(f"Preprocessing {len(rr_matches)} Round Robin matches...")
    
    # Group by tournament and date
    grouped = rr_matches.groupby(['tourney_id', 'tourney_date'])
    
    for (tourney_id, tourney_date), group in grouped:
        # Determine sort order by looking at Final round match_num
        tournament_matches = df[(df['tourney_id'] == tourney_id) & 
                               (df['tourney_date'] == tourney_date)]
        
        final_matches = tournament_matches[tournament_matches['round'] == 'F']
        
        # Default to ascending order
        ascending_order = True
        
        if len(final_matches) > 0:
            final_match_num = final_matches.iloc[0]['match_num']
            # If final has match_num = 1, use ascending order (early matches have lower numbers)
            # This logic can be adjusted based on actual data patterns
            ascending_order = (final_match_num <= 5)  # Assuming finals are typically early numbers
            print(f"Tournament {tourney_id}: Final match_num={final_match_num}, using {'ascending' if ascending_order else 'descending'} order")
        else:
            print(f"Tournament {tourney_id}: No final found, using default ascending order")
        
        # Sort by match_num according to determined order
        sorted_group = group.sort_values('match_num', ascending=ascending_order)
        
        # Assign round codes based on rank
        for rank, (idx, match) in enumerate(sorted_group.iterrows()):
            new_round_code = f"9{rank:02d}"  # e.g., "903"
            df.loc[idx, 'round'] = new_round_code
            
        print(f"Assigned RR codes for tournament {tourney_id} on {tourney_date}: {len(sorted_group)} matches")
    
    return df


def generate_match_id(match_row):
    """
    Generate unique match ID from match data.
    
    Format: {tourney_id:04d}{year:02d}{round_code:03d}{p1_initials}{p1_id:02d}_{p2_initials}{p2_id:02d}
    Example: "033925002JL03_RO87" for Brisbane 2025 Final between Jiri Lehecka (ID: 208103) and Reilly Opelka (ID: 124187)
    
    Year Logic: Tournaments from Dec 25th onwards use the following year for ID generation
    
    Args:
        match_row: pandas Series or dict with keys:
            - tourney_id: Tournament ID (string or int)
            - tourney_date: Tournament date (datetime or string)
            - round: Round name (e.g., "F", "SF", "QF", "R16", etc.)
            - player1_name/p1_name: First player full name
            - player2_name/p2_name: Second player full name
            - player1_id/p1_id: First player ID
            - player2_id/p2_id: Second player ID
    
    Returns:
        str: Unique match ID
    """
    # Tournament ID (4 digits with leading zeros)
    tourney_id = str(match_row.get('tourney_id', match_row.get('tournament_id', 0)))
    tourney_part = f"{int(float(tourney_id)):04d}"
    
    # Year (last 2 digits) - tournaments from Dec 25th onwards count as next year
    # Try multiple possible date column names from different data sources
    tourney_date = None
    date_columns = ['tourney_date', 'tournament_date', 'start_date', 'start', 'date_start']
    
    for col in date_columns:
        if col in match_row and match_row.get(col) is not None:
            tourney_date = match_row.get(col)
            break
    
    # Handle different date formats and missing dates
    if tourney_date is None:
        # Fallback to current year if no date available
        year_for_id = datetime.now().year
        print(f"Warning: No tournament date found for match, using current year {year_for_id}")
    else:
        # Convert string dates to datetime
        if isinstance(tourney_date, str):
            try:
                tourney_date = pd.to_datetime(tourney_date)
            except:
                # If date parsing fails, use current year
                year_for_id = datetime.now().year
                print(f"Warning: Could not parse tournament date '{tourney_date}', using current year {year_for_id}")
                tourney_date = None
        
        if tourney_date is not None:
            # If date is December 25th or later, use next year for ID
            if tourney_date.month == 12 and tourney_date.day >= 25:
                year_for_id = tourney_date.year + 1
            else:
                year_for_id = tourney_date.year
        else:
            year_for_id = datetime.now().year
    
    year_part = f"{year_for_id % 100:02d}"
    
    # Round code mapping
    round_codes = {
        'F': '002',      # Final
        'SF': '004',     # Semifinals  
        'QF': '008',     # Quarterfinals
        'R16': '016',    # Round of 16
        'R32': '032',    # Round of 32
        'R64': '064',    # Round of 64
        'R128': '128',   # Round of 128
        'ER': '256'      # Early Rounds (Qualifying)
    }
    round_name = match_row.get('round', 'UNK')
    
    # Handle Round Robin special case - should be preprocessed with 9XX format
    if isinstance(round_name, str) and len(round_name) == 3 and round_name.startswith('9'):
        # Use the 3-digit code directly (e.g., "903" -> "903")
        round_part = round_name
    elif round_name == 'RR':
        # Unprocessed RR match - use fallback
        print(f"Warning: Unprocessed RR match found. Use preprocess_rr_round_codes() first.")
        round_part = '999'
    else:
        round_part = round_codes.get(round_name, '999')
    
    # Player initials and IDs - handle multiple column name formats
    player1_name = (match_row.get('player1_name') or 
                   match_row.get('p1_name') or 
                   match_row.get('winner_name', ''))
    player2_name = (match_row.get('player2_name') or 
                   match_row.get('p2_name') or 
                   match_row.get('loser_name', ''))
    
    # Get player IDs
    player1_id = (match_row.get('player1_id') or 
                 match_row.get('p1_id') or 
                 match_row.get('winner_id', 0))
    player2_id = (match_row.get('player2_id') or 
                 match_row.get('p2_id') or 
                 match_row.get('loser_id', 0))
    
    # Generate initials and ID parts
    p1_initials = get_player_initials(player1_name)
    p2_initials = get_player_initials(player2_name)
    
    # Last 2 digits of player IDs
    p1_id_part = f"{int(player1_id) % 100:02d}"
    p2_id_part = f"{int(player2_id) % 100:02d}"
    
    # Format: player1initials + player1id + _ + player2initials + player2id
    player_part = f"{p1_initials}{p1_id_part}_{p2_initials}{p2_id_part}"
    
    return f"{tourney_part}{year_part}{round_part}{player_part}"


def get_player_initials(full_name):
    """
    Extract player initials from full name.
    
    Handles spaces, hyphens, and special characters.
    Takes first letter of each word.
    
    Args:
        full_name (str): Player's full name
        
    Returns:
        str: Player initials (uppercase)
        
    Examples:
        "Jiri Lehecka" -> "JL"
        "Giovanni Mpetshi Perricard" -> "GMP" 
        "Jean-Francois Smith" -> "JFS"
    """
    if not full_name or pd.isna(full_name):
        return "UNK"
    
    # Split on spaces and hyphens, remove empty strings
    words = [word.strip() for word in re.split(r'[\s\-]+', str(full_name).strip()) if word.strip()]
    
    # Take first letter of each word, uppercase
    initials = ''.join(word[0].upper() for word in words if word)
    
    return initials if initials else "UNK"


def create_processed_matches_structure():
    """
    Create empty processed matches dictionary structure.
    
    Returns:
        defaultdict: Multi-level nested dictionary for efficient lookups
    """
    return defaultdict(lambda: defaultdict(lambda: defaultdict(list)))


def is_match_processed(match_id, processed_dict):
    """
    Check if a match has already been processed.
    
    Args:
        match_id (str): Match ID to check
        processed_dict (dict): Processed matches dictionary
        
    Returns:
        bool: True if match has been processed, False otherwise
    """
    if not match_id or len(match_id) < 9:
        return False
        
    # Parse match ID components
    tourney_id = match_id[:4]
    year = match_id[4:6]
    round_code = match_id[6:9]
    player_initials = match_id[9:]
    
    # Check if match exists in processed dictionary
    return player_initials in processed_dict.get(tourney_id, {}).get(year, {}).get(round_code, [])


def add_processed_match(match_id, processed_dict):
    """
    Add a match to the processed matches dictionary.
    
    Args:
        match_id (str): Match ID to add
        processed_dict (dict): Processed matches dictionary to update
    """
    if not match_id or len(match_id) < 9:
        return
        
    # Parse match ID components
    tourney_id = match_id[:4]
    year = match_id[4:6]
    round_code = match_id[6:9]
    player_initials = match_id[9:]
    
    # Add to processed dictionary if not already present
    if player_initials not in processed_dict[tourney_id][year][round_code]:
        processed_dict[tourney_id][year][round_code].append(player_initials)


def get_processed_count(processed_dict):
    """
    Get total count of processed matches.
    
    Args:
        processed_dict (dict): Processed matches dictionary
        
    Returns:
        int: Total number of processed matches
    """
    count = 0
    for tourney_matches in processed_dict.values():
        for year_matches in tourney_matches.values():
            for round_matches in year_matches.values():
                count += len(round_matches)
    return count


def get_processed_matches_file_path():
    """Get the default path for the processed matches file."""
    return "data/processed_matches.json"


def get_processed_matches_by_tournament(tourney_id, processed_dict):
    """
    Get all processed matches for a specific tournament.
    
    Args:
        tourney_id (str): Tournament ID
        processed_dict (dict): Processed matches dictionary
        
    Returns:
        list: List of match IDs for the tournament
    """
    matches = []
    tourney_key = f"{int(tourney_id):04d}"
    
    if tourney_key in processed_dict:
        for year, year_matches in processed_dict[tourney_key].items():
            for round_code, round_matches in year_matches.items():
                for player_initials in round_matches:
                    match_id = f"{tourney_key}{year}{round_code}{player_initials}"
                    matches.append(match_id)
    
    return matches


def get_all_processed_ids(processed_dict):
    """
    Get flat list of all processed match IDs.
    
    Args:
        processed_dict (dict): Processed matches dictionary
        
    Returns:
        list: List of all processed match IDs
    """
    all_ids = []
    for tourney_id, tourney_matches in processed_dict.items():
        for year, year_matches in tourney_matches.items():
            for round_code, round_matches in year_matches.items():
                for player_initials in round_matches:
                    match_id = f"{tourney_id}{year}{round_code}{player_initials}"
                    all_ids.append(match_id)
    return all_ids


def save_processed_matches(processed_dict, filepath="data/processed_matches.json"):
    """
    Save processed matches dictionary to JSON file.
    
    Args:
        processed_dict (dict): Processed matches dictionary
        filepath (str): Path to save file
    """
    # Create directory if it doesn't exist
    dirname = os.path.dirname(filepath)
    if dirname:  # Only create directory if there is one
        os.makedirs(dirname, exist_ok=True)
    
    # Convert defaultdict to regular dict for JSON serialization
    regular_dict = {}
    for tourney_id, tourney_matches in processed_dict.items():
        regular_dict[tourney_id] = {}
        for year, year_matches in tourney_matches.items():
            regular_dict[tourney_id][year] = {}
            for round_code, round_matches in year_matches.items():
                regular_dict[tourney_id][year][round_code] = list(round_matches)
    
    # Save to file with metadata
    save_data = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "total_matches": get_processed_count(processed_dict),
            "version": "1.0"
        },
        "processed_matches": regular_dict
    }
    
    with open(filepath, 'w') as f:
        json.dump(save_data, f, indent=2)


def load_processed_matches(filepath="data/processed_matches.json"):
    """
    Load processed matches dictionary from JSON file.
    
    Args:
        filepath (str): Path to load file
        
    Returns:
        defaultdict: Processed matches dictionary
    """
    if not os.path.exists(filepath):
        return create_processed_matches_structure()
    
    try:
        with open(filepath, 'r') as f:
            save_data = json.load(f)
        
        # Convert regular dict back to defaultdict
        processed_dict = create_processed_matches_structure()
        regular_dict = save_data.get("processed_matches", {})
        
        for tourney_id, tourney_matches in regular_dict.items():
            for year, year_matches in tourney_matches.items():
                for round_code, round_matches in year_matches.items():
                    processed_dict[tourney_id][year][round_code] = round_matches
        
        return processed_dict
        
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error loading processed matches file: {e}")
        print("Creating new processed matches structure...")
        return create_processed_matches_structure()


def migrate_historical_data(csv_filepath="data/0cleanDataset.csv", 
                          save_filepath="data/processed_matches.json"):
    """
    One-time migration of existing dataset to match tracking system.
    
    Reads all historical matches and populates the processed_matches dictionary.
    
    Args:
        csv_filepath (str): Path to historical dataset CSV
        save_filepath (str): Path to save processed matches JSON
        
    Returns:
        defaultdict: Populated processed matches dictionary
    """
    print("Starting historical data migration...")
    
    if not os.path.exists(csv_filepath):
        print(f"Error: Historical dataset not found at {csv_filepath}")
        return create_processed_matches_structure()
    
    # Load historical data
    try:
        clean_data = pd.read_csv(csv_filepath)
        print(f"Loaded {len(clean_data)} historical matches")
    except Exception as e:
        print(f"Error loading historical data: {e}")
        return create_processed_matches_structure()
    
    # Convert tourney_date to datetime
    clean_data['tourney_date'] = pd.to_datetime(clean_data['tourney_date'])
    
    # Create processed matches structure
    processed_matches = create_processed_matches_structure()
    
    # Generate match IDs and populate dictionary
    successful_migrations = 0
    errors = 0
    
    for index, match_row in clean_data.iterrows():
        try:
            # Generate match ID
            match_id = generate_match_id(match_row)
            
            # Add to processed dictionary
            add_processed_match(match_id, processed_matches)
            successful_migrations += 1
            
            if successful_migrations % 5000 == 0:
                print(f"Processed {successful_migrations} matches...")
                
        except Exception as e:
            errors += 1
            if errors <= 10:  # Only print first 10 errors
                print(f"Error processing match at index {index}: {e}")
    
    print(f"\nMigration complete!")
    print(f"Successfully processed: {successful_migrations} matches")
    print(f"Errors encountered: {errors} matches")
    
    # Save processed matches
    save_processed_matches(processed_matches, save_filepath)
    print(f"Saved processed matches to {save_filepath}")
    
    return processed_matches


if __name__ == "__main__":
    # Example usage and testing
    
    # Test match ID generation
    sample_match = {
        'tourney_id': '339',
        'tourney_date': pd.to_datetime('2025-01-01'),
        'round': 'F',
        'player1_name': 'Jiri Lehecka',
        'player2_name': 'Reilly Opelka'
    }
    
    match_id = generate_match_id(sample_match)
    print(f"Sample match ID: {match_id}")
    
    # Test processed matches functionality
    processed = create_processed_matches_structure()
    add_processed_match(match_id, processed)
    print(f"Is match processed: {is_match_processed(match_id, processed)}")
    print(f"Total processed: {get_processed_count(processed)}")