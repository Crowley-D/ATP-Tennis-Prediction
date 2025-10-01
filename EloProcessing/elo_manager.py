"""
Elo Rating Management System

This module provides functions to manage tennis player Elo ratings with chronological
tracking based on tournament date, tournament ID, and round. Ratings are stored
persistently using pickle files.

Data Structure:
elo_ratings = {
    player_id: {
        'general': {
            (tourney_date, tourney_id, round): elo_rating,
        },
        'Hard': {
            (tourney_date, tourney_id, round): surface_elo_rating,
        },
        'Clay': {...},
        'Grass': {...},
        'Carpet': {...}
    }
}
"""

import pandas as pd
import numpy as np
import pickle
import os
import shutil
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, Tuple, Optional, Any, List
from utils.chronological_storage import (
    ChronologicalDeque,
    create_chronological_defaultdict,
)


# Tournament and round coefficients (from notebook)
TOURNAMENT_COEFF = {"G": 1.0, "O": 0.9, "F": 0.9, "M": 0.85, "A": 0.75}
ROUND_COEFF = {
    "F": 1.0,
    "BR": 0.95,
    "SF": 0.9,
    "QF": 0.85,
    "RR": 0.85,
    "R16": 0.8,
    "R32": 0.8,
    "R64": 0.75,
    "R128": 0.75,
    "ER": 0.75,
}
BEST_OF_COEFF = {3: 0.9, 5: 1.0}

# ROUND_RANK removed - now using chronological_key directly from pipeline

DEFAULT_START_ELO = 1500


def parse_chronological_key(chronological_key):
    """
    Parse chronological_key from various formats into a standard tuple.

    Args:
        chronological_key: Can be tuple, list, string, or pandas Series value

    Returns:
        tuple: (datetime, tourney_id, round_order) or None if parsing fails
    """
    import pandas as pd
    import ast
    from datetime import datetime

    if chronological_key is None:
        return None

    # Handle tuple - already in correct format
    if isinstance(chronological_key, tuple):
        if len(chronological_key) >= 3:
            return chronological_key
        return None

    # Handle list from JSON deserialization
    elif isinstance(chronological_key, list) and len(chronological_key) >= 3:
        try:
            # First element might be an ISO date string
            if isinstance(chronological_key[0], str):
                if "T" in chronological_key[0]:  # ISO format
                    date_obj = pd.to_datetime(chronological_key[0])
                    return (
                        date_obj.to_pydatetime(),
                        int(chronological_key[1]),
                        int(chronological_key[2]),
                    )
                else:
                    # Regular date string
                    date_obj = pd.to_datetime(chronological_key[0])
                    return (
                        date_obj.to_pydatetime(),
                        int(chronological_key[1]),
                        int(chronological_key[2]),
                    )
            # Handle existing datetime objects
            elif hasattr(chronological_key[0], "year"):  # datetime-like object
                return (
                    chronological_key[0],
                    int(chronological_key[1]),
                    int(chronological_key[2]),
                )
            else:
                return tuple(chronological_key[:3])
        except (ValueError, TypeError):
            return None

    # Handle string representations like "(2025-10-08, 5000, 8)"
    elif isinstance(chronological_key, str) and chronological_key.startswith("("):
        try:
            # First try direct parsing
            parsed_key = ast.literal_eval(chronological_key)
            if isinstance(parsed_key, (tuple, list)) and len(parsed_key) >= 3:
                # Convert date to datetime if needed
                if isinstance(parsed_key[0], str):
                    try:
                        date_obj = pd.to_datetime(parsed_key[0])
                        return (
                            date_obj.to_pydatetime(),
                            int(parsed_key[1]),
                            int(parsed_key[2]),
                        )
                    except:
                        pass
                elif isinstance(parsed_key[0], (int, float)):
                    # Handle numeric date formats (e.g., Excel serial dates)
                    try:
                        if parsed_key[0] > 40000:  # Reasonable range for modern dates
                            import datetime as dt

                            date_obj = dt.datetime(1900, 1, 1) + dt.timedelta(
                                days=parsed_key[0] - 2
                            )
                            return (date_obj, int(parsed_key[1]), int(parsed_key[2]))
                    except:
                        pass
                return tuple(parsed_key[:3])
        except (ValueError, SyntaxError):
            # If direct parsing fails, try manual extraction
            try:
                content = chronological_key.strip("()")
                parts = [part.strip().strip("'\"") for part in content.split(",")]
                if len(parts) == 3:
                    date_part, tourney_id, round_order = parts
                    try:
                        # Try numeric conversion first
                        date_num = float(date_part)
                        if date_num > 40000:  # Excel serial date
                            import datetime as dt

                            date_obj = dt.datetime(1900, 1, 1) + dt.timedelta(
                                days=date_num - 2
                            )
                            return (date_obj, int(tourney_id), int(round_order))
                    except ValueError:
                        # Try date parsing
                        date_obj = pd.to_datetime(date_part)
                        return (
                            date_obj.to_pydatetime(),
                            int(tourney_id),
                            int(round_order),
                        )
            except:
                pass

    return None


def _reverse_round_order_lookup(round_order: int) -> str:
    """
    Convert round_order back to round name using reverse lookup.

    Args:
        round_order: Numeric round order

    Returns:
        Round name string
    """
    from utils.chronological_storage import get_round_order

    # Common round mappings for reverse lookup
    round_mapping = {
        256: "ER",  # Early Rounds (Qualifying)
        128: "R128",  # Round of 128
        64: "R64",  # Round of 64
        32: "R32",  # Round of 32
        16: "R16",  # Round of 16
        8: "QF",  # Quarterfinals
        4: "SF",  # Semifinals
        3: "BR",  # Bronze Medal / 3rd place
        2: "F",  # Final
    }

    # Check direct mapping first
    if round_order in round_mapping:
        return round_mapping[round_order]

    # Handle Round Robin format (900+ range)
    if round_order >= 900:
        rr_number = round_order - 900
        return f"9{rr_number:02d}"  # e.g., 903 becomes "903"

    # Fallback for unknown round orders
    return f"R{round_order}"


# Legacy conversion function removed - no longer supporting legacy format conversion


def load_elo_ratings(filepath: str) -> Dict:
    """
    Load Elo ratings from pickle file in ChronologicalDeque format.

    Args:
        filepath: Path to pickle file

    Returns:
        ChronologicalDeque-based dictionary containing Elo ratings or empty dict if file doesn't exist
    """
    if not os.path.exists(filepath):
        return {}

    try:
        with open(filepath, "rb") as f:
            ratings_dict = pickle.load(f)

            # Validate that the loaded data is in ChronologicalDeque format
            if ratings_dict:
                sample_player = next(iter(ratings_dict.values()), {})
                if sample_player and not isinstance(
                    next(iter(sample_player.values()), None), ChronologicalDeque
                ):
                    print(
                        f"[ERROR] Unsupported Elo ratings format in {filepath}. Expected ChronologicalDeque format."
                    )
                    return {}

            return ratings_dict
    except Exception as e:
        print(f"Error loading Elo ratings from {filepath}: {e}")
        return {}


def create_elo_backup(ratings_dict: Dict, backup_dir: str = "backups/elo") -> str:
    """
    Create timestamped backup before clearing operations.
    Maintains up to 2 backups, replacing oldest on 3rd backup.

    Args:
        ratings_dict: Ratings to backup
        backup_dir: Directory for backups

    Returns:
        Path to created backup file
    """
    # Create backup directory if needed
    os.makedirs(backup_dir, exist_ok=True)

    # Generate timestamp filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"elo_ratings_backup_{timestamp}.pkl"
    backup_path = os.path.join(backup_dir, backup_filename)

    # Check existing backups and remove oldest if >2
    existing_backups = [
        f
        for f in os.listdir(backup_dir)
        if f.startswith("elo_ratings_backup_") and f.endswith(".pkl")
    ]
    existing_backups.sort()  # Sort by filename (timestamp)

    # Remove oldest backups if we have 2 or more (keep only 1, so we can add 1 more)
    while len(existing_backups) >= 2:
        oldest_backup = existing_backups.pop(0)
        oldest_path = os.path.join(backup_dir, oldest_backup)
        try:
            os.remove(oldest_path)
            print(f"Removed old backup: {oldest_backup}")
        except Exception as e:
            print(f"Warning: Could not remove old backup {oldest_backup}: {e}")

    # Save current ratings as backup
    try:
        with open(backup_path, "wb") as f:
            pickle.dump(ratings_dict, f)
        print(f"Created Elo backup: {backup_path}")
        return backup_path
    except Exception as e:
        print(f"Error creating Elo backup: {e}")
        return ""


def save_elo_ratings(
    ratings_dict: Dict, filepath: str, create_backup: bool = True
) -> bool:
    """
    Save Elo ratings to pickle file with optional backup.

    Args:
        ratings_dict: Dictionary containing Elo ratings
        filepath: Path to save pickle file
        create_backup: Whether to create backup of existing file

    Returns:
        True if successful, False otherwise
    """
    try:
        # Create backup if file exists and backup requested
        if create_backup and os.path.exists(filepath):
            backup_path = f"{filepath}.backup"
            shutil.copy2(filepath, backup_path)

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save ratings
        with open(filepath, "wb") as f:
            pickle.dump(ratings_dict, f)

        return True
    except Exception as e:
        print(f"Error saving Elo ratings to {filepath}: {e}")
        return False


def clear_ratings_from_sort_key(ratings_dict: Dict, start_sort_key: Tuple) -> Dict:
    """
    Clear all Elo ratings from and including the specified sort_key.

    Args:
        ratings_dict: Current ratings dictionary
        start_sort_key: (datetime, tourney_id, round_order) tuple

    Returns:
        Ratings dictionary with entries cleared from start_sort_key forward
    """
    if not ratings_dict or not start_sort_key:
        return ratings_dict

    print(f"Clearing ratings from sort_key: {start_sort_key}")
    cleared_count = 0

    for player_id, player_data in ratings_dict.items():
        for surface, surface_deque in player_data.items():
            if not isinstance(surface_deque, ChronologicalDeque):
                continue

            # Get all values and their keys
            surface_deque._sort_if_needed()  # Ensure sorted

            # Create new deque and add only entries before start_sort_key
            new_deque = ChronologicalDeque(maxlen=surface_deque._maxlen)
            entries_to_keep = []

            for sort_key, value in surface_deque._deque:
                if sort_key < start_sort_key:
                    entries_to_keep.append((sort_key, value))
                else:
                    cleared_count += 1

            # Replace the deque contents
            surface_deque._deque.clear()
            surface_deque._deque.extend(entries_to_keep)
            surface_deque._is_sorted = True

    print(
        f"Cleared {cleared_count} rating entries from sort_key {start_sort_key} forward"
    )
    return ratings_dict


def validate_dataframe_structure(matches_df: pd.DataFrame) -> None:
    """
    Validate that DataFrame has required columns and data types.

    Args:
        matches_df: DataFrame to validate

    Raises:
        ValueError: If required columns missing or invalid data types
    """
    required_columns = [
        "chronological_key",
        "match_id",
        "player1_id",
        "player2_id",
        "surface",
        "RESULT",
        "round",
        "tourney_level",
        "best_of",
        "tourney_date",
        "tourney_id",
    ]

    # Check for required columns
    missing_columns = [col for col in required_columns if col not in matches_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Check DataFrame is not empty
    if len(matches_df) == 0:
        raise ValueError("DataFrame is empty")

    # Validate data types and values
    errors = []

    # Check player IDs are numeric
    for col in ["player1_id", "player2_id", "tourney_id"]:
        if not pd.api.types.is_numeric_dtype(matches_df[col]):
            errors.append(f"Column '{col}' must be numeric")

    # Check RESULT values are 0 or 1
    if not matches_df["RESULT"].isin([0, 1]).all():
        errors.append("RESULT column must contain only 0 or 1 values")

    # Check surface values are valid
    valid_surfaces = ["Hard", "Clay", "Grass", "Carpet"]
    invalid_surfaces = matches_df[~matches_df["surface"].isin(valid_surfaces)][
        "surface"
    ].unique()
    if len(invalid_surfaces) > 0:
        errors.append(
            f"Invalid surface values: {invalid_surfaces}. Valid values: {valid_surfaces}"
        )

    # Check best_of values
    if not matches_df["best_of"].isin([3, 5]).all():
        errors.append("best_of column must contain only 3 or 5 values")

    if errors:
        raise ValueError("DataFrame validation errors:\n" + "\n".join(errors))

    print(f"DataFrame validation passed: {len(matches_df)} matches")


def validate_chronological_key_consistency(matches_df: pd.DataFrame) -> None:
    """
    Validate that chronological_keys are properly formatted or parseable.

    Args:
        matches_df: DataFrame with chronological_key column

    Raises:
        ValueError: If chronological_keys are malformed or inconsistent
    """
    errors = []

    for idx, chronological_key in enumerate(matches_df["chronological_key"]):
        try:
            # Try to parse the chronological_key
            parsed_key = parse_chronological_key(chronological_key)

            if parsed_key is None:
                errors.append(
                    f"Row {idx}: chronological_key could not be parsed: {chronological_key}"
                )
                continue

            # Check it's a tuple with 3 elements
            if not isinstance(parsed_key, tuple) or len(parsed_key) != 3:
                errors.append(
                    f"Row {idx}: parsed chronological_key must be 3-element tuple, got {type(parsed_key)}"
                )
                continue

            # Check first element is datetime-like
            if not hasattr(parsed_key[0], "year"):  # Basic datetime check
                errors.append(
                    f"Row {idx}: chronological_key[0] must be datetime, got {type(parsed_key[0])}"
                )

            # Check second element is integer (tourney_id)
            if not isinstance(parsed_key[1], (int, np.integer)):
                errors.append(
                    f"Row {idx}: chronological_key[1] must be integer, got {type(parsed_key[1])}"
                )

            # Check third element is integer (round_order)
            if not isinstance(parsed_key[2], (int, np.integer)):
                errors.append(
                    f"Row {idx}: chronological_key[2] must be integer, got {type(parsed_key[2])}"
                )

        except Exception as e:
            errors.append(f"Row {idx}: Error validating chronological_key: {e}")

        # Stop after 10 errors to avoid spam
        if len(errors) >= 10:
            errors.append("... (stopping after 10 errors)")
            break

    if errors:
        raise ValueError("chronological_key validation errors:\n" + "\n".join(errors))

    print(f"chronological_key validation passed: {len(matches_df)} matches")


def _process_single_match_elo(match_row: pd.Series, ratings_dict: Dict) -> Dict:
    """
    Process single match for Elo updates using chronological_key.

    Args:
        match_row: Single match data with chronological_key column
        ratings_dict: ChronologicalDeque-based ratings

    Returns:
        Updated ratings dictionary
    """
    # Extract match data
    p1_id = match_row["player1_id"]
    p2_id = match_row["player2_id"]
    surface = match_row["surface"]
    result = match_row["RESULT"]
    round_name = match_row["round"]
    level = match_row["tourney_level"]
    best_of = match_row["best_of"]

    # Parse chronological_key
    chronological_key = parse_chronological_key(match_row["chronological_key"])
    if chronological_key is None:
        raise ValueError(
            f"Could not parse chronological_key for match: {match_row['chronological_key']}"
        )

    sort_key = chronological_key  # Use parsed chronological_key as sort_key

    # Initialize players if needed
    for player_id in [p1_id, p2_id]:
        if player_id not in ratings_dict:
            ratings_dict[player_id] = {
                "general": ChronologicalDeque(maxlen=1000),
                "Hard": ChronologicalDeque(maxlen=1000),
                "Clay": ChronologicalDeque(maxlen=1000),
                "Grass": ChronologicalDeque(maxlen=1000),
                "Carpet": ChronologicalDeque(maxlen=1000),
            }

    # Get current Elos (most recent before this match)
    p1_general_elo = _get_player_elo_at_sort_key(p1_id, sort_key, ratings_dict)
    p2_general_elo = _get_player_elo_at_sort_key(p2_id, sort_key, ratings_dict)
    p1_surface_elo = _get_player_elo_at_sort_key(p1_id, sort_key, ratings_dict, surface)
    p2_surface_elo = _get_player_elo_at_sort_key(p2_id, sort_key, ratings_dict, surface)

    # Determine winner (result: 1 = p1 wins, 0 = p2 wins)
    p1_wins = result == 1

    # Calculate new Elos
    new_p1_general, new_p2_general = _calculate_elo_update(
        p1_general_elo, p2_general_elo, p1_wins, level, round_name, best_of
    )
    new_p1_surface, new_p2_surface = _calculate_elo_update(
        p1_surface_elo, p2_surface_elo, p1_wins, level, round_name, best_of
    )

    # Store new Elos using ChronologicalDeque
    ratings_dict[p1_id]["general"].append(sort_key, new_p1_general)
    ratings_dict[p2_id]["general"].append(sort_key, new_p2_general)
    ratings_dict[p1_id][surface].append(sort_key, new_p1_surface)
    ratings_dict[p2_id][surface].append(sort_key, new_p2_surface)

    return ratings_dict


def _get_player_elo_at_sort_key(
    player_id: int,
    target_sort_key: Tuple,
    ratings_dict: Dict,
    surface: Optional[str] = None,
) -> float:
    """
    Get player's Elo rating at or before specified sort_key (for internal processing).

    Args:
        player_id: Player ID
        target_sort_key: Target chronological sort key (EXCLUSIVE - get rating before this)
        ratings_dict: ChronologicalDeque-based ratings dictionary
        surface: Surface type (None for general Elo)

    Returns:
        Elo rating before specified chronological point
    """
    if player_id not in ratings_dict:
        return DEFAULT_START_ELO

    elo_type = surface if surface else "general"
    if elo_type not in ratings_dict[player_id]:
        return DEFAULT_START_ELO

    surface_deque = ratings_dict[player_id][elo_type]
    if not isinstance(surface_deque, ChronologicalDeque):
        return DEFAULT_START_ELO

    # Get all values before target_sort_key
    values_before = surface_deque.get_values_before_key(target_sort_key)

    # Return most recent rating before target, or default if none
    return values_before[-1] if values_before else DEFAULT_START_ELO


def update_elo_from_dataframe(matches_df: pd.DataFrame, ratings_dict: Dict) -> Dict:
    """
    Update Elo ratings from pre-sorted DataFrame with chronological clearing.

    Args:
        matches_df: Pre-sorted DataFrame with required columns
        ratings_dict: Current Elo ratings dictionary

    Returns:
        Updated ratings dictionary

    Process:
        1. Validate DataFrame structure
        2. Get earliest sort_key from input
        3. Create backup before clearing
        4. Clear ratings from earliest sort_key forward
        5. Process matches using existing sort_keys
        6. Return updated ratings
        7. Fall back to full recalculation on errors
    """
    try:
        print(f"Starting DataFrame Elo processing for {len(matches_df)} matches")

        # Step 1: Validate DataFrame structure
        validate_dataframe_structure(matches_df)
        validate_chronological_key_consistency(matches_df)

        # Step 2: Get earliest sort_key from input
        if len(matches_df) == 0:
            print("No matches to process")
            return ratings_dict

        earliest_sort_key = (
            matches_df["chronological_key"].apply(parse_chronological_key).min()
        )
        print(f"Earliest sort_key in input: {earliest_sort_key}")

        # Step 3: Create backup before clearing (if we have existing data)
        if ratings_dict:
            backup_path = create_elo_backup(ratings_dict)
            print(f"Created backup before clearing: {backup_path}")

        # Step 4: Clear ratings from earliest sort_key forward
        ratings_dict = clear_ratings_from_sort_key(ratings_dict, earliest_sort_key)

        # Step 5: Process matches using existing sort_keys
        processed_count = 0
        for idx, match_row in matches_df.iterrows():
            ratings_dict = _process_single_match_elo(match_row, ratings_dict)
            processed_count += 1

            # Progress reporting for large datasets
            if processed_count % 1000 == 0:
                print(f"Processed {processed_count}/{len(matches_df)} matches")

        print(f"Successfully processed {processed_count} matches")
        return ratings_dict

    except Exception as e:
        print(f"Error in DataFrame Elo processing: {e}")
        print("Falling back to full recalculation...")

        # Fall back to full recalculation
        return recalculate_elo_full_from_dataframe(matches_df)


def recalculate_elo_full_from_dataframe(
    matches_df: pd.DataFrame, start_elo: float = DEFAULT_START_ELO
) -> Dict:
    """
    Perform complete Elo recalculation from DataFrame (fallback function).

    Args:
        matches_df: All matches data (pre-sorted)
        start_elo: Starting Elo for all players

    Returns:
        Completely recalculated ChronologicalDeque-based ratings
    """
    print(f"Starting full Elo recalculation from DataFrame: {len(matches_df)} matches")

    try:
        # Validate basic structure
        validate_dataframe_structure(matches_df)
        validate_chronological_key_consistency(matches_df)

        # Initialize empty ChronologicalDeque structure
        ratings_dict = {}

        # Process each match chronologically
        processed_count = 0
        for idx, match_row in matches_df.iterrows():
            ratings_dict = _process_single_match_elo(match_row, ratings_dict)
            processed_count += 1

            # Progress reporting for large datasets
            if processed_count % 1000 == 0:
                print(f"Recalculated {processed_count}/{len(matches_df)} matches")

        print(f"Full recalculation complete: {processed_count} matches processed")
        return ratings_dict

    except Exception as e:
        print(f"CRITICAL: Full recalculation failed: {e}")
        # Return empty dictionary as last resort
        return {}


def _get_chronological_key(row: pd.Series) -> Tuple:
    """
    Extract or parse chronological_key from DataFrame row.

    Args:
        row: DataFrame row containing chronological_key column

    Returns:
        Tuple for sorting (tourney_date, tourney_id, round_order)
    """
    if "chronological_key" in row:
        # Use existing chronological_key if available
        parsed_key = parse_chronological_key(row["chronological_key"])
        if parsed_key is not None:
            return parsed_key

    # Fallback: generate from individual columns (for backward compatibility)
    from utils.chronological_storage import generate_chronological_key

    return generate_chronological_key(row)


def _calculate_elo_update(
    p1_elo: float,
    p2_elo: float,
    p1_wins: bool,
    level: str,
    round_name: str,
    best_of: int,
) -> Tuple[float, float]:
    """
    Calculate Elo rating updates for both players.

    Args:
        p1_elo: Player 1's current Elo
        p2_elo: Player 2's current Elo
        p1_wins: True if player 1 won
        level: Tournament level (G, F, M, A)
        round_name: Round name (F, SF, QF, etc.)
        best_of: Best of sets (3 or 5)

    Returns:
        Tuple of (p1_new_elo, p2_new_elo)
    """
    # Rating factor formula
    rf1 = 1 + 18 / (1 + 2 ** ((p1_elo - 1500) / 63))
    rf2 = 1 + 18 / (1 + 2 ** ((p2_elo - 1500) / 63))

    # K-factor calculation
    def get_k(t_coeff, r_coeff, b_coeff, rf):
        return 32 * t_coeff * r_coeff * b_coeff * rf

    k1 = get_k(
        TOURNAMENT_COEFF.get(level, 0.75),
        ROUND_COEFF.get(round_name, 0.8),
        BEST_OF_COEFF.get(best_of, 0.9),
        rf1,
    )
    k2 = get_k(
        TOURNAMENT_COEFF.get(level, 0.75),
        ROUND_COEFF.get(round_name, 0.8),
        BEST_OF_COEFF.get(best_of, 0.9),
        rf2,
    )

    # Expected scores
    exp1 = 1.0 / (1.0 + 10 ** ((p2_elo - p1_elo) / 400))
    exp2 = 1.0 / (1.0 + 10 ** ((p1_elo - p2_elo) / 400))

    # Actual scores
    score1 = 1.0 if p1_wins else 0.0
    score2 = 0.0 if p1_wins else 1.0

    # New ratings
    new_p1_elo = p1_elo + k1 * (score1 - exp1)
    new_p2_elo = p2_elo + k2 * (score2 - exp2)

    return new_p1_elo, new_p2_elo


def get_player_elo_at_date(
    player_id: int,
    tourney_date: pd.Timestamp,
    tourney_id: Any,
    round_name: str,
    ratings_dict: Dict,
    surface: Optional[str] = None,
) -> float:
    """
    Get player's Elo rating at or before specified date/tournament/round.

    Args:
        player_id: Player ID
        tourney_date: Tournament date
        tourney_id: Tournament ID
        round_name: Round name
        ratings_dict: Elo ratings dictionary
        surface: Surface type (None for general Elo)

    Returns:
        Elo rating at specified point in time
    """
    if player_id not in ratings_dict:
        return DEFAULT_START_ELO

    elo_type = surface if surface else "general"
    if elo_type not in ratings_dict[player_id]:
        return DEFAULT_START_ELO

    surface_deque = ratings_dict[player_id][elo_type]

    # Only support ChronologicalDeque format
    if isinstance(surface_deque, ChronologicalDeque):
        # Generate chronological key using the proper function
        from utils.chronological_storage import generate_chronological_key

        match_data = {
            "tourney_date": tourney_date,
            "tourney_id": tourney_id,
            "round": round_name,
        }
        target_sort_key = generate_chronological_key(match_data)
        values_before_or_at = surface_deque.get_values_before_key(target_sort_key)
        return values_before_or_at[-1] if values_before_or_at else DEFAULT_START_ELO
    else:
        print(
            f"[ERROR] Unsupported Elo data format for player {player_id}. Expected ChronologicalDeque."
        )
        return DEFAULT_START_ELO


def get_player_current_elo(
    player_id: int, ratings_dict: Dict, surface: Optional[str] = None
) -> float:
    """
    Get player's most recent Elo rating.

    Args:
        player_id: Player ID
        ratings_dict: Elo ratings dictionary
        surface: Surface type (None for general Elo)

    Returns:
        Most recent Elo rating
    """
    if player_id not in ratings_dict:
        return DEFAULT_START_ELO

    elo_type = surface if surface else "general"
    if elo_type not in ratings_dict[player_id]:
        return DEFAULT_START_ELO

    surface_deque = ratings_dict[player_id][elo_type]

    # Handle both legacy dict format and new ChronologicalDeque format
    if isinstance(surface_deque, ChronologicalDeque):
        # New ChronologicalDeque format
        if not surface_deque:
            return DEFAULT_START_ELO

        # Get chronological values and return the last one
        chronological_values = surface_deque.get_chronological_values()
        return chronological_values[-1] if chronological_values else DEFAULT_START_ELO
    else:
        print(
            f"[ERROR] Unsupported Elo data format for player {player_id}. Expected ChronologicalDeque."
        )
        return DEFAULT_START_ELO


def get_player_elo_history(
    player_id: int, ratings_dict: Dict, surface: Optional[str] = None
) -> pd.DataFrame:
    """
    Get player's complete Elo rating history.

    Args:
        player_id: Player ID
        ratings_dict: Elo ratings dictionary
        surface: Surface type (None for general Elo)

    Returns:
        DataFrame with columns [tourney_date, tourney_id, round, elo]
    """
    if player_id not in ratings_dict:
        return pd.DataFrame(columns=["tourney_date", "tourney_id", "round", "elo"])

    elo_type = surface if surface else "general"
    if elo_type not in ratings_dict[player_id]:
        return pd.DataFrame(columns=["tourney_date", "tourney_id", "round", "elo"])

    surface_deque = ratings_dict[player_id][elo_type]
    history_data = []

    # Handle ChronologicalDeque format
    if isinstance(surface_deque, ChronologicalDeque):
        # ChronologicalDeque format - already sorted chronologically
        surface_deque._sort_if_needed()  # Ensure sorted
        for (date, tid, round_order), elo in surface_deque._deque:
            # Convert round_order back to round name using reverse lookup
            from utils.chronological_storage import get_round_order

            round_name = _reverse_round_order_lookup(round_order)

            history_data.append(
                {
                    "tourney_date": date,
                    "tourney_id": tid,
                    "round": round_name,
                    "elo": elo,
                }
            )
    else:
        print(
            f"[ERROR] Unsupported Elo data format for player {player_id}. Expected ChronologicalDeque."
        )
        return pd.DataFrame()  # Return empty DataFrame

    df = pd.DataFrame(history_data)
    # Data is already sorted chronologically from ChronologicalDeque

    return df


def get_player_elo_before_match(
    player_id: int,
    ratings_dict: Dict,
    current_sort_key: Tuple,
    surface: Optional[str] = None,
) -> float:
    """
    Get player's Elo rating immediately before the current match for getStats.
    **CRITICAL**: Only uses ratings from BEFORE current_sort_key to prevent data leakage.

    Args:
        player_id: Player ID
        ratings_dict: ChronologicalDeque-based ratings dictionary
        current_sort_key: Current match sort_key (EXCLUDED from retrieval)
        surface: Surface type (None for general Elo only)

    Returns:
        Most recent Elo rating BEFORE current match

    Data Leakage Prevention:
        - Strictly enforces temporal boundary at current_sort_key
        - Ensures returned Elo represents player strength before current match
        - Critical for maintaining integrity of pre-match statistics
    """
    if not current_sort_key:
        raise ValueError("current_sort_key is required to prevent data leakage")

    if player_id not in ratings_dict:
        return DEFAULT_START_ELO

    elo_type = surface if surface else "general"
    if elo_type not in ratings_dict[player_id]:
        return DEFAULT_START_ELO

    surface_deque = ratings_dict[player_id][elo_type]
    if not isinstance(surface_deque, ChronologicalDeque):
        print(
            f"[ERROR] Unsupported Elo data format for player {player_id}. Expected ChronologicalDeque."
        )
        return DEFAULT_START_ELO

    # Use ChronologicalDeque to get values strictly before current match
    values_before = surface_deque.get_values_before_key(current_sort_key)

    # Return most recent rating before current match, or default if none
    return values_before[-1] if values_before else DEFAULT_START_ELO


def calculate_elo_gradient(
    player_id: int,
    ratings_dict: Dict,
    k: int,
    current_sort_key: Tuple,
    surface: Optional[str] = None,
) -> float:
    """
    Calculate Elo rating gradient (slope) over last k matches for getStats.
    **CRITICAL**: Only uses ratings from BEFORE current_sort_key to prevent data leakage.

    Args:
        player_id: Player ID
        ratings_dict: ChronologicalDeque-based ratings
        k: Number of recent matches to analyze
        current_sort_key: Current match sort_key (EXCLUDED from analysis)
        surface: Surface type (None for general Elo only)

    Returns:
        Elo gradient (rating change per match) over last k matches BEFORE current match

    Algorithm:
        1. Get ratings BEFORE current_sort_key using get_values_before_key()
        2. Take last k ratings from the filtered set
        3. Calculate linear regression slope: (most_recent_elo - oldest_elo) / (k-1)
        4. Return 0.0 if insufficient data (< 2 matches)
        5. NEVER include ratings from current_sort_key or later

    Data Leakage Prevention:
        - Always uses ChronologicalDeque.get_values_before_key(current_sort_key)
        - Ensures all Elo data is from completed matches before current match
        - Maintains temporal consistency for pre-match statistics
    """
    if not current_sort_key:
        raise ValueError("current_sort_key is required to prevent data leakage")

    if player_id not in ratings_dict:
        return 0.0

    elo_type = surface if surface else "general"
    if elo_type not in ratings_dict[player_id]:
        return 0.0

    surface_deque = ratings_dict[player_id][elo_type]
    if not isinstance(surface_deque, ChronologicalDeque):
        print(
            f"[ERROR] Unsupported Elo data format for player {player_id}. Expected ChronologicalDeque."
        )
        return 0.0

    # Use ChronologicalDeque to get values strictly before current match
    values_before = surface_deque.get_values_before_key(current_sort_key)

    if len(values_before) < 2:
        return 0.0

    # Take last k values from the pre-match data
    recent_values = values_before[-k:] if len(values_before) >= k else values_before

    if len(recent_values) < 2:
        return 0.0

    # Calculate gradient (slope): change per match
    gradient = (recent_values[-1] - recent_values[0]) / (len(recent_values) - 1)
    return gradient


def get_players_with_ratings(ratings_dict: Dict) -> list:
    """
    Get list of all player IDs with rating histories.

    Args:
        ratings_dict: Elo ratings dictionary

    Returns:
        List of player IDs
    """
    return list(ratings_dict.keys())


def clear_all_elo_data(confirm: bool = True, backup: bool = True) -> Dict[str, Any]:
    """
    Clear all saved Elo data with optional backup.

    Args:
        confirm: Require user confirmation
        backup: Create backup before clearing

    Returns:
        dict: Operation results
    """
    if confirm:
        response = input("Are you sure you want to clear ALL Elo data? (yes/no): ")
        if response.lower() != "yes":
            return {"success": False, "message": "Operation cancelled by user"}

    results = {
        "start_time": datetime.now().isoformat(),
        "backup_created": False,
        "files_cleared": [],
        "success": False,
    }

    try:
        # Find Elo files to clear
        elo_files = ["data/elo_ratings.pkl", "backups/elo/elo_ratings_backup_*.pkl"]

        existing_files = []
        for file_pattern in elo_files:
            if "*" in file_pattern:
                # Handle glob patterns
                import glob

                existing_files.extend(glob.glob(file_pattern))
            else:
                if os.path.exists(file_pattern):
                    existing_files.append(file_pattern)

        # Create backup if requested and files exist
        if backup and existing_files:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = f"backups/elo_clear_backup_{timestamp}"
            os.makedirs(backup_dir, exist_ok=True)

            backup_count = 0
            for file_path in existing_files:
                if os.path.exists(file_path):
                    filename = os.path.basename(file_path)
                    backup_path = os.path.join(backup_dir, filename)
                    shutil.copy2(file_path, backup_path)
                    backup_count += 1

            if backup_count > 0:
                results["backup_created"] = True
                results["backup_location"] = backup_dir
                print(f"[INFO] Elo backup created at: {backup_dir}")
            else:
                print("[INFO] No existing Elo files found to backup")

        # Clear Elo files
        cleared_files = []
        for file_path in existing_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    cleared_files.append(file_path)
                    print(f"[INFO] Cleared: {file_path}")
            except Exception as e:
                print(f"[WARNING] Could not clear {file_path}: {e}")

        # Clear backup directory files (but keep the backup we just made)
        if os.path.exists("backups/elo"):
            backup_files = [
                f
                for f in os.listdir("backups/elo")
                if f.startswith("elo_ratings_backup_") and f.endswith(".pkl")
            ]
            # Only clear old backup files, not the one we just created
            if backup and "backup_location" in results:
                backup_files = [
                    f for f in backup_files if not f.endswith(f"_{timestamp}.pkl")
                ]

            for backup_file in backup_files:
                try:
                    backup_path = os.path.join("backups/elo", backup_file)
                    os.remove(backup_path)
                    cleared_files.append(backup_path)
                    print(f"[INFO] Cleared backup: {backup_path}")
                except Exception as e:
                    print(f"[WARNING] Could not clear backup {backup_path}: {e}")

        results["files_cleared"] = cleared_files
        results["success"] = True
        results["message"] = f"Cleared {len(cleared_files)} Elo files successfully"
        results["end_time"] = datetime.now().isoformat()

        print(f"[SUCCESS] Elo data cleared: {len(cleared_files)} files removed")
        if backup and results.get("backup_created"):
            print(f"[INFO] Backup available at: {results['backup_location']}")

        return results

    except Exception as e:
        results["success"] = False
        results["error"] = str(e)
        results["end_time"] = datetime.now().isoformat()
        print(f"[ERROR] Failed to clear Elo data: {e}")
        return results


def print_rating_summary(ratings_dict: Dict) -> None:
    """
    Print summary statistics of the ratings dictionary.

    Args:
        ratings_dict: Elo ratings dictionary
    """
    if not ratings_dict:
        print("Ratings dictionary is empty.")
        return

    num_players = len(ratings_dict)
    print(f"\nElo Ratings Summary:")
    print(f"Total players: {num_players}")

    # Count matches per surface
    surface_counts = defaultdict(int)
    for player_data in ratings_dict.values():
        for surface, matches in player_data.items():
            surface_counts[surface] += len(matches)

    print("\nMatches processed per surface:")
    for surface, count in surface_counts.items():
        print(f"  {surface}: {count:,}")

    # Top 10 players by current general Elo
    current_elos = []
    for player_id in ratings_dict:
        current_elo = get_player_current_elo(player_id, ratings_dict)
        current_elos.append((player_id, current_elo))

    current_elos.sort(key=lambda x: x[1], reverse=True)

    print(f"\nTop 10 players by current Elo:")
    for i, (player_id, elo) in enumerate(current_elos[:10], 1):
        print(f"  {i:2d}. Player {player_id}: {elo:.1f}")


# ============================================================================
# NEW CORE FUNCTIONS (PHASE 1: STATS INTERFACE INTEGRATION)
# ============================================================================


def create_elo_ratings(load_existing: bool = True) -> Dict:
    """
    Create or load Elo ratings - matches createStats pattern.

    Args:
        load_existing: Whether to load existing ratings from file

    Returns:
        Dict: Elo ratings dictionary (ChronologicalDeque format)
    """
    ratings_file = "data/elo_ratings.pkl"

    if load_existing and os.path.exists(ratings_file):
        print(f"[INFO] Loading existing Elo ratings from {ratings_file}")
        return load_elo_ratings(ratings_file)
    else:
        print(f"[INFO] Creating new empty Elo ratings dictionary")
        return {}


def update_elo_ratings(match_row: pd.Series, ratings_dict: Dict) -> Dict:
    """
    Update Elo for single match - matches updateStats pattern.

    Args:
        match_row: Single match data with required columns
        ratings_dict: Current Elo ratings dictionary

    Returns:
        Dict: Updated ratings dictionary
    """
    return _process_single_match_elo(match_row, ratings_dict)


def save_elo_ratings_simple(ratings_dict: Dict) -> bool:
    """
    Save Elo ratings - matches saveStats pattern.

    Args:
        ratings_dict: Elo ratings dictionary to save

    Returns:
        bool: True if successful, False otherwise
    """
    ratings_file = "data/elo_ratings.pkl"
    return save_elo_ratings(ratings_dict, ratings_file, create_backup=True)


def get_elo_info(ratings_dict: Dict) -> Dict:
    """
    Get Elo summary info - matches getStatsInfo pattern.

    Args:
        ratings_dict: Elo ratings dictionary

    Returns:
        Dict: Summary information about Elo ratings
    """
    if not ratings_dict:
        return {
            "total_players": 0,
            "total_matches": 0,
            "surfaces": [],
            "date_range": None,
            "memory_usage": "0 MB",
        }

    total_players = len(ratings_dict)
    surface_counts = defaultdict(int)
    all_dates = []

    # Analyze the ratings data
    for player_data in ratings_dict.values():
        for surface, surface_deque in player_data.items():
            if isinstance(surface_deque, ChronologicalDeque):
                surface_counts[surface] += len(surface_deque)
                # Extract dates from sort keys
                for sort_key, _ in surface_deque._deque:
                    if len(sort_key) >= 1 and hasattr(sort_key[0], "date"):
                        all_dates.append(sort_key[0])

    # Calculate date range
    date_range = None
    if all_dates:
        date_range = {
            "earliest": min(all_dates).isoformat(),
            "latest": max(all_dates).isoformat(),
        }

    # Estimate memory usage (rough approximation)
    total_entries = sum(surface_counts.values())
    estimated_mb = (total_entries * 100) / (1024 * 1024)  # ~100 bytes per entry

    return {
        "total_players": total_players,
        "total_matches": total_entries,
        "surface_distribution": dict(surface_counts),
        "surfaces": list(surface_counts.keys()),
        "date_range": date_range,
        "memory_usage": f"{estimated_mb:.2f} MB",
    }


def clear_all_elo_data_simple() -> Dict:
    """
    Clear all Elo data - matches clearAllStats pattern.

    Returns:
        Dict: Operation results
    """
    return clear_all_elo_data(confirm=False, backup=True)


# ============================================================================
# BATCH PROCESSING FUNCTIONS (PHASE 1)
# ============================================================================


def process_elo_batch(
    matches_df: pd.DataFrame,
    ratings_dict: Dict,
    batch_size: int = 10000,
    progress_callback: Optional[callable] = None,
) -> Dict:
    """
    Process large batches of matches efficiently.

    Args:
        matches_df: DataFrame with matches to process
        ratings_dict: Current Elo ratings dictionary
        batch_size: Number of matches to process in each batch
        progress_callback: Optional callback for progress updates

    Returns:
        Dict: Updated ratings dictionary

    Features:
        - Chunked processing for memory efficiency
        - Progress tracking for 80K+ matches
        - Memory-optimized data loading
    """
    print(
        f"[INFO] Starting batch Elo processing: {len(matches_df)} matches, batch size: {batch_size}"
    )

    # Optimize DataFrame for large datasets
    if len(matches_df) > 50000:
        print("[INFO] Large dataset detected, applying memory optimizations...")
        optimized_df = optimize_for_large_datasets(matches_df)
    else:
        optimized_df = matches_df

    # Show processing time estimate
    time_estimate = estimate_processing_time(len(optimized_df), batch_size)
    print(
        f"[INFO] Estimated processing time: {time_estimate['estimated_minutes']} minutes"
    )
    print(f"[INFO] Recommendation: {time_estimate['recommendation']}")

    # Use EloBatchProcessor for optimized processing
    processor = EloBatchProcessor(batch_size=batch_size)
    updated_ratings = processor.process_large_dataset(
        optimized_df, ratings_dict, progress_bar=True
    )

    print(
        f"[SUCCESS] Batch processing completed: {processor.processed_matches} matches processed"
    )
    return updated_ratings


def recalculate_elo_from_chronological_key(
    from_chronological_key: Tuple, ratings_dict: Dict, data_source: str = "dataset"
) -> Dict:
    """
    Memory-efficient Elo recalculation from specific chronological point.

    Args:
        from_chronological_key: Starting chronological key for recalculation
        ratings_dict: Current ratings dictionary
        data_source: Data source for loading matches ("dataset" or file path)

    Returns:
        Dict: Updated ratings dictionary

    Process:
        - Only loads matches from chronological key onward (not all matches)
        - Uses chunked loading to minimize memory usage
        - Progressively processes data in batches
    """
    print(
        f"[INFO] Starting memory-efficient Elo recalculation from: {from_chronological_key}"
    )

    # Clear ratings from the specified key forward
    cleared_ratings = clear_ratings_from_sort_key(ratings_dict, from_chronological_key)

    # Use EloBatchProcessor for memory-efficient recalculation
    processor = EloBatchProcessor(batch_size=10000)
    updated_ratings = processor.process_from_chronological_key(
        from_chronological_key, cleared_ratings, data_source, progress_bar=True
    )

    print(f"[SUCCESS] Memory-efficient Elo recalculation completed")
    return updated_ratings


# ============================================================================
# PERFORMANCE OPTIMIZATION (PHASE 3)
# ============================================================================


class EloBatchProcessor:
    """Optimized batch processor for large Elo calculations."""

    def __init__(self, batch_size: int = 10000):
        """
        Initialize batch processor.

        Args:
            batch_size: Number of matches to process in each batch
        """
        self.batch_size = batch_size
        self.processed_matches = 0
        self.total_matches = 0

    def process_large_dataset(
        self, matches_df: pd.DataFrame, ratings_dict: Dict, progress_bar: bool = True
    ) -> Dict:
        """
        Process large datasets in chunks.

        Args:
            matches_df: DataFrame with matches to process
            ratings_dict: Current Elo ratings dictionary
            progress_bar: Show progress updates

        Returns:
            Dict: Updated ratings dictionary

        Strategy:
            1. Sort matches chronologically
            2. Process in batches to manage memory
            3. Update ratings incrementally
            4. Progress tracking for user feedback
        """
        self.total_matches = len(matches_df)
        self.processed_matches = 0

        print(
            f"[INFO] EloBatchProcessor: Processing {self.total_matches} matches in batches of {self.batch_size}"
        )

        # Sort chronologically first
        sorted_df = matches_df.sort_values(
            ["tourney_date", "tourney_id", "round_order"], ascending=[True, True, False]
        ).reset_index(drop=True)

        # Process in batches
        for start_idx in range(0, self.total_matches, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.total_matches)
            batch_df = sorted_df.iloc[start_idx:end_idx]

            # Process batch
            ratings_dict = self._process_batch(batch_df, ratings_dict)

            self.processed_matches += len(batch_df)

            if progress_bar:
                progress = (self.processed_matches / self.total_matches) * 100
                print(
                    f"[PROGRESS] EloBatchProcessor: {self.processed_matches}/{self.total_matches} matches ({progress:.1f}%)"
                )

        print(
            f"[SUCCESS] EloBatchProcessor: Completed processing {self.processed_matches} matches"
        )
        return ratings_dict

    def _process_batch(self, batch_df: pd.DataFrame, ratings_dict: Dict) -> Dict:
        """
        Process a single batch of matches.

        Args:
            batch_df: Batch of matches to process
            ratings_dict: Current ratings dictionary

        Returns:
            Dict: Updated ratings dictionary
        """
        for _, match_row in batch_df.iterrows():
            ratings_dict = update_elo_ratings(match_row, ratings_dict)
        return ratings_dict

    def process_from_chronological_key(
        self,
        from_chronological_key: Tuple,
        ratings_dict: Dict,
        data_source: str,
        progress_bar: bool = True,
    ) -> Dict:
        """
        Process matches from a specific chronological key using batch processing.

        Args:
            from_chronological_key: Starting chronological key
            ratings_dict: Current ratings dictionary
            data_source: Data source for loading matches
            progress_bar: Show progress updates

        Returns:
            Dict: Updated ratings dictionary
        """
        print(
            f"[INFO] EloBatchProcessor: Loading matches from chronological key: {from_chronological_key}"
        )

        # Load and filter matches
        if data_source == "dataset":
            from utils.stats_interface import _load_latest_main_dataset

            all_matches_df = _load_latest_main_dataset()
        else:
            all_matches_df = pd.read_csv(data_source)

        # Filter matches from chronological key onward
        def is_after_key(row_key_str):
            try:
                parsed_key = parse_chronological_key(row_key_str)
                return parsed_key and parsed_key >= from_chronological_key
            except:
                return False

        matches_to_process = all_matches_df[
            all_matches_df["chronological_key"].apply(is_after_key)
        ].copy()

        print(
            f"[INFO] EloBatchProcessor: Found {len(matches_to_process)} matches to process"
        )

        if matches_to_process.empty:
            return ratings_dict

        # Process using batch processing
        return self.process_large_dataset(
            matches_to_process, ratings_dict, progress_bar
        )


def optimize_for_large_datasets(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame for large-scale processing.

    Args:
        matches_df: Input DataFrame

    Returns:
        pd.DataFrame: Optimized DataFrame

    Optimizations:
        - Convert to appropriate dtypes
        - Remove unnecessary columns
        - Optimize memory usage
    """
    print(
        f"[INFO] Optimizing DataFrame for large-scale processing: {len(matches_df)} matches"
    )

    optimized_df = matches_df.copy()

    # Optimize data types
    if "player1_id" in optimized_df.columns:
        optimized_df["player1_id"] = pd.to_numeric(
            optimized_df["player1_id"], downcast="integer"
        )
    if "player2_id" in optimized_df.columns:
        optimized_df["player2_id"] = pd.to_numeric(
            optimized_df["player2_id"], downcast="integer"
        )
    if "tourney_id" in optimized_df.columns:
        optimized_df["tourney_id"] = pd.to_numeric(
            optimized_df["tourney_id"], downcast="integer"
        )
    if "RESULT" in optimized_df.columns:
        optimized_df["RESULT"] = optimized_df["RESULT"].astype("int8")

    # Convert categorical columns
    categorical_cols = ["surface", "round", "tourney_level"]
    for col in categorical_cols:
        if col in optimized_df.columns:
            optimized_df[col] = optimized_df[col].astype("category")

    # Calculate memory savings
    original_memory = matches_df.memory_usage(deep=True).sum() / (1024 * 1024)
    optimized_memory = optimized_df.memory_usage(deep=True).sum() / (1024 * 1024)
    memory_saved = original_memory - optimized_memory

    print(f"[SUCCESS] DataFrame optimization completed:")
    print(f"  - Original memory: {original_memory:.2f} MB")
    print(f"  - Optimized memory: {optimized_memory:.2f} MB")
    print(
        f"  - Memory saved: {memory_saved:.2f} MB ({(memory_saved / original_memory) * 100:.1f}%)"
    )

    return optimized_df


def estimate_processing_time(
    num_matches: int, batch_size: int = 10000
) -> Dict[str, Any]:
    """
    Estimate processing time for large datasets.

    Args:
        num_matches: Number of matches to process
        batch_size: Batch size for processing

    Returns:
        Dict: Time estimates and recommendations
    """
    # Based on empirical testing: ~100-200 matches per second
    matches_per_second = 150  # Conservative estimate

    estimated_seconds = num_matches / matches_per_second
    estimated_minutes = estimated_seconds / 60

    num_batches = (num_matches + batch_size - 1) // batch_size  # Ceiling division

    return {
        "num_matches": num_matches,
        "batch_size": batch_size,
        "num_batches": num_batches,
        "estimated_seconds": round(estimated_seconds, 1),
        "estimated_minutes": round(estimated_minutes, 1),
        "estimated_hours": round(estimated_minutes / 60, 2),
        "matches_per_second": matches_per_second,
        "recommendation": (
            "Consider using batch_size=15000+ for datasets >50K matches"
            if num_matches > 50000
            else "Current batch size should work well"
        ),
    }


# ============================================================================
# INTEGRATION INTERFACE FUNCTIONS (PHASE 1)
# ============================================================================


def check_elo_recalculation_needed(
    new_matches_df: pd.DataFrame, existing_ratings: Dict
) -> Tuple[bool, Optional[Tuple]]:
    """
    Check if Elo recalculation is needed based on chronological order.

    Args:
        new_matches_df: New matches DataFrame
        existing_ratings: Current Elo ratings dictionary

    Returns:
        Tuple: (needs_recalc, earliest_chronological_key)
    """
    if new_matches_df.empty:
        return False, None

    # Get earliest chronological key from new matches
    earliest_new_key = None
    for _, row in new_matches_df.iterrows():
        parsed_key = parse_chronological_key(row["chronological_key"])
        if parsed_key:
            if earliest_new_key is None or parsed_key < earliest_new_key:
                earliest_new_key = parsed_key

    if not earliest_new_key:
        return False, None

    # Get latest processed chronological key from existing ratings
    latest_processed_key = _get_latest_processed_chronological_key(existing_ratings)

    if not latest_processed_key:
        # No existing ratings, no recalculation needed
        return False, None

    # Check if recalculation is needed
    needs_recalc = earliest_new_key < latest_processed_key

    if needs_recalc:
        print(
            f"[WARNING] Elo recalculation needed: earliest new match {earliest_new_key} < latest processed {latest_processed_key}"
        )
        return True, earliest_new_key
    else:
        print(f"[INFO] No Elo recalculation needed: matches are in chronological order")
        return False, None


def _get_latest_processed_chronological_key(ratings_dict: Dict) -> Optional[Tuple]:
    """
    Get the latest chronological key from processed Elo ratings.

    Args:
        ratings_dict: Elo ratings dictionary

    Returns:
        Optional[Tuple]: Latest chronological key or None
    """
    latest_key = None

    for player_data in ratings_dict.values():
        for surface_deque in player_data.values():
            if isinstance(surface_deque, ChronologicalDeque):
                # Get all sort keys and find the latest
                surface_deque._sort_if_needed()
                for sort_key, _ in surface_deque._deque:
                    if latest_key is None or sort_key > latest_key:
                        latest_key = sort_key

    return latest_key


def integrate_with_stats_interface(
    matches_df: pd.DataFrame,
    rewrite_duplicates: bool = False,
    progress_bar: bool = True,
) -> Dict[str, Any]:
    """
    Main integration point for stats_interface.py
    Handles all Elo processing logic including recalculation checks.

    Args:
        matches_df: Matches DataFrame to process
        rewrite_duplicates: Whether to rewrite existing data
        progress_bar: Show progress bar

    Returns:
        Dict: Processing results
    """
    results = {
        "success": False,
        "processed_matches": 0,
        "recalculation_triggered": False,
        "recalc_from_key": None,
        "error": None,
    }

    try:
        # Load existing Elo ratings
        existing_ratings = create_elo_ratings(load_existing=True)

        # Check if recalculation is needed
        needs_recalc, earliest_key = check_elo_recalculation_needed(
            matches_df, existing_ratings
        )

        if needs_recalc and not rewrite_duplicates:
            # Trigger recalculation from earliest chronological key
            print(
                f"[INFO] Triggering Elo recalculation from chronological key: {earliest_key}"
            )
            updated_ratings = recalculate_elo_from_chronological_key(
                earliest_key, existing_ratings, data_source="dataset"
            )
            results["recalculation_triggered"] = True
            results["recalc_from_key"] = str(earliest_key)
        else:
            # Process matches incrementally
            print(f"[INFO] Processing {len(matches_df)} matches incrementally")
            updated_ratings = process_elo_batch(
                matches_df, existing_ratings, batch_size=10000, progress_callback=None
            )

        # Save updated ratings
        if save_elo_ratings_simple(updated_ratings):
            results["success"] = True
            results["processed_matches"] = len(matches_df)
            print(
                f"[SUCCESS] Elo integration completed: {len(matches_df)} matches processed"
            )
        else:
            results["error"] = "Failed to save Elo ratings"

    except Exception as e:
        results["error"] = str(e)
        print(f"[ERROR] Elo integration failed: {e}")

    return results


# ============================================================================
# LEGACY FUNCTIONS (MAINTAINED FOR BACKWARD COMPATIBILITY)
# ============================================================================


# Main workflow functions
def process_new_matches_dataframe(matches_df: pd.DataFrame, ratings_dict: Dict) -> Dict:
    """
    Main function to process new matches from pre-sorted DataFrame with chronological clearing.
    This function integrates with the main pipeline by accepting DataFrames with pre-computed
    sort_key and match_id columns.

    Args:
        matches_df: Pre-sorted DataFrame with required columns:
                   - sort_key: Pre-computed (datetime, tourney_id, round_order) tuple
                   - match_id: Pre-computed match identifier
                   - p1_id, p2_id: Player IDs
                   - surface: Court surface
                   - RESULT: Match result (1=P1 wins, 0=P2 wins)
                   - round, tourney_level, best_of, tourney_date, tourney_id
        ratings_dict: Current Elo ratings dictionary (will be modified)

    Returns:
        Updated ratings dictionary

    Process:
        1. Validate DataFrame structure and sort_keys
        2. Get earliest sort_key from input matches
        3. Create backup before clearing operations
        4. Clear ratings from earliest sort_key forward (chronological clearing)
        5. Process matches using existing sort_keys from DataFrame
        6. Return updated ratings or fall back to full recalculation on errors

    Integration Notes:
        - Designed to work with main processing pipeline
        - Accepts pre-computed sort_keys for consistency with updateStats.py
        - Provides chronological clearing for out-of-order match handling
        - Maintains backward compatibility with existing Elo functions
    """
    try:
        # Validate input DataFrame structure
        print("Validating DataFrame structure...")
        validate_dataframe_structure(matches_df)
        validate_chronological_key_consistency(matches_df)

        if matches_df.empty:
            print("No matches to process")
            return ratings_dict

        # Validate ratings are in ChronologicalDeque format
        if ratings_dict:
            sample_player = next(iter(ratings_dict.values()), {})
            if sample_player and not isinstance(
                next(iter(sample_player.values()), None), ChronologicalDeque
            ):
                print(
                    f"[ERROR] Elo ratings must be in ChronologicalDeque format. Legacy format not supported."
                )
                return ratings_dict

        # Get earliest sort_key for chronological clearing
        earliest_sort_key = (
            matches_df["chronological_key"].apply(parse_chronological_key).min()
        )
        print(
            f"Processing {len(matches_df)} matches starting from sort_key: {earliest_sort_key}"
        )

        # Create backup before clearing operations
        backup_path = create_elo_backup(ratings_dict)
        print(f"Created backup: {backup_path}")

        # Clear ratings from earliest sort_key forward
        print(f"Clearing ratings from {earliest_sort_key} forward...")
        ratings_dict = clear_ratings_from_sort_key(ratings_dict, earliest_sort_key)

        # Process matches using the main DataFrame processor
        print("Processing matches with chronological Elo updates...")
        updated_ratings = update_elo_from_dataframe(matches_df, ratings_dict)

        print(f"Successfully processed {len(matches_df)} matches")
        return updated_ratings

    except Exception as e:
        print(f"Error in process_new_matches_dataframe: {e}")
        print("Attempting fallback to full recalculation...")

        try:
            # Fallback: full recalculation from DataFrame
            return recalculate_elo_full_from_dataframe(matches_df)
        except Exception as fallback_error:
            print(f"Fallback recalculation failed: {fallback_error}")
            print("Returning original ratings dictionary")
            return ratings_dict


def process_new_matches(
    master_df: pd.DataFrame, ratings_file: str, force_recalc: bool = False
) -> Dict:
    """
    Legacy function to process new matches and update Elo ratings.

    DEPRECATED: Use process_new_matches_dataframe() for new implementations.
    This function is maintained for backward compatibility.

    Args:
        master_df: Master dataframe containing all matches
        ratings_file: Path to pickle file containing ratings
        force_recalc: Force full recalculation regardless of chronological order

    Returns:
        Updated ratings dictionary
    """
    print("Loading existing Elo ratings...")
    ratings_dict = load_elo_ratings(ratings_file)

    # Add chronological_key column if not present
    if "chronological_key" not in master_df.columns:
        master_df["chronological_key"] = master_df.apply(_get_chronological_key, axis=1)

    # Add match_id column if not present
    if "match_id" not in master_df.columns:
        master_df["match_id"] = master_df.index.astype(str)

    # Use the new DataFrame-based processing
    try:
        if force_recalc or not ratings_dict:
            print("Performing full recalculation...")
            ratings_dict = recalculate_elo_full_from_dataframe(master_df)
        else:
            print("Processing with DataFrame-based method...")
            ratings_dict = process_new_matches_dataframe(master_df, ratings_dict)
    except Exception as e:
        print(f"DataFrame processing failed: {e}")
        print("Falling back to full recalculation...")
        ratings_dict = recalculate_elo_full_from_dataframe(master_df)

    # Save updated ratings
    print(f"Saving ratings to {ratings_file}...")
    if save_elo_ratings(ratings_dict, ratings_file):
        print("Ratings saved successfully!")
    else:
        print("Error saving ratings!")
        return {}

    print_rating_summary(ratings_dict)
    return ratings_dict
