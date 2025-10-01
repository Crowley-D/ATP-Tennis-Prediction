#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Matches Dataset Pipeline

This module processes new match data through multiple stages to add it to the main
tennis matches dataset. The pipeline handles tournament ID unification, player ID
unification, duplicate detection, and chronological ordering.

Author: Tennis Prediction System
Created: 2025-01-15
Reference: Main_Matches_Dataset_Pipeline_Implementation_Plan.md
"""

import pandas as pd
import numpy as np
import json
import os
import shutil
import glob
import re
import math
import logging
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, Set, List
import warnings
from fuzzywuzzy import fuzz, process
from tqdm import tqdm


# ============================================================================
# CONFIGURATION CONSTANTS FOR STAGE 8: ODDS ENRICHMENT
# ============================================================================

# Odds data configuration
ODDS_DATASET_PATH = "data/Odds/"  # Directory with yearly Excel files (2001-2025)
FUZZY_MATCH_THRESHOLD = 90
DATE_BUFFER_BASE = 5
DATE_BUFFER_MULTIPLIER = 2
ATP_POINTS_TOLERANCE = 200
CHUNK_SIZE = 1000  # For memory management during batch processing
BATCH_REVIEW_SIZE = 10  # Number of uncertain matches to review at once
RESUME_CHECKPOINT_FILE = "databases/MatchesMain/odds_enrichment_checkpoint.json"

# New columns to add to main dataset
NEW_ENRICHMENT_COLUMNS = [
    "p1_max_odds",  # float: Maximum odds for player1
    "p2_max_odds",  # float: Maximum odds for player2
    "p1_odds",  # float: Pinnacle/Bet365 odds for player1
    "p2_odds",  # float: Pinnacle/Bet365 odds for player2
    "odds_source",  # str: 'pinnacle', 'bet365', or 'max_only'
    "enrichment_confidence",  # float: Match confidence score (0-100)
    "enrichment_method",  # str: 'auto_rank', 'auto_rank_round', 'auto_score', 'auto_name', 'manual', 'failed'
]

# Round normalization mapping from odds dataset to main dataset format
# Note: Numbered rounds (1st, 2nd, etc.) are handled dynamically in normalize_odds_round()
ROUND_MAPPING = {
    "Final": "F",
    "Finals": "F",
    "Semifinals": "SF",
    "Semi-finals": "SF",
    "Quarterfinals": "QF",
    "Quarter-finals": "QF",
    "Round of 128": "R128",
    "Round of 64": "R64",
    "Round of 32": "R32",
    "Round of 16": "R16",
}


# ============================================================================
# STAGE 1: DATA VALIDATION AND PREPARATION
# ============================================================================


def validate_input_dataframe(df: pd.DataFrame) -> None:
    """
    Validate input DataFrame has all required columns and proper structure.

    Args:
        df (pd.DataFrame): Input DataFrame to validate

    Raises:
        ValueError: If required columns are missing or data is invalid
    """
    required_columns = [
        "tourney_name",
        "tourney_id",
        "round",
        "player1_name",
        "player1_id",
        "player2_name",
        "player2_id",
        "score",
        "tourney_date",
        "surface",
        "tourney_level",
        "best_of",
        "match_num",
        "source",
    ]

    # Check for missing columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Check if DataFrame is empty
    if df.empty:
        raise ValueError("Input DataFrame is empty")

    # Validate data types can be converted
    try:
        df["tourney_date"] = pd.to_datetime(df["tourney_date"])
    except:
        raise ValueError("tourney_date column cannot be converted to datetime")

    # Keep tourney_id as string - don't force to numeric
    # (Tournament IDs may have various formats: "580", "2024-580", etc.)
    try:
        df["tourney_id"] = df["tourney_id"].astype(str)
    except:
        raise ValueError("tourney_id column cannot be converted to string")

    player_columns = ["player1_id", "player2_id"]
    for col in player_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)

    print(f"[PASS] Input validation passed: {len(df)} rows, {len(df.columns)} columns")


def prepare_input_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare input DataFrame with all required transformations and randomization.

    Args:
        df (pd.DataFrame): Input DataFrame to prepare

    Returns:
        pd.DataFrame: Prepared DataFrame with additional computed columns and randomization
    """
    print("[INFO] Preparing input data...")

    # Create working copy
    prepared_df = df.copy()

    # Convert date column
    prepared_df["tourney_date"] = pd.to_datetime(prepared_df["tourney_date"])

    # Map source to numeric codes if needed
    if prepared_df["source"].dtype == "object":
        source_mapping = {"main_dataset": 0, "infosys_api": 1, "tennis_api": 2}
        # Handle unmapped sources by assigning them to a default category
        prepared_df["source"] = prepared_df["source"].map(source_mapping).fillna(0)
        print(f"[INFO] Mapped source values to numeric codes")

    # Calculate RESULT column from score (always returns 1 since player1 is always winner in input)
    prepared_df["RESULT"] = 1

    # Add advanced stats calculation
    print("[INFO] Calculating advanced stats...")
    prepared_df = calculate_advanced_stats(prepared_df)

    # Add score parsing for set-by-set scores
    print("[INFO] Parsing set-by-set scores...")
    prepared_df = parse_match_scores(prepared_df)

    # Randomize player1 and player2 columns (similar to cell 58 in 0.CleanData.ipynb)
    print("[INFO] Applying randomization to player columns...")
    mask = np.random.rand(len(prepared_df)) < 0.5

    # Identify player1 and player2 columns
    player1_cols = [
        col for col in prepared_df.columns if "player1" in col or "p1_" in col
    ]
    player2_cols = [
        col for col in prepared_df.columns if "player2" in col or "p2_" in col
    ]

    # Update RESULT column based on randomization (1 = player1 wins, 0 = player2 wins)
    prepared_df["RESULT"] = np.where(mask, 0, 1)

    # Swap player columns where mask is True
    if player1_cols and player2_cols:
        prepared_df.loc[mask, player1_cols], prepared_df.loc[mask, player2_cols] = (
            prepared_df.loc[mask, player2_cols].values,
            prepared_df.loc[mask, player1_cols].values,
        )
        swapped_count = mask.sum()
        print(
            f"[INFO] Randomized {swapped_count}/{len(prepared_df)} matches (swapped player1/player2)"
        )
    else:
        print("[WARNING] No player1/player2 columns found for randomization")

    # Ensure required numeric columns are properly typed
    numeric_columns = ["best_of", "draw_size", "match_num"]
    for col in numeric_columns:
        if col in prepared_df.columns:
            prepared_df[col] = pd.to_numeric(prepared_df[col], errors="coerce")

    # Handle missing draw_size if not present
    if "draw_size" not in prepared_df.columns:
        prepared_df["draw_size"] = np.nan

    print(
        f"[SUCCESS] Data preparation completed: {len(prepared_df)} rows prepared with randomization, advanced stats, and score parsing"
    )
    return prepared_df


def calculate_advanced_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate advanced tennis statistics based on the 0.CleanData.ipynb methodology.

    Args:
        df (pd.DataFrame): DataFrame with tennis match data

    Returns:
        pd.DataFrame: DataFrame with additional advanced statistics columns
    """
    print("[INFO] Calculating break point conversions and advanced metrics...")

    result_df = df.copy()

    # Break point conversions (bp converted)
    result_df["p1_bpconv"] = result_df["p2_bpFaced"] - result_df["p2_bpSaved"]
    result_df["p2_bpconv"] = result_df["p1_bpFaced"] - result_df["p1_bpSaved"]

    # Break points per return game (bp/rg)
    result_df["p1_bp/rg"] = result_df["p2_bpFaced"] / result_df["p2_SvGms"]
    result_df["p2_bp/rg"] = result_df["p1_bpFaced"] / result_df["p1_SvGms"]

    # Total points won (tpw) - service points + return points
    result_df["p1_tpw"] = (
        result_df["p1_1stWon"]
        + result_df["p1_2ndWon"]
        + (result_df["p2_svpt"] - (result_df["p2_1stWon"] + result_df["p2_2ndWon"]))
    )
    result_df["p2_tpw"] = (
        result_df["p2_1stWon"]
        + result_df["p2_2ndWon"]
        + (result_df["p1_svpt"] - (result_df["p1_1stWon"] + result_df["p1_2ndWon"]))
    )

    # Handle any division by zero or missing values
    result_df["p1_bp/rg"] = result_df["p1_bp/rg"].replace([np.inf, -np.inf], np.nan)
    result_df["p2_bp/rg"] = result_df["p2_bp/rg"].replace([np.inf, -np.inf], np.nan)

    print(f"[SUCCESS] Advanced stats calculated for {len(result_df)} rows")
    return result_df


def parse_match_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse match scores into set-by-set columns based on 0.CleanData.ipynb methodology.
    Creates p1_set1, p1_set2, etc. columns. For best_of < 5, empty sets filled with 0.
    Tiebreak scores in brackets (e.g., 7-6(3)) are ignored - only game scores extracted.

    Args:
        df (pd.DataFrame): DataFrame with 'score' column

    Returns:
        pd.DataFrame: DataFrame with additional set score columns
    """
    print("[INFO] Parsing match scores into set-by-set columns...")

    result_df = df.copy()

    # Initialize set columns with 0 for all 5 potential sets
    for i in range(1, 6):
        result_df[f"p1_set{i}"] = 0
        result_df[f"p2_set{i}"] = 0

    if "score" not in result_df.columns:
        print("[WARNING] 'score' column not found, setting all set scores to 0")
        return result_df

    # Regex pattern to extract set scores, ignoring tiebreak scores in brackets
    # Matches patterns like "6-4", "7-6(3)", etc. but only captures the games (6-4, 7-6)
    pattern = re.compile(r"(\d+)-(\d+)(?:\(\d+\))?")

    for idx, row in result_df.iterrows():
        score = str(row["score"])
        if pd.isna(score) or score == "nan":
            continue

        # Find all set scores in the match
        matches = pattern.findall(score)

        for set_num, (p1_games, p2_games) in enumerate(matches, 1):
            if set_num <= 5:  # Only process up to 5 sets
                result_df.at[idx, f"p1_set{set_num}"] = int(p1_games)
                result_df.at[idx, f"p2_set{set_num}"] = int(p2_games)

    # For matches with best_of < 5, ensure unused sets remain 0 (already initialized)
    # Calculate total games won per player (sw = sets won in games)
    set_cols_p1 = [f"p1_set{i}" for i in range(1, 6)]
    set_cols_p2 = [f"p2_set{i}" for i in range(1, 6)]

    result_df["p1_sw"] = result_df[set_cols_p1].sum(axis=1)
    result_df["p2_sw"] = result_df[set_cols_p2].sum(axis=1)

    print(f"[SUCCESS] Score parsing completed for {len(result_df)} rows")
    return result_df


# ============================================================================
# STAGE 2: TOURNAMENT ID PROCESSING
# ============================================================================


def process_tournament_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process tournament IDs using the tennis tournament matching system.

    Args:
        df (pd.DataFrame): DataFrame with tournament data

    Returns:
        pd.DataFrame: DataFrame with unified tournament IDs
    """
    print("[INFO] Processing tournament IDs...")

    # Initialize GUI BEFORE importing (this fixes the dialog visibility issue)
    os.environ["TOURNAMENT_MATCHING_GUI"] = "1"
    print("[INFO] GUI mode enabled - dialog windows will appear for confirmations")

    try:
        from tennis_matching.main import process_dataframe_programmatically

        # Prepare DataFrame for tournament matching
        tournament_df = df[
            ["source", "tourney_level", "tourney_id", "tourney_name"]
        ].copy()

        # Process tournament matching (keep original column names)
        unified_tournament_df = process_dataframe_programmatically(
            tournament_df, database_path="databases/tennis_tournaments.db"
        )

        # Update main DataFrame with unified tournament IDs
        result_df = df.copy()
        result_df["tourney_id"] = unified_tournament_df["tourney_id"]

        # Count how many tournaments were unified
        unique_tournaments = len(result_df["tourney_id"].unique())
        print(
            f"[SUCCESS] Tournament processing completed: {unique_tournaments} unique tournaments"
        )

        return result_df

    except Exception as e:
        print(f"[WARNING] Tournament processing failed: {e}")
        print("[INFO] Continuing with original tournament IDs...")
        return df


# ============================================================================
# STAGE 3: PLAYER ID PROCESSING
# ============================================================================


def process_player_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process player IDs using the player matching system with metadata enrichment.

    Args:
        df (pd.DataFrame): DataFrame with player data

    Returns:
        pd.DataFrame: DataFrame with unified player IDs
    """
    print("[INFO] Processing player IDs...")

    try:
        from player_matching.integration.dataframe_processor import (
            process_players_dataframe,
        )

        # Process player matching with metadata enrichment
        unified_player_df = process_players_dataframe(
            df,
            auto_resolve=True,  # Skip manual prompts for automated processing
            enrich_metadata=True,  # Fill empty metadata from database
            validate_metadata=True,  # Ensure data quality
            update_database_metadata=False,  # Don't update database during main pipeline
            database_path="databases/tennis_players.db",
        )

        # Update DataFrame with unified player IDs
        result_df = unified_player_df.copy()
        # The player processing system modifies player1_id and player2_id in place
        result_df["p1_id"] = unified_player_df[
            "player1_id"
        ]  # Now contains unified numeric IDs
        result_df["p2_id"] = unified_player_df[
            "player2_id"
        ]  # Now contains unified numeric IDs

        # Original IDs are already preserved in original_player1_id and original_player2_id

        # Count unified players
        unique_p1 = len(result_df["p1_id"].unique())
        unique_p2 = len(result_df["p2_id"].unique())
        print(
            f"[SUCCESS] Player processing completed: {unique_p1} unique player1s, {unique_p2} unique player2s"
        )

        return result_df

    except Exception as e:
        print(f"[WARNING] Player processing failed: {e}")
        print("[INFO] Continuing with original player IDs...")
        # Fallback: use original IDs
        result_df = df.copy()
        result_df["p1_id"] = df["player1_id"]
        result_df["p2_id"] = df["player2_id"]
        result_df["original_player1_id"] = df[
            "player1_id"
        ]  # Match the naming convention
        result_df["original_player2_id"] = df["player2_id"]
        return result_df


# ============================================================================
# STAGE 4: MATCH ID AND CHRONOLOGICAL KEY GENERATION
# ============================================================================


def generate_match_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate unique match IDs and chronological sort keys.

    Args:
        df (pd.DataFrame): DataFrame with processed tournament and player data

    Returns:
        pd.DataFrame: DataFrame with match_id and chronological_key columns
    """
    print("[INFO] Generating match identifiers...")

    try:
        from utils.match_tracking import generate_match_id, preprocess_rr_round_codes
        from utils.chronological_storage import generate_chronological_key

        result_df = df.copy()

        # Preprocess Round Robin matches before generating identifiers
        if "RR" in result_df["round"].values:
            print("[INFO] Preprocessing Round Robin matches...")
            result_df = preprocess_rr_round_codes(result_df)

        # Generate match IDs using unified player IDs (p1_id, p2_id should be numeric now)
        # Create a temporary dataframe with the correct column names for match_id generation
        match_id_df = result_df.copy()
        match_id_df["player1_id"] = result_df["p1_id"]  # Use unified numeric IDs
        match_id_df["player2_id"] = result_df["p2_id"]  # Use unified numeric IDs

        result_df["match_id"] = match_id_df.apply(generate_match_id, axis=1)

        # Generate chronological sort keys
        result_df["chronological_key"] = result_df.apply(
            generate_chronological_key, axis=1
        )

        # Extract round_order from chronological_key tuple (3rd component)
        def extract_round_order(chrono_key):
            if isinstance(chrono_key, tuple) and len(chrono_key) >= 3:
                return chrono_key[2]  # round_order is the 3rd component
            return None

        result_df["round_order"] = result_df["chronological_key"].apply(
            extract_round_order
        )

        # Count successful ID generation
        successful_ids = result_df["match_id"].notna().sum()
        successful_keys = result_df["chronological_key"].notna().sum()
        successful_round_orders = result_df["round_order"].notna().sum()

        print(
            f"[SUCCESS] Generated {successful_ids} match IDs, {successful_keys} chronological keys, and {successful_round_orders} round orders"
        )

        return result_df

    except Exception as e:
        print(f"[ERROR] Match identifier generation failed: {e}")
        raise


# ============================================================================
# STAGE 5: DUPLICATE DETECTION AND HANDLING
# ============================================================================


def get_latest_dataset_file() -> Optional[str]:
    """
    Get the path to the most recent dataset file.

    Returns:
        str: Path to latest dataset file, or None if none exist
    """
    dataset_files = glob.glob("databases/MatchesMain/matches_dataset_*.csv")
    if dataset_files:
        return max(dataset_files, key=os.path.getctime)
    return None


def load_main_dataset() -> pd.DataFrame:
    """
    Load the most recent main dataset.

    Returns:
        pd.DataFrame: Most recent dataset, or empty DataFrame if none exist
    """
    latest_file = get_latest_dataset_file()
    if latest_file:
        print(f"[INFO] Loading existing dataset: {latest_file}")
        df = pd.read_csv(latest_file)

        # Ensure proper data types for sorting columns
        if "tourney_date" in df.columns:
            df["tourney_date"] = pd.to_datetime(df["tourney_date"])
        if "tourney_id" in df.columns:
            df["tourney_id"] = pd.to_numeric(df["tourney_id"], errors="coerce").astype(
                "Int64"
            )
        if "round_order" in df.columns:
            df["round_order"] = pd.to_numeric(
                df["round_order"], errors="coerce"
            ).astype("Int64")

        print(f"[INFO] Loaded existing dataset: {len(df)} rows")
        return df

    print("[INFO] No existing dataset found")
    return pd.DataFrame()


def detect_duplicates(
    new_df: pd.DataFrame, existing_df: pd.DataFrame
) -> Tuple[Set[str], pd.DataFrame, pd.DataFrame]:
    """
    Detect duplicate matches between new and existing data.

    Args:
        new_df (pd.DataFrame): New matches DataFrame
        existing_df (pd.DataFrame): Existing matches DataFrame

    Returns:
        Tuple[Set[str], pd.DataFrame, pd.DataFrame]: (duplicate_ids, new_matches, duplicate_matches)
    """
    if existing_df.empty:
        return set(), new_df, pd.DataFrame()

    # Check for duplicate match_ids
    existing_match_ids = (
        set(existing_df["match_id"]) if "match_id" in existing_df.columns else set()
    )
    new_match_ids = set(new_df["match_id"]) if "match_id" in new_df.columns else set()

    duplicate_ids = new_match_ids.intersection(existing_match_ids)

    new_matches = new_df[~new_df["match_id"].isin(duplicate_ids)]
    duplicate_matches = new_df[new_df["match_id"].isin(duplicate_ids)]

    print(
        f"[INFO] Found {len(duplicate_ids)} duplicate matches, {len(new_matches)} new matches"
    )

    return duplicate_ids, new_matches, duplicate_matches


def handle_duplicates(
    existing_df: pd.DataFrame, duplicate_matches: pd.DataFrame, replace: bool = False
) -> pd.DataFrame:
    """
    Handle duplicate matches based on replace toggle.

    Args:
        existing_df (pd.DataFrame): Existing dataset
        duplicate_matches (pd.DataFrame): Duplicate matches to handle
        replace (bool): Whether to replace existing matches

    Returns:
        pd.DataFrame: Updated dataset
    """
    if duplicate_matches.empty:
        return existing_df

    if replace:
        print(f"[INFO] Replacing {len(duplicate_matches)} duplicate matches")
        # Remove old versions and prepare to add new ones
        updated_df = existing_df[
            ~existing_df["match_id"].isin(duplicate_matches["match_id"])
        ]
        return updated_df
    else:
        print(
            f"[INFO] Keeping existing matches, skipping {len(duplicate_matches)} duplicates"
        )
        return existing_df


# ============================================================================
# STAGE 6: DATA INSERTION IN CHRONOLOGICAL ORDER
# ============================================================================


def insert_matches_chronologically(
    existing_df: pd.DataFrame, new_matches_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Insert new matches in the correct chronological position.

    Args:
        existing_df (pd.DataFrame): Existing dataset
        new_matches_df (pd.DataFrame): New matches to insert

    Returns:
        pd.DataFrame: Combined dataset in chronological order
    """
    if new_matches_df.empty:
        return existing_df

    if existing_df.empty:
        combined_df = new_matches_df.copy()
    else:
        combined_df = pd.concat([existing_df, new_matches_df], ignore_index=True)

    # Sort by chronological order using separate columns
    if not combined_df.empty and all(
        col in combined_df.columns
        for col in ["tourney_date", "tourney_id", "round_order"]
    ):
        # Ensure proper data types for sorting
        combined_df["tourney_date"] = pd.to_datetime(combined_df["tourney_date"])
        combined_df["tourney_id"] = pd.to_numeric(
            combined_df["tourney_id"], errors="coerce"
        ).astype("Int64")
        combined_df["round_order"] = pd.to_numeric(
            combined_df["round_order"], errors="coerce"
        ).astype("Int64")

        # Sort by tourney_date, then tourney_id, then round_order (descending)
        combined_df = combined_df.sort_values(
            ["tourney_date", "tourney_id", "round_order"],
            ascending=[True, True, False],
            na_position="last",
        ).reset_index(drop=True)

        print(
            f"[INFO] Sorted {len(combined_df)} matches chronologically by tourney_date/tourney_id/round_order"
        )
    elif not combined_df.empty:
        print(
            "[WARNING] Missing required sorting columns (tourney_date, tourney_id, round_order)"
        )

    return combined_df


# ============================================================================
# STAGE 7: DATASET SAVING WITH BACKUP PROTOCOL
# ============================================================================


def create_backup(existing_dataset_file: Optional[str]) -> Optional[str]:
    """
    Create backup of existing dataset file.

    Args:
        existing_dataset_file (Optional[str]): Path to existing dataset file

    Returns:
        Optional[str]: Path to backup file, or None if no backup created
    """
    if existing_dataset_file and os.path.exists(existing_dataset_file):
        backup_file = (
            f"{existing_dataset_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        shutil.copy2(existing_dataset_file, backup_file)
        print(f"[INFO] Created backup: {backup_file}")
        return backup_file
    return None


def get_expected_columns() -> List[str]:
    """
    Define the expected column order for the final dataset.

    Returns:
        List[str]: Ordered list of expected column names
    """
    return [
        # Tournament info
        "tourney_name",
        "tourney_id",
        "tourney_date",
        "tourney_level",
        "surface",
        "draw_size",
        "match_num",
        "round",
        "best_of",
        "minutes",
        # Player identification
        "player1_id",
        "player1_name",
        "player2_id",
        "player2_name",
        # Player rankings and attributes
        "player1_rank",
        "player1_atp_points",
        "player2_rank",
        "player2_atp_points",
        "player1_ht",
        "player1_hand",
        "player2_ht",
        "player2_hand",
        # Match statistics
        "p1_ace",
        "p2_ace",
        "p1_df",
        "p2_df",
        "p1_SvGms",
        "p2_SvGms",
        "p1_1stIn",
        "p2_1stIn",
        "p1_1stWon",
        "p2_1stWon",
        "p1_2ndWon",
        "p2_2ndWon",
        "p1_svpt",
        "p2_svpt",
        "p1_bpSaved",
        "p2_bpSaved",
        "p1_bpFaced",
        "p2_bpFaced",
        # Match outcome
        "score",
        "source",
        "RESULT",
        # Calculated statistics
        "p1_bpconv",
        "p2_bpconv",
        "p1_bp/rg",
        "p2_bp/rg",
        "p1_tpw",
        "p2_tpw",
        "p1_set1",
        "p2_set1",
        "p1_set2",
        "p2_set2",
        "p1_set3",
        "p2_set3",
        "p1_set4",
        "p2_set4",
        "p1_set5",
        "p2_set5",
        "p1_sw",
        "p2_sw",
        # Metadata and tracking
        "original_player1_id",
        "original_player2_id",
        "player1_match_confidence",
        "player1_match_action",
        "player2_match_confidence",
        "player2_match_action",
        "player1_metadata_updated",
        "player1_metadata_source",
        "player2_metadata_updated",
        "player2_metadata_source",
        "p1_id",
        "p2_id",
        "match_id",
        "chronological_key",
        "round_order",
        # Odds data
        "p1_max_odds",
        "p2_max_odds",
        "p1_odds",
        "p2_odds",
        "odds_source",
        "enrichment_confidence",
        "enrichment_method",
    ]


def filter_and_order_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter DataFrame to only include expected columns in the correct order.
    Missing columns will be added as NaN.

    Args:
        df (pd.DataFrame): Input DataFrame with potentially extra or missing columns

    Returns:
        pd.DataFrame: Filtered and ordered DataFrame with only expected columns
    """
    expected_cols = get_expected_columns()

    # Get columns present in both expected and actual
    present_cols = [col for col in expected_cols if col in df.columns]
    missing_cols = [col for col in expected_cols if col not in df.columns]
    extra_cols = [col for col in df.columns if col not in expected_cols]

    # Log filtering info
    if extra_cols:
        print(
            f"[INFO] Filtering out {len(extra_cols)} extra columns: {extra_cols[:5]}{'...' if len(extra_cols) > 5 else ''}"
        )
    if missing_cols:
        print(
            f"[INFO] Adding {len(missing_cols)} missing columns with NaN values: {missing_cols[:5]}{'...' if len(missing_cols) > 5 else ''}"
        )

    # Create filtered DataFrame with expected columns
    filtered_df = df[present_cols].copy()

    # Add missing columns with NaN
    for col in missing_cols:
        filtered_df[col] = np.nan

    # Reorder columns to match expected order
    filtered_df = filtered_df[expected_cols]

    print(
        f"[SUCCESS] Columns filtered and ordered: {len(expected_cols)} columns retained"
    )
    return filtered_df


def save_dataset_with_timestamp(df: pd.DataFrame) -> str:
    """
    Save dataset with timestamp, filtering to only expected columns.

    Args:
        df (pd.DataFrame): Dataset to save

    Returns:
        str: Path to saved file
    """
    # Ensure databases directory exists
    os.makedirs("databases/MatchesMain", exist_ok=True)

    # Filter to expected columns only
    filtered_df = filter_and_order_columns(df)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"databases/MatchesMain/matches_dataset_{timestamp}.csv"
    filtered_df.to_csv(filename, index=False)
    print(f"[SUCCESS] Dataset saved: {filename}")
    return filename


def update_process_log(log_data: Dict[str, Any]) -> None:
    """
    Update the process log with new entry.

    Args:
        log_data (Dict[str, Any]): Log data to add
    """
    # Ensure databases directory exists
    os.makedirs("databases/MatchesMain", exist_ok=True)

    log_file = "databases/MatchesMain/pipeline_log.json"

    # Load existing log or create new
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            log_history = json.load(f)
    else:
        log_history = []

    # Add new entry
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "dataset_file": log_data["dataset_file"],
        "rows_added": log_data["rows_added"],
        "rows_replaced": log_data["rows_replaced"],
        "source": log_data["source"],
        "total_rows": log_data["total_rows"],
        "backup_created": log_data.get("backup_file"),
    }

    log_history.append(log_entry)

    # Save updated log
    with open(log_file, "w") as f:
        json.dump(log_history, f, indent=2)

    print(f"[INFO] Process log updated: {log_file}")


# ============================================================================
# STAGE 8: ODDS DATA ENRICHMENT
# ============================================================================


def convert_odds_series_to_level(series: str) -> str:
    """
    Convert odds 'Series' column to standard tourney_level format.

    Args:
        series: Series value from odds dataset

    Returns:
        str: Standard tourney_level code (G, M, A)
    """
    series_mapping = {
        "Grand Slam": "G",
        "Masters 1000": "M",
        "ATP500": "A",
        "ATP250": "A",
    }
    return series_mapping.get(series, "A")  # Default to "A" if unknown


def match_tournament_for_odds(
    tournament_name: str, location: str, series: str
) -> Tuple[Optional[List[Dict]], float, str]:
    """
    Match odds tournament to database tournament WITHOUT storing source_id.

    This function is specifically for odds datasets where:
    - Tournament IDs change yearly (1-42 per year)
    - We only need the mapping for enrichment, not permanent storage
    - Matching is based on name + location (equal weight)

    Args:
        tournament_name: Tournament name from odds dataset
        location: Location from odds dataset
        series: Series from odds dataset (for level conversion)

    Returns:
        Tuple: (list of candidates or None, confidence_score, match_method)
        - Returns list of candidates for manual review (50-79%)
        - Returns single-item list for auto-accept (≥80%)
        - Returns None for no match (<50%)
    """
    from tennis_matching.matching.text_processing import (
        normalize_tournament_name,
        combined_similarity_score,
    )
    from tennis_matching.database.crud_operations import (
        get_all_tournaments_for_fuzzy_matching,
    )

    # Normalize inputs
    normalized_name = normalize_tournament_name(tournament_name)
    normalized_location = normalize_tournament_name(location) if location else ""
    tourney_level = convert_odds_series_to_level(series)

    # Get all tournaments from database
    all_tournaments = get_all_tournaments_for_fuzzy_matching()

    if not all_tournaments:
        return None, 0.0, "no_database_tournaments"

    # Calculate similarity scores with equal weight for name and location
    scored_tournaments = []

    for tournament in all_tournaments:
        db_name = tournament["source_name_variant"]
        db_level = tournament.get("tourney_level", "")

        # Calculate name similarity
        name_similarity = combined_similarity_score(normalized_name, db_name)

        # Calculate location similarity (match location against tournament name)
        location_similarity = 0.0
        if normalized_location:
            location_similarity = combined_similarity_score(normalized_location, db_name)

        # Score is MAX of name and location (not average)
        best_score_component = max(name_similarity, location_similarity)

        # Small boost for level match
        level_boost = 0.02 if db_level == tourney_level else 0.0
        total_score = min(best_score_component + level_boost, 1.0)

        scored_tournaments.append({
            "tournament_id": tournament["tournament_id"],
            "primary_name": tournament["primary_name"],
            "tourney_level": db_level,
            "name_similarity": name_similarity,
            "location_similarity": location_similarity,
            "total_score": total_score,
        })

    # Sort by total score (desc), then name similarity (desc), then alphabetically
    scored_tournaments.sort(
        key=lambda x: (-x["total_score"], -x["name_similarity"], x["primary_name"])
    )

    if not scored_tournaments:
        return None, 0.0, "no_matches"

    # Filter candidates with score ≥50%
    candidates_above_threshold = [t for t in scored_tournaments if t["total_score"] >= 0.50]

    if not candidates_above_threshold:
        # No matches above 50%
        return None, scored_tournaments[0]["total_score"], "below_threshold"

    best_match = candidates_above_threshold[0]
    best_score = best_match["total_score"]

    # Auto-match conditions:
    # 1. Best match ≥80%
    # 2. No other candidates ≥50%
    if best_score >= 0.80 and len(candidates_above_threshold) == 1:
        # Only one candidate above 50% and it's above 80% - auto-accept
        return [best_match], best_score, "auto_accept"
    elif best_score >= 0.995:
        # Very high confidence (99.5%+) - auto-accept even if others present
        return [best_match], best_score, "auto_accept_high_confidence"
    elif len(candidates_above_threshold) > 0:
        # Multiple candidates ≥50% OR single candidate 50-79% - manual review
        return candidates_above_threshold[:5], best_score, "manual_review_candidate"
    else:
        # Shouldn't reach here, but handle just in case
        return None, best_score, "below_threshold"


def load_all_odds_data(years_range: List[int] = None) -> pd.DataFrame:
    """
    Load and concatenate all yearly odds Excel files with tournament matching.

    Args:
        years_range: List of years to load (default: 2001-2025)

    Returns:
        Combined DataFrame with all odds data including tourney_id column
    """
    if years_range is None:
        years_range = list(range(2001, 2026))  # 2001-2025

    print(f"[INFO] Loading odds data for years: {years_range[0]}-{years_range[-1]}")

    odds_dataframes = []

    for year in tqdm(years_range, desc="Loading odds files"):
        # Determine file extension
        if year <= 2012:
            file_path = os.path.join(ODDS_DATASET_PATH, f"{year}.xls")
        else:
            file_path = os.path.join(ODDS_DATASET_PATH, f"{year}.xlsx")

        if os.path.exists(file_path):
            try:
                df_year = pd.read_excel(file_path)
                df_year["year"] = year  # Add year column for reference
                odds_dataframes.append(df_year)
            except Exception as e:
                print(f"[WARNING] Failed to load {file_path}: {e}")
        else:
            print(f"[WARNING] File not found: {file_path}")

    if not odds_dataframes:
        print("[ERROR] No odds data files could be loaded")
        return pd.DataFrame()

    # Combine all years
    combined_odds = pd.concat(odds_dataframes, ignore_index=True)

    # Ensure Date column is datetime
    if "Date" in combined_odds.columns:
        combined_odds["Date"] = pd.to_datetime(combined_odds["Date"], errors="coerce")

    print(
        f"[SUCCESS] Loaded {len(combined_odds)} odds records from {len(odds_dataframes)} files"
    )

    # Add tournament ID matching
    print("[INFO] Matching odds tournaments to database...")
    combined_odds = add_tournament_ids_to_odds(combined_odds)

    return combined_odds


def add_tournament_ids_to_odds(odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add tourney_id column to odds DataFrame by matching tournaments.

    Args:
        odds_df: Odds DataFrame with Tournament, Location, Series columns

    Returns:
        DataFrame with added tourney_id column
    """
    # Initialize tourney_id column
    odds_df["tourney_id"] = None

    # Get unique tournaments in the odds dataset
    unique_tournaments = odds_df[["Tournament", "Location", "Series"]].drop_duplicates()

    # Create tournament mapping cache
    tournament_mapping = {}
    auto_matched = 0
    manual_needed = 0
    failed = 0
    manual_review_items = []

    print(f"[INFO] Matching {len(unique_tournaments)} unique tournaments...")

    for _, row in tqdm(
        unique_tournaments.iterrows(),
        total=len(unique_tournaments),
        desc="Matching tournaments",
    ):
        tournament_name = row["Tournament"]
        location = row.get("Location", "")
        series = row.get("Series", "")

        # Create cache key
        cache_key = f"{tournament_name}|{location}|{series}"

        # Match tournament (returns list of candidates or None)
        candidates, confidence, method = match_tournament_for_odds(
            tournament_name, location, series
        )

        if candidates:
            if method in ["auto_accept_high_confidence", "auto_accept"]:
                # Auto-accept (single candidate in list)
                tournament_mapping[cache_key] = candidates[0]["tournament_id"]
                auto_matched += 1
            else:
                # Manual review candidate (50-79%, multiple candidates possible)
                manual_review_items.append({
                    "cache_key": cache_key,
                    "tournament_name": tournament_name,
                    "location": location,
                    "series": series,
                    "candidates": candidates,  # Now a list
                    "confidence": confidence
                })
                manual_needed += 1
        else:
            tournament_mapping[cache_key] = None
            failed += 1

    # Process manual review items
    if manual_review_items:
        print(f"\n[INFO] {len(manual_review_items)} tournaments need manual review (50-79% confidence)")
        print("Options: [1-5] Select candidate, [n] Skip, [q] Quit review (skip all remaining)")

        for item in manual_review_items:
            print(f"\n{'='*70}")
            print(f"--- Odds Tournament ---")
            print(f"  Name: {item['tournament_name']}")
            print(f"  Location: {item['location']}")
            print(f"  Series: {item['series']}")

            candidates = item["candidates"]
            print(f"\n--- Suggested Matches (top {len(candidates)}) ---")

            for idx, candidate in enumerate(candidates, 1):
                name_sim = candidate["name_similarity"] * 100
                loc_sim = candidate["location_similarity"] * 100
                total_sim = candidate["total_score"] * 100
                max_component = max(name_sim, loc_sim)

                print(f"\n[{idx}] {candidate['primary_name']} (ID: {candidate['tournament_id']})")
                print(f"    Level: {candidate['tourney_level']}")
                print(f"    Name similarity: {name_sim:.1f}%")
                print(f"    Location similarity: {loc_sim:.1f}%")
                print(f"    Best score: {max_component:.1f}% (max of name/location)")
                print(f"    Total confidence: {total_sim:.1f}% (with level boost)")

            choice = input(f"\nSelect option [1-{len(candidates)}/n/q]: ").strip().lower()

            if choice == 'q':
                print("[INFO] Skipping remaining manual reviews")
                break
            elif choice == 'n':
                tournament_mapping[item["cache_key"]] = None
                manual_needed -= 1
                failed += 1
            elif choice.isdigit() and 1 <= int(choice) <= len(candidates):
                selected_candidate = candidates[int(choice) - 1]
                tournament_mapping[item["cache_key"]] = selected_candidate["tournament_id"]
                auto_matched += 1  # Count as matched
                manual_needed -= 1
                print(f"[INFO] Matched to: {selected_candidate['primary_name']}")
            else:
                print("[WARNING] Invalid choice, skipping...")
                tournament_mapping[item["cache_key"]] = None
                manual_needed -= 1
                failed += 1

    # Apply mapping to all rows
    odds_df["tourney_id"] = odds_df.apply(
        lambda row: tournament_mapping.get(
            f"{row['Tournament']}|{row.get('Location', '')}|{row.get('Series', '')}",
            None,
        ),
        axis=1,
    )

    # Report results
    print(f"\n[INFO] Tournament matching results:")
    print(f"  - Auto-matched (≥80%): {auto_matched}")
    print(f"  - Manual review needed (50-79%): {manual_needed}")
    print(f"  - Failed (<50%): {failed}")

    matched_count = odds_df["tourney_id"].notna().sum()
    total_count = len(odds_df)
    print(
        f"[SUCCESS] Matched {matched_count}/{total_count} odds records ({matched_count / total_count * 100:.1f}%)"
    )

    return odds_df


def calculate_date_range(
    tourney_date: datetime, draw_size: int
) -> Tuple[datetime, datetime]:
    """
    Calculate date range for filtering odds dataset.

    Formula: [tourney_date - 5, tourney_date + (3 + 2*log2(draw_size))]

    Examples:
    - Draw 32: 3 + 2*log2(32) = 3 + 2*5 = 13 days
    - Draw 128: 3 + 2*log2(128) = 3 + 2*7 = 17 days
    """
    base_buffer = DATE_BUFFER_BASE  # Days before tournament
    end_buffer = 3 + 2 * math.log2(draw_size) if draw_size > 0 else 7

    start_date = tourney_date - pd.Timedelta(days=base_buffer)
    end_date = tourney_date + pd.Timedelta(days=end_buffer)

    return start_date, end_date


def filter_odds_by_date(
    odds_df: pd.DataFrame, start_date: datetime, end_date: datetime
) -> pd.DataFrame:
    """Filter odds dataset to matches within date range."""
    return odds_df[
        (odds_df["Date"] >= start_date) & (odds_df["Date"] <= end_date)
    ].copy()


def normalize_odds_round(odds_round: str, draw_size: int) -> str:
    """
    Normalize odds dataset round to main dataset format.

    Handle numbered rounds based on draw size:
    - Draw 128: '1st Round' → 'R128', '2nd Round' → 'R64', '3rd Round' → 'R32', '4th Round' → 'R16'
    - Draw 64: '1st Round' → 'R64', '2nd Round' → 'R32', '3rd Round' → 'R16'
    - Draw 32: '1st Round' → 'R32', '2nd Round' → 'R16'
    """
    if pd.isna(odds_round) or not odds_round:
        return ""

    odds_round_str = str(odds_round).strip()

    # Handle numbered rounds dynamically based on draw size
    if "Round" in odds_round_str and any(
        num in odds_round_str for num in ["1st", "2nd", "3rd", "4th"]
    ):
        # Extract round number
        round_num = None
        if "1st" in odds_round_str:
            round_num = 1
        elif "2nd" in odds_round_str:
            round_num = 2
        elif "3rd" in odds_round_str:
            round_num = 3
        elif "4th" in odds_round_str:
            round_num = 4

        if round_num:
            # Calculate the round based on draw size and round number
            # Round 1 = draw_size, Round 2 = draw_size/2, Round 3 = draw_size/4, etc.
            target_size = draw_size // (2 ** (round_num - 1))

            if target_size >= 128:
                return "R128"
            elif target_size >= 64:
                return "R64"
            elif target_size >= 32:
                return "R32"
            elif target_size >= 16:
                return "R16"
            elif target_size >= 8:
                return "QF"  # Quarterfinals (8 players)
            elif target_size >= 4:
                return "SF"  # Semifinals (4 players)
            elif target_size >= 2:
                return "F"  # Final (2 players)
            else:
                return odds_round_str  # Fallback to original

    # Handle explicit round names
    return ROUND_MAPPING.get(odds_round_str, odds_round_str)


def create_score_key(
    p1_s1: int,
    p2_s1: int,
    p1_s2: int,
    p2_s2: int,
    p1_s3: int,
    p2_s3: int,
    p1_s4: int,
    p2_s4: int,
    p1_s5: int,
    p2_s5: int,
) -> str:
    """
    Create 10-character score key for matching.

    Format: p1_s1 + p2_s1 + p1_s2 + p2_s2 + p1_s3 + p2_s3 + p1_s4 + p2_s4 + p1_s5 + p2_s5
    Handle NaN/None as 0.

    Examples:
    - Match 6-4 6-2: "6462000000"
    - Match 7-6 4-6 6-3: "7646630000"
    - Match 6-4 3-6 6-2 7-5: "6436627500"
    """
    scores = [p1_s1, p2_s1, p1_s2, p2_s2, p1_s3, p2_s3, p1_s4, p2_s4, p1_s5, p2_s5]

    # Convert NaN/None/empty strings to 0, then to string
    def safe_int(s):
        if pd.isna(s):
            return "0"
        try:
            # Handle strings with spaces or empty strings
            s_str = str(s).strip()
            if not s_str or s_str == "":
                return "0"
            return str(int(float(s_str)))
        except (ValueError, TypeError):
            return "0"

    score_str = "".join(safe_int(s) for s in scores)
    return score_str


def create_main_score_key(row: pd.Series) -> str:
    """Create score key from main dataset row."""
    return create_score_key(
        row.get("p1_set1", 0),
        row.get("p2_set1", 0),
        row.get("p1_set2", 0),
        row.get("p2_set2", 0),
        row.get("p1_set3", 0),
        row.get("p2_set3", 0),
        row.get("p1_set4", 0),
        row.get("p2_set4", 0),
        row.get("p1_set5", 0),
        row.get("p2_set5", 0),
    )


def create_odds_score_key(row: pd.Series) -> str:
    """Create score key from odds dataset row."""
    return create_score_key(
        row.get("W1", 0),
        row.get("L1", 0),
        row.get("W2", 0),
        row.get("L2", 0),
        row.get("W3", 0),
        row.get("L3", 0),
        row.get("W4", 0),
        row.get("L4", 0),
        row.get("W5", 0),
        row.get("L5", 0),
    )


def extract_name_components(odds_name: str) -> Dict[str, str]:
    """
    Extract name components from odds dataset format.

    Input: "Federer R." or "Murray A.M." or "Van Der Meer J."
    Output: {"last": "Federer", "first_initials": "R", "full_parts": ["Federer", "R"]}
    """
    if pd.isna(odds_name) or not odds_name:
        return {"last": "", "first_initials": "", "full_parts": []}

    parts = str(odds_name).strip().split()
    if len(parts) < 2:
        return {"last": odds_name, "first_initials": "", "full_parts": [odds_name]}

    # Last name could be multiple parts (Van Der Meer)
    # First initials are typically the last part
    initials = parts[-1]
    last_name_parts = parts[:-1]

    return {
        "last": " ".join(last_name_parts),
        "first_initials": initials.replace(".", ""),
        "full_parts": parts,
    }


def calculate_name_similarity(
    main_name: str, odds_name_components: Dict[str, str]
) -> float:
    """
    Calculate similarity between main dataset name and odds dataset name.

    Strategy:
    1. Compare full names (handle hyphens as spaces)
    2. Compare last name only
    3. Check first initial match
    4. Return best similarity score
    """
    if pd.isna(main_name) or not main_name:
        return 0.0

    main_clean = str(main_name).replace("-", " ").lower()
    odds_last = odds_name_components["last"].lower()
    odds_initials = odds_name_components["first_initials"].lower()

    # Full name similarity
    full_similarity = fuzz.ratio(main_clean, odds_last)

    # Last name only similarity
    main_parts = main_clean.split()
    if len(main_parts) >= 2:
        main_last = main_parts[-1]
        last_similarity = fuzz.ratio(main_last, odds_last)
    else:
        last_similarity = 0

    # First initial check
    if len(main_parts) >= 1 and len(odds_initials) >= 1:
        first_initial_match = main_parts[0][0] == odds_initials[0]
        initial_bonus = 10 if first_initial_match else 0
    else:
        initial_bonus = 0

    return max(full_similarity, last_similarity) + initial_bonus


def find_name_matches(
    main_p1: str,
    main_p2: str,
    odds_candidates: pd.DataFrame,
    threshold: float = FUZZY_MATCH_THRESHOLD,
) -> List[Dict]:
    """
    Find name matches between main dataset players and odds candidates.

    Returns list of matches with confidence scores.
    """
    matches = []

    for idx, odds_row in odds_candidates.iterrows():
        winner_components = extract_name_components(odds_row.get("Winner", ""))
        loser_components = extract_name_components(odds_row.get("Loser", ""))

        # Try both orientations (main p1/p2 vs odds winner/loser)
        # Orientation 1: p1=winner, p2=loser
        p1_winner_sim = calculate_name_similarity(main_p1, winner_components)
        p2_loser_sim = calculate_name_similarity(main_p2, loser_components)
        orientation1_score = (p1_winner_sim + p2_loser_sim) / 2

        # Orientation 2: p1=loser, p2=winner
        p1_loser_sim = calculate_name_similarity(main_p1, loser_components)
        p2_winner_sim = calculate_name_similarity(main_p2, winner_components)
        orientation2_score = (p1_loser_sim + p2_winner_sim) / 2

        # Choose better orientation
        if orientation1_score >= orientation2_score:
            best_score = orientation1_score
            match_orientation = "p1_winner"
        else:
            best_score = orientation2_score
            match_orientation = "p1_loser"

        if best_score >= threshold:
            matches.append(
                {
                    "odds_index": idx,
                    "odds_row": odds_row,
                    "confidence": best_score,
                    "orientation": match_orientation,
                    "p1_similarity": p1_winner_sim
                    if match_orientation == "p1_winner"
                    else p1_loser_sim,
                    "p2_similarity": p2_loser_sim
                    if match_orientation == "p1_winner"
                    else p2_winner_sim,
                }
            )

    return sorted(matches, key=lambda x: x["confidence"], reverse=True)


def enrich_match_with_odds(main_row: pd.Series, odds_match: Dict) -> pd.Series:
    """
    Add odds data to main dataset row.

    Handle player orientation (p1 could be winner or loser in odds dataset).
    """
    enriched_row = main_row.copy()
    odds_row = odds_match["odds_row"]
    orientation = odds_match["orientation"]

    if orientation == "p1_winner":
        # Player1 is winner in odds dataset
        enriched_row["p1_max_odds"] = odds_row.get("MaxW", np.nan)
        enriched_row["p2_max_odds"] = odds_row.get("MaxL", np.nan)
        enriched_row["p1_odds"] = odds_row.get("PSW", odds_row.get("B365W", np.nan))
        enriched_row["p2_odds"] = odds_row.get("PSL", odds_row.get("B365L", np.nan))

        # Fill missing ATP data
        if pd.isna(enriched_row.get("player1_rank")):
            enriched_row["player1_rank"] = odds_row.get("WRank", np.nan)
        if pd.isna(enriched_row.get("player1_atp_points")):
            enriched_row["player1_atp_points"] = odds_row.get("WPts", np.nan)
        if pd.isna(enriched_row.get("player2_rank")):
            enriched_row["player2_rank"] = odds_row.get("LRank", np.nan)
        if pd.isna(enriched_row.get("player2_atp_points")):
            enriched_row["player2_atp_points"] = odds_row.get("LPts", np.nan)

    else:  # p1_loser
        # Player1 is loser in odds dataset
        enriched_row["p1_max_odds"] = odds_row.get("MaxL", np.nan)
        enriched_row["p2_max_odds"] = odds_row.get("MaxW", np.nan)
        enriched_row["p1_odds"] = odds_row.get("PSL", odds_row.get("B365L", np.nan))
        enriched_row["p2_odds"] = odds_row.get("PSW", odds_row.get("B365W", np.nan))

        # Fill missing ATP data (swapped)
        if pd.isna(enriched_row.get("player1_rank")):
            enriched_row["player1_rank"] = odds_row.get("LRank", np.nan)
        if pd.isna(enriched_row.get("player1_atp_points")):
            enriched_row["player1_atp_points"] = odds_row.get("LPts", np.nan)
        if pd.isna(enriched_row.get("player2_rank")):
            enriched_row["player2_rank"] = odds_row.get("WRank", np.nan)
        if pd.isna(enriched_row.get("player2_atp_points")):
            enriched_row["player2_atp_points"] = odds_row.get("WPts", np.nan)

    # Set enrichment metadata
    enriched_row["enrichment_confidence"] = odds_match["confidence"]
    enriched_row["enrichment_method"] = odds_match.get("method", "auto")

    # Determine odds source
    if pd.notna(odds_row.get("PSW")) and pd.notna(odds_row.get("PSL")):
        enriched_row["odds_source"] = "pinnacle"
    elif pd.notna(odds_row.get("B365W")) and pd.notna(odds_row.get("B365L")):
        enriched_row["odds_source"] = "bet365"
    else:
        enriched_row["odds_source"] = "max_only"

    return enriched_row


def score_name_match_for_rank_tiebreak(main_name: str, odds_name: str) -> float:
    """
    Score name match for tie-breaking when multiple rank matches exist.

    Strategy:
    - Split full name into words (e.g., "Albert Portas" → ["albert", "portas"])
    - Split abbreviated name (e.g., "Portas A." → ["portas", "a"])
    - Last name match: 2 points
    - First initial match: 1 point

    Args:
        main_name: Full name from main dataset (e.g., "Albert Portas")
        odds_name: Abbreviated name from odds dataset (e.g., "Portas A.")

    Returns:
        float: Match score (0-3)
    """
    if pd.isna(main_name) or pd.isna(odds_name):
        return 0.0

    # Normalize names: lowercase, remove dots, split into words
    main_parts = str(main_name).lower().replace(".", " ").replace("-", " ").split()
    odds_parts = str(odds_name).lower().replace(".", " ").replace("-", " ").split()

    if not main_parts or not odds_parts:
        return 0.0

    score = 0.0

    # Assume last word in main_parts is last name
    # Assume first word in odds_parts is last name (format: "Lastname F.")
    if len(main_parts) >= 1 and len(odds_parts) >= 1:
        main_last = main_parts[-1]  # Last word of full name
        odds_last = odds_parts[0]  # First word of abbreviated name

        # Last name exact match
        if main_last == odds_last:
            score += 2.0
        # Partial last name match (e.g., "del potro" vs "potro")
        elif main_last in odds_last or odds_last in main_last:
            score += 1.5

    # First initial match
    if len(main_parts) >= 1 and len(odds_parts) >= 2:
        main_first_initial = main_parts[0][0]  # First char of first name
        odds_initials = odds_parts[
            1
        ]  # Second word in abbreviated name (e.g., "a" from "Portas A.")

        # Check if first initial matches
        if odds_initials and main_first_initial == odds_initials[0]:
            score += 1.0

    return score


def resolve_multiple_rank_matches_with_names(
    main_row: pd.Series, matches_df: pd.DataFrame, orientation: str, draw_size: int
) -> Dict[str, Any]:
    """
    Resolve multiple rank matches by filtering on round, then using name matching.

    Args:
        main_row: Main dataset row
        matches_df: DataFrame with multiple rank matches
        orientation: 'p1_winner' or 'p1_loser'
        draw_size: Tournament draw size for round normalization

    Returns:
        Dict with status ('enriched', 'uncertain', 'failed') and match data
    """
    main_round = main_row.get("round", "")

    # Filter by round
    round_matches = matches_df[
        matches_df["Round"].apply(
            lambda x: normalize_odds_round(str(x), int(draw_size))
        )
        == main_round
    ]

    if len(round_matches) == 1:
        # Single match after round filtering
        match_data = {
            "odds_row": round_matches.iloc[0],
            "confidence": 98.0,  # Higher confidence with rank + round match
            "orientation": orientation,
            "method": "auto_rank_round",
        }
        return {"status": "enriched", "match": match_data}
    elif len(round_matches) > 1:
        # Multiple matches after round filtering - try name-based tie-breaking
        best_match = resolve_by_name_matching(
            main_row,
            round_matches,
            orientation,
            confidence=98.0,
            method="auto_rank_round",
        )
        if best_match:
            return best_match

        # Name matching didn't resolve - return as candidates
        candidates = []
        for idx, row in round_matches.iterrows():
            candidates.append(
                {
                    "odds_row": row,
                    "confidence": 98.0,
                    "orientation": orientation,
                    "method": "auto_rank_round",
                }
            )
        return {"status": "uncertain", "candidates": candidates}
    else:
        # No round matches - try name matching on all rank matches
        best_match = resolve_by_name_matching(
            main_row, matches_df, orientation, confidence=97.0, method="auto_rank"
        )
        if best_match:
            return best_match

        # Return all rank matches as candidates
        candidates = []
        for idx, row in matches_df.iterrows():
            candidates.append(
                {
                    "odds_row": row,
                    "confidence": 97.0,
                    "orientation": orientation,
                    "method": "auto_rank",
                }
            )
        return {"status": "uncertain", "candidates": candidates}


def resolve_by_name_matching(
    main_row: pd.Series,
    candidates_df: pd.DataFrame,
    orientation: str,
    confidence: float,
    method: str,
) -> Dict[str, Any]:
    """
    Try to resolve multiple rank matches using name matching.

    Args:
        main_row: Main dataset row with player names
        candidates_df: DataFrame with candidate matches
        orientation: 'p1_winner' or 'p1_loser'
        confidence: Confidence score to assign
        method: Method string for tracking

    Returns:
        Dict with status 'enriched' and match data if single best match found, None otherwise
    """
    # Get main dataset player names
    main_p1_name = main_row.get("player1_name", "")
    main_p2_name = main_row.get("player2_name", "")

    # Score each candidate
    candidate_scores = []
    for idx, odds_row in candidates_df.iterrows():
        if orientation == "p1_winner":
            # p1 = Winner, p2 = Loser
            odds_p1_name = odds_row.get("Winner", "")
            odds_p2_name = odds_row.get("Loser", "")
        else:  # p1_loser
            # p1 = Loser, p2 = Winner
            odds_p1_name = odds_row.get("Loser", "")
            odds_p2_name = odds_row.get("Winner", "")

        # Calculate match scores for both players
        p1_score = score_name_match_for_rank_tiebreak(main_p1_name, odds_p1_name)
        p2_score = score_name_match_for_rank_tiebreak(main_p2_name, odds_p2_name)
        total_score = p1_score + p2_score

        candidate_scores.append(
            {
                "odds_row": odds_row,
                "total_score": total_score,
                "p1_score": p1_score,
                "p2_score": p2_score,
            }
        )

    # Sort by total score (highest first)
    candidate_scores.sort(key=lambda x: x["total_score"], reverse=True)

    # Check if we have a clear winner (best score > second best score)
    if len(candidate_scores) >= 1:
        best = candidate_scores[0]

        # Require both players to have at least some match (avoid false positives)
        if best["p1_score"] >= 2.0 and best["p2_score"] >= 2.0:
            # Check if this is clearly the best match
            if (
                len(candidate_scores) == 1
                or best["total_score"] > candidate_scores[1]["total_score"]
            ):
                match_data = {
                    "odds_row": best["odds_row"],
                    "confidence": confidence,
                    "orientation": orientation,
                    "method": f"{method}_name_tiebreak",
                }
                return {"status": "enriched", "match": match_data}

    return None


def resolve_multiple_rank_matches(
    main_row: pd.Series, matches_df: pd.DataFrame, orientation: str, draw_size: int
) -> Dict[str, Any]:
    """
    Resolve multiple rank matches by filtering on round, then using name matching.

    Args:
        main_row: Main dataset row
        matches_df: DataFrame with multiple rank matches
        orientation: 'p1_winner' or 'p1_loser'
        draw_size: Tournament draw size for round normalization

    Returns:
        Dict with status ('enriched', 'uncertain', 'failed') and match data
    """
    # Use the enhanced version with name matching
    return resolve_multiple_rank_matches_with_names(
        main_row, matches_df, orientation, draw_size
    )


def match_by_rank(
    main_row: pd.Series, date_filtered_df: pd.DataFrame, draw_size: int
) -> Dict[str, Any]:
    """
    Match by exact player ranks if available.
    If multiple matches found, try to resolve by round.

    Returns:
        Dict with status ('enriched', 'uncertain', 'failed') and match data if found
    """
    # Check if main dataset has both player ranks
    p1_rank = main_row.get("player1_rank")
    p2_rank = main_row.get("player2_rank")

    if pd.isna(p1_rank) or pd.isna(p2_rank):
        return {"status": "failed", "reason": "no_ranks_in_main"}

    # Filter odds data to rows with both ranks available
    odds_with_ranks = date_filtered_df.dropna(subset=["WRank", "LRank"])

    if odds_with_ranks.empty:
        return {"status": "failed", "reason": "no_ranks_in_odds"}

    # Try matching with p1 as winner orientation
    p1_winner_matches = odds_with_ranks[
        (odds_with_ranks["WRank"] == p1_rank) & (odds_with_ranks["LRank"] == p2_rank)
    ]

    if len(p1_winner_matches) == 1:
        match_data = {
            "odds_row": p1_winner_matches.iloc[0],
            "confidence": 97.0,
            "orientation": "p1_winner",
            "method": "auto_rank",
        }
        return {"status": "enriched", "match": match_data}
    elif len(p1_winner_matches) > 1:
        # Multiple rank matches - try to resolve by round
        return resolve_multiple_rank_matches(
            main_row, p1_winner_matches, "p1_winner", draw_size
        )

    # Try matching with p1 as loser orientation
    p1_loser_matches = odds_with_ranks[
        (odds_with_ranks["LRank"] == p1_rank) & (odds_with_ranks["WRank"] == p2_rank)
    ]

    if len(p1_loser_matches) == 1:
        match_data = {
            "odds_row": p1_loser_matches.iloc[0],
            "confidence": 97.0,
            "orientation": "p1_loser",
            "method": "auto_rank",
        }
        return {"status": "enriched", "match": match_data}
    elif len(p1_loser_matches) > 1:
        # Multiple rank matches - try to resolve by round
        return resolve_multiple_rank_matches(
            main_row, p1_loser_matches, "p1_loser", draw_size
        )

    return {"status": "failed", "reason": "no_rank_matches"}


def process_single_match(main_row: pd.Series, odds_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Process a single match through the enhanced 5-stage matching algorithm.

    Stage 1: Tournament + Date filtering (NEW: uses tourney_id)
    Stage 2: Rank matching (if ranks available)
    Stage 3: Round filtering (if rank matching fails)
    Stage 4: Score key matching
    Stage 5: Name-based resolution

    Returns:
        Dict with status ('enriched', 'uncertain', 'failed') and relevant data
    """
    # Stage 1: Tournament + Date Range Filtering
    draw_size = main_row.get("draw_size", 32)
    if pd.isna(draw_size) or draw_size <= 0:
        draw_size = 32  # Default

    # First filter by tournament ID if available
    main_tourney_id = main_row.get("tourney_id")
    if pd.notna(main_tourney_id) and "tourney_id" in odds_df.columns:
        # Filter by exact tournament ID
        tourney_filtered = odds_df[odds_df["tourney_id"] == main_tourney_id].copy()

        if not tourney_filtered.empty:
            # Use tournament-filtered data (much more precise)
            date_filtered = tourney_filtered
        else:
            # Fallback to date filtering if no tournament match
            start_date, end_date = calculate_date_range(
                main_row["tourney_date"], int(draw_size)
            )
            date_filtered = filter_odds_by_date(odds_df, start_date, end_date)
    else:
        # No tourney_id available - use date filtering
        start_date, end_date = calculate_date_range(
            main_row["tourney_date"], int(draw_size)
        )
        date_filtered = filter_odds_by_date(odds_df, start_date, end_date)

    if date_filtered.empty:
        return {"status": "failed", "reason": "no_tournament_or_date_matches"}

    # Stage 2: Rank Matching (NEW)
    rank_result = match_by_rank(main_row, date_filtered, draw_size)
    if rank_result["status"] in ["enriched", "uncertain"]:
        # Rank matching succeeded - return immediately
        return rank_result

    # Stage 3: Round Filtering (only if rank matching failed)
    main_round = main_row.get("round", "")
    round_filtered = date_filtered[
        date_filtered["Round"].apply(
            lambda x: normalize_odds_round(str(x), int(draw_size))
        )
        == main_round
    ]

    if round_filtered.empty:
        return {"status": "failed", "reason": "no_round_matches"}

    # Stage 3: Score Key Matching
    main_score_key = create_main_score_key(main_row)
    round_filtered = round_filtered.copy()  # Avoid SettingWithCopyWarning
    round_filtered["odds_score_key"] = round_filtered.apply(
        create_odds_score_key, axis=1
    )
    score_matches = round_filtered[round_filtered["odds_score_key"] == main_score_key]

    if score_matches.empty:
        return {"status": "failed", "reason": "no_score_matches"}

    if len(score_matches) == 1:
        # Single score match - high confidence
        match_data = {
            "odds_row": score_matches.iloc[0],
            "confidence": 98.0,
            "orientation": "p1_winner",  # Will be determined by result matching
            "method": "auto_score",
        }
        return {"status": "enriched", "match": match_data}

    # Stage 4: Name-based resolution for multiple score matches
    main_p1 = main_row.get("player1_name", "")
    main_p2 = main_row.get("player2_name", "")

    name_matches = find_name_matches(
        main_p1, main_p2, score_matches, FUZZY_MATCH_THRESHOLD
    )

    if len(name_matches) == 1:
        # Single name match
        name_matches[0]["method"] = "auto_name"
        return {"status": "enriched", "match": name_matches[0]}
    elif len(name_matches) > 1:
        # Multiple matches - needs manual review
        return {"status": "uncertain", "candidates": name_matches}
    else:
        # No name matches above threshold
        return {"status": "failed", "reason": "no_name_matches"}


def interactive_batch_review(
    uncertain_matches: List[Dict], batch_size: int = BATCH_REVIEW_SIZE
) -> List[Dict]:
    """
    Interactive review system for uncertain matches.

    Returns list of resolved matches with user decisions.
    """
    resolved_matches = []

    print(
        f"\n[INFO] Manual review required for {len(uncertain_matches)} uncertain matches"
    )
    print(
        "Options: [1-9] Accept candidate, [s] Skip, [q] Quit, [Enter] Accept highest confidence"
    )

    for i, uncertain in enumerate(uncertain_matches[:batch_size]):
        main_row = uncertain["main_row"]
        candidates = uncertain["candidates"]
        original_index = uncertain["original_index"]  # Extract the original index

        print(f"\n--- Match #{i + 1} ---")
        print(f"Date: {main_row.get('tourney_date', 'N/A')}")
        print(
            f'Main Dataset: "{main_row.get("player1_name", "")}" vs "{main_row.get("player2_name", "")}"'
        )
        print(f"Round: {main_row.get('round', '')}")

        print("\nCandidates from Odds Dataset:")
        for j, candidate in enumerate(candidates[:9]):  # Show up to 9 candidates
            odds_row = candidate["odds_row"]
            round_info = odds_row.get("Round", "N/A")
            print(
                f'[{j + 1}] "{odds_row.get("Winner", "")}" vs "{odds_row.get("Loser", "")}" | Round: {round_info} | Confidence: {candidate["confidence"]:.1f}%'
            )

        while True:
            try:
                choice = input("Choice: ").strip().lower()

                if choice == "q":
                    print("Quitting batch review...")
                    return resolved_matches
                elif choice == "s":
                    resolved_matches.append(
                        {
                            "main_row": main_row,
                            "action": "skip",
                            "reason": "manual_skip",
                            "original_index": original_index,  # Include original index
                        }
                    )
                    break
                elif choice == "" and candidates:
                    # Accept highest confidence
                    selected_match = candidates[0]
                    selected_match["method"] = "manual"
                    resolved_matches.append(
                        {
                            "main_row": main_row,
                            "action": "accept",
                            "selected_match": selected_match,
                            "original_index": original_index,  # Include original index
                        }
                    )
                    break
                elif choice.isdigit():
                    choice_num = int(choice)
                    if 1 <= choice_num <= min(len(candidates), 9):
                        selected_match = candidates[choice_num - 1]
                        selected_match["method"] = "manual"
                        resolved_matches.append(
                            {
                                "main_row": main_row,
                                "action": "accept",
                                "selected_match": selected_match,
                                "original_index": original_index,  # Include original index
                            }
                        )
                        break
                    else:
                        print(
                            f"Invalid choice. Please enter 1-{min(len(candidates), 9)}, 's', 'q', or Enter."
                        )
                else:
                    print("Invalid choice. Please enter a number, 's', 'q', or Enter.")
            except KeyboardInterrupt:
                print("\nInterrupted. Quitting batch review...")
                return resolved_matches

    return resolved_matches


def enrich_with_odds_data(
    df: pd.DataFrame, process_all_years: bool = True, enable_manual_review: bool = True
) -> pd.DataFrame:
    """
    Stage 8: Enrich dataset with odds data using sophisticated matching.

    This function integrates as Stage 8 in the existing pipeline.
    """
    print("\n[INFO] Stage 8: Starting odds data enrichment...")

    # Load all odds data
    odds_df = load_all_odds_data()
    if odds_df.empty:
        print("[WARNING] No odds data loaded. Skipping enrichment.")
        return df

    print(f"[INFO] Loaded {len(odds_df)} odds records from 2001-2025")

    # Initialize new columns
    enriched_df = df.copy()

    # Ensure tourney_date is datetime format for date arithmetic
    if "tourney_date" in enriched_df.columns:
        enriched_df["tourney_date"] = pd.to_datetime(
            enriched_df["tourney_date"], errors="coerce"
        )

    for col in NEW_ENRICHMENT_COLUMNS:
        # Use object dtype to allow both numeric and string values
        enriched_df[col] = pd.Series(dtype='object')

    # Process matches
    total_matches = len(enriched_df)
    enriched_count = 0
    uncertain_matches = []
    failed_count = 0

    print(f"[INFO] Processing {total_matches} matches for odds enrichment...")

    for idx, row in tqdm(
        enriched_df.iterrows(), total=total_matches, desc="Enriching matches"
    ):
        result = process_single_match(row, odds_df)

        if result["status"] == "enriched":
            # Apply enrichment
            enriched_row = enrich_match_with_odds(row, result["match"])
            # Update only the enrichment columns to avoid Series assignment issues
            for col in NEW_ENRICHMENT_COLUMNS:
                if col in enriched_row.index:
                    enriched_df.at[idx, col] = enriched_row[col]
            # Also update any ATP data that was filled in
            for col in [
                "player1_rank",
                "player1_atp_points",
                "player2_rank",
                "player2_atp_points",
            ]:
                if col in enriched_row.index and pd.notna(enriched_row[col]):
                    enriched_df.at[idx, col] = enriched_row[col]
            enriched_count += 1
        elif result["status"] == "uncertain":
            # Queue for manual review
            uncertain_matches.append(
                {
                    "main_row": row,
                    "candidates": result["candidates"],
                    "original_index": idx,
                }
            )
        else:
            # No match found - keep original row with failed enrichment method
            enriched_df.loc[idx, "enrichment_method"] = "failed"
            failed_count += 1

    print(
        f"[INFO] Automatic matching completed: {enriched_count} enriched, {len(uncertain_matches)} uncertain, {failed_count} failed"
    )

    # Process uncertain matches if manual review is enabled
    if enable_manual_review and uncertain_matches:
        print(f"[INFO] Processing {len(uncertain_matches)} uncertain matches...")

        # Process in batches
        batch_start = 0
        while batch_start < len(uncertain_matches):
            batch_end = min(batch_start + BATCH_REVIEW_SIZE, len(uncertain_matches))
            batch = uncertain_matches[batch_start:batch_end]

            resolved_matches = interactive_batch_review(batch)

            # Apply resolved matches
            for resolved in resolved_matches:
                idx = resolved.get("original_index")
                if resolved["action"] == "accept":
                    enriched_row = enrich_match_with_odds(
                        resolved["main_row"], resolved["selected_match"]
                    )
                    # Update only the enrichment columns to avoid Series assignment issues
                    for col in NEW_ENRICHMENT_COLUMNS:
                        if col in enriched_row.index:
                            enriched_df.at[idx, col] = enriched_row[col]
                    # Also update any ATP data that was filled in
                    for col in [
                        "player1_rank",
                        "player1_atp_points",
                        "player2_rank",
                        "player2_atp_points",
                    ]:
                        if col in enriched_row.index and pd.notna(enriched_row[col]):
                            enriched_df.at[idx, col] = enriched_row[col]
                    enriched_count += 1
                else:
                    enriched_df.at[idx, "enrichment_method"] = "manual_skip"

            batch_start = batch_end

            # Check if user wants to continue
            if batch_start < len(uncertain_matches):
                continue_choice = (
                    input(
                        f"\nContinue with next batch? ({len(uncertain_matches) - batch_start} matches remaining) [y/n]: "
                    )
                    .strip()
                    .lower()
                )
                if continue_choice != "y":
                    print(
                        "Stopping manual review. Remaining matches marked as uncertain."
                    )
                    # Mark remaining as uncertain
                    for i in range(batch_start, len(uncertain_matches)):
                        idx = uncertain_matches[i].get("original_index")
                        enriched_df.loc[idx, "enrichment_method"] = "uncertain"
                    break

    print(
        f"[SUCCESS] Stage 8 completed: {enriched_count}/{total_matches} matches enriched with odds data"
    )
    return enriched_df


# ============================================================================
# MAIN PIPELINE ORCHESTRATION FUNCTION
# ============================================================================


def process_matches_pipeline(
    input_df: pd.DataFrame,
    replace: bool = False,
    backup_existing: bool = True,
    source_name: str = None,
) -> Dict[str, Any]:
    """
    Main pipeline function for processing new match data through all 9 stages.

    Args:
        input_df: DataFrame with new match data (must include 'source' column)
        replace: Whether to replace duplicate matches
        backup_existing: Whether to create backup before processing
        source_name: Optional source identifier for logging (extracted from DataFrame if not provided)

    Returns:
        Dictionary with processing results and statistics
    """
    start_time = datetime.now()

    # Extract source from DataFrame if not provided
    if source_name is None:
        if "source" not in input_df.columns:
            raise ValueError(
                "DataFrame must contain 'source' column or source_name must be provided"
            )

        # Map numeric source codes to names for logging
        source_mapping_reverse = {0: "main_dataset", 1: "infosys_api", 2: "tennis_api"}
        unique_sources = input_df["source"].unique()

        if len(unique_sources) == 1:
            source_code = unique_sources[0]
            source_name = source_mapping_reverse.get(
                source_code, f"source_{source_code}"
            )
        else:
            source_name = f"mixed_sources_{len(unique_sources)}"

    processing_results = {
        "start_time": start_time.isoformat(),
        "source": source_name,
        "input_rows": len(input_df),
        "rows_added": 0,
        "rows_replaced": 0,
        "duplicates_found": 0,
        "dataset_file": None,
        "backup_file": None,
        "total_rows": 0,
        "success": False,
        "errors": [],
    }

    try:
        print(f"[INFO] Starting Main Matches Dataset Pipeline for {source_name}")
        print(f"[INFO] Processing {len(input_df)} input rows")
        print(f"[DEBUG] Input null tourney_id: {input_df['tourney_id'].isna().sum()}")

        # Stage 1: Initial Stats Calculation and Data Validation
        print("\n[INFO] Stage 1: Validating and preparing input data...")
        validate_input_dataframe(input_df)
        print(
            f"[DEBUG] After validation, null tourney_id: {input_df['tourney_id'].isna().sum()}"
        )

        processed_df = prepare_input_data(input_df.copy())
        print(
            f"[DEBUG] After prepare_input_data, null tourney_id: {processed_df['tourney_id'].isna().sum()}"
        )
        print(f"[SUCCESS] Stage 1 completed: {len(processed_df)} rows prepared")

        # Stage 2: Tournament ID Processing
        print("\n[INFO] Stage 2: Processing tournament IDs...")
        processed_df = process_tournament_ids(processed_df)
        print(f"[SUCCESS] Stage 2 completed: Tournament IDs processed")

        # Stage 3: Player ID Processing
        print("\n[INFO] Stage 3: Processing player IDs...")
        processed_df = process_player_ids(processed_df)
        print(f"[SUCCESS] Stage 3 completed: Player IDs processed")

        # Stage 4: Generate Match IDs and Chronological Keys
        print("\n[INFO] Stage 4: Generating match IDs and chronological keys...")
        processed_df = generate_match_identifiers(processed_df)
        print(
            f"[SUCCESS] Stage 4 completed: {len(processed_df)} match identifiers generated"
        )

        # Stage 5: Load existing dataset and detect duplicates
        print("\n[INFO] Stage 5: Loading existing dataset and detecting duplicates...")
        existing_df = load_main_dataset()
        duplicate_ids, new_matches, duplicate_matches = detect_duplicates(
            processed_df, existing_df
        )

        processing_results["duplicates_found"] = len(duplicate_ids)
        processing_results["rows_added"] = len(new_matches)
        print(
            f"[SUCCESS] Stage 5 completed: {len(duplicate_ids)} duplicates found, {len(new_matches)} new matches"
        )

        # Track match IDs that need odds enrichment (new + replaced matches)
        matches_to_enrich_ids = set()
        if "match_id" in new_matches.columns:
            matches_to_enrich_ids.update(new_matches["match_id"].tolist())

        # Handle duplicates based on replace toggle
        updated_existing = existing_df
        if replace and len(duplicate_matches) > 0:
            print(f"[INFO] Replacing {len(duplicate_matches)} duplicate matches...")
            updated_existing = handle_duplicates(
                existing_df, duplicate_matches, replace=True
            )
            processing_results["rows_replaced"] = len(duplicate_matches)
            # Add replaced matches to enrichment list
            if "match_id" in duplicate_matches.columns:
                matches_to_enrich_ids.update(duplicate_matches["match_id"].tolist())
        elif len(duplicate_matches) > 0:
            print(
                f"[INFO] Keeping existing matches, skipping {len(duplicate_matches)} duplicates"
            )

        # Store the match IDs to enrich for Stage 8
        processing_results["matches_to_enrich_ids"] = list(matches_to_enrich_ids)
        print(
            f"[INFO] {len(matches_to_enrich_ids)} matches will be enriched with odds data"
        )

        # Stage 6: Insert new matches chronologically
        print("\n[INFO] Stage 6: Inserting new matches in chronological order...")
        combined_df = insert_matches_chronologically(updated_existing, new_matches)
        print(
            f"[SUCCESS] Stage 6 completed: {len(combined_df)} total matches in chronological order"
        )

        # Stage 7: Save dataset with backup protocol
        print("\n[INFO] Stage 7: Saving updated dataset with backup protocol...")

        # Create backup if requested
        if backup_existing:
            existing_file = get_latest_dataset_file()
            backup_file = create_backup(existing_file)
            processing_results["backup_file"] = backup_file

        # Save new dataset with timestamp
        new_dataset_file = save_dataset_with_timestamp(combined_df)
        processing_results["dataset_file"] = new_dataset_file
        processing_results["total_rows"] = len(combined_df)

        # Update process log
        log_data = {
            "dataset_file": new_dataset_file,
            "rows_added": processing_results["rows_added"],
            "rows_replaced": processing_results["rows_replaced"],
            "source": source_name,
            "total_rows": processing_results["total_rows"],
            "backup_file": processing_results["backup_file"],
        }
        update_process_log(log_data)

        print(f"[SUCCESS] Stage 7 completed: Dataset saved and logged")

        # Mark as successful
        processing_results["success"] = True
        processing_results["end_time"] = datetime.now().isoformat()

        # Summary
        print(f"\n[SUCCESS] Main Matches Dataset Pipeline completed successfully!")
        print(f"  - Source: {source_name}")
        print(f"  - Input rows: {processing_results['input_rows']}")
        print(f"  - Rows added: {processing_results['rows_added']}")
        print(f"  - Rows replaced: {processing_results['rows_replaced']}")
        print(f"  - Total rows in dataset: {processing_results['total_rows']}")
        print(f"  - Dataset saved: {new_dataset_file}")
        if processing_results["backup_file"]:
            print(f"  - Backup created: {processing_results['backup_file']}")

        return processing_results

    except Exception as e:
        processing_results["success"] = False
        processing_results["errors"].append(str(e))
        processing_results["end_time"] = datetime.now().isoformat()

        print(f"[ERROR] Main Matches Dataset Pipeline failed: {e}")
        return processing_results


def process_matches_pipeline_with_odds(
    input_df: pd.DataFrame,
    replace: bool = False,
    backup_existing: bool = True,
    enrich_odds: bool = True,
    enable_manual_review: bool = True,
    source_name: str = None,
) -> Dict[str, Any]:
    """
    Extended pipeline including Stage 8: Odds Enrichment

    Args:
        input_df: DataFrame with new match data
        replace: Whether to replace duplicate matches
        backup_existing: Whether to create backup before processing
        enrich_odds: Whether to enable odds enrichment (Stage 8)
        enable_manual_review: Whether to enable manual review for uncertain matches
        source_name: Optional source identifier for logging

    Returns:
        Dictionary with processing results and statistics
    """
    print(f"[INFO] Starting Extended Pipeline (with odds enrichment: {enrich_odds})")

    # Run existing Stages 1-7
    results = process_matches_pipeline(input_df, replace, backup_existing, source_name)

    if not results["success"]:
        print("[WARNING] Stages 1-7 failed, skipping odds enrichment")
        return results

    if not enrich_odds:
        print("[INFO] Odds enrichment disabled, pipeline completed")
        return results

    # Stage 8: Odds enrichment
    try:
        print(f"\n[INFO] Loading latest dataset for odds enrichment...")
        latest_dataset_file = results["dataset_file"]
        if not latest_dataset_file or not os.path.exists(latest_dataset_file):
            print("[ERROR] Cannot find dataset file for enrichment")
            results["odds_enrichment"] = False
            results["enrichment_error"] = "Dataset file not found"
            return results

        df = pd.read_csv(latest_dataset_file)
        print(f"[INFO] Loaded {len(df)} total matches from dataset")

        # Get list of match IDs to enrich (only new/replaced matches)
        matches_to_enrich_ids = results.get("matches_to_enrich_ids", [])

        if not matches_to_enrich_ids:
            print("[INFO] No new matches to enrich, skipping odds enrichment")
            results["odds_enrichment"] = False
            results["enrichment_error"] = "No new matches to enrich"
            return results

        print(
            f"[INFO] Will enrich {len(matches_to_enrich_ids)} new/replaced matches only"
        )

        # Filter to only the matches that need enrichment
        matches_to_enrich = df[df["match_id"].isin(matches_to_enrich_ids)].copy()
        print(
            f"[INFO] Filtered to {len(matches_to_enrich)} matches for odds enrichment"
        )

        # Run Stage 8 only on new matches
        enriched_new_matches = enrich_with_odds_data(
            matches_to_enrich, enable_manual_review=enable_manual_review
        )

        # Merge enriched matches back into the full dataset
        # Remove the old versions of these matches
        df_without_enriched = df[~df["match_id"].isin(matches_to_enrich_ids)].copy()

        # Add the newly enriched matches
        enriched_df = pd.concat(
            [df_without_enriched, enriched_new_matches], ignore_index=True
        )

        # Re-sort by chronological order
        if all(
            col in enriched_df.columns
            for col in ["tourney_date", "tourney_id", "round_order"]
        ):
            enriched_df["tourney_date"] = pd.to_datetime(enriched_df["tourney_date"])
            enriched_df["tourney_id"] = pd.to_numeric(
                enriched_df["tourney_id"], errors="coerce"
            ).astype("Int64")
            enriched_df["round_order"] = pd.to_numeric(
                enriched_df["round_order"], errors="coerce"
            ).astype("Int64")
            enriched_df = enriched_df.sort_values(
                ["tourney_date", "tourney_id", "round_order"],
                ascending=[True, True, False],
                na_position="last",
            ).reset_index(drop=True)
            print(f"[INFO] Re-sorted dataset chronologically")

        print(f"[INFO] Final dataset has {len(enriched_df)} matches")

        # Create backup of pre-enrichment dataset if requested
        if backup_existing:
            backup_file = create_backup(latest_dataset_file)
            results["pre_enrichment_backup"] = backup_file

        # Save enriched dataset (creates new timestamped file)
        enriched_file = save_dataset_with_timestamp(enriched_df)

        # Update results
        results["enriched_dataset_file"] = enriched_file
        results["odds_enrichment"] = True
        results["enrichment_success"] = True

        # Count enrichment statistics (only for newly enriched matches)
        enrichment_stats = {
            "total_matches_in_dataset": len(enriched_df),
            "new_matches_processed": len(enriched_new_matches),
            "enriched_matches": len(
                enriched_new_matches[
                    enriched_new_matches["enrichment_method"].notna()
                    & (enriched_new_matches["enrichment_method"] != "failed")
                ]
            ),
            "failed_matches": len(
                enriched_new_matches[
                    enriched_new_matches["enrichment_method"] == "failed"
                ]
            ),
            "manual_reviews": len(
                enriched_new_matches[
                    enriched_new_matches["enrichment_method"] == "manual"
                ]
            ),
            "auto_rank_matches": len(
                enriched_new_matches[
                    enriched_new_matches["enrichment_method"].isin(
                        ["auto_rank", "auto_rank_round"]
                    )
                ]
            ),
            "auto_score_matches": len(
                enriched_new_matches[
                    enriched_new_matches["enrichment_method"] == "auto_score"
                ]
            ),
            "auto_name_matches": len(
                enriched_new_matches[
                    enriched_new_matches["enrichment_method"] == "auto_name"
                ]
            ),
        }
        results["enrichment_stats"] = enrichment_stats

        # Update process log with enrichment info
        log_data = {
            "dataset_file": enriched_file,
            "rows_added": results["rows_added"],
            "rows_replaced": results["rows_replaced"],
            "source": results["source"],
            "total_rows": len(enriched_df),
            "backup_file": results.get("backup_file"),
            "enrichment_stats": enrichment_stats,
        }
        update_process_log(log_data)

        print(f"[SUCCESS] Odds enrichment completed!")
        print(f"  - Enriched dataset saved: {enriched_file}")
        print(
            f"  - Total matches in dataset: {enrichment_stats['total_matches_in_dataset']}"
        )
        print(f"  - New matches processed: {enrichment_stats['new_matches_processed']}")
        print(
            f"  - Successfully enriched: {enrichment_stats['enriched_matches']}/{enrichment_stats['new_matches_processed']}"
        )
        print(f"  - Auto rank matches: {enrichment_stats['auto_rank_matches']}")
        print(f"  - Auto score matches: {enrichment_stats['auto_score_matches']}")
        print(f"  - Auto name matches: {enrichment_stats['auto_name_matches']}")
        print(f"  - Manual reviews: {enrichment_stats['manual_reviews']}")
        print(f"  - Failed matches: {enrichment_stats['failed_matches']}")

        return results

    except Exception as e:
        import traceback

        error_msg = f"{type(e).__name__}: {str(e)}" if str(e) else f"{type(e).__name__}"
        print(f"[ERROR] Odds enrichment failed: {error_msg}")
        print(f"[ERROR] Traceback:\n{traceback.format_exc()}")
        results["odds_enrichment"] = False
        results["enrichment_success"] = False
        results["enrichment_error"] = error_msg
        return results


if __name__ == "__main__":
    # Enable GUI dialogs for tournament matching
    os.environ["TOURNAMENT_MATCHING_GUI"] = "1"

    print("=" * 70)
    print("MAIN MATCHES DATASET PIPELINE")
    print("=" * 70)
    print()
    print("GUI Mode: ENABLED")
    print("Dialog windows will appear for tournament matching confirmations.")
    print("If dialogs don't appear, check your taskbar or try:")
    print("  python run_pipeline_with_gui.py")
    print("=" * 70)
    print()

    input_dataset = pd.read_csv("./data/PreCleanedMatches/CurrentYear.csv")
    process_matches_pipeline_with_odds(
        input_df=input_dataset,
        replace=True,
        backup_existing=True,
        source_name="infosys",
    )
