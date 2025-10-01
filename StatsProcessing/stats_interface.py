#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stats Interface Module

This module provides an easy-to-use interface for interacting with the updateStats.py system.
It allows for flexible input sources (CSV files or dataset portions), duplicate handling,
and prematch stats retrieval with comprehensive filtering options.

Author: Tennis Prediction System
Created: 2025-01-15
Reference: docs/stats-interface-implementation-plan.md
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
import warnings
from tqdm import tqdm
import glob


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================


def load_data_source(
    source: Union[str, pd.DataFrame], source_type: str = "auto", **filters
) -> pd.DataFrame:
    """
    Load data from various sources with optional filtering.

    Args:
        source: Path to CSV file, DataFrame, or 'dataset' for main dataset
        source_type: 'csv', 'dataset', 'dataframe', or 'auto'
        **filters: Filtering options:
            - tourney_date_start: Start date (datetime or str)
            - tourney_date_end: End date (datetime or str)
            - tourney_id: Specific tournament ID or list of IDs
            - round_order: Specific round order or list of orders
            - surface: Surface type(s) to filter
            - tourney_level: Tournament level(s) to filter

    Returns:
        pd.DataFrame: Filtered dataset
    """
    print(f"[INFO] Loading data source: {source}")

    # Auto-detect source type
    if source_type == "auto":
        if isinstance(source, pd.DataFrame):
            source_type = "dataframe"
        elif isinstance(source, str):
            if source.lower() == "dataset" or "matches_dataset_" in source:
                source_type = "dataset"
            else:
                source_type = "csv"
        else:
            raise ValueError(f"Cannot auto-detect source type for: {type(source)}")

    # Load the data
    if source_type == "dataframe":
        df = source.copy()
    elif source_type == "csv":
        if not os.path.exists(source):
            raise FileNotFoundError(f"CSV file not found: {source}")
        df = pd.read_csv(source)
    elif source_type == "dataset":
        if source.lower() == "dataset":
            # Load latest dataset
            df = _load_latest_main_dataset()
        else:
            # Load specific dataset file
            if not os.path.exists(source):
                raise FileNotFoundError(f"Dataset file not found: {source}")
            df = pd.read_csv(source)
    else:
        raise ValueError(f"Unknown source_type: {source_type}")

    if df.empty:
        raise ValueError("Loaded dataset is empty")

    print(f"[INFO] Loaded {len(df)} rows from {source_type} source")

    # Apply filters
    df_filtered = apply_filters(df, **filters)

    print(f"[INFO] After filtering: {len(df_filtered)} rows remaining")
    return df_filtered


def _load_latest_main_dataset() -> pd.DataFrame:
    """
    Load the most recent main dataset from databases/matches_main/.

    Returns:
        pd.DataFrame: Most recent dataset
    """
    dataset_files = glob.glob("databases/matches_main/matches_dataset_*.csv")
    if not dataset_files:
        raise FileNotFoundError("No dataset files found in databases/matches_main/")

    latest_file = max(dataset_files, key=os.path.getctime)
    print(f"[INFO] Loading latest dataset: {latest_file}")

    df = pd.read_csv(latest_file)

    # Ensure proper data types for sorting columns
    if "tourney_date" in df.columns:
        df["tourney_date"] = pd.to_datetime(df["tourney_date"])
    if "tourney_id" in df.columns:
        df["tourney_id"] = pd.to_numeric(df["tourney_id"], errors="coerce").astype(
            "Int64"
        )
    if "round_order" in df.columns:
        df["round_order"] = pd.to_numeric(df["round_order"], errors="coerce").astype(
            "Int64"
        )

    print(f"[INFO] Loaded existing dataset: {len(df)} rows")
    return df


def apply_filters(df: pd.DataFrame, **filters) -> pd.DataFrame:
    """
    Apply various filters to the dataset.

    Args:
        df: Input DataFrame
        **filters: Filter criteria

    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    filtered_df = df.copy()

    # Ensure date column is datetime
    if "tourney_date" in filtered_df.columns:
        filtered_df["tourney_date"] = pd.to_datetime(filtered_df["tourney_date"])

    # Date filters
    if "tourney_date_start" in filters:
        start_date = pd.to_datetime(filters["tourney_date_start"])
        filtered_df = filtered_df[filtered_df["tourney_date"] >= start_date]
        print(f"[FILTER] Applied start date filter: {start_date}")

    if "tourney_date_end" in filters:
        end_date = pd.to_datetime(filters["tourney_date_end"])
        filtered_df = filtered_df[filtered_df["tourney_date"] <= end_date]
        print(f"[FILTER] Applied end date filter: {end_date}")

    # Tournament ID filter
    if "tourney_id" in filters:
        tourney_ids = filters["tourney_id"]
        if not isinstance(tourney_ids, (list, tuple)):
            tourney_ids = [tourney_ids]
        filtered_df = filtered_df[filtered_df["tourney_id"].isin(tourney_ids)]
        print(f"[FILTER] Applied tournament ID filter: {tourney_ids}")

    # Round order filter
    if "round_order" in filters:
        round_orders = filters["round_order"]
        if not isinstance(round_orders, (list, tuple)):
            round_orders = [round_orders]
        if "round_order" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["round_order"].isin(round_orders)]
            print(f"[FILTER] Applied round order filter: {round_orders}")
        else:
            print(f"[WARNING] round_order column not found, skipping filter")

    # Surface filter
    if "surface" in filters:
        surfaces = filters["surface"]
        if not isinstance(surfaces, (list, tuple)):
            surfaces = [surfaces]
        if "surface" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["surface"].isin(surfaces)]
            print(f"[FILTER] Applied surface filter: {surfaces}")
        else:
            print(f"[WARNING] surface column not found, skipping filter")

    # Tournament level filter
    if "tourney_level" in filters:
        levels = filters["tourney_level"]
        if not isinstance(levels, (list, tuple)):
            levels = [levels]
        if "tourney_level" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["tourney_level"].isin(levels)]
            print(f"[FILTER] Applied tournament level filter: {levels}")
        else:
            print(f"[WARNING] tourney_level column not found, skipping filter")

    return filtered_df


def validate_match_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and prepare match data for stats processing.

    Args:
        df: Input DataFrame

    Returns:
        pd.DataFrame: Validated and prepared DataFrame
    """
    print("[INFO] Validating match data for stats processing...")

    validated_df = df.copy()

    # Required columns for updateStats
    required_cols = [
        "player1_id",
        "player2_id",
        "surface",
        "RESULT",
        "round",
        "tourney_level",
        "tourney_id",
        "match_id",
        "chronological_key",
    ]

    missing_cols = [col for col in required_cols if col not in validated_df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns for stats processing: {missing_cols}"
        )

    # Ensure proper data types
    validated_df["player1_id"] = pd.to_numeric(
        validated_df["player1_id"], errors="coerce"
    )
    validated_df["player2_id"] = pd.to_numeric(
        validated_df["player2_id"], errors="coerce"
    )
    validated_df["tourney_id"] = pd.to_numeric(
        validated_df["tourney_id"], errors="coerce"
    )
    validated_df["RESULT"] = pd.to_numeric(validated_df["RESULT"], errors="coerce")

    # Parse chronological_key if it's a string
    if validated_df["chronological_key"].dtype == object:
        print("[INFO] Parsing chronological_key from string format...")

        def parse_chrono_key_safe(key_str):
            """Parse chronological key from string with datetime.datetime() format."""
            import ast
            import re
            from datetime import datetime

            if pd.isna(key_str) or key_str is None:
                return None

            if isinstance(key_str, tuple):
                return key_str

            try:
                # Handle format: "(datetime.datetime(1990, 12, 31, 0, 0), 2000, 32)"
                if "datetime.datetime" in str(key_str):
                    # Extract datetime components and other values
                    match = re.match(r'\(datetime\.datetime\((\d+),\s*(\d+),\s*(\d+)(?:,\s*\d+)*\),\s*(\d+),\s*(\d+)\)', str(key_str))
                    if match:
                        year, month, day, tourney_id, round_order = match.groups()
                        dt = datetime(int(year), int(month), int(day))
                        return (dt, int(tourney_id), int(round_order))

                # Try standard parsing
                parsed = ast.literal_eval(str(key_str))
                if isinstance(parsed, (tuple, list)) and len(parsed) >= 3:
                    if isinstance(parsed[0], str):
                        dt = pd.to_datetime(parsed[0]).to_pydatetime()
                        return (dt, int(parsed[1]), int(parsed[2]))
                    return tuple(parsed[:3])
            except:
                pass

            return None

        validated_df["chronological_key"] = validated_df["chronological_key"].apply(
            parse_chrono_key_safe
        )

    # Remove rows with invalid essential data
    before_count = len(validated_df)
    validated_df = validated_df.dropna(
        subset=["player1_id", "player2_id", "match_id", "chronological_key"]
    )
    after_count = len(validated_df)

    if before_count != after_count:
        print(
            f"[WARNING] Removed {before_count - after_count} rows with invalid essential data"
        )

    print(f"[SUCCESS] Validation completed: {len(validated_df)} valid rows")
    return validated_df


# ============================================================================
# CORE PROCESSING FUNCTIONS
# ============================================================================


def update_stats_from_source(
    source: Union[str, pd.DataFrame],
    source_type: str = "auto",
    rewrite_duplicates: bool = False,
    progress_bar: bool = True,
    save_stats: bool = True,
    **filters,
) -> Dict[str, Any]:
    """
    Update player statistics from a data source with duplicate handling.

    Args:
        source: Data source (CSV path, DataFrame, or 'dataset')
        source_type: 'csv', 'dataset', 'dataframe', or 'auto'
        rewrite_duplicates: If True, overwrite existing data for duplicate match_ids
        progress_bar: Show progress bar for processing
        save_stats: Save statistics after processing
        **filters: Data filtering options

    Returns:
        Dict with processing results and statistics
    """
    from utils.updateStats import createStats, updateStats, saveStats, getStatsInfo
    from utils.match_tracking import is_match_processed

    print(f"[INFO] Starting stats update from source")
    print(
        f"[INFO] Duplicate handling mode: {'REWRITE' if rewrite_duplicates else 'SKIP'}"
    )

    start_time = datetime.now()
    results = {
        "start_time": start_time.isoformat(),
        "source": str(source),
        "rewrite_duplicates": rewrite_duplicates,
        "processed_matches": 0,
        "skipped_duplicates": 0,
        "overwritten_matches": 0,
        "errors": [],
        "success": False,
    }

    try:
        # Load and validate data
        df = load_data_source(source, source_type, **filters)
        df = validate_match_data(df)

        if df.empty:
            print("[WARNING] No valid matches found after filtering and validation")
            return results

        # Sort by chronological order before processing matches
        # This ensures matches are processed in the correct chronological sequence
        df = df.sort_values(
            ["tourney_date", "tourney_id", "round_order"], ascending=[True, True, False]
        ).reset_index(drop=True)
        print(f"[INFO] Sorted {len(df)} matches chronologically before processing")

        # Load or create stats
        prev_stats = createStats(load_existing=True)
        results["initial_stats"] = getStatsInfo(prev_stats)

        # Process matches
        iterator = (
            tqdm(df.iterrows(), total=len(df), desc="Processing matches")
            if progress_bar
            else df.iterrows()
        )

        for idx, match_row in iterator:
            try:
                match_id = match_row["match_id"]

                # Check if match already processed
                if is_match_processed(match_id, prev_stats["processed_matches"]):
                    if rewrite_duplicates:
                        # Remove existing data for this match from ChronologicalDeques
                        prev_stats = remove_match_from_chronological_data(
                            prev_stats, match_row
                        )
                        results["overwritten_matches"] += 1
                        if progress_bar:
                            iterator.set_postfix(
                                rewritten=results["overwritten_matches"]
                            )
                    else:
                        results["skipped_duplicates"] += 1
                        if progress_bar:
                            iterator.set_postfix(skipped=results["skipped_duplicates"])
                        continue

                # Update stats
                prev_stats = updateStats(match_row, prev_stats)
                results["processed_matches"] += 1

                if progress_bar:
                    iterator.set_postfix(
                        processed=results["processed_matches"],
                        skipped=results["skipped_duplicates"],
                        overwritten=results["overwritten_matches"],
                    )

            except Exception as e:
                import traceback

                error_msg = f"Error processing match {match_row.get('match_id', 'unknown')}: {str(e)}"
                results["errors"].append(error_msg)
                print(f"[ERROR] {error_msg}")
                print("=== FULL TRACEBACK ===")
                traceback.print_exc()
                print("=== END TRACEBACK ===")
                continue

        # Save stats if requested
        if save_stats:
            saveStats(prev_stats)
            print("[INFO] Statistics saved successfully")

        # Set success flag before Elo processing
        results["success"] = True

        # AUTOMATIC: Elo processing integration (always enabled)
        if results["success"]:
            try:
                from utils.elo_manager import integrate_with_stats_interface

                print("[INFO] Starting Elo processing...")

                # Process Elo ratings (no column mapping needed - pipeline uses player1_id/player2_id)
                elo_results = integrate_with_stats_interface(
                    df, rewrite_duplicates=rewrite_duplicates, progress_bar=progress_bar
                )

                results["elo_processing"] = elo_results

                if elo_results["success"]:
                    print(
                        f"[SUCCESS] Elo processing completed: {elo_results['processed_matches']} matches"
                    )
                    if elo_results.get("recalculation_triggered"):
                        print(
                            f"[INFO] Elo recalculation triggered from: {elo_results['recalc_from_key']}"
                        )
                else:
                    print(
                        f"[WARNING] Elo processing failed: {elo_results.get('error', 'Unknown error')}"
                    )

            except Exception as e:
                print(f"[WARNING] Elo processing failed: {e}")
                results["elo_processing"] = {"success": False, "error": str(e)}

        results["final_stats"] = getStatsInfo(prev_stats)
        results["end_time"] = datetime.now().isoformat()

        # Summary
        print(f"\n[SUCCESS] Stats update completed!")
        print(f"  - Processed matches: {results['processed_matches']}")
        print(f"  - Skipped duplicates: {results['skipped_duplicates']}")
        print(f"  - Overwritten matches: {results['overwritten_matches']}")
        print(f"  - Errors: {len(results['errors'])}")

        return results

    except Exception as e:
        results["success"] = False
        results["errors"].append(str(e))
        results["end_time"] = datetime.now().isoformat()
        print(f"[ERROR] Stats update failed: {e}")
        import traceback

        print("=== FULL TRACEBACK ===")
        traceback.print_exc()
        print("=== END TRACEBACK ===")
        return results


def update_stats_from_source_batch(
    source: Union[str, pd.DataFrame],
    source_type: str = "auto",
    rewrite_duplicates: bool = False,
    progress_bar: bool = True,
    batch_size: int = 10000,
    save_stats: bool = True,
    **filters,
) -> Dict[str, Any]:
    """
    Enhanced stats update function with optimized Elo batch processing for large datasets.

    Args:
        source: Data source (CSV path, DataFrame, or 'dataset')
        source_type: 'csv', 'dataset', 'dataframe', or 'auto'
        rewrite_duplicates: If True, overwrite existing data for duplicate match_ids
        progress_bar: Show progress bar for processing
        elo_batch_size: Batch size for Elo processing (optimized for large datasets)
        save_stats: Save statistics after processing
        **filters: Data filtering options

    Returns:
        Dict with processing results and statistics
    """
    from utils.updateStats import createStats, updateStats, saveStats, getStatsInfo
    from utils.match_tracking import is_match_processed
    from utils.elo_manager import integrate_with_stats_interface, process_elo_batch

    print(f"[INFO] Starting enhanced stats update with Elo batch processing")
    print(f"[INFO] Elo batch size: {batch_size}")
    print(
        f"[INFO] Duplicate handling mode: {'REWRITE' if rewrite_duplicates else 'SKIP'}"
    )

    start_time = datetime.now()
    results = {
        "start_time": start_time.isoformat(),
        "source": str(source),
        "rewrite_duplicates": rewrite_duplicates,
        "elo_batch_size": batch_size,
        "processed_matches": 0,
        "skipped_duplicates": 0,
        "overwritten_matches": 0,
        "errors": [],
        "success": False,
    }

    try:
        # Load and validate data (reuse if already a DataFrame)
        if isinstance(source, pd.DataFrame):
            df = source.copy()
        else:
            df = load_data_source(source, source_type, **filters)
            df = validate_match_data(df)

        if df.empty:
            print("[WARNING] No valid matches found after filtering and validation")
            return results

        # Sort by chronological order before processing matches
        df = df.sort_values(
            ["tourney_date", "tourney_id", "round_order"], ascending=[True, True, False]
        ).reset_index(drop=True)
        print(f"[INFO] Sorted {len(df)} matches chronologically for batch processing")

        # Load or create stats
        prev_stats = createStats(load_existing=True)
        results["initial_stats"] = getStatsInfo(prev_stats)

        # Process matches in batches for better memory management
        batch_size = min(5000, len(df) // 4)  # Adaptive batch size for stats processing
        total_matches = len(df)
        processed_count = 0

        print(f"[INFO] Processing {total_matches} matches in batches of {batch_size}")

        for batch_start in range(0, total_matches, batch_size):
            batch_end = min(batch_start + batch_size, total_matches)
            batch_df = df.iloc[batch_start:batch_end].copy()

            print(
                f"[INFO] Processing batch {batch_start // batch_size + 1}: matches {batch_start + 1}-{batch_end}"
            )

            # Process this batch of matches for stats
            iterator = (
                tqdm(
                    batch_df.iterrows(),
                    total=len(batch_df),
                    desc=f"Processing batch {batch_start // batch_size + 1}",
                )
                if progress_bar
                else batch_df.iterrows()
            )

            for idx, match_row in iterator:
                try:
                    match_id = match_row["match_id"]

                    # Check if match already processed
                    if is_match_processed(match_id, prev_stats["processed_matches"]):
                        if rewrite_duplicates:
                            prev_stats = remove_match_from_chronological_data(
                                prev_stats, match_row
                            )
                            results["overwritten_matches"] += 1
                        else:
                            results["skipped_duplicates"] += 1
                            continue

                    # Update stats
                    prev_stats = updateStats(match_row, prev_stats)
                    results["processed_matches"] += 1
                    processed_count += 1

                except Exception as e:
                    error_msg = f"Error processing match {match_row.get('match_id', 'unknown')}: {str(e)}"
                    results["errors"].append(error_msg)
                    print(f"[ERROR] {error_msg}")
                    continue

        # Save stats after batch processing
        if save_stats:
            saveStats(prev_stats)
            print("[INFO] Statistics saved successfully")

        # Enhanced Elo processing with larger batch sizes for large datasets
        print(
            f"\n[INFO] Starting enhanced Elo batch processing with batch size: {batch_size}"
        )
        try:
            from utils.elo_manager import (
                create_elo_ratings,
                process_elo_batch,
                save_elo_ratings_simple,
            )

            # Load existing Elo ratings
            existing_elo_ratings = create_elo_ratings(load_existing=True)

            # Process Elo in larger batches for efficiency
            updated_elo_ratings = process_elo_batch(
                df, existing_elo_ratings, batch_size=batch_size, progress_callback=None
            )

            # Save Elo ratings
            if save_elo_ratings_simple(updated_elo_ratings):
                elo_results = {
                    "success": True,
                    "processed_matches": len(df),
                    "batch_size_used": batch_size,
                    "recalculation_triggered": False,
                }
                print(
                    f"[SUCCESS] Enhanced Elo batch processing completed: {len(df)} matches"
                )
            else:
                elo_results = {"success": False, "error": "Failed to save Elo ratings"}

        except Exception as e:
            elo_results = {"success": False, "error": str(e)}
            print(f"[WARNING] Enhanced Elo batch processing failed: {e}")

        results["elo_processing"] = elo_results
        results["final_stats"] = getStatsInfo(prev_stats)
        results["success"] = True
        results["end_time"] = datetime.now().isoformat()

        # Summary
        print(f"\n[SUCCESS] Enhanced stats update with Elo batch processing completed!")
        print(f"  - Processed matches: {results['processed_matches']}")
        print(f"  - Skipped duplicates: {results['skipped_duplicates']}")
        print(f"  - Overwritten matches: {results['overwritten_matches']}")
        print(f"  - Elo batch size used: {batch_size}")
        print(f"  - Errors: {len(results['errors'])}")

        return results

    except Exception as e:
        results["success"] = False
        results["errors"].append(str(e))
        results["end_time"] = datetime.now().isoformat()
        print(f"[ERROR] Enhanced stats update failed: {e}")
        return results


def get_prematch_stats_from_source(
    source: Union[str, pd.DataFrame],
    source_type: str = "auto",
    stats_mode: str = "prematch",
    progress_bar: bool = True,
    include_metadata: bool = True,
    **filters,
) -> pd.DataFrame:
    """
    Get statistics for matches with flexible timing options.

    Args:
        source: Data source (CSV path, DataFrame, or 'dataset')
        source_type: 'csv', 'dataset', 'dataframe', or 'auto'
        stats_mode: 'prematch', 'current', or 'both'
        progress_bar: Show progress bar for processing
        include_metadata: Include match metadata in results
        **filters: Data filtering options

    Returns:
        pd.DataFrame: DataFrame with stats for each match
    """
    from utils.updateStats import createStats, getStats

    print(f"[INFO] Getting {stats_mode} stats from source")

    try:
        # Load and validate data
        df = load_data_source(source, source_type, **filters)
        df = validate_match_data(df)

        if df.empty:
            print("[WARNING] No valid matches found")
            return pd.DataFrame()

        # Load stats and Elo ratings (load once for efficiency)
        prev_stats = createStats(load_existing=True)

        # Load Elo ratings once for batch processing efficiency
        from utils.elo_manager import create_elo_ratings

        elo_ratings = create_elo_ratings(load_existing=True)

        # Sort by chronological order to ensure proper before_time filtering
        # round_order should be descending (higher numbers = earlier rounds)
        df_sorted = df.sort_values(
            ["tourney_date", "tourney_id", "round_order"], ascending=[True, True, False]
        ).reset_index(drop=True)

        results_list = []
        iterator = (
            tqdm(
                df_sorted.iterrows(),
                total=len(df_sorted),
                desc=f"Getting {stats_mode} stats",
            )
            if progress_bar
            else df_sorted.iterrows()
        )

        for idx, match_row in iterator:
            try:
                # Prepare player data
                player1 = {
                    "ID": match_row["player1_id"],
                    "ATP_POINTS": match_row.get("p1_atp_points", 0),
                    "ATP_RANK": match_row.get("p1_atp_rank", 999),
                    "AGE": match_row.get("p1_age", 25),
                    "HEIGHT": match_row.get("p1_height", 180),
                }

                player2 = {
                    "ID": match_row["player2_id"],
                    "ATP_POINTS": match_row.get("p2_atp_points", 0),
                    "ATP_RANK": match_row.get("p2_atp_rank", 999),
                    "AGE": match_row.get("p2_age", 25),
                    "HEIGHT": match_row.get("p2_height", 180),
                }

                # Prepare match context
                match_context = {
                    "SURFACE": match_row.get("surface", "Hard"),
                    "ROUND": match_row.get("round", "R32"),
                    "TOURNEY_ID": match_row["tourney_id"],
                    "TOURNEY_LEVEL": match_row.get("tourney_level", "ATP250"),
                    "BEST_OF": match_row.get("best_of", 3),
                    "DRAW_SIZE": match_row.get("draw_size", 32),
                    "TOURNEY_DATE": match_row["tourney_date"],
                }

                # Get stats based on mode
                if stats_mode == "prematch":
                    # Stats before this match's chronological key
                    before_time = match_row["chronological_key"]
                    stats = getStats(
                        player1,
                        player2,
                        match_context,
                        prev_stats,
                        before_time=before_time,
                        elo_ratings=elo_ratings,
                    )
                elif stats_mode == "current":
                    # Most up-to-date stats available
                    stats = getStats(
                        player1,
                        player2,
                        match_context,
                        prev_stats,
                        before_time=None,
                        elo_ratings=elo_ratings,
                    )
                elif stats_mode == "both":
                    # Get both prematch and current stats
                    before_time = match_row["chronological_key"]
                    prematch_stats = getStats(
                        player1,
                        player2,
                        match_context,
                        prev_stats,
                        before_time=before_time,
                        elo_ratings=elo_ratings,
                    )
                    current_stats = getStats(
                        player1,
                        player2,
                        match_context,
                        prev_stats,
                        before_time=None,
                        elo_ratings=elo_ratings,
                    )

                    # Combine with prefixes
                    stats = {}
                    for key, value in prematch_stats.items():
                        stats[f"prematch_{key}"] = value
                    for key, value in current_stats.items():
                        stats[f"current_{key}"] = value
                else:
                    raise ValueError(f"Unknown stats_mode: {stats_mode}")

                # Add match metadata if requested
                if include_metadata:
                    stats["match_id"] = match_row["match_id"]
                    stats["tourney_date"] = match_row["tourney_date"]
                    stats["tourney_id"] = match_row["tourney_id"]
                    stats["round"] = match_row["round"]
                    stats["surface"] = match_row["surface"]
                    stats["player1_id"] = match_row["player1_id"]
                    stats["player2_id"] = match_row["player2_id"]
                    stats["actual_result"] = match_row["RESULT"]

                    # Add odds columns
                    stats["p1_odds"] = match_row.get("p1_odds", None)
                    stats["p2_odds"] = match_row.get("p2_odds", None)
                    stats["p1_max_odds"] = match_row.get("p1_max_odds", None)
                    stats["p2_max_odds"] = match_row.get("p2_max_odds", None)

                    # Add best_of and draw_size from original data
                    stats["best_of"] = match_row.get("best_of", None)
                    stats["draw_size"] = match_row.get("draw_size", None)

                    # Add ATP rank and points
                    stats["p1_atp_rank"] = match_row.get("player1_rank", None)
                    stats["p1_atp_points"] = match_row.get("player1_atp_points", None)
                    stats["p2_atp_rank"] = match_row.get("player2_rank", None)
                    stats["p2_atp_points"] = match_row.get("player2_atp_points", None)

                    # Add age and height
                    stats["p1_age"] = match_row.get("p1_age", None)
                    stats["p2_age"] = match_row.get("p2_age", None)
                    stats["p1_height"] = match_row.get("p1_height", None)
                    stats["p2_height"] = match_row.get("p2_height", None)

                results_list.append(stats)

            except Exception as e:
                print(
                    f"[ERROR] Failed to get stats for match {match_row.get('match_id', 'unknown')}: {e}"
                )
                continue

        # Convert to DataFrame
        results_df = pd.DataFrame(results_list)

        print(f"[SUCCESS] Retrieved {stats_mode} stats for {len(results_df)} matches")
        return results_df

    except Exception as e:
        print(f"[ERROR] Failed to get {stats_mode} stats: {e}")
        return pd.DataFrame()


# ============================================================================
# DUPLICATE MANAGEMENT
# ============================================================================


def remove_match_from_chronological_data(
    prev_stats: Dict, match_row: pd.Series
) -> Dict:
    """
    Remove specific match data from ChronologicalDeques when rewriting duplicates.

    Args:
        prev_stats: Current statistics dictionary
        match_row: Match data to remove

    Returns:
        Dict: Updated statistics dictionary
    """
    from utils.common import getWinnerLoserIDS

    print(f"[INFO] Removing match data for rewrite: {match_row['match_id']}")

    try:
        # Get match details
        match_id = match_row["match_id"]
        sort_key = match_row["chronological_key"]
        player1_id, player2_id = match_row["player1_id"], match_row["player2_id"]
        result = match_row["RESULT"]
        w_id, l_id = getWinnerLoserIDS(player1_id, player2_id, result)

        # Parse the sort key if it's a string
        if isinstance(sort_key, str):
            from utils.chronological_storage import generate_chronological_key
            import pandas as pd
            import re

            # Try to parse the string manually to avoid eval() issues
            try:
                # Extract components using regex: (YYYY-MM-DD, number, number)
                match = re.match(r"\((\d{4}-\d{2}-\d{2}),\s*(\d+),\s*(\d+)\)", sort_key)
                if match:
                    date_str, tourney_id, round_order = match.groups()
                    date_obj = pd.to_datetime(date_str).to_pydatetime()
                    parsed_key = (date_obj, int(tourney_id), int(round_order))
                    sort_key = parsed_key
                else:
                    raise ValueError("Regex parsing failed")
            except Exception as e:
                print(
                    f"[DEBUG] Failed to parse sort key ({e}), regenerating from match data"
                )
                sort_key = generate_chronological_key(match_row)

        # Remove from processed matches
        if match_id in prev_stats["processed_matches"]:
            del prev_stats["processed_matches"][match_id]

        # Remove from all ChronologicalDeques using enhanced methods
        chronological_stats = [
            "last_k_matches",
            "matches_played",
            "championships",
            "games_diff",
            "last_150_round_results",
        ]

        total_removed = 0
        for stat_name in chronological_stats:
            if w_id in prev_stats[stat_name]:
                removed = prev_stats[stat_name][w_id].remove_by_sort_key(sort_key)
                total_removed += removed
            if l_id in prev_stats[stat_name]:
                removed = prev_stats[stat_name][l_id].remove_by_sort_key(sort_key)
                total_removed += removed

        # Remove from nested ChronologicalDeques
        nested_stats = [
            "last_k_matches_stats",
            "tourney_history",
            "level_history",
            "round_history",
        ]

        for stat_name in nested_stats:
            for player_id in [w_id, l_id]:
                if player_id in prev_stats[stat_name]:
                    for sub_key in prev_stats[stat_name][player_id]:
                        removed = prev_stats[stat_name][player_id][
                            sub_key
                        ].remove_by_sort_key(sort_key)
                        total_removed += removed

        # Remove from chronological H2H data (both directions)
        surface = match_row.get("surface", "Hard")

        # Remove from general H2H ChronologicalDeques
        if w_id in prev_stats["h2h"] and l_id in prev_stats["h2h"][w_id]:
            removed = prev_stats["h2h"][w_id][l_id].remove_by_sort_key(sort_key)
            total_removed += removed
        if l_id in prev_stats["h2h"] and w_id in prev_stats["h2h"][l_id]:
            removed = prev_stats["h2h"][l_id][w_id].remove_by_sort_key(sort_key)
            total_removed += removed

        # Remove from surface-specific H2H ChronologicalDeques
        if surface in prev_stats["h2h_surface"]:
            if (
                w_id in prev_stats["h2h_surface"][surface]
                and l_id in prev_stats["h2h_surface"][surface][w_id]
            ):
                removed = prev_stats["h2h_surface"][surface][w_id][
                    l_id
                ].remove_by_sort_key(sort_key)
                total_removed += removed
            if (
                l_id in prev_stats["h2h_surface"][surface]
                and w_id in prev_stats["h2h_surface"][surface][l_id]
            ):
                removed = prev_stats["h2h_surface"][surface][l_id][
                    w_id
                ].remove_by_sort_key(sort_key)
                total_removed += removed

        print(
            f"[SUCCESS] Removed {total_removed} chronological entries for match: {match_id}"
        )

    except Exception as e:
        print(f"[WARNING] Could not fully remove match data: {e}")

    return prev_stats


# ============================================================================
# BATCH PROCESSING
# ============================================================================


def update_getstats_batch(
    source: Union[str, pd.DataFrame],
    source_type: str = "auto",
    batch_size: int = 5000,
    progress_bar: bool = True,
    **filters,
) -> pd.DataFrame:
    """
    Process matches in chronological order, getting prematch stats then updating stats for each row.

    This function processes data in batches where each row gets prematch stats (using current state)
    then immediately updates the stats, so subsequent rows have access to the updated statistics.
    The data must be chronologically after the latest existing stats.

    Args:
        source: Data source (CSV path, DataFrame, or 'dataset')
        source_type: Source type ('csv', 'dataset', 'dataframe', or 'auto')
        batch_size: Number of matches to process per batch (default: 5000)
        progress_bar: Show progress bar for processing
        **filters: Data filtering options

    Returns:
        pd.DataFrame: Prematch stats for each match (similar to get_prematch_stats_from_source)

    Raises:
        ValueError: If input data precedes existing stats chronologically
    """
    from utils.updateStats import createStats, updateStats, saveStats, getStats
    from utils.elo_manager import create_elo_ratings

    print(f"[INFO] Starting update_getstats_batch processing")
    start_time = datetime.now()

    try:
        # Load and validate data
        df = load_data_source(source, source_type, **filters)
        df = validate_match_data(df)

        if df.empty:
            print("[WARNING] No valid matches found after filtering and validation")
            return pd.DataFrame()

        # Sort by chronological order
        df = df.sort_values(
            ["tourney_date", "tourney_id", "round_order"], ascending=[True, True, False]
        ).reset_index(drop=True)
        print(f"[INFO] Sorted {len(df)} matches chronologically")

        # Load existing stats and Elo ratings
        prev_stats = createStats(load_existing=True)
        elo_ratings = create_elo_ratings(load_existing=True)

        # Validate chronological order against existing stats
        if prev_stats.get("processed_matches"):
            # Get the latest chronological key from existing stats
            existing_matches = prev_stats["processed_matches"]
            if existing_matches:
                # Find the latest chronological key
                latest_keys = []
                for match_data in existing_matches.values():
                    if "chronological_key" in match_data:
                        latest_keys.append(match_data["chronological_key"])

                if latest_keys:
                    # Parse and compare chronological keys
                    from utils.chronological_storage import parse_chronological_key

                    latest_parsed = max(
                        [parse_chronological_key(key) for key in latest_keys]
                    )
                    first_input_key = parse_chronological_key(
                        df.iloc[0]["chronological_key"]
                    )

                    if first_input_key <= latest_parsed:
                        raise ValueError(
                            f"Input data chronologically precedes existing stats. "
                            f"Latest existing: {latest_parsed}, First input: {first_input_key}"
                        )

        print(f"[INFO] Chronological validation passed")

        # Initialize results collection
        results_list = []
        total_processed = 0

        # Set up single progress bar for all rows
        total_rows = len(df)
        iterator = (
            tqdm(
                range(0, total_rows, batch_size),
                desc="Processing batches",
                unit="batch",
            )
            if progress_bar
            else range(0, total_rows, batch_size)
        )

        # Process in batches
        for batch_start in iterator:
            batch_end = min(batch_start + batch_size, total_rows)
            batch_df = df.iloc[batch_start:batch_end].copy()

            # Process each row in the batch
            for idx, match_row in batch_df.iterrows():
                try:
                    # Prepare player data
                    player1 = {
                        "ID": match_row["player1_id"],
                        "ATP_POINTS": match_row.get("p1_atp_points", 0),
                        "ATP_RANK": match_row.get("p1_atp_rank", 999),
                        "AGE": match_row.get("p1_age", 25),
                        "HEIGHT": match_row.get("p1_height", 180),
                    }

                    player2 = {
                        "ID": match_row["player2_id"],
                        "ATP_POINTS": match_row.get("p2_atp_points", 0),
                        "ATP_RANK": match_row.get("p2_atp_rank", 999),
                        "AGE": match_row.get("p2_age", 25),
                        "HEIGHT": match_row.get("p2_height", 180),
                    }

                    # Prepare match context
                    match_context = {
                        "SURFACE": match_row.get("surface", "Hard"),
                        "ROUND": match_row.get("round", "R32"),
                        "TOURNEY_ID": match_row["tourney_id"],
                        "TOURNEY_LEVEL": match_row.get("tourney_level", "ATP250"),
                        "BEST_OF": match_row.get("best_of", 3),
                        "DRAW_SIZE": match_row.get("draw_size", 32),
                        "TOURNEY_DATE": match_row["tourney_date"],
                    }

                    # Get prematch stats (no before_time needed since data is already sorted)
                    stats = getStats(
                        player1,
                        player2,
                        match_context,
                        prev_stats,
                        before_time=None,  # Use current state since data is pre-sorted
                        elo_ratings=elo_ratings,
                    )

                    # Add match metadata
                    stats["match_id"] = match_row["match_id"]
                    stats["tourney_date"] = match_row["tourney_date"]
                    stats["tourney_id"] = match_row["tourney_id"]
                    stats["round"] = match_row["round"]
                    stats["surface"] = match_row["surface"]
                    stats["player1_id"] = match_row["player1_id"]
                    stats["player2_id"] = match_row["player2_id"]
                    stats["actual_result"] = match_row["RESULT"]

                    # Add odds columns
                    stats["p1_odds"] = match_row.get("p1_odds", None)
                    stats["p2_odds"] = match_row.get("p2_odds", None)
                    stats["p1_max_odds"] = match_row.get("p1_max_odds", None)
                    stats["p2_max_odds"] = match_row.get("p2_max_odds", None)

                    # Add best_of and draw_size from original data
                    stats["best_of"] = match_row.get("best_of", None)
                    stats["draw_size"] = match_row.get("draw_size", None)

                    # Add ATP rank and points
                    stats["p1_atp_rank"] = match_row.get("player1_rank", None)
                    stats["p1_atp_points"] = match_row.get("player1_atp_points", None)
                    stats["p2_atp_rank"] = match_row.get("player2_rank", None)
                    stats["p2_atp_points"] = match_row.get("player2_atp_points", None)

                    # Add age and height
                    stats["p1_age"] = match_row.get("p1_age", None)
                    stats["p2_age"] = match_row.get("p2_age", None)
                    stats["p1_height"] = match_row.get("p1_height", None)
                    stats["p2_height"] = match_row.get("p2_height", None)

                    results_list.append(stats)

                    # Immediately update stats for next row
                    prev_stats = updateStats(match_row, prev_stats)
                    total_processed += 1

                    # Update progress bar description
                    if progress_bar and hasattr(iterator, "set_postfix"):
                        iterator.set_postfix(processed=total_processed)

                except Exception as e:
                    print(
                        f"[ERROR] Failed to process match {match_row.get('match_id', 'unknown')}: {e}"
                    )
                    continue

        # Save updated stats automatically
        saveStats(prev_stats)
        print(
            f"[INFO] Statistics saved successfully after processing {total_processed} matches"
        )

        # Convert results to DataFrame
        results_df = pd.DataFrame(results_list)

        end_time = datetime.now()
        runtime = (end_time - start_time).total_seconds()

        print(f"[SUCCESS] update_getstats_batch completed!")
        print(f"  - Processed matches: {total_processed}")
        print(f"  - Runtime: {runtime:.2f} seconds")
        print(f"  - Matches per second: {total_processed / runtime:.2f}")

        return results_df

    except Exception as e:
        print(f"[ERROR] update_getstats_batch failed: {e}")
        import traceback

        print("=== FULL TRACEBACK ===")
        traceback.print_exc()
        print("=== END TRACEBACK ===")
        return pd.DataFrame()


def get_prematch_stats_from_source_batch(
    source: Union[str, pd.DataFrame],
    source_type: str = "auto",
    progress_bar: bool = True,
    include_metadata: bool = True,
    stream_to_csv: Optional[str] = None,
    **filters,
) -> pd.DataFrame:
    """
    Batch processing version of get_prematch_stats_from_source for large datasets.
    
    This function is optimized for pure retrieval of prematch statistics with efficient
    memory management for large datasets. Uses before_time filtering for each match.

    Args:
        source: Data source (CSV path, DataFrame, or 'dataset')
        source_type: Source type ('csv', 'dataset', 'dataframe', or 'auto')
        progress_bar: Show progress bar for batch processing
        include_metadata: Include match metadata in results
        stream_to_csv: Optional file path to stream results directly to CSV (memory efficient)
        **filters: Data filtering options

    Returns:
        pd.DataFrame: Prematch stats for each match (empty if stream_to_csv is used)
    """
    from utils.updateStats import createStats, getStats
    from utils.elo_manager import create_elo_ratings

    print(f"[INFO] Starting get_prematch_stats_from_source_batch processing")
    start_time = datetime.now()

    try:
        # Load and validate data
        df = load_data_source(source, source_type, **filters)
        df = validate_match_data(df)

        if df.empty:
            print("[WARNING] No valid matches found after filtering and validation")
            return pd.DataFrame()

        # Sort by chronological order to ensure proper before_time filtering
        df_sorted = df.sort_values(
            ["tourney_date", "tourney_id", "round_order"], 
            ascending=[True, True, False]
        ).reset_index(drop=True)
        print(f"[INFO] Sorted {len(df_sorted)} matches chronologically")

        # Load stats and Elo ratings once for efficiency (no updates needed)
        print("[INFO] Loading statistics and Elo ratings...")
        prev_stats = createStats(load_existing=True)
        elo_ratings = create_elo_ratings(load_existing=True)
        print("[INFO] Stats and Elo ratings loaded successfully")

        # Adaptive batch sizing based on dataset size
        total_matches = len(df_sorted)
        if total_matches <= 1000:
            batch_size = 500
        elif total_matches <= 10000:
            batch_size = 2000
        elif total_matches <= 50000:
            batch_size = 5000
        else:
            batch_size = 10000  # Large datasets get bigger batches for efficiency
            
        print(f"[INFO] Using adaptive batch size: {batch_size} for {total_matches} matches")

        # Initialize result collection
        results_list = []
        total_processed = 0
        
        # Setup CSV streaming if requested
        csv_file = None
        if stream_to_csv:
            import csv
            print(f"[INFO] Streaming results directly to: {stream_to_csv}")

        # Process in batches with per-batch progress tracking
        num_batches = (total_matches + batch_size - 1) // batch_size
        batch_iterator = (
            tqdm(range(num_batches), desc="Processing batches", unit="batch")
            if progress_bar
            else range(num_batches)
        )

        for batch_idx in batch_iterator:
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, total_matches)
            batch_df = df_sorted.iloc[batch_start:batch_end].copy()
            
            batch_results = []
            batch_processed = 0
            
            # Inner progress for matches within batch
            match_iterator = (
                tqdm(
                    batch_df.iterrows(), 
                    total=len(batch_df),
                    desc=f"Batch {batch_idx + 1}/{num_batches}",
                    leave=False
                )
                if progress_bar
                else batch_df.iterrows()
            )

            for idx, match_row in match_iterator:
                try:
                    # Prepare player data
                    player1 = {
                        "ID": match_row["player1_id"],
                        "ATP_POINTS": match_row.get("p1_atp_points", 0),
                        "ATP_RANK": match_row.get("p1_atp_rank", 999),
                        "AGE": match_row.get("p1_age", 25),
                        "HEIGHT": match_row.get("p1_height", 180),
                    }

                    player2 = {
                        "ID": match_row["player2_id"],
                        "ATP_POINTS": match_row.get("p2_atp_points", 0),
                        "ATP_RANK": match_row.get("p2_atp_rank", 999),
                        "AGE": match_row.get("p2_age", 25),
                        "HEIGHT": match_row.get("p2_height", 180),
                    }

                    # Prepare match context
                    match_context = {
                        "SURFACE": match_row.get("surface", "Hard"),
                        "ROUND": match_row.get("round", "R32"),
                        "TOURNEY_ID": match_row["tourney_id"],
                        "TOURNEY_LEVEL": match_row.get("tourney_level", "ATP250"),
                        "BEST_OF": match_row.get("best_of", 3),
                        "DRAW_SIZE": match_row.get("draw_size", 32),
                        "TOURNEY_DATE": match_row["tourney_date"],
                    }

                    # Get prematch stats with before_time filtering
                    before_time = match_row["chronological_key"]
                    stats = getStats(
                        player1,
                        player2,
                        match_context,
                        prev_stats,
                        before_time=before_time,
                        elo_ratings=elo_ratings,
                    )

                    # Add match metadata if requested
                    if include_metadata:
                        stats["match_id"] = match_row["match_id"]
                        stats["tourney_date"] = match_row["tourney_date"]
                        stats["tourney_id"] = match_row["tourney_id"]
                        stats["round"] = match_row["round"]
                        stats["surface"] = match_row["surface"]
                        stats["player1_id"] = match_row["player1_id"]
                        stats["player2_id"] = match_row["player2_id"]
                        stats["actual_result"] = match_row["RESULT"]

                        # Add odds columns
                        stats["p1_odds"] = match_row.get("p1_odds", None)
                        stats["p2_odds"] = match_row.get("p2_odds", None)
                        stats["p1_max_odds"] = match_row.get("p1_max_odds", None)
                        stats["p2_max_odds"] = match_row.get("p2_max_odds", None)

                        # Add best_of and draw_size from original data
                        stats["best_of"] = match_row.get("best_of", None)
                        stats["draw_size"] = match_row.get("draw_size", None)

                        # Add ATP rank and points
                        stats["p1_atp_rank"] = match_row.get("player1_rank", None)
                        stats["p1_atp_points"] = match_row.get("player1_atp_points", None)
                        stats["p2_atp_rank"] = match_row.get("player2_rank", None)
                        stats["p2_atp_points"] = match_row.get("player2_atp_points", None)

                        # Add age and height
                        stats["p1_age"] = match_row.get("p1_age", None)
                        stats["p2_age"] = match_row.get("p2_age", None)
                        stats["p1_height"] = match_row.get("p1_height", None)
                        stats["p2_height"] = match_row.get("p2_height", None)

                    batch_results.append(stats)
                    batch_processed += 1
                    total_processed += 1

                except Exception as e:
                    print(f"[ERROR] Failed to get stats for match {match_row.get('match_id', 'unknown')}: {e}")
                    continue

            # Handle batch results
            if stream_to_csv:
                # Stream this batch to CSV with proper quoting
                batch_df_results = pd.DataFrame(batch_results)
                if batch_idx == 0:
                    # First batch - create file with header
                    batch_df_results.to_csv(stream_to_csv, index=False, quoting=1)  # QUOTE_ALL
                else:
                    # Subsequent batches - append without header
                    batch_df_results.to_csv(stream_to_csv, mode='a', header=False, index=False, quoting=1)
            else:
                # Accumulate in memory
                results_list.extend(batch_results)
            
            # Update progress
            if progress_bar and hasattr(batch_iterator, 'set_postfix'):
                batch_iterator.set_postfix(
                    processed=total_processed,
                    batch_matches=batch_processed
                )

        # Create final DataFrame (empty if streaming)
        if stream_to_csv:
            results_df = pd.DataFrame()  # Empty since results streamed to file
            print(f"[INFO] Results streamed to: {stream_to_csv}")
        else:
            # Build final DataFrame efficiently
            print("[INFO] Building final results DataFrame...")
            results_df = pd.DataFrame(results_list)

        end_time = datetime.now()
        runtime = (end_time - start_time).total_seconds()
        
        print(f"[SUCCESS] get_prematch_stats_from_source_batch completed!")
        print(f"  - Processed matches: {total_processed}")
        print(f"  - Batch size used: {batch_size}")
        print(f"  - Number of batches: {num_batches}")
        print(f"  - Runtime: {runtime:.2f} seconds")
        print(f"  - Matches per second: {total_processed / runtime:.2f}")
        if stream_to_csv:
            print(f"  - Output streamed to: {stream_to_csv}")
        else:
            print(f"  - Result DataFrame shape: {results_df.shape}")

        return results_df

    except Exception as e:
        print(f"[ERROR] get_prematch_stats_from_source_batch failed: {e}")
        import traceback
        print("=== FULL TRACEBACK ===")
        traceback.print_exc()
        print("=== END TRACEBACK ===")
        return pd.DataFrame()


def _generate_batch_report(
    results: Dict[str, Any], output_prefix: str
) -> Dict[str, Any]:
    """
    Generate comprehensive batch processing report.

    Args:
        results: Batch processing results
        output_prefix: Output file prefix

    Returns:
        Dict: Comprehensive report
    """
    report = {
        "batch_summary": {
            "output_prefix": output_prefix,
            "total_runtime": None,
            "phases_completed": results.get("phases_completed", []),
            "success": results.get("success", False),
        },
        "input_analysis": results.get("input_data", {}),
        "performance_metrics": {},
    }

    # Calculate runtime
    if "start_time" in results and "end_time" in results:
        start = datetime.fromisoformat(results["start_time"])
        end = datetime.fromisoformat(results["end_time"])
        runtime = (end - start).total_seconds()
        report["batch_summary"]["total_runtime"] = f"{runtime:.2f} seconds"

    # Performance metrics
    if "input_data" in results:
        total_matches = results["input_data"].get("total_matches", 0)
        if total_matches > 0 and "total_runtime" in report["batch_summary"]:
            runtime_seconds = float(report["batch_summary"]["total_runtime"].split()[0])
            report["performance_metrics"] = {
                "matches_per_second": round(total_matches / runtime_seconds, 2),
                "seconds_per_match": round(runtime_seconds / total_matches, 4),
            }

    # Operation-specific reporting
    if "stats_retrieval" in results:
        report["stats_retrieval_summary"] = {
            "modes_processed": list(results["stats_retrieval"].keys()),
            "total_files_created": len(
                [r for r in results["stats_retrieval"].values() if "file" in r]
            ),
        }

    if "stats_update" in results:
        update_data = results["stats_update"]
        report["stats_update_summary"] = {
            "processed_matches": update_data.get("processed_matches", 0),
            "skipped_duplicates": update_data.get("skipped_duplicates", 0),
            "overwritten_matches": update_data.get("overwritten_matches", 0),
            "errors": len(update_data.get("errors", [])),
            "success": update_data.get("success", False),
        }

    return report


def _prepare_for_json(obj):
    """
    Prepare object for JSON serialization by converting problematic types.

    Args:
        obj: Object to prepare

    Returns:
        JSON-serializable object
    """
    if isinstance(obj, dict):
        return {key: _prepare_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_prepare_for_json(item) for item in obj]
    elif isinstance(obj, pd.DataFrame):
        return f"DataFrame({len(obj)} rows, {len(obj.columns)} columns)"
    elif hasattr(obj, "isoformat"):  # datetime objects
        return obj.isoformat()
    else:
        return obj


def clear_all_stats_data(confirm: bool = True, backup: bool = True) -> Dict[str, Any]:
    """
    Clear all saved statistics data with optional backup.

    Args:
        confirm: Require user confirmation
        backup: Create backup before clearing

    Returns:
        dict: Operation results
    """
    from utils.stats_io import clear_all_stats_with_confirmation, clear_all_stats
    from utils.updateStats import createStats
    import shutil

    if confirm:
        response = input(
            "Are you sure you want to clear ALL statistics data? (yes/no): "
        )
        if response.lower() != "yes":
            return {"success": False, "message": "Operation cancelled by user"}

    results = {
        "start_time": datetime.now().isoformat(),
        "backup_created": False,
        "files_cleared": [],
        "success": False,
    }

    try:
        # Create backup if requested
        if backup:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = f"data/stats_backup_{timestamp}"
            os.makedirs(backup_dir, exist_ok=True)

            # Backup stats files
            stats_files = [
                "data/player_stats.pkl",
                "data/processed_matches.json",
                "data/elo_ratings.pkl",
            ]

            backup_count = 0
            for file_path in stats_files:
                if os.path.exists(file_path):
                    shutil.copy2(file_path, backup_dir)
                    backup_count += 1

            if backup_count > 0:
                results["backup_created"] = True
                results["backup_location"] = backup_dir
                print(f"[INFO] Backup created at: {backup_dir}")
            else:
                print("[INFO] No existing stats files found to backup")

        # Clear all stats data
        fresh_stats = clear_all_stats()
        results["files_cleared"] = [
            "player_stats.pkl",
            "processed_matches.json",
            "chronological_deques_data",
        ]

        results["success"] = True
        results["message"] = "All statistics data cleared successfully"
        results["end_time"] = datetime.now().isoformat()

        print(f"[SUCCESS] Statistics data cleared")
        if backup and results.get("backup_created"):
            print(f"[INFO] Backup available at: {results['backup_location']}")

        return results

    except Exception as e:
        results["success"] = False
        results["error"] = str(e)
        results["end_time"] = datetime.now().isoformat()
        print(f"[ERROR] Failed to clear stats data: {e}")
        return results


def get_stats_summary(prev_stats: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Get a comprehensive summary of current statistics.

    Args:
        prev_stats: Statistics dictionary (loads if None)

    Returns:
        Dict with statistics summary
    """
    from utils.updateStats import createStats, getStatsInfo

    if prev_stats is None:
        prev_stats = createStats(load_existing=True)

    # Get basic summary from existing function
    basic_summary = getStatsInfo(prev_stats)

    # Enhanced summary with additional details
    enhanced_summary = {
        "basic_info": basic_summary,
        "chronological_stats": {
            "last_k_matches_players": len(prev_stats.get("last_k_matches", {})),
            "matches_played_players": len(prev_stats.get("matches_played", {})),
            "championship_winners": len(prev_stats.get("championships", {})),
            "h2h_pairs": len(prev_stats.get("h2h", {})),
            "surface_h2h_surfaces": len(prev_stats.get("h2h_surface", {})),
        },
        "data_distribution": {},
        "storage_info": {},
    }

    try:
        # Analyze data distribution
        if "processed_matches" in prev_stats:
            processed_matches = prev_stats["processed_matches"]
            enhanced_summary["data_distribution"]["processed_matches_count"] = len(
                processed_matches
            )

        # Calculate average deque sizes
        avg_deque_sizes = {}
        for stat_name in [
            "last_k_matches",
            "matches_played",
            "championships",
            "games_diff",
        ]:
            if stat_name in prev_stats:
                deques = prev_stats[stat_name]
                if deques:
                    sizes = [len(deque) for deque in deques.values()]
                    avg_deque_sizes[stat_name] = {
                        "avg_size": sum(sizes) / len(sizes) if sizes else 0,
                        "max_size": max(sizes) if sizes else 0,
                        "min_size": min(sizes) if sizes else 0,
                    }

        enhanced_summary["chronological_stats"]["deque_sizes"] = avg_deque_sizes

        # Storage information
        stats_files = {
            "player_stats.pkl": "data/player_stats.pkl",
            "processed_matches.json": "data/processed_matches.json",
            "elo_ratings.pkl": "data/elo_ratings.pkl",
        }

        file_info = {}
        total_size = 0
        for name, path in stats_files.items():
            if os.path.exists(path):
                size = os.path.getsize(path)
                file_info[name] = {
                    "exists": True,
                    "size_bytes": size,
                    "size_mb": round(size / (1024 * 1024), 2),
                }
                total_size += size
            else:
                file_info[name] = {"exists": False}

        enhanced_summary["storage_info"] = {
            "files": file_info,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }

    except Exception as e:
        enhanced_summary["analysis_errors"] = [str(e)]
        print(f"[WARNING] Some summary analysis failed: {e}")

    return enhanced_summary


# ============================================================================
# MODULE TESTING
# ============================================================================

if __name__ == "__main__":
    print("Stats Interface Module - Core Functions Loaded")
    print("Available functions:")
    print("  - load_data_source()")
    print("  - apply_filters()")
    print("  - validate_match_data()")
    print("  - update_stats_from_source()")
    print("  - get_prematch_stats_from_source()")
    print("  - remove_match_from_chronological_data()")
    print("  - clear_all_stats_data()")
    print("  - get_stats_summary()")
    print("  - batch_process_with_options()")
    print("\nAll core functions implemented and ready for use!")
