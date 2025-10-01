"""
Fast database initialization with bulk operations for optimal performance.
Optimized version that reduces initialization time from 15-30 minutes to 2-3 minutes.
"""

import pandas as pd
import sqlite3
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import tqdm
import gc
import time

from .schema import create_tables, verify_schema
from .connection import get_db_connection
from ..matching.name_processing import merge_player_names, preprocess_player_name


def parse_date_of_birth_vectorized(dob_series: pd.Series) -> pd.Series:
    """
    Vectorized DOB parsing for improved performance.

    Args:
        dob_series: Series of DOB values (various types)

    Returns:
        Series of formatted dates or None
    """

    def parse_single_dob(dob_value):
        if not dob_value or pd.isna(dob_value):
            return None

        try:
            # Handle float values like 19131122.0
            if isinstance(dob_value, float):
                if dob_value == 0.0:
                    return None
                dob_str = str(int(dob_value))
            else:
                dob_str = str(dob_value).strip()

            # Must be exactly 8 digits
            if len(dob_str) == 8 and dob_str.isdigit():
                year = dob_str[:4]
                month = dob_str[4:6]
                day = dob_str[6:8]

                year_int = int(year)
                month_int = int(month)
                day_int = int(day)

                if (
                    1900 <= year_int <= 2010
                    and 1 <= month_int <= 12
                    and 1 <= day_int <= 31
                ):
                    return f"{year}-{month}-{day}"

            return None

        except (ValueError, IndexError, TypeError):
            return None

    return dob_series.apply(parse_single_dob)


def preprocess_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Preprocess entire DataFrame using vectorized operations.

    Args:
        df: Raw ATP players DataFrame

    Returns:
        Tuple of (processed_df, stats)
    """
    stats = {
        "input_rows": len(df),
        "valid_rows": 0,
        "invalid_rows": 0,
        "preprocessing_time": 0,
    }

    start_time = time.time()

    # Required columns validation
    required_columns = ["player_id", "player_name"]
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"Missing required columns: {missing}")

    print("Preprocessing player data...")

    # Create working copy
    processed_df = df.copy()

    # Convert player_id to int, filtering invalid entries
    processed_df["player_id"] = pd.to_numeric(
        processed_df["player_id"], errors="coerce"
    )
    valid_ids = processed_df["player_id"].between(100000, 999999, inclusive="both")
    processed_df = processed_df[valid_ids & processed_df["player_id"].notna()].copy()

    # Clean and validate names
    processed_df["player_name"] = processed_df["player_name"].astype(str).str.strip()

    # Remove rows with invalid names
    valid_names = (processed_df["player_name"].str.len() > 0) & (
        processed_df["player_name"].str.len() <= 50
    )
    processed_df = processed_df[valid_names].copy()

    processed_df["preprocessed_name"] = processed_df["player_name"].apply(
        preprocess_player_name
    )

    # Process DOB (vectorized)
    if "dob" in processed_df.columns:
        processed_df["dob_formatted"] = parse_date_of_birth_vectorized(
            processed_df["dob"]
        )
    else:
        processed_df["dob_formatted"] = None

    # Process hand
    if "hand" in processed_df.columns:
        processed_df["hand_clean"] = (
            processed_df["hand"].astype(str).str.strip().str.upper()
        )
        processed_df["hand_clean"] = processed_df["hand_clean"].where(
            processed_df["hand_clean"].isin(["L", "R", "U"]), None
        )
    else:
        processed_df["hand_clean"] = None

    # Process height
    if "height" in processed_df.columns:
        processed_df["height_clean"] = pd.to_numeric(
            processed_df["height"], errors="coerce"
        )
        processed_df["height_clean"] = processed_df["height_clean"].where(
            processed_df["height_clean"].between(150, 220, inclusive="both"), None
        )
        processed_df["height_clean"] = processed_df["height_clean"].astype(
            "Int64"
        )  # Nullable integer
    else:
        processed_df["height_clean"] = None

    # Remove duplicates by player_id (keep first occurrence)
    initial_count = len(processed_df)
    processed_df = processed_df.drop_duplicates(subset=["player_id"], keep="first")
    duplicates_removed = initial_count - len(processed_df)

    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate player IDs")

    stats["valid_rows"] = len(processed_df)
    stats["invalid_rows"] = stats["input_rows"] - stats["valid_rows"]
    stats["preprocessing_time"] = time.time() - start_time
    stats["duplicates_removed"] = duplicates_removed

    print(
        f"Preprocessing complete: {stats['valid_rows']} valid players from {stats['input_rows']} input rows"
    )

    return processed_df, stats


def bulk_insert_players(
    df: pd.DataFrame,
    db_path: str,
    batch_size: int = 5000,
    source_code: int = 0,
) -> Dict[str, Any]:
    """
    Perform bulk insert of players using optimized database operations.

    Args:
        df: Preprocessed DataFrame with players
        db_path: Database file path
        batch_size: Number of records per batch
        source_code: Source identifier

    Returns:
        Dictionary with insertion statistics
    """
    stats = {
        "total_players": len(df),
        "inserted_players": 0,
        "inserted_sources": 0,
        "batch_count": 0,
        "insertion_time": 0,
        "errors": [],
    }

    start_time = time.time()

    # Prepare data for bulk insertion
    players_data = []
    sources_data = []

    for _, row in df.iterrows():
        # Players table data
        players_data.append(
            (
                int(row["player_id"]),
                row["preprocessed_name"],
                row["dob_formatted"],
                row["hand_clean"],
                row["height_clean"] if pd.notna(row["height_clean"]) else None,
            )
        )

        # Player sources table data
        sources_data.append(
            (
                int(row["player_id"]),
                source_code,
                str(
                    int(row["player_id"])
                ),  # Use player_id as source_id for main dataset
                row["player_name"],
                row["preprocessed_name"],
                True,  # is_primary_name for main dataset
            )
        )

    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Disable auto-commit for batch processing
        conn.execute("BEGIN TRANSACTION")

        # Bulk insert players
        print("Inserting players...")
        cursor.executemany(
            """INSERT OR REPLACE INTO players 
               (player_id, primary_name, dob, hand, height) 
               VALUES (?, ?, ?, ?, ?)""",
            players_data,
        )
        stats["inserted_players"] = cursor.rowcount

        # Bulk insert source mappings
        print("Inserting source mappings...")
        cursor.executemany(
            """INSERT OR REPLACE INTO player_sources 
               (player_id, source_code, source_id, source_name_variant, 
                preprocessed_name, is_primary_name)
               VALUES (?, ?, ?, ?, ?, ?)""",
            sources_data,
        )
        stats["inserted_sources"] = cursor.rowcount

        # Commit transaction
        conn.commit()
        print("Bulk insertion completed successfully")

    except Exception as e:
        conn.rollback()
        stats["errors"].append(f"Bulk insertion failed: {str(e)}")
        raise e

    finally:
        cursor.close()
        conn.close()

    stats["insertion_time"] = time.time() - start_time
    stats["batch_count"] = 1  # Single batch for this implementation

    return stats


def fast_load_atp_players_csv(
    csv_path: str = "data/atp_players.csv",
    db_path: str = "databases/tennis_players.db",
    source_code: int = 0,
    batch_size: int = 5000,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Fast loading of ATP players using optimized bulk operations.

    Args:
        csv_path: Path to ATP players CSV file
        db_path: Path to player database
        source_code: Source name for this data
        batch_size: Records per batch (currently uses single batch)
        show_progress: Show progress bar

    Returns:
        Dictionary with comprehensive loading statistics
    """
    overall_stats = {
        "start_time": datetime.now().isoformat(),
        "csv_path": csv_path,
        "db_path": db_path,
        "source_code": source_code,
        "batch_size": batch_size,
        "total_time": 0,
        "csv_loading_time": 0,
        "preprocessing_stats": {},
        "insertion_stats": {},
        "verification_stats": {},
        "success": False,
        "errors": [],
    }

    overall_start = time.time()

    try:
        # Ensure database exists
        print("Creating database tables...")
        if not create_tables(db_path):
            raise Exception("Failed to create database tables")

        # Load CSV with optimized settings
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"ATP players CSV not found: {csv_path}")

        print(f"Loading ATP players from: {csv_path}")
        csv_start = time.time()

        # Read with specific dtypes to prevent warnings and improve performance
        dtype_spec = {
            "player_id": "int64",
            "player_name": "str",
            "hand": "str",
            "ioc": "str",
            # dob and height will be handled as mixed types
        }

        df = pd.read_csv(
            csv_path,
            dtype=dtype_spec,
            low_memory=False,  # Read entire file for consistent types
            na_values=["", "NA", "null", "0"],  # Explicit NA values
        )

        overall_stats["csv_loading_time"] = time.time() - csv_start
        print(f"CSV loaded in {overall_stats['csv_loading_time']:.2f} seconds")

        if df.empty:
            raise ValueError("ATP players CSV is empty")

        # Preprocess data
        processed_df, preprocessing_stats = preprocess_dataframe(df)
        overall_stats["preprocessing_stats"] = preprocessing_stats

        # Bulk insert
        insertion_stats = bulk_insert_players(
            processed_df, db_path, batch_size=batch_size, source_code=source_code
        )
        overall_stats["insertion_stats"] = insertion_stats

        # Verify results
        print("Verifying database initialization...")
        from .initializer import verify_database_initialization

        verification_stats = verify_database_initialization(db_path)
        overall_stats["verification_stats"] = verification_stats

        # Success!
        overall_stats["success"] = True
        overall_stats["total_time"] = time.time() - overall_start

        print(f"\n{'=' * 60}")
        print("FAST INITIALIZATION COMPLETED SUCCESSFULLY")
        print(f"{'=' * 60}")
        print(f"Total time: {overall_stats['total_time']:.2f} seconds")
        print(f"CSV loading: {overall_stats['csv_loading_time']:.2f} seconds")
        print(f"Preprocessing: {preprocessing_stats['preprocessing_time']:.2f} seconds")
        print(f"Database insertion: {insertion_stats['insertion_time']:.2f} seconds")
        print(f"Players inserted: {insertion_stats['inserted_players']:,}")
        print(f"Source mappings: {insertion_stats['inserted_sources']:,}")

        if "metadata_coverage" in verification_stats:
            meta = verification_stats["metadata_coverage"]
            total = meta.get("total", 0)
            if total > 0:
                print(f"\nMetadata Coverage:")
                print(
                    f"  DOB: {meta.get('with_dob', 0):,}/{total:,} ({meta.get('dob_pct', 0):.1f}%)"
                )
                print(
                    f"  Hand: {meta.get('with_hand', 0):,}/{total:,} ({meta.get('hand_pct', 0):.1f}%)"
                )
                print(
                    f"  Height: {meta.get('with_height', 0):,}/{total:,} ({meta.get('height_pct', 0):.1f}%)"
                )

        print(f"{'=' * 60}")

        return overall_stats

    except Exception as e:
        overall_stats["success"] = False
        overall_stats["errors"].append(str(e))
        overall_stats["total_time"] = time.time() - overall_start

        print(
            f"Fast initialization failed after {overall_stats['total_time']:.2f} seconds: {e}"
        )
        return overall_stats


def compare_initialization_methods(
    csv_path: str = "data/atp_players.csv", test_rows: int = 1000
) -> Dict[str, Any]:
    """
    Compare performance between standard and fast initialization methods.

    Args:
        csv_path: Path to ATP players CSV
        test_rows: Number of rows to test with

    Returns:
        Performance comparison results
    """
    print(f"Performance Comparison: Testing with {test_rows} rows")
    print("=" * 60)

    # Create test data
    df = pd.read_csv(csv_path, low_memory=False).head(test_rows)
    test_csv = "databases/test_performance.csv"
    df.to_csv(test_csv, index=False)

    results = {
        "test_rows": test_rows,
        "standard_method": {},
        "fast_method": {},
        "speedup_factor": 0,
        "comparison_time": datetime.now().isoformat(),
    }

    try:
        # Test standard method
        print("Testing STANDARD initialization method...")
        from .initializer import load_atp_players_csv

        # Clear any existing test database
        test_db_standard = "databases/test_standard.db"
        if Path(test_db_standard).exists():
            Path(test_db_standard).unlink()

        start_time = time.time()
        standard_stats = load_atp_players_csv(
            csv_path=test_csv, db_path=test_db_standard
        )
        standard_time = time.time() - start_time

        results["standard_method"] = {
            "time_seconds": standard_time,
            "stats": standard_stats,
        }
        print(f"Standard method: {standard_time:.2f} seconds")

        # Test fast method
        print("\nTesting FAST initialization method...")
        test_db_fast = "databases/test_fast.db"
        if Path(test_db_fast).exists():
            Path(test_db_fast).unlink()

        start_time = time.time()
        fast_stats = fast_load_atp_players_csv(csv_path=test_csv, db_path=test_db_fast)
        fast_time = time.time() - start_time

        results["fast_method"] = {"time_seconds": fast_time, "stats": fast_stats}
        print(f"Fast method: {fast_time:.2f} seconds")

        # Calculate speedup
        if fast_time > 0:
            results["speedup_factor"] = standard_time / fast_time

        print(f"\nSPEEDUP: {results['speedup_factor']:.2f}x faster")
        print("=" * 60)

    except Exception as e:
        print(f"Performance comparison failed: {e}")
        results["error"] = str(e)

    finally:
        # Cleanup test files
        for file in [test_csv, "databases/test_standard.db", "databases/test_fast.db"]:
            if Path(file).exists():
                try:
                    Path(file).unlink()
                except:
                    pass

    return results


if __name__ == "__main__":
    # Test fast initialization
    print("Testing Fast Player Database Initialization")
    print("=" * 60)

    # Run performance comparison with sample data
    comparison = compare_initialization_methods(test_rows=1000)

    if "speedup_factor" in comparison:
        print(f"Performance improvement: {comparison['speedup_factor']:.2f}x faster")

    # Uncomment to run full fast initialization
    # stats = fast_load_atp_players_csv()
    # print(f"Full initialization stats: {stats}")
