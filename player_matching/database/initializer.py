"""
Database initialization from existing ATP players data.
"""

import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import re

from .schema import create_tables, verify_schema
from .connection import get_db_connection
from .crud_operations import create_player, add_source_id_to_player
from ..matching.name_processing import merge_player_names, preprocess_player_name


def parse_date_of_birth(dob_value) -> Optional[str]:
    """
    Parse date of birth from ATP format (YYYYMMDD) to SQL format (YYYY-MM-DD).

    Args:
        dob_value: Date value (string, int, float) in format YYYYMMDD

    Returns:
        Date string in SQL format or None if invalid
    """
    if not dob_value or pd.isna(dob_value):
        return None

    try:
        # Convert to string and handle float values like 19131122.0
        if isinstance(dob_value, float):
            if dob_value == 0.0:  # Handle 0.0 as missing
                return None
            dob_str = str(int(dob_value))  # Remove .0 suffix
        else:
            dob_str = str(dob_value).strip()

        # Must be exactly 8 digits
        if len(dob_str) == 8 and dob_str.isdigit():
            year = dob_str[:4]
            month = dob_str[4:6]
            day = dob_str[6:8]

            # Basic validation
            year_int = int(year)
            month_int = int(month)
            day_int = int(day)

            if 1900 <= year_int <= 2010 and 1 <= month_int <= 12 and 1 <= day_int <= 31:
                return f"{year}-{month}-{day}"

        return None

    except (ValueError, IndexError, TypeError):
        return None


def validate_player_data(row: pd.Series) -> tuple[bool, str]:
    """
    Validate a player data row from ATP CSV.

    Args:
        row: Pandas Series with player data

    Returns:
        Tuple of (is_valid, error_message)
    """
    required_fields = ["player_id", "name_first", "name_last"]

    for field in required_fields:
        if field not in row or pd.isna(row[field]) or str(row[field]).strip() == "":
            return False, f"Missing or empty required field: {field}"

    # Validate player ID
    try:
        player_id = int(row["player_id"])
        if player_id < 100000 or player_id > 999999:
            return (
                False,
                f"Player ID {player_id} is not a valid 6-digit number (100000-999999)",
            )
    except (ValueError, TypeError):
        return False, f"Player ID '{row['player_id']}' is not a valid integer"

    # Validate names
    first_name = str(row["name_first"]).strip()
    last_name = str(row["name_last"]).strip()

    if len(first_name) < 1 or len(last_name) < 1:
        return False, "First and last names must be at least 1 character"

    if len(first_name) > 50 or len(last_name) > 50:
        return False, "Names cannot be longer than 50 characters"

    return True, ""


def load_atp_players_csv(
    csv_path: str = "data/atp_players.csv",
    db_path: str = "databases/tennis_players.db",
    source_name: str = "main_dataset",
    update_existing: bool = False,
) -> Dict[str, Any]:
    """
    Load ATP players data from CSV into the database.

    Args:
        csv_path: Path to ATP players CSV file
        db_path: Path to player database
        source_name: Source name for this data ('main_dataset')
        update_existing: Whether to update existing players with new metadata

    Returns:
        Dictionary with loading statistics
    """
    stats = {
        "total_rows": 0,
        "processed": 0,
        "created_new": 0,
        "updated_existing": 0,
        "skipped_invalid": 0,
        "errors": 0,
        "error_details": [],
    }

    try:
        # Ensure database exists
        if not create_tables(db_path):
            raise Exception("Failed to create database tables")

        # Load CSV
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"ATP players CSV not found: {csv_path}")

        print(f"Loading ATP players from: {csv_path}")
        df = pd.read_csv(csv_path)

        if df.empty:
            raise ValueError("ATP players CSV is empty")

        stats["total_rows"] = len(df)
        print(f"Found {stats['total_rows']} player records")

        # Expected columns: player_id,name_first,name_last,hand,dob,ioc,height,wikidata_id
        required_columns = ["player_id", "name_first", "name_last"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Process each player
        db = get_db_connection(db_path)

        for idx, row in df.iterrows():
            try:
                # Validate row data
                valid, error_msg = validate_player_data(row)
                if not valid:
                    stats["skipped_invalid"] += 1
                    stats["error_details"].append(f"Row {idx}: {error_msg}")
                    continue

                # Extract data
                player_id = int(row["player_id"])
                first_name = str(row["name_first"]).strip()
                last_name = str(row["name_last"]).strip()

                # Create full name and preprocessed version
                full_name = merge_player_names(first_name, last_name)
                preprocessed_name = preprocess_player_name(full_name)

                # Parse optional fields
                dob = parse_date_of_birth(row.get("dob", "")) if "dob" in row else None
                hand = (
                    str(row["hand"]).strip().upper()
                    if "hand" in row and not pd.isna(row["hand"])
                    else None
                )
                if hand and hand not in ["L", "R", "U"]:
                    hand = None

                height = None
                if "height" in row and not pd.isna(row["height"]):
                    try:
                        height = int(float(row["height"]))
                        if height < 150 or height > 220:  # Reasonable bounds
                            height = None
                    except (ValueError, TypeError):
                        height = None

                # Check if player already exists
                existing_player = db.execute_query(
                    "SELECT player_id FROM players WHERE player_id = ?", (player_id,)
                )

                if existing_player:
                    if update_existing:
                        # Update metadata
                        updates = []
                        params = []

                        if dob:
                            updates.append("dob = ?")
                            params.append(dob)
                        if hand:
                            updates.append("hand = ?")
                            params.append(hand)
                        if height:
                            updates.append("height = ?")
                            params.append(height)

                        if updates:
                            updates.append("last_updated = ?")
                            params.append(datetime.now().isoformat())
                            params.append(player_id)

                            query = f"UPDATE players SET {', '.join(updates)} WHERE player_id = ?"
                            db.execute_update(query, tuple(params))

                        stats["updated_existing"] += 1
                    else:
                        # Player exists, skip
                        pass
                else:
                    # Create new player
                    new_player_id = create_player(
                        primary_name=preprocessed_name,
                        player_id=player_id,
                        dob=dob,
                        hand=hand,
                        height=height,
                        db_path=db_path,
                    )

                    if new_player_id != player_id:
                        raise Exception(
                            f"Created player ID {new_player_id} doesn't match expected {player_id}"
                        )

                    # Add source mapping
                    success = add_source_id_to_player(
                        player_id=player_id,
                        source_name=source_name,
                        source_id=str(
                            player_id
                        ),  # Use player_id as source_id for main dataset
                        source_name_variant=full_name,
                        preprocessed_name=preprocessed_name,
                        is_primary_name=True,
                        db_path=db_path,
                    )

                    if not success:
                        # This might happen if source mapping already exists
                        pass

                    stats["created_new"] += 1

                stats["processed"] += 1

                # Progress update every 1000 players
                if stats["processed"] % 1000 == 0:
                    print(
                        f"Processed {stats['processed']}/{stats['total_rows']} players..."
                    )

            except Exception as e:
                stats["errors"] += 1
                stats["error_details"].append(
                    f"Row {idx} (ID: {row.get('player_id', 'unknown')}): {str(e)}"
                )
                continue

        print(
            f"Loading complete: {stats['processed']} processed, {stats['created_new']} new, {stats['updated_existing']} updated"
        )

        if stats["skipped_invalid"] > 0:
            print(f"Skipped {stats['skipped_invalid']} invalid records")

        if stats["errors"] > 0:
            print(f"Encountered {stats['errors']} errors")
            # Show first few errors
            for error in stats["error_details"][:5]:
                print(f"  - {error}")
            if len(stats["error_details"]) > 5:
                print(f"  ... and {len(stats['error_details']) - 5} more errors")

        return stats

    except Exception as e:
        stats["errors"] += 1
        stats["error_details"].append(f"Critical error: {str(e)}")
        print(f"Failed to load ATP players: {e}")
        return stats


def verify_database_initialization(
    db_path: str = "databases/tennis_players.db",
) -> Dict[str, Any]:
    """
    Verify that the database has been properly initialized with player data.

    Args:
        db_path: Path to player database

    Returns:
        Dictionary with verification results
    """
    try:
        db = get_db_connection(db_path)

        # Check basic counts
        player_count = db.execute_query("SELECT COUNT(*) FROM players")[0][0]
        source_count = db.execute_query("SELECT COUNT(*) FROM player_sources")[0][0]

        # Check source distribution
        source_dist = db.execute_query("""
            SELECT source_name, COUNT(*) 
            FROM player_sources 
            GROUP BY source_name
        """)

        # Check data quality
        players_with_metadata = db.execute_query("""
            SELECT 
                COUNT(*) as total,
                COUNT(dob) as with_dob,
                COUNT(hand) as with_hand,
                COUNT(height) as with_height
            FROM players
        """)[0]

        results = {
            "total_players": player_count,
            "total_source_mappings": source_count,
            "sources": dict(source_dist),
            "metadata_coverage": {
                "total": players_with_metadata[0],
                "with_dob": players_with_metadata[1],
                "with_hand": players_with_metadata[2],
                "with_height": players_with_metadata[3],
            },
        }

        # Calculate percentages
        if results["metadata_coverage"]["total"] > 0:
            total = results["metadata_coverage"]["total"]
            results["metadata_coverage"]["dob_pct"] = round(
                results["metadata_coverage"]["with_dob"] / total * 100, 1
            )
            results["metadata_coverage"]["hand_pct"] = round(
                results["metadata_coverage"]["with_hand"] / total * 100, 1
            )
            results["metadata_coverage"]["height_pct"] = round(
                results["metadata_coverage"]["with_height"] / total * 100, 1
            )

        return results

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # Test database initialization
    print("Testing database initialization...")

    # Load test data (first 10 rows of real data for testing)
    try:
        csv_path = "data/atp_players.csv"
        if Path(csv_path).exists():
            # Load small sample for testing
            df = pd.read_csv(csv_path)

            # Initialize database
            stats = load_atp_players_csv(
                csv_path=df, db_path="databases/tennis_players.db"
            )

            print(f"Initialization stats: {stats}")

            # Verify
            verification = verify_database_initialization(
                "databases/test_tennis_players.db"
            )
            print(f"Verification results: {verification}")

        else:
            print(f"ATP players CSV not found at: {csv_path}")

    except Exception as e:
        print(f"Database initialization: FAILED - {e}")
