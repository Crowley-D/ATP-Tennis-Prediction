"""
Core metadata processing functions for dataframe enrichment.
"""

import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Set
import numpy as np
import warnings

from ..database.crud_operations import (
    get_players_metadata_batch,
    update_players_metadata_batch,
    get_metadata_coverage_stats
)
from ..matching.metadata_validation import (
    validate_player_metadata,
    get_metadata_validation_summary
)


def detect_metadata_columns(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """
    Auto-detect metadata columns in dataframe using common naming patterns.
    
    Args:
        df: Input dataframe
        
    Returns:
        Dict mapping player ('player1', 'player2') to metadata column mapping
        {
            'player1': {'height': 'player1_ht', 'dob': 'player1_dob', 'hand': 'player1_hand'},
            'player2': {'height': 'player2_ht', 'dob': 'player2_dob', 'hand': 'player2_hand'}
        }
    """
    column_mapping = {'player1': {}, 'player2': {}}
    
    # Define column patterns for each metadata type
    patterns = {
        'height': {
            'player1': ['player1_ht', 'player1_height', 'p1_ht', 'p1_height', 'winner_ht', 'winner_height'],
            'player2': ['player2_ht', 'player2_height', 'p2_ht', 'p2_height', 'loser_ht', 'loser_height']
        },
        'dob': {
            'player1': ['player1_dob', 'player1_birth', 'player1_birthdate', 'p1_dob', 'p1_birth', 'winner_dob'],
            'player2': ['player2_dob', 'player2_birth', 'player2_birthdate', 'p2_dob', 'p2_birth', 'loser_dob']
        },
        'hand': {
            'player1': ['player1_hand', 'player1_dominant', 'p1_hand', 'p1_dominant', 'winner_hand'],
            'player2': ['player2_hand', 'player2_dominant', 'p2_hand', 'p2_dominant', 'loser_hand']
        }
    }
    
    # Look for columns in dataframe
    df_columns_lower = {col.lower(): col for col in df.columns}
    
    for metadata_type, player_patterns in patterns.items():
        for player, column_patterns in player_patterns.items():
            for pattern in column_patterns:
                if pattern.lower() in df_columns_lower:
                    column_mapping[player][metadata_type] = df_columns_lower[pattern.lower()]
                    break  # Use first match found
    
    return column_mapping


def collect_unique_player_ids(
    df: pd.DataFrame,
    player1_id_col: str = 'player1_id',
    player2_id_col: str = 'player2_id'
) -> Set[int]:
    """
    Collect all unique player IDs from both player columns.
    
    Args:
        df: Input dataframe
        player1_id_col: Player 1 ID column name
        player2_id_col: Player 2 ID column name
        
    Returns:
        Set of unique player IDs
    """
    unique_ids = set()
    
    # Collect from player1 column
    if player1_id_col in df.columns:
        p1_ids = df[player1_id_col].dropna()
        unique_ids.update(p1_ids.astype(int))
    
    # Collect from player2 column  
    if player2_id_col in df.columns:
        p2_ids = df[player2_id_col].dropna()
        unique_ids.update(p2_ids.astype(int))
    
    return unique_ids


def enrich_dataframe_metadata(
    df: pd.DataFrame,
    player1_id_col: str = 'player1_id',
    player2_id_col: str = 'player2_id',
    metadata_columns: Optional[Dict[str, Dict[str, str]]] = None,
    batch_size: int = 1000,
    validate_metadata: bool = True,
    db_path: str = "databases/tennis_players.db"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Enrich dataframe with metadata from database for empty fields.
    
    Args:
        df: Input dataframe
        player1_id_col: Player 1 ID column name
        player2_id_col: Player 2 ID column name  
        metadata_columns: Column mapping (auto-detected if None)
        batch_size: Batch size for database queries
        validate_metadata: Whether to validate metadata before updating
        db_path: Database path
        
    Returns:
        Tuple of (enriched_dataframe, enrichment_statistics)
    """
    df_enriched = df.copy()
    stats = {
        'total_rows': len(df),
        'unique_players_processed': 0,
        'metadata_retrievals': 0,
        'fields_updated': {'height': 0, 'dob': 0, 'hand': 0},
        'validation_failures': 0,
        'database_queries': 0
    }
    
    # Auto-detect metadata columns if not provided
    if metadata_columns is None:
        metadata_columns = detect_metadata_columns(df)
        print(f"Auto-detected metadata columns: {metadata_columns}")
    
    # Check if we have any metadata columns to work with
    has_metadata_cols = any(
        bool(player_cols) for player_cols in metadata_columns.values()
    )
    if not has_metadata_cols:
        print("No metadata columns detected - skipping enrichment")
        return df_enriched, stats
    
    # Collect unique player IDs
    unique_player_ids = collect_unique_player_ids(df, player1_id_col, player2_id_col)
    stats['unique_players_processed'] = len(unique_player_ids)
    
    if not unique_player_ids:
        print("No player IDs found - skipping enrichment")
        return df_enriched, stats
    
    print(f"Enriching metadata for {len(unique_player_ids)} unique players...")
    
    # Process in batches
    player_ids_list = list(unique_player_ids)
    database_metadata = {}
    
    for i in range(0, len(player_ids_list), batch_size):
        batch_ids = player_ids_list[i:i + batch_size]
        batch_metadata = get_players_metadata_batch(batch_ids, db_path)
        database_metadata.update(batch_metadata)
        stats['database_queries'] += 1
        
        print(f"Retrieved metadata batch {i//batch_size + 1}: {len(batch_metadata)} players")
    
    stats['metadata_retrievals'] = len(database_metadata)
    
    # Create tracking columns for what was updated
    df_enriched['player1_metadata_updated'] = False
    df_enriched['player1_metadata_source'] = None
    df_enriched['player2_metadata_updated'] = False
    df_enriched['player2_metadata_source'] = None
    
    # Process each row
    for idx, row in df_enriched.iterrows():
        # Process Player 1
        if not pd.isna(row[player1_id_col]):
            player1_id = int(row[player1_id_col])
            updated_p1 = _enrich_player_metadata_in_row(
                df_enriched, idx, player1_id, 'player1',
                metadata_columns['player1'], database_metadata,
                validate_metadata, stats
            )
            if updated_p1:
                df_enriched.at[idx, 'player1_metadata_updated'] = True
                df_enriched.at[idx, 'player1_metadata_source'] = 'database'
        
        # Process Player 2
        if not pd.isna(row[player2_id_col]):
            player2_id = int(row[player2_id_col])
            updated_p2 = _enrich_player_metadata_in_row(
                df_enriched, idx, player2_id, 'player2',
                metadata_columns['player2'], database_metadata,
                validate_metadata, stats
            )
            if updated_p2:
                df_enriched.at[idx, 'player2_metadata_updated'] = True
                df_enriched.at[idx, 'player2_metadata_source'] = 'database'
    
    print(f"Enrichment complete:")
    print(f"  Fields updated - Height: {stats['fields_updated']['height']}, "
          f"DOB: {stats['fields_updated']['dob']}, Hand: {stats['fields_updated']['hand']}")
    print(f"  Validation failures: {stats['validation_failures']}")
    
    return df_enriched, stats


def _enrich_player_metadata_in_row(
    df: pd.DataFrame,
    row_idx: int,
    player_id: int,
    player_prefix: str,  # 'player1' or 'player2'
    metadata_col_mapping: Dict[str, str],
    database_metadata: Dict[int, Dict[str, Any]],
    validate_metadata: bool,
    stats: Dict[str, Any]
) -> bool:
    """
    Enrich metadata for a single player in a specific row.
    
    Returns:
        True if any metadata was updated, False otherwise
    """
    if player_id not in database_metadata:
        return False
    
    db_metadata = database_metadata[player_id]
    updated = False
    
    # Process each metadata field
    for metadata_type, column_name in metadata_col_mapping.items():
        if column_name not in df.columns:
            continue
        
        # Check if dataframe field is empty and database has value
        df_value = df.at[row_idx, column_name]
        db_value = db_metadata.get(metadata_type)
        
        # Only update if DF field is empty/null and DB has value
        if _is_empty_metadata_value(df_value) and db_value is not None:
            # Validate database value if requested
            if validate_metadata:
                validation_result = validate_player_metadata({metadata_type: db_value})
                validated_value = validation_result.get(metadata_type)
                
                if validated_value is None:
                    stats['validation_failures'] += 1
                    continue
                    
                db_value = validated_value
            
            # Update the dataframe
            df.at[row_idx, column_name] = db_value
            stats['fields_updated'][metadata_type] += 1
            updated = True
    
    return updated


def _is_empty_metadata_value(value: Any) -> bool:
    """
    Check if a metadata value should be considered empty and eligible for update.
    
    Args:
        value: Value to check
        
    Returns:
        True if value is empty/null, False otherwise
    """
    if pd.isna(value):
        return True
    
    if value is None:
        return True
    
    # Convert to string and check for empty-equivalent values
    str_value = str(value).strip().lower()
    empty_equivalents = ['', '0', 'nan', 'none', 'null', 'unknown', 'n/a', 'na']
    
    return str_value in empty_equivalents


def sync_metadata_to_database(
    df: pd.DataFrame,
    player1_id_col: str = 'player1_id',
    player2_id_col: str = 'player2_id',
    metadata_columns: Optional[Dict[str, Dict[str, str]]] = None,
    batch_size: int = 1000,
    validate_metadata: bool = True,
    db_path: str = "databases/tennis_players.db"
) -> Tuple[int, Dict[str, Any]]:
    """
    Sync non-empty dataframe metadata to database.
    
    Args:
        df: Input dataframe with metadata
        player1_id_col: Player 1 ID column name
        player2_id_col: Player 2 ID column name
        metadata_columns: Column mapping (auto-detected if None)
        batch_size: Batch size for database updates
        validate_metadata: Whether to validate metadata before updating
        db_path: Database path
        
    Returns:
        Tuple of (players_updated_count, sync_statistics)
    """
    stats = {
        'total_rows': len(df),
        'unique_players_processed': 0,
        'database_updates': 0,
        'fields_synced': {'height': 0, 'dob': 0, 'hand': 0},
        'validation_failures': 0,
        'database_queries': 0
    }
    
    # Auto-detect metadata columns if not provided
    if metadata_columns is None:
        metadata_columns = detect_metadata_columns(df)
        print(f"Auto-detected metadata columns for sync: {metadata_columns}")
    
    # Check if we have metadata columns
    has_metadata_cols = any(
        bool(player_cols) for player_cols in metadata_columns.values()
    )
    if not has_metadata_cols:
        print("No metadata columns detected - skipping database sync")
        return 0, stats
    
    # Collect metadata updates for each unique player
    player_updates = {}
    
    for idx, row in df.iterrows():
        # Process Player 1
        if not pd.isna(row[player1_id_col]):
            player1_id = int(row[player1_id_col])
            _collect_player_metadata_for_sync(
                player1_id, row, metadata_columns['player1'],
                player_updates, validate_metadata, stats
            )
        
        # Process Player 2
        if not pd.isna(row[player2_id_col]):
            player2_id = int(row[player2_id_col])
            _collect_player_metadata_for_sync(
                player2_id, row, metadata_columns['player2'],
                player_updates, validate_metadata, stats
            )
    
    stats['unique_players_processed'] = len(player_updates)
    
    if not player_updates:
        print("No metadata updates to sync to database")
        return 0, stats
    
    print(f"Syncing metadata to database for {len(player_updates)} players...")
    
    # Process updates in batches
    update_list = [{'player_id': pid, **metadata} for pid, metadata in player_updates.items()]
    updates_processed = 0
    
    for i in range(0, len(update_list), batch_size):
        batch_updates = update_list[i:i + batch_size]
        batch_updated_count = update_players_metadata_batch(batch_updates, db_path)
        updates_processed += batch_updated_count
        stats['database_queries'] += 1
        
        print(f"Synced batch {i//batch_size + 1}: {batch_updated_count}/{len(batch_updates)} players updated")
    
    stats['database_updates'] = updates_processed
    
    print(f"Database sync complete:")
    print(f"  Players updated: {updates_processed}")
    print(f"  Fields synced - Height: {stats['fields_synced']['height']}, "
          f"DOB: {stats['fields_synced']['dob']}, Hand: {stats['fields_synced']['hand']}")
    
    return updates_processed, stats


def _collect_player_metadata_for_sync(
    player_id: int,
    row: pd.Series,
    metadata_col_mapping: Dict[str, str],
    player_updates: Dict[int, Dict[str, Any]],
    validate_metadata: bool,
    stats: Dict[str, Any]
) -> None:
    """
    Collect metadata from dataframe row for database sync.
    
    Args:
        player_id: Player ID to collect metadata for
        row: DataFrame row containing metadata
        metadata_col_mapping: Column mapping for this player
        player_updates: Dict to store updates (modified in place)
        validate_metadata: Whether to validate metadata
        stats: Statistics dict (modified in place)
    """
    if player_id not in player_updates:
        player_updates[player_id] = {}
    
    for metadata_type, column_name in metadata_col_mapping.items():
        if column_name in row.index:
            df_value = row[column_name]
            
            # Only sync non-empty values
            if not _is_empty_metadata_value(df_value):
                # Validate if requested
                if validate_metadata:
                    validation_result = validate_player_metadata({metadata_type: df_value})
                    validated_value = validation_result.get(metadata_type)
                    
                    if validated_value is None:
                        stats['validation_failures'] += 1
                        continue
                    
                    df_value = validated_value
                
                # Store for batch update (latest value wins if multiple rows for same player)
                player_updates[player_id][metadata_type] = df_value
                stats['fields_synced'][metadata_type] += 1


def get_metadata_processing_summary(
    enrichment_stats: Dict[str, Any],
    sync_stats: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate combined summary of metadata processing operations.
    
    Args:
        enrichment_stats: Statistics from enrichment operation
        sync_stats: Optional statistics from sync operation
        
    Returns:
        Combined processing summary
    """
    summary = {
        'enrichment': {
            'total_rows': enrichment_stats.get('total_rows', 0),
            'unique_players_processed': enrichment_stats.get('unique_players_processed', 0),
            'fields_updated': enrichment_stats.get('fields_updated', {}),
            'validation_failures': enrichment_stats.get('validation_failures', 0)
        }
    }
    
    if sync_stats:
        summary['sync'] = {
            'unique_players_processed': sync_stats.get('unique_players_processed', 0),
            'database_updates': sync_stats.get('database_updates', 0),
            'fields_synced': sync_stats.get('fields_synced', {}),
            'validation_failures': sync_stats.get('validation_failures', 0)
        }
    
    # Calculate totals
    total_fields_updated = sum(enrichment_stats.get('fields_updated', {}).values())
    total_validation_failures = enrichment_stats.get('validation_failures', 0)
    
    if sync_stats:
        total_validation_failures += sync_stats.get('validation_failures', 0)
    
    summary['totals'] = {
        'fields_updated': total_fields_updated,
        'validation_failures': total_validation_failures,
        'database_operations': enrichment_stats.get('database_queries', 0) + 
                              (sync_stats.get('database_queries', 0) if sync_stats else 0)
    }
    
    return summary


if __name__ == "__main__":
    # Test metadata processing functions
    print("Testing metadata processing functions...")
    
    try:
        # Create test dataframe
        test_df = pd.DataFrame({
            'player1_id': [100001, 100002, 100003],
            'player1_name': ['Roger Federer', 'Rafael Nadal', 'Andy Murray'],
            'player1_ht': [None, 185, ''],          # Mixed empty values
            'player1_dob': ['', '1986-06-03', None], # Mixed empty values  
            'player1_hand': ['R', '', 'L'],          # Mixed empty values
            'player2_id': [100004, 100005, 100006],
            'player2_name': ['Novak Djokovic', 'Stan Wawrinka', 'David Ferrer'],
            'player2_ht': [188, None, 175],
            'player2_dob': ['1987-05-22', '', '1982-04-02'],
            'player2_hand': ['', 'R', '']
        })
        
        print(f"Test DataFrame shape: {test_df.shape}")
        print("Original metadata columns:")
        for col in ['player1_ht', 'player1_dob', 'player1_hand', 'player2_ht', 'player2_dob', 'player2_hand']:
            print(f"  {col}: {test_df[col].tolist()}")
        
        # Test column detection
        detected_columns = detect_metadata_columns(test_df)
        print(f"\nDetected metadata columns: {detected_columns}")
        
        # Test unique player ID collection
        unique_ids = collect_unique_player_ids(test_df)
        print(f"Unique player IDs: {sorted(unique_ids)}")
        
        # Test enrichment (this would need actual database)
        print(f"\nMetadata processing tests completed!")
        
    except Exception as e:
        print(f"Metadata processing test: FAILED - {e}")
        import traceback
        traceback.print_exc()