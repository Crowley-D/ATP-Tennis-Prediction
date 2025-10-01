"""
DataFrame processing for player ID matching across dual columns.
"""

import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from tqdm import tqdm

from ..database.connection import get_db_connection, close_all_connections
from ..matching.matching_engine import PlayerMatchingEngine, MatchResult
from ..matching.name_processing import (
    merge_player_names, preprocess_player_name, get_source_name
)
from .metadata_processor import (
    enrich_dataframe_metadata, sync_metadata_to_database,
    get_metadata_processing_summary, detect_metadata_columns
)


def validate_dataframe_structure(
    df: pd.DataFrame,
    source_col: str = 'source',
    player1_id_col: str = 'player1_id',
    player1_name_col: str = 'player1_name',
    player2_id_col: str = 'player2_id',
    player2_name_col: str = 'player2_name',
    player1_first_col: Optional[str] = None,
    player1_last_col: Optional[str] = None,
    player2_first_col: Optional[str] = None,
    player2_last_col: Optional[str] = None,
    player1_height_col: str = 'player1_ht',
    player2_height_col: str = 'player2_ht',
    player1_dob_col: str = 'player1_dob',
    player2_dob_col: str = 'player2_dob',
    player1_hand_col: str = 'player1_hand',
    player2_hand_col: str = 'player2_hand'
) -> Tuple[bool, str]:
    """
    Validate DataFrame structure for player processing.
    
    Args:
        df: DataFrame to validate
        source_col: Source column name
        player1_id_col: Player 1 ID column name
        player1_name_col: Player 1 name column name
        player2_id_col: Player 2 ID column name
        player2_name_col: Player 2 name column name
        player1_first_col: Optional player 1 first name column
        player1_last_col: Optional player 1 last name column
        player2_first_col: Optional player 2 first name column
        player2_last_col: Optional player 2 last name column
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    # Check for required columns
    required_columns = [source_col, player1_id_col, player2_id_col]
    
    # Check if we have name columns OR separate first/last columns
    if player1_first_col and player1_last_col:
        required_columns.extend([player1_first_col, player1_last_col])
    else:
        required_columns.append(player1_name_col)
    
    if player2_first_col and player2_last_col:
        required_columns.extend([player2_first_col, player2_last_col])
    else:
        required_columns.append(player2_name_col)
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f"Missing required columns: {missing_columns}"
    
    # Check source column values
    unique_sources = df[source_col].unique()
    invalid_sources = [s for s in unique_sources if s not in [0, 1, 2]]
    if invalid_sources:
        return False, f"Invalid source codes found: {invalid_sources}. Must be 0, 1, or 2."
    
    # Check for completely empty ID/name columns
    empty_p1_ids = df[player1_id_col].isna().sum()
    empty_p2_ids = df[player2_id_col].isna().sum()
    
    if empty_p1_ids == len(df):
        return False, f"All values in {player1_id_col} are empty"
    if empty_p2_ids == len(df):
        return False, f"All values in {player2_id_col} are empty"
    
    return True, ""


def prepare_player_names(
    df: pd.DataFrame,
    player1_name_col: str = 'player1_name',
    player2_name_col: str = 'player2_name',
    player1_first_col: Optional[str] = None,
    player1_last_col: Optional[str] = None,
    player2_first_col: Optional[str] = None,
    player2_last_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Prepare player names from either combined name columns or separate first/last columns.
    
    Args:
        df: Input DataFrame
        player1_name_col: Player 1 name column
        player2_name_col: Player 2 name column
        player1_first_col: Optional player 1 first name column
        player1_last_col: Optional player 1 last name column
        player2_first_col: Optional player 2 first name column
        player2_last_col: Optional player 2 last name column
        
    Returns:
        DataFrame with prepared name columns
    """
    df = df.copy()
    
    # Handle player 1 names
    if player1_first_col and player1_last_col and player1_first_col in df.columns and player1_last_col in df.columns:
        # Merge first and last names
        df[player1_name_col] = df.apply(
            lambda row: merge_player_names(row[player1_first_col], row[player1_last_col]),
            axis=1
        )
    
    # Handle player 2 names
    if player2_first_col and player2_last_col and player2_first_col in df.columns and player2_last_col in df.columns:
        # Merge first and last names
        df[player2_name_col] = df.apply(
            lambda row: merge_player_names(row[player2_first_col], row[player2_last_col]),
            axis=1
        )
    
    return df


def process_players_dataframe(
    df: pd.DataFrame,
    matches_df: Optional[pd.DataFrame] = None,
    source_col: str = 'source',
    player1_id_col: str = 'player1_id',
    player1_name_col: str = 'player1_name',
    player2_id_col: str = 'player2_id',
    player2_name_col: str = 'player2_name',
    player1_first_col: Optional[str] = None,
    player1_last_col: Optional[str] = None,
    player2_first_col: Optional[str] = None,
    player2_last_col: Optional[str] = None,
    player1_height_col: str = 'player1_ht',
    player2_height_col: str = 'player2_ht',
    player1_dob_col: str = 'player1_dob',
    player2_dob_col: str = 'player2_dob',
    player1_hand_col: str = 'player1_hand',
    player2_hand_col: str = 'player2_hand',
    database_path: str = "databases/tennis_players.db",
    auto_resolve: bool = False,
    enrich_metadata: bool = True,
    update_database_metadata: bool = False,
    metadata_batch_size: int = 1000,
    validate_metadata: bool = True,
    metadata_columns: Optional[Dict[str, Dict[str, str]]] = None
) -> pd.DataFrame:
    """
    Process both player columns in a DataFrame to add unified player IDs and enrich metadata.
    
    Args:
        df: Input DataFrame with player data
        matches_df: Optional DataFrame with match history for disambiguation
        source_col: Column containing source codes (0/1/2)
        player1_id_col: Column with player 1 IDs
        player1_name_col: Column with player 1 names
        player2_id_col: Column with player 2 IDs
        player2_name_col: Column with player 2 names
        player1_first_col: Optional player 1 first name column
        player1_last_col: Optional player 1 last name column
        player2_first_col: Optional player 2 first name column
        player2_last_col: Optional player 2 last name column
        player1_height_col: Player 1 height column name (default: 'player1_ht')
        player2_height_col: Player 2 height column name (default: 'player2_ht')
        player1_dob_col: Player 1 date of birth column name (default: 'player1_dob')
        player2_dob_col: Player 2 date of birth column name (default: 'player2_dob')
        player1_hand_col: Player 1 hand column name (default: 'player1_hand')
        player2_hand_col: Player 2 hand column name (default: 'player2_hand')
        database_path: Path to SQLite database
        auto_resolve: Whether to automatically resolve ambiguous matches
        enrich_metadata: Whether to enrich empty metadata fields from database (default: True)
        update_database_metadata: Whether to update database with dataframe metadata (default: False)
        metadata_batch_size: Batch size for metadata database operations (default: 1000)
        validate_metadata: Whether to validate metadata before updating (default: True)
        metadata_columns: Custom metadata column mapping (auto-detected if None)
        
    Returns:
        DataFrame with replaced player ID columns and enriched metadata
        (original IDs stored in original_player1_id/original_player2_id)
    """
    # Validate input
    valid, error_msg = validate_dataframe_structure(
        df, source_col, player1_id_col, player1_name_col, player2_id_col, player2_name_col,
        player1_first_col, player1_last_col, player2_first_col, player2_last_col,
        player1_height_col, player2_height_col, player1_dob_col, player2_dob_col,
        player1_hand_col, player2_hand_col
    )
    
    if not valid:
        raise ValueError(f"DataFrame validation failed: {error_msg}")
    
    print(f"Processing DataFrame with {len(df)} rows...")
    
    # Prepare names if using separate first/last columns
    df = prepare_player_names(
        df, player1_name_col, player2_name_col,
        player1_first_col, player1_last_col, player2_first_col, player2_last_col
    )
    
    # Initialize matching engine
    matching_engine = PlayerMatchingEngine(matches_df=matches_df, db_path=database_path)
    
    # Store original IDs for backup and initialize metadata columns
    df['original_player1_id'] = df[player1_id_col].copy()
    df['original_player2_id'] = df[player2_id_col].copy()
    df['player1_match_confidence'] = None
    df['player1_match_action'] = None
    df['player2_match_confidence'] = None
    df['player2_match_action'] = None
    
    # Track unique players to avoid redundant processing
    processed_players = {}
    
    # Sort by tournament date if available to process recent matches first
    if 'tourney_date' in df.columns:
        df = df.sort_values('tourney_date', ascending=False)
    elif 'date' in df.columns:
        df = df.sort_values('date', ascending=False)
    
    # Process each row with progress bar (configured for Windows terminal compatibility)
    progress_bar = tqdm(
        total=len(df), 
        desc="Processing player matches", 
        unit="rows",
        dynamic_ncols=True,  # Adjust width dynamically
        leave=True,          # Keep progress bar after completion
        miniters=1,          # Update every iteration
        mininterval=0.1      # Update at most every 0.1 seconds
    )
    
    try:
        for idx, row in df.iterrows():
            source_code = int(row[source_col])
            
            # Process Player 1
            p1_key = (source_code, str(row[player1_id_col]))
            
            if p1_key not in processed_players and not pd.isna(row[player1_id_col]) and str(row[player1_name_col]).strip():
                try:
                    result = matching_engine.match_player(
                        source_code=source_code,
                        source_id=str(row[player1_id_col]),
                        original_name=str(row[player1_name_col]),
                        auto_resolve=auto_resolve
                    )
                    processed_players[p1_key] = result
                except Exception as e:
                    print(f"Error processing player 1 at row {idx}: {e}")
                    processed_players[p1_key] = MatchResult(
                        action='error',
                        player_id=0,
                        confidence='low',
                        message=str(e)
                    )
            
            # Set Player 1 results (replace original column)
            if p1_key in processed_players:
                result = processed_players[p1_key]
                df.at[idx, player1_id_col] = result.player_id if result.player_id > 0 else None
                df.at[idx, 'player1_match_confidence'] = result.confidence
                df.at[idx, 'player1_match_action'] = result.action
            
            # Process Player 2
            p2_key = (source_code, str(row[player2_id_col]))
            
            if p2_key not in processed_players and not pd.isna(row[player2_id_col]) and str(row[player2_name_col]).strip():
                try:
                    result = matching_engine.match_player(
                        source_code=source_code,
                        source_id=str(row[player2_id_col]),
                        original_name=str(row[player2_name_col]),
                        auto_resolve=auto_resolve
                    )
                    processed_players[p2_key] = result
                except Exception as e:
                    print(f"Error processing player 2 at row {idx}: {e}")
                    processed_players[p2_key] = MatchResult(
                        action='error',
                        player_id=0,
                        confidence='low',
                        message=str(e)
                    )
            
            # Set Player 2 results (replace original column)
            if p2_key in processed_players:
                result = processed_players[p2_key]
                df.at[idx, player2_id_col] = result.player_id if result.player_id > 0 else None
                df.at[idx, 'player2_match_confidence'] = result.confidence
                df.at[idx, 'player2_match_action'] = result.action
            
            # Update progress bar with throttled updates
            if idx % 50 == 0 or idx == len(df) - 1:  # Update every 50 rows or on last row
                progress_bar.update(50 if idx % 50 == 0 else (idx % 50) + 1)
                progress_bar.set_postfix({
                    'unique_players': len(processed_players),
                    'row': f"{idx + 1}/{len(df)}"
                })
    
    finally:
        progress_bar.close()
    
    print(f"Processing complete. Processed {len(processed_players)} unique players.")
    
    # Generate processing report
    all_results = list(processed_players.values())
    stats = matching_engine.get_match_statistics(all_results)
    
    print("\nPlayer ID Processing Statistics:")
    print("-" * 35)
    print(f"Total unique players: {stats['total']}")
    print(f"Linked (exact): {stats['linked_exact']} ({stats['linked_exact_pct']:.1f}%)")
    print(f"Linked (fuzzy): {stats['linked_fuzzy']} ({stats['linked_fuzzy_pct']:.1f}%)")
    print(f"Created new: {stats['created_new']} ({stats['created_new_pct']:.1f}%)")
    print(f"Manual needed: {stats['manual_needed']} ({stats['manual_needed_pct']:.1f}%)")
    print(f"Errors: {stats['errors']} ({stats['errors_pct']:.1f}%)")
    
    # ========================================
    # METADATA PROCESSING
    # ========================================
    
    enrichment_stats = {}
    sync_stats = {}
    
    if enrich_metadata or update_database_metadata:
        print("\n" + "="*50)
        print("METADATA PROCESSING")
        print("="*50)
        
        # Build metadata column mapping if not provided
        if metadata_columns is None:
            metadata_columns = {
                'player1': {
                    'height': player1_height_col,
                    'dob': player1_dob_col,
                    'hand': player1_hand_col
                },
                'player2': {
                    'height': player2_height_col,
                    'dob': player2_dob_col,
                    'hand': player2_hand_col
                }
            }
            
            # Auto-detect columns if they exist in dataframe
            detected = detect_metadata_columns(df)
            for player in ['player1', 'player2']:
                if detected[player]:  # Update with detected columns if found
                    metadata_columns[player].update(detected[player])
    
    # Enrich metadata from database
    if enrich_metadata:
        print("\nEnriching empty metadata fields from database...")
        df, enrichment_stats = enrich_dataframe_metadata(
            df,
            player1_id_col=player1_id_col,
            player2_id_col=player2_id_col,
            metadata_columns=metadata_columns,
            batch_size=metadata_batch_size,
            validate_metadata=validate_metadata,
            db_path=database_path
        )
    
    # Sync metadata to database  
    if update_database_metadata:
        print("\nSyncing dataframe metadata to database...")
        updated_count, sync_stats = sync_metadata_to_database(
            df,
            player1_id_col=player1_id_col,
            player2_id_col=player2_id_col,
            metadata_columns=metadata_columns,
            batch_size=metadata_batch_size,
            validate_metadata=validate_metadata,
            db_path=database_path
        )
    
    # Generate combined metadata processing summary
    if enrich_metadata or update_database_metadata:
        metadata_summary = get_metadata_processing_summary(
            enrichment_stats, 
            sync_stats if update_database_metadata else None
        )
        
        print("\nMetadata Processing Summary:")
        print("-" * 30)
        if enrichment_stats:
            enrich = metadata_summary['enrichment']
            print(f"Enrichment:")
            print(f"  Fields updated: {sum(enrich['fields_updated'].values())}")
            for field, count in enrich['fields_updated'].items():
                if count > 0:
                    print(f"    {field}: {count}")
        
        if sync_stats:
            sync = metadata_summary['sync']
            print(f"Database Sync:")
            print(f"  Players updated: {sync['database_updates']}")
            print(f"  Fields synced: {sum(sync['fields_synced'].values())}")
        
        if metadata_summary['totals']['validation_failures'] > 0:
            print(f"Validation failures: {metadata_summary['totals']['validation_failures']}")
    
    return df


def process_dataframe_programmatically(
    df: pd.DataFrame,
    source_name: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Main entry point for programmatic DataFrame processing.
    
    This is the primary function users should call to process their DataFrames.
    
    Args:
        df: Input DataFrame with player data
        source_name: Optional fixed source name (legacy mode)
        **kwargs: Additional arguments passed to process_players_dataframe
        
    Returns:
        DataFrame with unified player IDs
    """
    if source_name:
        # Legacy mode: all rows treated as same source
        source_mapping = {
            'main_dataset': 0,
            'infosys_api': 1,
            'tennis_api': 2
        }
        
        if source_name not in source_mapping:
            raise ValueError(f"Invalid source_name: {source_name}. Must be one of: {list(source_mapping.keys())}")
        
        df = df.copy()
        df['source'] = source_mapping[source_name]
    
    return process_players_dataframe(df, **kwargs)


def get_processing_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get a summary of DataFrame processing results.
    
    Args:
        df: Processed DataFrame with unified player IDs
        
    Returns:
        Summary statistics dictionary
    """
    # Check for processing metadata columns instead of unified columns
    if 'player1_match_confidence' not in df.columns or 'player2_match_confidence' not in df.columns:
        return {'error': 'DataFrame has not been processed yet'}
    
    total_matches = len(df)
    
    # Detect the actual player ID column names from the DataFrame
    player1_id_col = 'player1_id'  # Default
    player2_id_col = 'player2_id'  # Default
    
    for col in df.columns:
        if col.startswith('player1') and 'id' in col and 'original' not in col and 'confidence' not in col and 'action' not in col:
            player1_id_col = col
        elif col.startswith('player2') and 'id' in col and 'original' not in col and 'confidence' not in col and 'action' not in col:
            player2_id_col = col
    
    p1_matched = df[player1_id_col].notna().sum()
    p2_matched = df[player2_id_col].notna().sum()
    both_matched = ((df[player1_id_col].notna()) & (df[player2_id_col].notna())).sum()
    
    # Confidence distribution
    p1_high_conf = (df['player1_match_confidence'] == 'high').sum()
    p1_med_conf = (df['player1_match_confidence'] == 'medium').sum()
    p1_low_conf = (df['player1_match_confidence'] == 'low').sum()
    
    p2_high_conf = (df['player2_match_confidence'] == 'high').sum()
    p2_med_conf = (df['player2_match_confidence'] == 'medium').sum()
    p2_low_conf = (df['player2_match_confidence'] == 'low').sum()
    
    # Action distribution
    actions = {}
    for col in ['player1_match_action', 'player2_match_action']:
        if col in df.columns:
            action_counts = df[col].value_counts()
            for action, count in action_counts.items():
                actions[action] = actions.get(action, 0) + count
    
    # Metadata statistics
    metadata_stats = {}
    if 'player1_metadata_updated' in df.columns and 'player2_metadata_updated' in df.columns:
        p1_metadata_updated = df['player1_metadata_updated'].sum() if 'player1_metadata_updated' in df.columns else 0
        p2_metadata_updated = df['player2_metadata_updated'].sum() if 'player2_metadata_updated' in df.columns else 0
        
        metadata_stats = {
            'player1_metadata_updated': p1_metadata_updated,
            'player2_metadata_updated': p2_metadata_updated,
            'total_metadata_updated': p1_metadata_updated + p2_metadata_updated,
            'metadata_update_rate': ((p1_metadata_updated + p2_metadata_updated) / (total_matches * 2)) * 100 if total_matches > 0 else 0
        }
        
        # Source distribution for metadata
        metadata_source_counts = {}
        for col in ['player1_metadata_source', 'player2_metadata_source']:
            if col in df.columns:
                source_counts = df[col].value_counts()
                for source, count in source_counts.items():
                    if source is not None:  # Skip None values
                        metadata_source_counts[source] = metadata_source_counts.get(source, 0) + count
        
        metadata_stats['metadata_source_distribution'] = metadata_source_counts
    
    result = {
        'total_matches': total_matches,
        'player1_matched': p1_matched,
        'player2_matched': p2_matched,
        'both_players_matched': both_matched,
        'match_completion_rate': both_matched / total_matches * 100 if total_matches > 0 else 0,
        'confidence_distribution': {
            'high': p1_high_conf + p2_high_conf,
            'medium': p1_med_conf + p2_med_conf,
            'low': p1_low_conf + p2_low_conf
        },
        'action_distribution': actions
    }
    
    # Add metadata stats if available
    if metadata_stats:
        result['metadata'] = metadata_stats
    
    return result


if __name__ == "__main__":
    # Test DataFrame processing
    print("Testing DataFrame processing...")
    
    try:
        # Create test DataFrame
        test_df = pd.DataFrame({
            'source': [0, 0, 1, 1],
            'player1_id': ['100001', '100002', 'inf_001', 'inf_002'],
            'player1_name': ['Test Player One', 'Test Player Two', 'Player One', 'Player Two'],
            'player2_id': ['100003', '100004', 'inf_003', 'inf_004'],
            'player2_name': ['Test Player Three', 'Test Player Four', 'Player Three', 'Player Four'],
            'match_id': ['M001', 'M002', 'M003', 'M004']
        })
        
        print(f"Test DataFrame shape: {test_df.shape}")
        
        # Process DataFrame
        result_df = process_players_dataframe(test_df, auto_resolve=True)
        
        print(f"Result DataFrame shape: {result_df.shape}")
        print(f"New columns: {[col for col in result_df.columns if col not in test_df.columns]}")
        print(f"Player1 IDs replaced: {result_df['player1_id'].notna().sum()}/{len(result_df)}")
        print(f"Player2 IDs replaced: {result_df['player2_id'].notna().sum()}/{len(result_df)}")
        
        # Get summary
        summary = get_processing_summary(result_df)
        print(f"Processing summary: {summary}")
        
        print("DataFrame processing test: SUCCESS")
        
    except Exception as e:
        print(f"DataFrame processing test: FAILED - {e}")
    
    finally:
        close_all_connections()