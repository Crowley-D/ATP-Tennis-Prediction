"""
DataFrame processing integration for tennis tournament matching.
"""

import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(iterable, **kwargs):
        return iterable

from tennis_matching.matching.matching_engine import match_tournament, check_data_quality_issues
from tennis_matching.matching.manual_review import display_error_summary, prompt_for_missing_tourney_level


def map_source_code_to_name(source_code: int) -> str:
    """
    Map source code to source name.
    
    Args:
        source_code: Integer source code (0, 1, or 2)
        
    Returns:
        str: Source name
        
    Raises:
        ValueError: If source code is not valid
    """
    source_mapping = {
        0: 'main_dataset',
        1: 'infosys_api', 
        2: 'tennis_api'
    }
    
    if source_code not in source_mapping:
        raise ValueError(f"Invalid source code: {source_code}. Must be 0, 1, or 2.")
    
    return source_mapping[source_code]


def process_matches_dataframe(
    df: pd.DataFrame,
    source_name: str = None,
    source_col: str = 'source',
    tourney_id_col: str = 'tourney_id',
    tournament_name_col: str = 'tourney_name',
    tourney_level_col: str = 'tourney_level'
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Process a DataFrame of tennis matches and update tournament IDs with level-based tournament IDs.
    
    Args:
        df: Input DataFrame with tournament information
        source_name: Fixed source name (deprecated, use source_col instead)
        source_col: Column name containing source codes (0=main_dataset, 1=infosys_api, 2=tennis_api)
        tourney_id_col: Column name containing tournament IDs
        tournament_name_col: Column name containing tournament names
        tourney_level_col: Column name containing tournament level (G, M, F, D, A, C, S)
        
    Returns:
        Tuple[pd.DataFrame, List[Dict]]: (updated_df_with_tourney_ids, processing_report)
    """
    # Determine how to get source information
    if source_name is not None:
        # Legacy mode - use fixed source name
        print(f"\nProcessing {len(df)} matches from {source_name}")
        use_source_column = False
    else:
        # New mode - use source column
        if source_col not in df.columns:
            raise ValueError(f"Source column '{source_col}' not found in DataFrame")
        unique_sources = df[source_col].unique()
        source_names = [map_source_code_to_name(code) for code in unique_sources]
        print(f"\nProcessing {len(df)} matches from sources: {', '.join(source_names)}")
        use_source_column = True
    
    print("=" * 60)
    
    # Validate required columns
    if tourney_id_col not in df.columns:
        raise ValueError(f"Column '{tourney_id_col}' not found in DataFrame")
    if tournament_name_col not in df.columns:
        raise ValueError(f"Column '{tournament_name_col}' not found in DataFrame")
    if tourney_level_col not in df.columns:
        raise ValueError(f"Column '{tourney_level_col}' not found in DataFrame")
    
    # Initialize results
    updated_df = df.copy()
    updated_df['tourney_id'] = None  # New tournament ID (4-digit)
    
    processing_report = []
    match_stats = {
        'total_processed': 0,
        'auto_matched': 0,
        'manual_confirmed': 0,
        'new_tournaments': 0,
        'errors': 0,
        'skipped': 0
    }
    
    # Group by unique combinations
    if use_source_column:
        unique_tournaments = df[[source_col, tourney_id_col, tournament_name_col, tourney_level_col]].drop_duplicates()
        print(f"Found {len(unique_tournaments)} unique source/tournament ID/name/level combinations")
    else:
        unique_tournaments = df[[tourney_id_col, tournament_name_col, tourney_level_col]].drop_duplicates()
        print(f"Found {len(unique_tournaments)} unique tournament ID/name/level combinations")
    
    print("Starting tournament matching process...\n")
    
    # Process each unique tournament
    tourney_id_map = {}  # Cache for (source_name, source_id, tournament_name) -> tourney_id mapping
    
    progress_iter = tqdm(unique_tournaments.iterrows(), total=len(unique_tournaments), desc="Processing tournaments")
    
    for _, row in progress_iter:
        tourney_id = str(row[tourney_id_col])
        tournament_name = str(row[tournament_name_col])
        tourney_level = row[tourney_level_col]
        
        # Handle missing tourney_level
        if pd.isna(tourney_level) or tourney_level == '':
            if use_source_column:
                source_code = row[source_col]
                current_source_name = map_source_code_to_name(source_code)
            else:
                current_source_name = source_name
            tourney_level = prompt_for_missing_tourney_level(tournament_name, current_source_name)
            # Update the row in the original DataFrame
            mask = (df[tourney_id_col] == row[tourney_id_col]) & (df[tournament_name_col] == row[tournament_name_col])
            if use_source_column:
                mask = mask & (df[source_col] == row[source_col])
            df.loc[mask, tourney_level_col] = tourney_level
            updated_df.loc[mask, tourney_level_col] = tourney_level
        
        # Determine source name for this row
        if use_source_column:
            source_code = row[source_col]
            current_source_name = map_source_code_to_name(source_code)
        else:
            current_source_name = source_name
        
        # Skip if we've already processed this combination
        cache_key = (current_source_name, tourney_id, tournament_name)
        if cache_key in tourney_id_map:
            continue
        
        try:
            # Perform tournament matching
            new_tourney_id, confidence, action, stored_new_id = match_tournament(
                source_name=current_source_name,
                source_id=tourney_id,
                tournament_name=tournament_name,
                tourney_level=tourney_level
            )
            
            # Cache the result
            tourney_id_map[cache_key] = {
                'tourney_id': new_tourney_id,
                'confidence': confidence,
                'action': action
            }
            
            # Update statistics
            match_stats['total_processed'] += 1
            
            if new_tourney_id:
                if 'auto' in action or 'strict' in action:
                    match_stats['auto_matched'] += 1
                elif 'manual' in action or 'confirmed' in action:
                    match_stats['manual_confirmed'] += 1
                elif 'created' in action or 'new' in action:
                    match_stats['new_tournaments'] += 1
            else:
                match_stats['skipped'] += 1
            
            # Add to processing report
            processing_report.append({
                'source_tourney_id': tourney_id,
                'tournament_name': tournament_name,
                'tourney_level': tourney_level,
                'assigned_tourney_id': new_tourney_id,
                'confidence': confidence,
                'action': action,
                'source_name': current_source_name
            })
            
        except Exception as e:
            print(f"\nError processing tournament {tourney_id} '{tournament_name}': {e}")
            match_stats['errors'] += 1
            
            processing_report.append({
                'source_tourney_id': tourney_id,
                'tournament_name': tournament_name,
                'tourney_level': tourney_level if 'tourney_level' in locals() else 'unknown',
                'assigned_tourney_id': None,
                'confidence': 'error',
                'action': f'error: {str(e)}',
                'source_name': current_source_name
            })
    
    # Apply results to all rows in the DataFrame
    print("\nApplying results to DataFrame...")
    for idx, row in updated_df.iterrows():
        # Get the original tournament ID from the original DataFrame
        source_tourney_id = str(df.iloc[idx][tourney_id_col])
        
        # Determine source name for this row
        if use_source_column:
            source_code = row[source_col]
            current_source_name = map_source_code_to_name(source_code)
        else:
            current_source_name = source_name
            
        source_tournament_name = str(row[tournament_name_col])
        cache_key = (current_source_name, source_tourney_id, source_tournament_name)
        
        if cache_key in tourney_id_map:
            result = tourney_id_map[cache_key]
            updated_df.at[idx, 'tourney_id'] = result['tourney_id']  # Set the new 4-digit tournament ID
            updated_df.at[idx, 'match_confidence'] = result['confidence']
            updated_df.at[idx, 'match_action'] = result['action']
    
    # Display processing summary
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total matches processed: {len(df)}")
    print(f"Unique tournaments: {match_stats['total_processed']}")
    print(f"Auto-matched: {match_stats['auto_matched']}")
    print(f"Manual confirmations: {match_stats['manual_confirmed']}")
    print(f"New tournaments created: {match_stats['new_tournaments']}")
    print(f"Skipped/Manual needed: {match_stats['skipped']}")
    print(f"Errors: {match_stats['errors']}")
    
    if match_stats['total_processed'] > 0:
        auto_match_rate = (match_stats['auto_matched'] / match_stats['total_processed']) * 100
        print(f"Auto-match rate: {auto_match_rate:.1f}%")
    
    return updated_df, processing_report


def validate_dataframe_structure(
    df: pd.DataFrame,
    tourney_id_col: str = 'tourney_id',
    tournament_name_col: str = 'tourney_name',
    tourney_level_col: str = 'tourney_level',
    source_col: str = 'source',
    require_source_col: bool = True
) -> List[str]:
    """
    Validate DataFrame structure and content for processing.
    
    Args:
        df: DataFrame to validate
        tourney_id_col: Column name for tournament IDs
        tournament_name_col: Column name for tournament names
        tourney_level_col: Column name for tournament level
        source_col: Column name for source codes
        require_source_col: Whether source column is required
        
    Returns:
        List[str]: List of validation issues found
    """
    issues = []
    
    # Check required columns exist
    if tourney_id_col not in df.columns:
        issues.append(f"Missing required column: '{tourney_id_col}'")
    if tournament_name_col not in df.columns:
        issues.append(f"Missing required column: '{tournament_name_col}'")
    if tourney_level_col not in df.columns:
        issues.append(f"Missing required column: '{tourney_level_col}'")
    if require_source_col and source_col not in df.columns:
        issues.append(f"Missing required source column: '{source_col}'")
    
    if issues:  # Can't continue validation without required columns
        return issues
    
    # Check for null/empty values
    null_ids = df[tourney_id_col].isnull().sum()
    if null_ids > 0:
        issues.append(f"{null_ids} rows have null tournament IDs")
    
    null_names = df[tournament_name_col].isnull().sum()
    if null_names > 0:
        issues.append(f"{null_names} rows have null tournament names")
    
    empty_names = (df[tournament_name_col] == '').sum()
    if empty_names > 0:
        issues.append(f"{empty_names} rows have empty tournament names")
    
    # Check data types
    if not pd.api.types.is_string_dtype(df[tournament_name_col]):
        issues.append(f"Tournament name column should be string type, got {df[tournament_name_col].dtype}")
    
    # Check tourney_level column values
    if tourney_level_col in df.columns:
        valid_levels = ['G', 'M', 'F', 'D', 'A', 'C', 'S', 'O']
        non_null_levels = df[df[tourney_level_col].notna()][tourney_level_col]
        invalid_levels = non_null_levels[~non_null_levels.isin(valid_levels)].unique()
        if len(invalid_levels) > 0:
            issues.append(f"Invalid tournament levels found: {list(invalid_levels)}. Must be one of: {valid_levels}")
        
        missing_levels = df[tourney_level_col].isnull().sum()
        if missing_levels > 0:
            issues.append(f"{missing_levels} rows have missing tournament levels (will prompt user)")
    
    # Check source column values if present
    if require_source_col and source_col in df.columns:
        invalid_sources = df[~df[source_col].isin([0, 1, 2])][source_col].unique()
        if len(invalid_sources) > 0:
            issues.append(f"Invalid source codes found: {list(invalid_sources)}. Must be 0, 1, or 2.")
    
    # Check for reasonable data
    if len(df) == 0:
        issues.append("DataFrame is empty")
    
    if require_source_col and source_col in df.columns:
        unique_tournaments = df[[source_col, tourney_id_col, tournament_name_col, tourney_level_col]].drop_duplicates()
    else:
        unique_tournaments = df[[tourney_id_col, tournament_name_col, tourney_level_col]].drop_duplicates()
        
    if len(unique_tournaments) == 0:
        issues.append("No unique tournament combinations found")
    
    return issues


def run_data_quality_check(source_name: str) -> None:
    """
    Run data quality checks on processed tournaments and display results.
    
    Args:
        source_name: Name of the source to check
    """
    print(f"\nRunning data quality checks for {source_name}...")
    
    issues = check_data_quality_issues(source_name)
    display_error_summary(issues)


def export_processing_report(
    processing_report: List[Dict[str, Any]],
    output_file: str = "tournament_processing_report.csv"
) -> None:
    """
    Export processing report to CSV for review.
    
    Args:
        processing_report: List of processing results
        output_file: Output CSV file path
    """
    if not processing_report:
        print("No processing report data to export.")
        return
    
    report_df = pd.DataFrame(processing_report)
    report_df.to_csv(output_file, index=False)
    print(f"\nProcessing report exported to: {output_file}")


def get_unmatched_tournaments(
    processing_report: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Get list of tournaments that were not matched and need manual review.
    
    Args:
        processing_report: List of processing results
        
    Returns:
        List[Dict]: Tournaments requiring manual review
    """
    unmatched = []
    
    for entry in processing_report:
        if (entry['assigned_tourney_id'] is None and 
            'skip' in entry['action'].lower() and 
            'error' not in entry['action'].lower()):
            unmatched.append(entry)
    
    return unmatched


def batch_process_multiple_sources(
    dataframes_and_sources: List[Tuple[pd.DataFrame, str]],
    tourney_id_col: str = 'tourney_id',
    tournament_name_col: str = 'tourney_name',
    tourney_level_col: str = 'tourney_level'
) -> Dict[str, Tuple[pd.DataFrame, List[Dict[str, Any]]]]:
    """
    Process multiple DataFrames from different sources in batch.
    
    Args:
        dataframes_and_sources: List of (DataFrame, source_name) tuples
        tourney_id_col: Column name for tournament IDs
        tournament_name_col: Column name for tournament names
        tourney_level_col: Column name for tournament level
        
    Returns:
        Dict: Results keyed by source_name
    """
    results = {}
    
    print(f"\nBATCH PROCESSING {len(dataframes_and_sources)} SOURCES")
    print("=" * 80)
    
    for i, (df, source_name) in enumerate(dataframes_and_sources, 1):
        print(f"\nProcessing source {i}/{len(dataframes_and_sources)}: {source_name}")
        
        # Validate DataFrame
        validation_issues = validate_dataframe_structure(df, tourney_id_col, tournament_name_col, tourney_level_col)
        if validation_issues:
            print(f"Validation issues found for {source_name}:")
            for issue in validation_issues:
                print(f"  - {issue}")
            continue
        
        # Process the DataFrame
        try:
            updated_df, report = process_matches_dataframe(
                df, source_name, tourney_id_col=tourney_id_col, 
                tournament_name_col=tournament_name_col, tourney_level_col=tourney_level_col
            )
            results[source_name] = (updated_df, report)
            
            # Run data quality check
            run_data_quality_check(source_name)
            
        except Exception as e:
            print(f"Error processing {source_name}: {e}")
            results[source_name] = (None, [])
    
    print(f"\nBATCH PROCESSING COMPLETE - {len(results)} sources processed")
    return results