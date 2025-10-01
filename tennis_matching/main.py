"""
Main entry point for the tennis tournament matching system.
"""

import pandas as pd
import argparse
import sys
from pathlib import Path
from typing import Optional

from tennis_matching.database.connection import set_db_path
from tennis_matching.integration.dataframe_processor import (
    process_matches_dataframe,
    validate_dataframe_structure,
    run_data_quality_check,
    export_processing_report,
    batch_process_multiple_sources
)
from tennis_matching.integration.batch_processing import BatchProcessor, estimate_processing_time


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Tennis Tournament Matching System')
    
    parser.add_argument('--input-file', '-i', type=str, required=True,
                       help='Input CSV file with match data')
    parser.add_argument('--source-name', '-s', type=str, required=True,
                       help='Source name (main_dataset, infosys_api, tennis_api)')
    parser.add_argument('--output-file', '-o', type=str,
                       help='Output CSV file (optional)')
    parser.add_argument('--database-path', '-d', type=str, 
                       default='databases/tennis_tournaments.db',
                       help='SQLite database path')
    parser.add_argument('--tournament-id-col', type=str, 
                       default='tourney_id',
                       help='Column name for tournament IDs')
    parser.add_argument('--tournament-name-col', type=str,
                       default='tourney_name', 
                       help='Column name for tournament names')
    parser.add_argument('--tourney-level-col', type=str,
                       default='tourney_level',
                       help='Column name for tournament level (G, M, F, D, A, C, S)')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Batch size for processing')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate DataFrame structure')
    parser.add_argument('--quality-check', action='store_true',
                       help='Run data quality check on existing data')
    parser.add_argument('--estimate-time', action='store_true',
                       help='Estimate processing time only')
    
    args = parser.parse_args()
    
    # Set database path
    set_db_path(args.database_path)
    print(f"Using database: {args.database_path}")
    
    # Handle quality check only
    if args.quality_check:
        print(f"Running data quality check for {args.source_name}...")
        run_data_quality_check(args.source_name)
        return
    
    # Load input file
    try:
        print(f"Loading data from: {args.input_file}")
        df = pd.read_csv(args.input_file)
        print(f"Loaded {len(df)} rows")
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)
    
    # Validate DataFrame structure
    validation_issues = validate_dataframe_structure(
        df, args.tournament_id_col, args.tournament_name_col, args.tourney_level_col
    )
    
    if validation_issues:
        print("Validation issues found:")
        for issue in validation_issues:
            print(f"  - {issue}")
        
        if args.validate_only:
            sys.exit(1)
        
        response = input("Continue processing despite validation issues? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    elif args.validate_only:
        print("DataFrame validation passed")
        return
    
    # Estimate processing time if requested
    if args.estimate_time:
        estimate = estimate_processing_time(df, args.tournament_id_col, args.tournament_name_col)
        print(f"\nPROCESSING TIME ESTIMATE")
        print(f"Unique tournaments: {estimate['tournaments']}")
        print(f"Total matches: {estimate['total_matches']}")
        print(f"Estimated time: {estimate['estimated_minutes']:.1f} minutes")
        print(f"                ({estimate['estimated_hours']:.1f} hours)")
        return
    
    # Process the DataFrame
    try:
        if len(df) > args.batch_size * 2:  # Use batch processing for larger files
            print(f"Using batch processing (batch size: {args.batch_size})")
            
            batch_processor = BatchProcessor(batch_size=args.batch_size)
            updated_df, processing_report = batch_processor.process_dataframe_in_batches(
                df, args.source_name, args.tournament_id_col, args.tournament_name_col, args.tourney_level_col
            )
        else:
            print("Using standard processing")
            updated_df, processing_report = process_matches_dataframe(
                df, args.source_name, tourney_id_col=args.tournament_id_col, 
                tournament_name_col=args.tournament_name_col, tourney_level_col=args.tourney_level_col
            )
        
        # Save results
        if args.output_file:
            output_path = args.output_file
        else:
            # Generate output filename
            input_path = Path(args.input_file)
            output_path = input_path.parent / f"{input_path.stem}_with_tournament_ids{input_path.suffix}"
        
        print(f"\nSaving results to: {output_path}")
        updated_df.to_csv(output_path, index=False)
        
        # Export processing report
        report_path = Path(output_path).parent / f"{Path(output_path).stem}_report.csv"
        export_processing_report(processing_report, str(report_path))
        
        # Run data quality check
        run_data_quality_check(args.source_name)
        
        print(f"\nProcessing complete! Results saved to: {output_path}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)


def process_dataframe_programmatically(
    df: pd.DataFrame,
    source_name: str = None,
    source_col: str = 'source',
    tourney_id_col: str = 'tourney_id',
    tournament_name_col: str = 'tourney_name',
    tourney_level_col: str = 'tourney_level',
    database_path: str = 'databases/tennis_tournaments.db'
) -> pd.DataFrame:
    """
    Programmatic interface for processing DataFrames from Python code.
    
    Args:
        df: Input DataFrame with tournament information
        source_name: Fixed source name (deprecated, use source_col instead)
        source_col: Column name containing source codes (0=main_dataset, 1=infosys_api, 2=tennis_api)
        tournament_id_col: Column name containing tournament IDs
        tournament_name_col: Column name containing tournament names
        tourney_level_col: Column name containing tournament level (G, M, F, D, A, C, S)
        database_path: Path to SQLite database
        
    Returns:
        pd.DataFrame: DataFrame with added tournament_id column (4-digit level-based ID)
        
    Example:
        >>> import pandas as pd
        >>> from tennis_matching.main import process_dataframe_programmatically
        >>> 
        >>> # Load your data with source column
        >>> df = pd.read_csv('matches.csv')  # Must have 'source' column with 0/1/2 values
        >>> 
        >>> # Process and get tournament IDs (new method - using source column)
        >>> updated_df = process_dataframe_programmatically(df)
        >>> 
        >>> # Or legacy method (fixed source name)
        >>> updated_df = process_dataframe_programmatically(df, source_name='main_dataset')
        >>> 
        >>> # Use the updated DataFrame with tournament IDs
        >>> print(updated_df[['tournament_id', 'tournament_name', 'tourney_level']].head())
    """
    # Set database path
    set_db_path(database_path)

    # Validate DataFrame structure (INPUT validation)
    print(f"[DEBUG] Validating INPUT DataFrame: {len(df)} rows")
    print(f"[DEBUG] Input null tourney_id: {df[tourney_id_col].isna().sum()}")

    require_source_col = source_name is None
    validation_issues = validate_dataframe_structure(
        df, tourney_id_col, tournament_name_col, tourney_level_col, source_col, require_source_col
    )
    if validation_issues:
        raise ValueError(f"DataFrame validation failed (INPUT): {', '.join(validation_issues)}")

    print(f"[DEBUG] Input validation PASSED")

    # Process the DataFrame
    print(f"[DEBUG] Starting tournament processing...")
    updated_df, processing_report = process_matches_dataframe(
        df, source_name, source_col, tourney_id_col, tournament_name_col, tourney_level_col
    )

    print(f"[DEBUG] Processing completed")
    print(f"[DEBUG] Output null tourney_id: {updated_df['tourney_id'].isna().sum()}")

    # Validate OUTPUT
    output_null_ids = updated_df['tourney_id'].isna().sum()
    if output_null_ids > 0:
        raise ValueError(f"DataFrame validation failed: {output_null_ids} rows have null tournament IDs")

    return updated_df


if __name__ == '__main__':
    main()