"""
Batch processing utilities for large-scale tournament matching operations.
"""

import pandas as pd
import time
from typing import List, Dict, Any, Callable, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tennis_matching.matching.matching_engine import match_tournament
from tennis_matching.integration.dataframe_processor import process_matches_dataframe


class BatchProcessor:
    """Handles batch processing of large DataFrames with progress tracking."""
    
    def __init__(self, batch_size: int = 1000, max_workers: int = 1):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Number of unique tournaments to process in each batch
            max_workers: Number of worker threads (should be 1 for SQLite)
        """
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.stats = {
            'total_batches': 0,
            'completed_batches': 0,
            'total_tournaments': 0,
            'processed_tournaments': 0,
            'start_time': None,
            'errors': []
        }
    
    def process_dataframe_in_batches(
        self,
        df: pd.DataFrame,
        source_name: str,
        tourney_id_col: str = 'tourney_id',
        tournament_name_col: str = 'tourney_name',
        tourney_level_col: str = 'tourney_level',
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Process a large DataFrame in batches to manage memory and provide progress updates.
        
        Args:
            df: Input DataFrame
            source_name: Name of the data source
            tourney_id_col: Column name for tournament IDs
            tournament_name_col: Column name for tournament names
            tourney_level_col: Column name for tournament level
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Tuple[pd.DataFrame, List[Dict]]: (processed_df, processing_report)
        """
        self._reset_stats()
        self.stats['start_time'] = time.time()
        
        print(f"\nBATCH PROCESSING: {len(df)} matches from {source_name}")
        print(f"Batch size: {self.batch_size} tournaments")
        print("=" * 70)
        
        # Get unique tournaments
        unique_tournaments = df[[tourney_id_col, tournament_name_col, tourney_level_col]].drop_duplicates()
        self.stats['total_tournaments'] = len(unique_tournaments)
        
        # Split into batches
        batches = self._create_batches(unique_tournaments)
        self.stats['total_batches'] = len(batches)
        
        print(f"Split into {len(batches)} batches")
        
        # Initialize results
        updated_df = df.copy()
        updated_df['tourney_id'] = None  # New 4-digit tournament ID
        
        all_reports = []
        tourney_id_map = {}
        
        # Process each batch
        for batch_idx, batch_df in enumerate(batches):
            batch_start_time = time.time()
            
            try:
                print(f"\nProcessing batch {batch_idx + 1}/{len(batches)} "
                      f"({len(batch_df)} tournaments)...")
                
                # Process this batch
                batch_results = self._process_batch(
                    batch_df, source_name, tourney_id_col, tournament_name_col, tourney_level_col
                )
                
                # Update results
                for result in batch_results:
                    cache_key = (source_name, result['source_tourney_id'])
                    tourney_id_map[cache_key] = {
                        'tourney_id': result['assigned_tourney_id'],
                        'confidence': result['confidence'],
                        'action': result['action']
                    }
                
                all_reports.extend(batch_results)
                
                # Update statistics
                self.stats['completed_batches'] += 1
                self.stats['processed_tournaments'] += len(batch_df)
                
                batch_time = time.time() - batch_start_time
                avg_time_per_tournament = batch_time / len(batch_df)
                
                print(f"Batch {batch_idx + 1} completed in {batch_time:.1f}s "
                      f"({avg_time_per_tournament:.2f}s per tournament)")
                
                # Call progress callback if provided
                if progress_callback:
                    progress_info = {
                        'batch_number': batch_idx + 1,
                        'total_batches': len(batches),
                        'batch_size': len(batch_df),
                        'completed_tournaments': self.stats['processed_tournaments'],
                        'total_tournaments': self.stats['total_tournaments'],
                        'batch_time': batch_time,
                        'avg_time_per_tournament': avg_time_per_tournament
                    }
                    progress_callback(progress_info)
                
            except Exception as e:
                error_msg = f"Error in batch {batch_idx + 1}: {str(e)}"
                print(f"Error: {error_msg}")
                self.stats['errors'].append(error_msg)
                continue
        
        # Apply results to full DataFrame
        print(f"\nApplying results to {len(df)} total matches...")
        self._apply_results_to_dataframe(
            updated_df, tourney_id_map, source_name, tourney_id_col
        )
        
        # Print final statistics
        self._print_final_stats()
        
        return updated_df, all_reports
    
    def _reset_stats(self):
        """Reset processing statistics."""
        self.stats = {
            'total_batches': 0,
            'completed_batches': 0,
            'total_tournaments': 0,
            'processed_tournaments': 0,
            'start_time': None,
            'errors': []
        }
    
    def _create_batches(self, unique_tournaments: pd.DataFrame) -> List[pd.DataFrame]:
        """Split unique tournaments into batches."""
        batches = []
        for i in range(0, len(unique_tournaments), self.batch_size):
            batch = unique_tournaments.iloc[i:i + self.batch_size]
            batches.append(batch)
        return batches
    
    def _process_batch(
        self,
        batch_df: pd.DataFrame,
        source_name: str,
        tourney_id_col: str,
        tournament_name_col: str,
        tourney_level_col: str
    ) -> List[Dict[str, Any]]:
        """Process a single batch of tournaments."""
        batch_results = []
        
        for _, row in batch_df.iterrows():
            tourney_id = str(row[tourney_id_col])
            tournament_name = str(row[tournament_name_col])
            tourney_level = row[tourney_level_col]
            
            try:
                # Perform tournament matching
                new_tourney_id, confidence, action, stored_new_id = match_tournament(
                    source_name=source_name,
                    source_id=tourney_id,
                    tournament_name=tournament_name,
                    tourney_level=tourney_level
                )
                
                batch_results.append({
                    'source_tourney_id': tourney_id,
                    'tournament_name': tournament_name,
                    'tourney_level': tourney_level,
                    'assigned_tourney_id': new_tourney_id,
                    'confidence': confidence,
                    'action': action,
                    'source_name': source_name
                })
                
            except Exception as e:
                batch_results.append({
                    'source_tourney_id': tourney_id,
                    'tournament_name': tournament_name,
                    'tourney_level': tourney_level,
                    'assigned_tourney_id': None,
                    'confidence': 'error',
                    'action': f'error: {str(e)}',
                    'source_name': source_name
                })
        
        return batch_results
    
    def _apply_results_to_dataframe(
        self,
        df: pd.DataFrame,
        tourney_id_map: Dict,
        source_name: str,
        tourney_id_col: str
    ):
        """Apply processing results to the full DataFrame."""
        for idx, row in df.iterrows():
            source_tourney_id = str(row[tourney_id_col])
            cache_key = (source_name, source_tourney_id)
            
            if cache_key in tourney_id_map:
                result = tourney_id_map[cache_key]
                df.at[idx, 'tourney_id'] = result['tourney_id']  # Set the new 4-digit tournament ID
    
    def _print_final_stats(self):
        """Print final processing statistics."""
        total_time = time.time() - self.stats['start_time']
        
        print("\n" + "=" * 70)
        print("BATCH PROCESSING COMPLETE")
        print("=" * 70)
        print(f"Total processing time: {total_time:.1f}s")
        print(f"Batches completed: {self.stats['completed_batches']}/{self.stats['total_batches']}")
        print(f"Tournaments processed: {self.stats['processed_tournaments']}/{self.stats['total_tournaments']}")
        
        if self.stats['processed_tournaments'] > 0:
            avg_time = total_time / self.stats['processed_tournaments']
            print(f"Average time per tournament: {avg_time:.2f}s")
            
            tournaments_per_hour = 3600 / avg_time
            print(f"Processing rate: {tournaments_per_hour:.0f} tournaments/hour")
        
        if self.stats['errors']:
            print(f"Errors encountered: {len(self.stats['errors'])}")
            for error in self.stats['errors'][:5]:  # Show first 5 errors
                print(f"  - {error}")
            if len(self.stats['errors']) > 5:
                print(f"  ... and {len(self.stats['errors']) - 5} more errors")


def create_processing_checkpoint(
    df: pd.DataFrame,
    processing_report: List[Dict[str, Any]],
    checkpoint_file: str
) -> None:
    """
    Create a checkpoint file to resume processing if interrupted.
    
    Args:
        df: Current state of the DataFrame
        processing_report: Current processing report
        checkpoint_file: Path to checkpoint file
    """
    checkpoint_data = {
        'dataframe': df.to_dict('records'),
        'processing_report': processing_report,
        'timestamp': time.time()
    }
    
    # Save as pickle for efficiency
    import pickle
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    print(f"Checkpoint saved to: {checkpoint_file}")


def load_processing_checkpoint(checkpoint_file: str) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Load processing state from checkpoint file.
    
    Args:
        checkpoint_file: Path to checkpoint file
        
    Returns:
        Tuple[pd.DataFrame, List[Dict]]: (dataframe, processing_report)
    """
    import pickle
    with open(checkpoint_file, 'rb') as f:
        checkpoint_data = pickle.load(f)
    
    df = pd.DataFrame(checkpoint_data['dataframe'])
    processing_report = checkpoint_data['processing_report']
    
    print(f"Checkpoint loaded from: {checkpoint_file}")
    print(f"Checkpoint created: {time.ctime(checkpoint_data['timestamp'])}")
    
    return df, processing_report


def estimate_processing_time(
    df: pd.DataFrame,
    tourney_id_col: str = 'tourney_id',
    tournament_name_col: str = 'tourney_name',
    avg_time_per_tournament: float = 1.0
) -> Dict[str, float]:
    """
    Estimate processing time for a DataFrame.
    
    Args:
        df: DataFrame to process
        tourney_id_col: Column name for tournament IDs
        tournament_name_col: Column name for tournament names
        avg_time_per_tournament: Average seconds per tournament (from past runs)
        
    Returns:
        Dict: Time estimates in various units
    """
    unique_tournaments = df[[tourney_id_col, tournament_name_col]].drop_duplicates()
    num_tournaments = len(unique_tournaments)
    
    total_seconds = num_tournaments * avg_time_per_tournament
    
    return {
        'tournaments': num_tournaments,
        'total_matches': len(df),
        'estimated_seconds': total_seconds,
        'estimated_minutes': total_seconds / 60,
        'estimated_hours': total_seconds / 3600,
        'avg_time_per_tournament': avg_time_per_tournament
    }