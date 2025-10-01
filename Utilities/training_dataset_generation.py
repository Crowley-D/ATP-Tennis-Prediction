import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.stats_interface import (
    update_stats_from_source_batch,
    get_prematch_stats_from_source_batch,
)

# 1. Process a date range of matches through stats + Elo
results = update_stats_from_source_batch(
    source="databases/MatchesMain/matches_dataset_20250930_194930.csv",
    source_type="csv",
    rewrite_duplicates=False,  # Set True to reprocess existing matches
    progress_bar=True,
    batch_size=10000,
    save_stats=True,
    # Optional filters:
    tourney_date_start="1995-01-01",  # Start date
    tourney_date_end="2025-12-31",  # End date
    # surface="Hard",                  # Specific surface
    # tourney_level="G",                # Grand Slams only
)

# 2. Generate training dataset with pre-match stats
training_df = get_prematch_stats_from_source_batch(
    source="databases/MatchesMain/matches_dataset_20250930_194930.csv",
    source_type="csv",
    progress_bar=True,
    include_metadata=True,
    stream_to_csv="Training/Datasets/training_dataset_1995-2025.csv",  # Save directly
    # Same filters as above:
    tourney_date_start="1995-01-01",
    tourney_date_end="2025-12-31",
)
