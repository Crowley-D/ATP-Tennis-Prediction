"""
Core matching engine for player identification across data sources.
"""

from typing import Optional, Dict, Any, List, NamedTuple
from datetime import date, datetime
import pandas as pd

from ..database.crud_operations import (
    get_player_by_source_id,
    search_players_by_name,
    create_player,
    add_source_id_to_player,
    get_next_available_player_id
)
from .name_processing import (
    calculate_name_similarity,
    preprocess_player_name,
    validate_name_input,
    get_source_name
)
from .manual_review import ManualReviewSystem


class MatchResult(NamedTuple):
    """Result of a player matching operation."""
    action: str  # 'linked_exact', 'linked_fuzzy', 'created_new', 'manual_needed'
    player_id: int
    confidence: str  # 'high', 'medium', 'low'
    similarity: Optional[float] = None
    message: Optional[str] = None


class PlayerMatchingEngine:
    """
    Core engine for matching players across different data sources.
    """
    
    def __init__(self, matches_df: Optional[pd.DataFrame] = None, db_path: str = "databases/tennis_players.db"):
        """
        Initialize the matching engine.
        
        Args:
            matches_df: Optional DataFrame with match history for disambiguation
            db_path: Path to the player database
        """
        self.matches_df = matches_df
        self.db_path = db_path
        
        # Matching thresholds
        self.high_similarity_threshold = 0.8  # Auto-accept with confirmation
        self.medium_similarity_threshold = 0.6  # Manual review required
        
    def get_player_recent_match(self, player_id: int) -> Optional[Dict[str, Any]]:
        """
        Get most recent match for a player ID.
        
        Args:
            player_id: Player ID to look up
            
        Returns:
            Recent match info dict or None if no matches or no matches_df
        """
        if self.matches_df is None or self.matches_df.empty:
            return None
        
        try:
            # Find matches where player participated
            player_matches = self.matches_df[
                (self.matches_df['player1_id'] == player_id) |
                (self.matches_df['player2_id'] == player_id)
            ].copy()
            
            if player_matches.empty:
                return None
            
            # Get most recent match
            player_matches = player_matches.sort_values('date', ascending=False)
            recent_match = player_matches.iloc[0]
            
            # Determine opponent
            if recent_match['player1_id'] == player_id:
                opponent_name = recent_match['player2_name']
            else:
                opponent_name = recent_match['player1_name']
            
            return {
                'match_id': recent_match['match_id'],
                'date': recent_match['date'],
                'opponent': opponent_name,
                'tournament': recent_match.get('tournament_name', 'Unknown'),
                'round': recent_match.get('round', 'Unknown')
            }
            
        except Exception as e:
            # If match history DataFrame doesn't have expected columns, return None
            return None
    
    def match_player(
        self, 
        source_code: int, 
        source_id: str, 
        original_name: str,
        auto_resolve: bool = False
    ) -> MatchResult:
        """
        Match a player from a data source to the unified player database.
        
        Args:
            source_code: Source code (0=main_dataset, 1=infosys_api, 2=tennis_api)
            source_id: Original ID from the source
            original_name: Original player name from the source
            auto_resolve: If True, automatically resolve ambiguous matches
            
        Returns:
            MatchResult with the outcome
        """
        # Validate input
        valid, error_msg = validate_name_input(original_name)
        if not valid:
            return MatchResult(
                action='error',
                player_id=0,
                confidence='low',
                message=f"Invalid name input: {error_msg}"
            )
        
        # Preprocess the name
        preprocessed_name = preprocess_player_name(original_name)
        if not preprocessed_name:
            return MatchResult(
                action='error',
                player_id=0,
                confidence='low',
                message="Name preprocessing resulted in empty string"
            )
        
        # Step 1: Check if source_id already exists for this source
        existing_player = get_player_by_source_id(source_code, source_id, self.db_path)
        
        if existing_player:
            # Validate name similarity with existing mapping
            similarity = calculate_name_similarity(
                preprocessed_name, 
                existing_player['preprocessed_name']
            )
            
            if similarity >= self.high_similarity_threshold:
                # Good match, just add as variant if not already present
                return MatchResult(
                    action='linked_exact',
                    player_id=existing_player['player_id'],
                    confidence='high',
                    similarity=similarity,
                    message="Matched to existing player mapping"
                )
            else:
                # Low similarity - potential data quality issue
                return MatchResult(
                    action='manual_needed',
                    player_id=existing_player['player_id'],
                    confidence='low', 
                    similarity=similarity,
                    message=f"Name mismatch for existing mapping (similarity: {similarity:.1%})"
                )
        
        # Step 2: Search by preprocessed name
        matches = search_players_by_name(
            preprocessed_name=preprocessed_name,
            db_path=self.db_path
        )
        
        if not matches:
            # No matches - create new player
            return self._create_new_player(
                source_code=source_code,
                source_id=source_id,
                original_name=original_name,
                preprocessed_name=preprocessed_name
            )
        
        # Step 3: Evaluate matches
        exact_matches = [m for m in matches if m['similarity'] >= 0.99]
        
        if len(exact_matches) == 1:
            # Single exact match - link to it
            match = exact_matches[0]
            success = add_source_id_to_player(
                player_id=match['player_id'],
                source_code=source_code,
                source_id=source_id,
                source_name_variant=original_name,
                preprocessed_name=preprocessed_name,
                db_path=self.db_path
            )
            
            if success:
                return MatchResult(
                    action='linked_exact',
                    player_id=match['player_id'],
                    confidence='high',
                    similarity=match['similarity'],
                    message="Linked to existing player via exact name match"
                )
            else:
                return MatchResult(
                    action='error',
                    player_id=match['player_id'],
                    confidence='low',
                    message="Failed to add source mapping (may already exist)"
                )
        
        # Step 4: Handle multiple or fuzzy matches
        if auto_resolve:
            # Take the best match if it's above medium threshold
            best_match = matches[0]  # Already sorted by similarity
            if best_match['similarity'] >= self.medium_similarity_threshold:
                success = add_source_id_to_player(
                    player_id=best_match['player_id'],
                    source_code=source_code,
                    source_id=source_id,
                    source_name_variant=original_name,
                    preprocessed_name=preprocessed_name,
                    db_path=self.db_path
                )
                
                if success:
                    return MatchResult(
                        action='linked_fuzzy',
                        player_id=best_match['player_id'],
                        confidence='medium' if best_match['similarity'] < self.high_similarity_threshold else 'high',
                        similarity=best_match['similarity'],
                        message=f"Auto-linked to best match (similarity: {best_match['similarity']:.1%})"
                    )
        
        # Manual review needed
        return MatchResult(
            action='manual_needed',
            player_id=matches[0]['player_id'] if matches else 0,
            confidence='low',
            similarity=matches[0]['similarity'] if matches else 0.0,
            message=f"Manual review needed - found {len(matches)} potential matches"
        )
    
    def _create_new_player(
        self,
        source_code: int,
        source_id: str,
        original_name: str,
        preprocessed_name: str
    ) -> MatchResult:
        """
        Create a new player in the database.
        
        Args:
            source_code: Source code (0/1/2)
            source_id: Original ID from source  
            original_name: Original name from source
            preprocessed_name: Processed name
            
        Returns:
            MatchResult with new player information
        """
        try:
            # Create new player
            player_id = create_player(
                primary_name=preprocessed_name,
                db_path=self.db_path
            )
            
            # Add source mapping
            success = add_source_id_to_player(
                player_id=player_id,
                source_code=source_code,
                source_id=source_id,
                source_name_variant=original_name,
                preprocessed_name=preprocessed_name,
                is_primary_name=True,
                db_path=self.db_path
            )
            
            if success:
                return MatchResult(
                    action='created_new',
                    player_id=player_id,
                    confidence='high',
                    message=f"Created new player with ID {player_id}"
                )
            else:
                return MatchResult(
                    action='error',
                    player_id=player_id,
                    confidence='low',
                    message="Created player but failed to add source mapping"
                )
                
        except Exception as e:
            return MatchResult(
                action='error',
                player_id=0,
                confidence='low',
                message=f"Failed to create new player: {str(e)}"
            )
    
    def process_player_batch(
        self,
        players_data: List[Dict[str, Any]],
        auto_resolve: bool = False
    ) -> List[MatchResult]:
        """
        Process a batch of players for matching.
        
        Args:
            players_data: List of player dictionaries with keys:
                         'source_code', 'source_id', 'original_name'
            auto_resolve: Whether to automatically resolve ambiguous matches
            
        Returns:
            List of MatchResult objects
        """
        results = []
        
        for player_data in players_data:
            result = self.match_player(
                source_code=player_data['source_code'],
                source_id=player_data['source_id'],
                original_name=player_data['original_name'],
                auto_resolve=auto_resolve
            )
            results.append(result)
        
        return results
    
    def get_match_statistics(self, results: List[MatchResult]) -> Dict[str, Any]:
        """
        Generate statistics for a batch of match results.
        
        Args:
            results: List of MatchResult objects
            
        Returns:
            Statistics dictionary
        """
        if not results:
            return {}
        
        stats = {
            'total': len(results),
            'linked_exact': len([r for r in results if r.action == 'linked_exact']),
            'linked_fuzzy': len([r for r in results if r.action == 'linked_fuzzy']),
            'created_new': len([r for r in results if r.action == 'created_new']),
            'manual_needed': len([r for r in results if r.action == 'manual_needed']),
            'errors': len([r for r in results if r.action == 'error']),
            'high_confidence': len([r for r in results if r.confidence == 'high']),
            'medium_confidence': len([r for r in results if r.confidence == 'medium']),
            'low_confidence': len([r for r in results if r.confidence == 'low'])
        }
        
        # Calculate percentages
        for key in ['linked_exact', 'linked_fuzzy', 'created_new', 'manual_needed', 'errors']:
            stats[f'{key}_pct'] = round(stats[key] / stats['total'] * 100, 1)
        
        return stats


if __name__ == "__main__":
    # Test matching engine
    print("Testing matching engine...")
    
    try:
        engine = PlayerMatchingEngine()
        
        # Test match player
        result = engine.match_player(
            source_code=0,  # main_dataset
            source_id="test_001",
            original_name="Test Player"
        )
        
        print(f"Match result: {result}")
        print("Matching engine test: SUCCESS")
        
    except Exception as e:
        print(f"Matching engine test: FAILED - {e}")