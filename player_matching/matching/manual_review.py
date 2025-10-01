"""
Manual review system for player matching disambiguation.
"""

import sys
from typing import Dict, Any, List, Optional, Union
from datetime import date
import pandas as pd

from ..database.crud_operations import (
    get_player_by_id, get_next_available_player_id, 
    create_player, add_source_id_to_player
)
from .name_processing import preprocess_player_name


class ManualReviewSystem:
    """
    Interactive system for manual player matching review.
    """
    
    def __init__(self, db_path: str = "databases/tennis_players.db"):
        """
        Initialize manual review system.
        
        Args:
            db_path: Path to player database
        """
        self.db_path = db_path
    
    def prompt_user_selection(
        self, 
        matches: List[Dict[str, Any]], 
        preprocessed_name: str,
        source_name: str,
        source_id: str
    ) -> Dict[str, Any]:
        """
        Present match options to user for selection.
        
        Args:
            matches: List of potential player matches
            preprocessed_name: Preprocessed name being matched
            source_name: Data source name
            source_id: Original source ID
            
        Returns:
            Dictionary with user selection result
        """
        if not matches:
            return self.prompt_manual_entry(preprocessed_name, source_name, source_id)
        
        print("\nPLAYER MATCH REVIEW")
        print("=" * 60)
        print(f"Source: {source_name}")
        print(f"Source ID: {source_id}")
        print(f"Player Name: '{preprocessed_name}'")
        print("=" * 60)
        print("Potential matches:\n")
        
        # Display matches with metadata
        for idx, match in enumerate(matches, 1):
            print(f"{idx}. {match['primary_name']} (ID: {match['player_id']})")
            
            # Show metadata if available
            metadata_parts = []
            if match.get('dob'):
                metadata_parts.append(f"DOB: {match['dob']}")
            if match.get('hand'):
                metadata_parts.append(f"Hand: {match['hand']}")
            if match.get('height'):
                metadata_parts.append(f"Height: {match['height']}cm")
            
            if metadata_parts:
                print(f"   {', '.join(metadata_parts)}")
            
            if 'sources' in match:
                print(f"   Sources: {', '.join(match['sources'])}")
            
            print(f"   Similarity: {match['similarity']:.1%}")
            print()
        
        # Add options for creating new or manual entry
        create_option = len(matches) + 1
        manual_option = create_option + 1
        
        print(f"{create_option}. Create new player")
        print(f"{manual_option}. Manual entry (provide player ID)")
        print("0. Skip this player")
        
        # Get user selection
        while True:
            try:
                choice = input(f"\nSelect option (0-{manual_option}): ").strip()
                choice_num = int(choice)
                
                if choice_num == 0:
                    return {'action': 'skip'}
                elif 1 <= choice_num <= len(matches):
                    selected_match = matches[choice_num - 1]
                    return {
                        'action': 'link_existing',
                        'player_id': selected_match['player_id'],
                        'similarity': selected_match['similarity']
                    }
                elif choice_num == create_option:
                    return self.prompt_new_player_creation(preprocessed_name, source_name, source_id)
                elif choice_num == manual_option:
                    return self.prompt_manual_player_id(preprocessed_name, source_name, source_id)
                else:
                    print(f"Invalid choice. Please enter 0-{manual_option}")
                    
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\nOperation cancelled by user")
                return {'action': 'skip'}
    
    def prompt_manual_entry(
        self, 
        preprocessed_name: str,
        source_name: str,
        source_id: str
    ) -> Dict[str, Any]:
        """
        Handle manual entry when no matches are found.
        
        Args:
            preprocessed_name: Preprocessed player name
            source_name: Data source name
            source_id: Original source ID
            
        Returns:
            Dictionary with manual entry result
        """
        print(f"\nNO MATCHES FOUND")
        print("=" * 30)
        print(f"Player: {preprocessed_name}")
        print(f"Source: {source_name}, ID: {source_id}")
        print("\nOptions:")
        print("1. Enter existing player ID (if you know it)")
        print("2. Create new player with auto-generated ID")
        print("3. Create new player with specific ID")
        print("4. Skip this player")
        
        while True:
            try:
                choice = input("\nSelect option (1-4): ").strip()
                
                if choice == '1':
                    return self.prompt_manual_player_id(preprocessed_name, source_name, source_id)
                elif choice == '2':
                    return self.prompt_new_player_creation(preprocessed_name, source_name, source_id)
                elif choice == '3':
                    return self.prompt_new_player_creation(preprocessed_name, source_name, source_id, allow_custom_id=True)
                elif choice == '4':
                    return {'action': 'skip'}
                else:
                    print("Invalid choice. Please enter 1-4")
                    
            except KeyboardInterrupt:
                print("\nOperation cancelled by user")
                return {'action': 'skip'}
    
    def prompt_manual_player_id(
        self, 
        preprocessed_name: str,
        source_name: str,
        source_id: str
    ) -> Dict[str, Any]:
        """
        Prompt user to enter an existing player ID manually.
        
        Args:
            preprocessed_name: Preprocessed player name
            source_name: Data source name
            source_id: Original source ID
            
        Returns:
            Dictionary with manual ID result
        """
        while True:
            try:
                player_id_input = input("Enter existing player ID (6-digit, 100000+): ").strip()
                
                if not player_id_input:
                    return {'action': 'skip'}
                
                player_id = int(player_id_input)
                
                if player_id < 100000 or player_id > 999999:
                    print("Player ID must be a 6-digit number between 100000-999999")
                    continue
                
                # Verify player exists
                existing_player = get_player_by_id(player_id, self.db_path)
                if not existing_player:
                    print(f"Player ID {player_id} not found in database")
                    retry = input("Try again? (y/n): ").strip().lower()
                    if retry != 'y':
                        return {'action': 'skip'}
                    continue
                
                print(f"Found player: {existing_player['primary_name']}")
                confirm = input("Link to this player? (y/n): ").strip().lower()
                
                if confirm == 'y':
                    return {
                        'action': 'link_existing',
                        'player_id': player_id,
                        'similarity': 0.0  # Manual entry, no calculated similarity
                    }
                else:
                    retry = input("Try different ID? (y/n): ").strip().lower()
                    if retry != 'y':
                        return {'action': 'skip'}
                    
            except ValueError:
                print("Invalid player ID format. Please enter a 6-digit number.")
            except KeyboardInterrupt:
                print("\nOperation cancelled by user")
                return {'action': 'skip'}
    
    def prompt_new_player_creation(
        self, 
        preprocessed_name: str,
        source_name: str,
        source_id: str,
        allow_custom_id: bool = False
    ) -> Dict[str, Any]:
        """
        Prompt user to create a new player.
        
        Args:
            preprocessed_name: Preprocessed player name
            source_name: Data source name
            source_id: Original source ID
            allow_custom_id: Whether to allow custom player ID
            
        Returns:
            Dictionary with new player result
        """
        print(f"\nCREATING NEW PLAYER")
        print("-" * 25)
        print(f"Name: {preprocessed_name}")
        print(f"Source: {source_name}, ID: {source_id}")
        
        # Get player ID
        if allow_custom_id:
            player_id_input = input("Enter custom player ID (6-digit, 100000+) or press Enter for auto: ").strip()
            if player_id_input:
                try:
                    player_id = int(player_id_input)
                    if player_id < 100000 or player_id > 999999:
                        print("Invalid ID range, using auto-generated ID")
                        player_id = None
                    else:
                        # Check if ID already exists
                        existing = get_player_by_id(player_id, self.db_path)
                        if existing:
                            print(f"Player ID {player_id} already exists, using auto-generated ID")
                            player_id = None
                except ValueError:
                    print("Invalid ID format, using auto-generated ID")
                    player_id = None
            else:
                player_id = None
        else:
            player_id = None
        
        if player_id is None:
            player_id = get_next_available_player_id(self.db_path)
        
        # Collect optional metadata
        print(f"\nNew Player ID will be: {player_id}")
        print("Optional metadata (press Enter to skip):")
        
        # Date of birth
        dob = None
        while True:
            dob_input = input("Date of birth (YYYY-MM-DD): ").strip()
            if not dob_input:
                break
            try:
                # Basic date validation
                year, month, day = dob_input.split('-')
                if len(year) == 4 and len(month) == 2 and len(day) == 2:
                    year, month, day = int(year), int(month), int(day)
                    if 1900 <= year <= 2010 and 1 <= month <= 12 and 1 <= day <= 31:
                        dob = dob_input
                        break
                print("Invalid date format or range. Use YYYY-MM-DD (1900-2010)")
            except ValueError:
                print("Invalid date format. Use YYYY-MM-DD")
        
        # Playing hand
        hand = None
        hand_input = input("Playing hand (L/R/U): ").strip().upper()
        if hand_input in ['L', 'R', 'U']:
            hand = hand_input
        
        # Height
        height = None
        while True:
            height_input = input("Height in cm (150-220): ").strip()
            if not height_input:
                break
            try:
                height_val = int(height_input)
                if 150 <= height_val <= 220:
                    height = height_val
                    break
                print("Height must be between 150-220 cm")
            except ValueError:
                print("Invalid height format. Enter number only")
        
        print(f"\nCreating new player:")
        print(f"  ID: {player_id}")
        print(f"  Name: {preprocessed_name}")
        if dob:
            print(f"  DOB: {dob}")
        if hand:
            print(f"  Hand: {hand}")
        if height:
            print(f"  Height: {height}cm")
        
        confirm = input("\nCreate this player? (y/n): ").strip().lower()
        
        if confirm == 'y':
            try:
                # Create the player
                created_id = create_player(
                    primary_name=preprocessed_name,
                    player_id=player_id,
                    dob=date.fromisoformat(dob) if dob else None,
                    hand=hand,
                    height=height,
                    db_path=self.db_path
                )
                
                if created_id == player_id:
                    return {
                        'action': 'created_new',
                        'player_id': player_id,
                        'metadata': {
                            'dob': dob,
                            'hand': hand,
                            'height': height
                        }
                    }
                else:
                    print(f"Error: Created player ID {created_id} doesn't match expected {player_id}")
                    return {'action': 'error', 'message': 'Player creation ID mismatch'}
                    
            except Exception as e:
                print(f"Error creating player: {e}")
                return {'action': 'error', 'message': str(e)}
        else:
            return {'action': 'skip'}
    
    def prompt_high_similarity_confirmation(
        self, 
        input_name: str,
        matched_name: str,
        similarity: float,
        player_id: int
    ) -> bool:
        """
        Prompt user to confirm a high-similarity automatic match.
        
        Args:
            input_name: Original input name
            matched_name: Matched player name
            similarity: Similarity score
            player_id: Matched player ID
            
        Returns:
            True if user confirms, False otherwise
        """
        print("\nHIGH SIMILARITY MATCH FOUND")
        print("=" * 35)
        print(f"Input name: '{input_name}'")
        print(f"Matched name: '{matched_name}'")
        print(f"Player ID: {player_id}")
        print(f"Similarity: {similarity:.1%}")
        
        while True:
            try:
                choice = input("Accept this match? (y/n): ").strip().lower()
                if choice in ['y', 'yes']:
                    return True
                elif choice in ['n', 'no']:
                    return False
                else:
                    print("Please enter 'y' for yes or 'n' for no")
            except KeyboardInterrupt:
                print("\nOperation cancelled by user")
                return False
    
    def prompt_name_mismatch_warning(
        self, 
        source_name: str,
        source_id: str,
        existing_name: str,
        new_name: str,
        similarity: float
    ) -> bool:
        """
        Warn user about name mismatch for existing player mapping.
        
        Args:
            source_name: Data source name
            source_id: Source ID that already exists
            existing_name: Name currently associated with this source ID
            new_name: New name found for the same source ID
            similarity: Similarity between old and new names
            
        Returns:
            True if user wants to proceed, False otherwise
        """
        print("\nNAME MISMATCH WARNING")
        print("=" * 30)
        print(f"Source: {source_name}")
        print(f"Source ID: {source_id}")
        print(f"Existing name: '{existing_name}'")
        print(f"New name: '{new_name}'")
        print(f"Similarity: {similarity:.1%}")
        print("\nThis source ID is already linked to a different player name.")
        print("This might indicate a data quality issue.")
        
        while True:
            try:
                choice = input("Accept this as an alternate name? (y/n): ").strip().lower()
                if choice in ['y', 'yes']:
                    return True
                elif choice in ['n', 'no']:
                    return False
                else:
                    print("Please enter 'y' for yes or 'n' for no")
            except KeyboardInterrupt:
                print("\nOperation cancelled by user")
                return False


if __name__ == "__main__":
    # Test manual review system
    print("Testing manual review system...")
    
    try:
        review_system = ManualReviewSystem()
        
        # Test with mock data
        mock_matches = [
            {
                'player_id': 100001,
                'primary_name': 'roger-federer',
                'dob': '1981-08-08',
                'hand': 'R',
                'height': 185,
                'sources': ['main_dataset'],
                'similarity': 0.95
            },
            {
                'player_id': 100002,
                'primary_name': 'r-federer',
                'dob': '1985-03-15',
                'hand': 'R',
                'height': 180,
                'sources': ['tennis_api'],
                'similarity': 0.75
            }
        ]
        
        print("Manual review system initialized successfully")
        print("Note: Interactive prompts would appear here in real usage")
        
        # Test high similarity confirmation
        confirmed = review_system.prompt_high_similarity_confirmation(
            "Roger Federer", "roger-federer", 0.95, 100001
        )
        print(f"High similarity confirmation result: {confirmed}")
        
        print("Manual review system test: SUCCESS")
        
    except Exception as e:
        print(f"Manual review system test: FAILED - {e}")
    
    finally:
        from ..database.connection import close_all_connections
        close_all_connections()