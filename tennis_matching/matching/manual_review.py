"""
Manual review functionality for tournament matching decisions.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from tennis_matching.database.crud_operations import get_level_range

# Try to import GUI dialogs, fall back to terminal if not available
_USE_GUI = os.environ.get('TOURNAMENT_MATCHING_GUI', '1') == '1'
_gui_manager = None

if _USE_GUI:
    try:
        from tennis_matching.ui.gui_prompts import get_dialog_manager
        _gui_manager = get_dialog_manager()
        print("[INFO] GUI dialog manager initialized successfully")
    except ImportError as e:
        _USE_GUI = False
        print(f"[INFO] GUI dialogs not available (ImportError: {e}), using terminal input")
    except Exception as e:
        _USE_GUI = False
        print(f"[WARNING] GUI initialization failed ({e}), using terminal input")


def prompt_user_for_match_confirmation(
    target_name: str,
    matches: List[Dict[str, Any]],
    source_name: str,
    source_id: str
) -> Tuple[Optional[int], str]:
    """
    Prompt user to confirm a tournament match from suggested candidates.

    Args:
        target_name: The tournament name to match
        matches: List of potential matches with similarity scores
        source_name: Source name for context
        source_id: Source ID for context

    Returns:
        Tuple[Optional[int], str]: (selected_tournament_id, action_taken)
    """
    # Use GUI if available
    if _USE_GUI and _gui_manager:
        try:
            return _gui_manager.prompt_for_match_confirmation(
                target_name, matches, source_name, source_id
            )
        except Exception as e:
            print(f"[WARNING] GUI dialog failed ({e}), falling back to terminal")

    # Terminal fallback
    print(f"\nTOURNAMENT MATCH REVIEW")
    print(f"Source: {source_name}")
    print(f"Source ID: {source_id}")
    print(f"Tournament Name: '{target_name}'")
    print("=" * 60)

    if not matches:
        print("No potential matches found.")
        return None, 'no_matches_found'

    print("Potential matches:")
    for i, match in enumerate(matches, 1):
        similarity = match['similarity_score']
        existing_name = match['source_name_variant']
        existing_source = match['source_name']
        tournament_id = match['tournament_id']

        print(f"{i}. '{existing_name}' (ID: {tournament_id}, from {existing_source})")
        print(f"   Similarity: {similarity:.1%}")
        print()

    print(f"{len(matches) + 1}. Create new tournament")
    print("0. Skip (manual assignment needed later)")

    while True:
        try:
            choice = input(f"\nSelect option (0-{len(matches) + 1}): ").strip()

            if choice == '0':
                return None, 'manual_skip'
            elif choice == str(len(matches) + 1):
                return None, 'create_new_tournament'
            else:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(matches):
                    selected_match = matches[choice_idx]
                    return selected_match['tournament_id'], 'manual_confirmation'
                else:
                    print(f"Invalid choice. Please enter a number between 0 and {len(matches) + 1}")
        except ValueError:
            print(f"Invalid input. Please enter a number between 0 and {len(matches) + 1}")
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled by user.")
            return None, 'user_cancelled'


def prompt_for_new_tournament_name(suggested_name: str) -> str:
    """
    Prompt user for the primary name when creating a new tournament.
    If user edits the name, checks for similar matches in the database.

    Args:
        suggested_name: Suggested name based on input

    Returns:
        str: User-confirmed primary name for the tournament
    """
    print(f"[DEBUG] prompt_for_new_tournament_name called with: '{suggested_name}'")
    print(f"[DEBUG] _USE_GUI={_USE_GUI}, _gui_manager={_gui_manager}")

    # Use GUI if available
    if _USE_GUI and _gui_manager:
        try:
            print(f"[GUI] Showing tournament name dialog for: '{suggested_name}'")
            result = _gui_manager.prompt_for_tournament_name(suggested_name)
            print(f"[GUI] User confirmed name: '{result}'")

            # Check if user edited the name
            if result != suggested_name:
                print(f"[INFO] User edited tournament name from '{suggested_name}' to '{result}'")
                print(f"[INFO] Checking for similar matches in database...")

                try:
                    # Import only when needed to avoid circular import at module load time
                    from tennis_matching.matching.matching_engine import fuzzy_match_names_all_sources

                    fuzzy_matches = fuzzy_match_names_all_sources(result)
                    print(f"[DEBUG] Found {len(fuzzy_matches)} total fuzzy matches")

                    # Filter to only matches >= 60%
                    significant_matches = [m for m in fuzzy_matches if m['similarity_score'] >= 0.6]
                    print(f"[DEBUG] Filtered to {len(significant_matches)} matches >= 60%")

                    if significant_matches:
                        print(f"[INFO] Found {len(significant_matches)} similar tournament(s) in database")
                        print(f"[DEBUG] Calling _prompt_for_edited_name_matches...")
                        # Show matches and ask user
                        match_result = _prompt_for_edited_name_matches(result, significant_matches)
                        print(f"[DEBUG] User selection result: {match_result}")
                        if match_result:
                            # User wants to use existing tournament - return special signal
                            print(f"[INFO] User selected existing tournament, returning: {match_result}")
                            return match_result
                        else:
                            print(f"[INFO] User chose to create new tournament")
                    else:
                        print(f"[INFO] No similar matches found (all below 60%), proceeding with new tournament")
                except Exception as match_error:
                    print(f"[ERROR] Error during fuzzy matching: {match_error}")
                    import traceback
                    traceback.print_exc()

            return result
        except Exception as e:
            print(f"[WARNING] GUI dialog failed ({e}), falling back to terminal")
            import traceback
            traceback.print_exc()

    # Terminal fallback
    print(f"\nNEW TOURNAMENT CREATION")
    print(f"Suggested primary name: '{suggested_name}'")

    while True:
        try:
            choice = input("Press Enter to accept, or type new name: ").strip()

            if not choice:
                return suggested_name
            elif choice:
                # User edited the name - check for similar matches
                print(f"[INFO] Checking for similar matches in database...")

                # Import only when needed to avoid circular import at module load time
                from tennis_matching.matching.matching_engine import fuzzy_match_names_all_sources

                fuzzy_matches = fuzzy_match_names_all_sources(choice)

                # Filter to only matches >= 60%
                significant_matches = [m for m in fuzzy_matches if m['similarity_score'] >= 0.6]

                if significant_matches:
                    print(f"[INFO] Found {len(significant_matches)} similar tournament(s) in database")
                    # Show matches and ask user
                    match_result = _prompt_for_edited_name_matches(choice, significant_matches)
                    if match_result:
                        return match_result
                    # If None, user rejected all matches, continue with new tournament
                    print(f"[INFO] Proceeding with new tournament creation")
                else:
                    print(f"[INFO] No similar matches found (all below 60%), proceeding with new tournament")

                return choice
            else:
                print("Please provide a valid name or press Enter to accept suggestion.")
        except (EOFError, KeyboardInterrupt):
            print(f"\n[AUTO] Accepting suggested name: '{suggested_name}'")
            return suggested_name


def _prompt_for_edited_name_matches(edited_name: str, matches: List[Dict[str, Any]]) -> Optional[str]:
    """
    Prompt user when edited tournament name matches existing tournaments.

    Args:
        edited_name: The name user entered
        matches: List of similar tournament matches (>= 60%)

    Returns:
        str: Special marker string with tournament ID if user selects existing tournament
        None: If user wants to proceed with new tournament
    """
    # Use GUI if available
    if _USE_GUI and _gui_manager:
        try:
            return _gui_manager.prompt_for_edited_name_matches(edited_name, matches)
        except Exception as e:
            print(f"[WARNING] GUI dialog failed ({e}), falling back to terminal")

    # Terminal fallback
    print(f"\n{'='*60}")
    print(f"SIMILAR TOURNAMENTS FOUND")
    print(f"{'='*60}")
    print(f"Edited name: '{edited_name}'")
    print(f"\nFound {len(matches)} similar tournament(s) in the database:")
    print()

    for i, match in enumerate(matches[:10], 1):  # Show up to 10 matches
        similarity = match['similarity_score']
        primary_name = match['primary_name']
        tournament_id = match['tournament_id']
        source_variant = match['source_name_variant']
        level = match['tourney_level']

        print(f"{i}. '{primary_name}' (ID: {tournament_id}, Level: {level})")
        print(f"   Matched variant: '{source_variant}'")
        print(f"   Similarity: {similarity:.1%}")
        print()

    print(f"{len(matches) + 1}. Proceed with new tournament for '{edited_name}'")
    print()

    while True:
        try:
            choice = input(f"Select option (1-{len(matches) + 1}): ").strip()

            if choice == str(len(matches) + 1):
                # User wants to create new tournament
                return None
            else:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(matches):
                    selected_match = matches[choice_idx]
                    # Return a special marker that signals to use existing tournament
                    # Format: __USE_EXISTING_TOURNAMENT__:tournament_id
                    return f"__USE_EXISTING_TOURNAMENT__:{selected_match['tournament_id']}"
                else:
                    print(f"Invalid choice. Please enter a number between 1 and {len(matches) + 1}")
        except ValueError:
            print(f"Invalid input. Please enter a number between 1 and {len(matches) + 1}")
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled, proceeding with new tournament.")
            return None


def display_match_summary(
    tournament_id: int,
    primary_name: str,
    action_taken: str,
    confidence_level: str
) -> None:
    """
    Display a summary of the matching result.
    
    Args:
        tournament_id: ID of the matched/created tournament
        primary_name: Primary name of the tournament
        action_taken: Description of the action taken
        confidence_level: Confidence level of the match
    """
    print(f"\nMATCH RESULT")
    print(f"Tournament ID: {tournament_id}")
    print(f"Primary Name: {primary_name}")
    print(f"Action: {action_taken}")
    print(f"Confidence: {confidence_level}")
    print("-" * 40)


def confirm_high_similarity_match(
    target_name: str,
    matched_name: str,
    similarity_score: float,
    source_name: str
) -> bool:
    """
    Confirm a high similarity match (80%+) with the user.

    Args:
        target_name: The tournament name being matched
        matched_name: The existing tournament name found
        similarity_score: Similarity score between names
        source_name: Source name for context

    Returns:
        bool: True if user confirms the match
    """
    # Use GUI if available
    if _USE_GUI and _gui_manager:
        try:
            print(f"[GUI] Showing high similarity confirmation for: '{target_name}' vs '{matched_name}' ({similarity_score:.1%})")
            result = _gui_manager.confirm_high_similarity_match(
                target_name, matched_name, similarity_score, source_name
            )
            if result:
                print(f"[GUI] User ACCEPTED match")
            else:
                print(f"[GUI] User REJECTED match - will create new tournament")
            return result
        except Exception as e:
            print(f"[WARNING] GUI dialog failed ({e}), falling back to terminal")

    # Terminal fallback
    print(f"\nHIGH SIMILARITY MATCH FOUND")
    print(f"Source: {source_name}")
    print(f"Input name: '{target_name}'")
    print(f"Matched name: '{matched_name}'")
    print(f"Similarity: {similarity_score:.1%}")

    while True:
        try:
            choice = input("Accept this match? (y/n): ").strip().lower()
            if choice in ['y', 'yes']:
                return True
            elif choice in ['n', 'no']:
                return False
            else:
                print("Please enter 'y' for yes or 'n' for no.")
        except (EOFError, KeyboardInterrupt):
            print("\n[AUTO] Accepting high similarity match")
            return True


def handle_validation_warning(
    tournament_name: str,
    existing_names: List[str],
    similarity_scores: List[float]
) -> bool:
    """
    Handle validation warnings when existing ID has low similarity with new name.
    
    Args:
        tournament_name: New tournament name
        existing_names: List of existing names for this tournament ID
        similarity_scores: Similarity scores with existing names
        
    Returns:
        bool: True if user wants to proceed despite warning
    """
    print(f"\nVALIDATION WARNING")
    print(f"New name: '{tournament_name}'")
    print("Existing names for this tournament ID:")
    
    for name, score in zip(existing_names, similarity_scores):
        print(f"  - '{name}' (similarity: {score:.1%})")
    
    print("\nThis might indicate a data quality issue.")
    
    while True:
        choice = input("Proceed anyway? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            return True
        elif choice in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' for yes or 'n' for no.")


def display_error_summary(errors: List[Dict[str, Any]]) -> None:
    """
    Display a summary of data quality errors found during processing.
    
    Args:
        errors: List of error dictionaries
    """
    if not errors:
        print("\nNo data quality issues detected.")
        return
    
    print(f"\nDATA QUALITY ISSUES DETECTED ({len(errors)} issues)")
    print("=" * 60)
    
    for i, error in enumerate(errors, 1):
        print(f"{i}. {error['description']}")
        if 'details' in error:
            for detail in error['details']:
                print(f"   - {detail}")
        print()
    
    print("Consider reviewing these issues to improve data quality.")
    print("=" * 60)


def prompt_for_tournament_creation_with_id(
    tournament_name: str,
    source_id: str,
    tourney_level: str,
    suggested_id: int,
    source_name: str
) -> int:
    """
    Prompt user for tournament creation with ID suggestion.

    Args:
        tournament_name: Tournament name
        source_id: Original source ID
        tourney_level: Tournament level classification
        suggested_id: Suggested tournament ID
        source_name: Source name for context

    Returns:
        int: Final tournament ID chosen by user
    """
    # Use GUI if available
    if _USE_GUI and _gui_manager:
        try:
            return _gui_manager.prompt_for_tournament_id(
                tournament_name, suggested_id, source_id, tourney_level
            )
        except Exception as e:
            print(f"[WARNING] GUI dialog failed ({e}), falling back to terminal")

    # Terminal fallback
    min_id, max_id = get_level_range(tourney_level)

    print(f"\nNEW TOURNAMENT CREATION")
    print(f"Tournament Name: '{tournament_name}'")
    print(f"Source ID: '{source_id}' (from {source_name})")
    print(f"Tournament Level: {tourney_level}")
    print(f"")
    print(f"Suggested Tournament ID: {suggested_id}")

    # Determine reason for suggestion
    if source_id.isdigit() and len(source_id) == 4:
        source_id_int = int(source_id)
        if min_id <= source_id_int <= max_id:
            print(f"Reason: Source ID fits format and is available")
        else:
            print(f"Reason: Source ID out of range, using lowest available in {min_id}-{max_id}")
    else:
        print(f"Reason: Source ID invalid format, using lowest available in {min_id}-{max_id}")

    while True:
        try:
            choice = input(f"\nPress Enter to accept, or enter custom 4-digit ID ({min_id}-{max_id}): ").strip()

            if not choice:
                return suggested_id

            # Validate user input
            try:
                custom_id = int(choice)
                if not (min_id <= custom_id <= max_id):
                    print(f"Error: ID must be between {min_id} and {max_id} for level {tourney_level}")
                    continue

                if len(str(custom_id)) != 4:
                    print("Error: Tournament ID must be exactly 4 digits")
                    continue

                return custom_id

            except ValueError:
                print("Error: Please enter a valid 4-digit number")
                continue

        except (EOFError, KeyboardInterrupt):
            print(f"\n[AUTO] Accepting suggested ID: {suggested_id}")
            return suggested_id


def prompt_for_missing_tourney_level(tournament_name: str, source_name: str) -> str:
    """
    Prompt user to provide tournament level for missing values.
    
    Args:
        tournament_name: Tournament name for context
        source_name: Source name for context
        
    Returns:
        str: Selected tournament level
    """
    print(f"\nMISSING TOURNAMENT LEVEL")
    print(f"Tournament: '{tournament_name}' (from {source_name})")
    print("Select tournament level:")
    print("G - Grand Slam (5000-5999)")
    print("M - Masters (4500-4999)")
    print("F - Finals (4500-4999)")
    print("D - Other Major (4000-4499)")
    print("A - ATP Tour (2000-3999)")
    print("C - Challenger (1000-1999)")
    print("S - Other/Satellite (1000-1999)")
    
    valid_levels = ['G', 'M', 'F', 'D', 'A', 'C', 'S']
    
    while True:
        choice = input("\nEnter level (G/M/F/D/A/C/S): ").strip().upper()
        
        if choice in valid_levels:
            return choice
        else:
            print("Invalid level. Please enter one of: G, M, F, D, A, C, S")


def handle_tournament_id_selection(
    primary_name: str,
    source_id: str, 
    tourney_level: str, 
    suggested_id: int,
    source_name: str = "unknown"
) -> int:
    """
    Handle tournament ID selection with user prompts.
    
    Args:
        primary_name: Primary tournament name
        source_id: Original source ID
        tourney_level: Tournament level classification
        suggested_id: Suggested tournament ID
        source_name: Source name for context
        
    Returns:
        int: Final tournament ID
    """
    return prompt_for_tournament_creation_with_id(
        primary_name, source_id, tourney_level, suggested_id, source_name
    )


def prompt_for_alternative_name_decision(
    existing_primary_name: str,
    alternative_name: str,
    max_similarity: float,
    source_name: str,
    source_id: str
) -> str:
    """
    Prompt user to decide whether to accept an alternative tournament name
    for an existing tournament when similarity is below 80%.
    
    Args:
        existing_primary_name: Current primary name of the tournament
        alternative_name: New alternative name being proposed
        max_similarity: Highest similarity score with existing names
        source_name: Source name for context
        source_id: Source ID for context
        
    Returns:
        str: 'accept' to add alternative name, 'reject' to create new tournament
    """
    print(f"\nALTERNATIVE NAME DETECTED")
    print(f"=" * 60)
    print(f"Existing tournament: \"{existing_primary_name}\"")
    print(f"Alternative name: \"{alternative_name}\"")
    print(f"Source: {source_name} (ID: {source_id})")
    print(f"Best similarity: {max_similarity:.1%}")
    print()
    print(f"The same source ID ({source_id}) from {source_name} has a different tournament name.")
    print()
    print("Options:")
    print(f"1. Accept and add \"{alternative_name}\" as alternative name for existing tournament")
    print(f"2. Reject and create new tournament for \"{alternative_name}\"")
    print()
    
    while True:
        try:
            choice = input("Select option (1-2): ").strip()
            
            if choice == '1':
                return 'accept'
            elif choice == '2':
                return 'reject'
            else:
                print("Invalid choice. Please enter 1 or 2.")
                
        except (EOFError, KeyboardInterrupt):
            print(f"\nEOF error during user input. Defaulting to reject (create new tournament).")
            return 'reject'