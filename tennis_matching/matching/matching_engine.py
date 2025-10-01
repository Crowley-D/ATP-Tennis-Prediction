"""
Core matching engine for tennis tournament identification.
"""

from typing import Optional, List, Dict, Any, Tuple
from tennis_matching.database.crud_operations import (
    query_by_source_id,
    check_strict_name_match,
    add_source_id_to_tournament,
    get_all_source_names_for_tournament,
    get_all_tournaments_for_fuzzy_matching,
    create_tournament,
    get_tournament_info,
    suggest_tournament_id,
)
from tennis_matching.matching.text_processing import (
    combined_similarity_score,
    get_similarity_category,
    find_best_matches,
    normalize_tournament_name,
)
from tennis_matching.matching.manual_review import (
    prompt_user_for_match_confirmation,
    prompt_for_new_tournament_name,
    confirm_high_similarity_match,
    handle_validation_warning,
    handle_tournament_id_selection,
    prompt_for_alternative_name_decision,
)


def validate_existing_match(
    tournament_id: int, tournament_name: str, source_name: str, source_id: str
) -> Tuple[Optional[int], str, str]:
    """
    Validate that a new tournament name matches existing names for a tournament.
    Implements enhanced alternative name handling with 80% threshold.

    Args:
        tournament_id: ID of existing tournament
        tournament_name: New tournament name to validate (original, not preprocessed)
        source_name: Source name for context and storage
        source_id: Source ID for context and storage

    Returns:
        Tuple[Optional[int], str, str]: (tournament_id or None, confidence_level, action_taken)
    """
    # Get tournament info to find source names
    tournament_info = get_tournament_info(tournament_id)
    if not tournament_info:
        return tournament_id, "low", "tournament_not_found_error"

    # Check similarity with all existing names (including primary name)
    existing_names = [
        mapping["source_name_variant"] for mapping in tournament_info["source_mappings"]
    ]
    primary_name = tournament_info["primary_name"]

    # Include primary name in similarity check
    all_names_to_check = existing_names + [primary_name]
    similarities = [
        combined_similarity_score(tournament_name, name) for name in all_names_to_check
    ]

    if not similarities:
        # First name for this tournament - auto-accept and store
        success = add_source_id_to_tournament(
            tournament_id, source_name, source_id, tournament_name
        )
        if success:
            return tournament_id, "high", "validated_first_name_stored"
        else:
            return tournament_id, "medium", "validated_first_name_duplicate"

    max_similarity = max(similarities)

    if max_similarity >= 0.8:
        # High similarity - auto-accept alternative name
        success = add_source_id_to_tournament(
            tournament_id, source_name, source_id, tournament_name
        )
        if success:
            return tournament_id, "high", "auto_accepted_alternative_name"
        else:
            return tournament_id, "high", "alternative_name_already_exists"
    else:
        # Below 80% similarity - prompt user for decision
        user_choice = prompt_for_alternative_name_decision(
            primary_name, tournament_name, max_similarity, source_name, source_id
        )

        if user_choice == "accept":
            # User accepts alternative name - add to tournament
            success = add_source_id_to_tournament(
                tournament_id, source_name, source_id, tournament_name
            )
            if success:
                return tournament_id, "medium", "user_accepted_alternative_name"
            else:
                return tournament_id, "medium", "alternative_name_already_exists"
        else:
            # User rejects - signal to create new tournament
            return None, "manual_new", "user_rejected_alternative_create_new"


def fuzzy_match_names_all_sources(tournament_name: str) -> List[Dict[str, Any]]:
    """
    Find fuzzy matches for a tournament name across all sources.
    Now includes both primary names and source variants as candidates.

    Args:
        tournament_name: Tournament name to find matches for (original, not preprocessed)

    Returns:
        List[Dict]: List of potential matches with similarity scores, deduplicated by tournament_id
    """
    # Get all tournaments from database (now includes both primary names and source variants)
    all_tournaments = get_all_tournaments_for_fuzzy_matching()

    if not all_tournaments:
        return []

    # Extract candidate names and calculate similarities (using original names)
    candidate_names = [t["source_name_variant"] for t in all_tournaments]
    matches = find_best_matches(tournament_name, candidate_names, min_threshold=0.6)

    # Convert to detailed match information with deduplication
    detailed_matches = []
    seen_tournaments = {}  # tournament_id -> best match info

    for candidate_name, similarity_score, category in matches:
        # Find the tournament info for this candidate
        for tournament in all_tournaments:
            if tournament["source_name_variant"] == candidate_name:
                tournament_id = tournament["tournament_id"]

                # Check if we already have a match for this tournament
                if tournament_id in seen_tournaments:
                    existing_match = seen_tournaments[tournament_id]
                    # Keep the better match, prioritize primary names on ties
                    if similarity_score > existing_match["similarity_score"] or (
                        similarity_score == existing_match["similarity_score"]
                        and tournament.get("is_primary_name", False)
                    ):
                        seen_tournaments[tournament_id] = {
                            "tournament_id": tournament_id,
                            "primary_name": tournament["primary_name"],
                            "tourney_level": tournament["tourney_level"],
                            "source_name_variant": candidate_name,
                            "source_name": tournament["source_name"],
                            "similarity_score": similarity_score,
                            "category": category,
                            "matched_against": "primary_name"
                            if tournament.get("is_primary_name", False)
                            else "source_variant",
                        }
                else:
                    # First match for this tournament
                    seen_tournaments[tournament_id] = {
                        "tournament_id": tournament_id,
                        "primary_name": tournament["primary_name"],
                        "tourney_level": tournament["tourney_level"],
                        "source_name_variant": candidate_name,
                        "source_name": tournament["source_name"],
                        "similarity_score": similarity_score,
                        "category": category,
                        "matched_against": "primary_name"
                        if tournament.get("is_primary_name", False)
                        else "source_variant",
                    }
                break

    # Convert to list and sort by similarity score (best first)
    detailed_matches = list(seen_tournaments.values())
    detailed_matches.sort(key=lambda x: x["similarity_score"], reverse=True)

    return detailed_matches


def handle_fuzzy_results(
    fuzzy_matches: List[Dict[str, Any]],
    tournament_name: str,
    source_name: str,
    source_id: str,
    tourney_level: str,
) -> Tuple[Optional[int], str, str]:
    """
    Handle fuzzy matching results with appropriate user interaction.

    Args:
        fuzzy_matches: List of fuzzy match results
        tournament_name: Original tournament name
        source_name: Source name for context
        source_id: Source ID for context
        tourney_level: Tournament level classification

    Returns:
        Tuple: (tournament_id, confidence_level, action_taken)
    """
    if not fuzzy_matches:
        return None, "no_match", "no_fuzzy_matches_found"

    # Separate auto-accept and manual review matches
    auto_accept_matches = [m for m in fuzzy_matches if m["category"] == "auto_accept"]
    manual_review_matches = [
        m for m in fuzzy_matches if m["category"] == "manual_review"
    ]

    # Handle auto-accept matches (≥80% similarity)
    if auto_accept_matches:
        best_match = auto_accept_matches[0]  # Already sorted by similarity

        # For 99.5%+ similarity, auto-accept without prompting
        if best_match["similarity_score"] >= 0.995:
            return best_match["tournament_id"], "high", "auto_accept_confirmed"

        # For 80-99.4% similarity, still confirm with user
        if confirm_high_similarity_match(
            tournament_name,
            best_match["source_name_variant"],
            best_match["similarity_score"],
            source_name,
        ):
            return best_match["tournament_id"], "high", "auto_accept_confirmed"
        else:
            # User rejected auto-accept - don't show manual review, just create new
            return None, "no_match", "user_rejected_high_similarity"

    # Handle manual review matches (60-79% similarity)
    if manual_review_matches:
        tournament_id, action = prompt_user_for_match_confirmation(
            tournament_name, manual_review_matches, source_name, source_id
        )

        if tournament_id:
            return tournament_id, "medium", action
        elif action == "create_new_tournament":
            return None, "manual_new", "user_requested_new_tournament"
        else:
            return None, "manual_skip", action

    return None, "no_match", "no_acceptable_matches"


def match_tournament(
    source_name: str, source_id: str, tournament_name: str, tourney_level: str
) -> Tuple[Optional[int], str, str, bool]:
    """
    Main tournament matching function following the 4-step algorithm.

    Args:
        source_name: Name of the data source
        source_id: ID from the source
        tournament_name: Tournament name to match (original name, not preprocessed)
        tourney_level: Tournament level classification

    Returns:
        Tuple: (matched_tournament_id, confidence_level, action_taken, should_store_new_id)
    """

    # Only normalize for basic whitespace/capitalization, but preserve full name
    normalized_name = normalize_tournament_name(tournament_name)

    # STEP 1: Check if this exact source_id already exists
    existing_id_match = query_by_source_id(source_name, source_id)
    if existing_id_match:
        # ID already exists - validate the name matches stored names
        tournament_id, confidence, action = validate_existing_match(
            existing_id_match["tournament_id"], normalized_name, source_name, source_id
        )

        if action == "user_rejected_alternative_create_new":
            # User wants new tournament - continue to Step 4 (create new)
            pass  # Fall through to new tournament creation
        else:
            # Alternative name accepted, auto-accepted, or stored
            return tournament_id, confidence, action, True  # Mark as stored

    # STEP 2: NEW ID - Check strict name match against same source
    strict_match = check_strict_name_match(source_name, normalized_name)
    if strict_match:
        # Found exact name match with different ID - add this ID to existing tournament
        success = add_source_id_to_tournament(
            strict_match["tournament_id"],
            source_name,
            source_id,
            normalized_name,  # Store the full normalized name, not preprocessed
        )
        if success:
            return (
                strict_match["tournament_id"],
                "high",
                "strict_name_match_new_id",
                True,
            )
        else:
            return (
                strict_match["tournament_id"],
                "medium",
                "strict_match_already_exists",
                False,
            )

    # STEP 3: NEW ID - Fuzzy match against all sources (cross-source matching)
    fuzzy_matches = fuzzy_match_names_all_sources(normalized_name)
    if fuzzy_matches:
        tournament_id, confidence, action = handle_fuzzy_results(
            fuzzy_matches, normalized_name, source_name, source_id, tourney_level
        )
        if tournament_id:
            # Add this source ID to the matched tournament (store full name)
            add_source_id_to_tournament(
                tournament_id, source_name, source_id, normalized_name
            )
            return tournament_id, confidence, action + "_new_id", True
        elif action in [
            "user_requested_new_tournament",
            "no_acceptable_matches",
            "user_rejected_high_similarity",
        ]:
            # User wants to create new tournament despite matches, OR rejected all matches
            # Fall through to Step 4 to create new tournament
            pass
        else:
            # User skipped - return None without creating
            return None, confidence, action, False

    # STEP 4: No matches - create new tournament with level-based ID
    suggested_id = suggest_tournament_id(source_id, tourney_level)
    primary_name = prompt_for_new_tournament_name(normalized_name)

    # Check if user selected an existing tournament after editing the name
    if primary_name.startswith("__USE_EXISTING_TOURNAMENT__:"):
        # Extract tournament ID from the special marker
        existing_tournament_id = int(primary_name.split(":")[1])
        print(f"[INFO] User selected existing tournament ID {existing_tournament_id}")

        # Add this source ID to the existing tournament
        add_source_id_to_tournament(
            existing_tournament_id, source_name, source_id, normalized_name
        )
        return existing_tournament_id, "high", "matched_after_name_edit", True

    # Proceed with creating new tournament
    final_id = handle_tournament_id_selection(
        primary_name, source_id, tourney_level, suggested_id, source_name
    )
    new_tournament_id = create_tournament(primary_name, tourney_level, final_id)
    add_source_id_to_tournament(
        new_tournament_id, source_name, source_id, normalized_name
    )
    return new_tournament_id, "high", "created_new_tournament", True


def check_data_quality_issues(source_name: str) -> List[Dict[str, Any]]:
    """
    Check for potential data quality issues within a source.

    Args:
        source_name: Name of the source to check

    Returns:
        List[Dict]: List of data quality issues found
    """
    issues = []

    # Get all tournaments for this source
    tournaments = get_all_tournaments_for_fuzzy_matching(source_name)

    # Group by tournament ID to check for inconsistent names
    id_to_names = {}
    for tournament in tournaments:
        tournament_id = tournament["tournament_id"]
        if tournament_id not in id_to_names:
            id_to_names[tournament_id] = []
        id_to_names[tournament_id].append(tournament["source_name_variant"])

    # Check for tournaments with very different name variants
    for tournament_id, names in id_to_names.items():
        if len(names) > 1:
            # Calculate similarities between all name pairs (using original names)
            similarities = []
            for i, name1 in enumerate(names):
                for name2 in names[i + 1 :]:
                    similarity = combined_similarity_score(name1, name2)
                    similarities.append((name1, name2, similarity))

            # Flag if any pair has low similarity
            low_similarity_pairs = [
                (n1, n2, s) for n1, n2, s in similarities if s < 0.6
            ]

            if low_similarity_pairs:
                issues.append(
                    {
                        "type": "inconsistent_names",
                        "description": f"Tournament {tournament_id} has inconsistent name variants",
                        "details": [
                            f"'{pair[0]}' vs '{pair[1]}' (similarity: {pair[2]:.1%})"
                            for pair in low_similarity_pairs
                        ],
                    }
                )

    return issues
