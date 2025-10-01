def createStats(load_existing=True):
    """
    Create or load player statistics dictionary.

    Args:
        load_existing (bool): Whether to try loading existing stats from file

    Returns:
        dict: Statistics dictionary with all required defaultdict structures
    """
    from collections import defaultdict, deque
    from utils.match_tracking import create_processed_matches_structure
    from utils.stats_persistence import load_player_stats, get_stats_summary
    from utils.chronological_storage import (
        ChronologicalDeque,
        create_chronological_defaultdict,
        create_nested_chronological_defaultdict,
    )

    # Try to load existing stats if requested
    if load_existing:
        existing_stats = load_player_stats()
        if existing_stats is not None:
            summary = get_stats_summary(existing_stats)
            print(
                f"Loaded existing stats: {summary['total_players']} players, {summary['processed_match_ids']} matches"
            )
            return existing_stats
        else:
            print("No existing stats found or failed to load. Creating fresh stats...")

    # Create fresh stats structure with chronological storage
    prev_stats = {}

    # Note: elo_grad_players removed - now handled by utils/elo_manager.py
    prev_stats["last_k_matches"] = create_chronological_defaultdict(maxlen=1000)
    prev_stats["last_k_matches_stats"] = create_nested_chronological_defaultdict(
        maxlen=1000
    )
    prev_stats["matches_played"] = create_chronological_defaultdict(maxlen=1000)
    prev_stats["championships"] = create_chronological_defaultdict(maxlen=1000)
    prev_stats["last_150_round_results"] = create_chronological_defaultdict(maxlen=150)
    # match_dates removed - using matches_played chronological data for date-based calculations
    prev_stats["tourney_history"] = create_nested_chronological_defaultdict(maxlen=1000)
    prev_stats["level_history"] = create_nested_chronological_defaultdict(maxlen=1000)
    prev_stats["round_history"] = create_nested_chronological_defaultdict(maxlen=1000)
    prev_stats["h2h"] = create_nested_chronological_defaultdict(maxlen=1000)
    prev_stats["h2h_surface"] = defaultdict(
        lambda: create_nested_chronological_defaultdict(maxlen=1000)
    )
    prev_stats["games_diff"] = create_chronological_defaultdict(maxlen=1000)

    # Add match tracking system
    prev_stats["processed_matches"] = create_processed_matches_structure()

    print("Created fresh stats structure")
    return prev_stats


def saveStats(prev_stats):
    """
    Save player statistics to persistent storage.

    Args:
        prev_stats (dict): The statistics dictionary to save
    """
    from utils.stats_persistence import save_player_stats
    from utils.match_tracking import save_processed_matches

    # Save player stats
    save_player_stats(prev_stats)

    # Save processed matches (already integrated but ensure it's saved)
    save_processed_matches(prev_stats["processed_matches"])

    print("Statistics saved successfully")


def getStatsInfo(prev_stats):
    """
    Get information about current statistics.

    Args:
        prev_stats (dict): The statistics dictionary

    Returns:
        dict: Summary information
    """
    from utils.stats_persistence import get_stats_summary

    return get_stats_summary(prev_stats)


def clearAllStats():
    """Legacy function - redirects to stats_io module"""
    from utils.stats_io import clear_all_stats

    return clear_all_stats()


def clearAllStatsWithConfirmation():
    """Legacy function - redirects to stats_io module"""
    from utils.stats_io import clear_all_stats_with_confirmation

    return clear_all_stats_with_confirmation()


# Import/Export functions moved to utils.stats_io module
# Use: from utils.stats_io import export_chronological_stats, import_chronological_stats


def export_chronological_stats(prev_stats, filepath):
    """Legacy function - redirects to stats_io module"""
    from utils.stats_io import export_chronological_stats as export_func

    return export_func(prev_stats, filepath)


def import_chronological_stats(filepath):
    """Legacy function - redirects to stats_io module"""
    from utils.stats_io import import_chronological_stats as import_func

    return import_func(filepath)


"""
INPUTS:
Match should be a row in the tennis dataset we created in 0.CleanData.ipynb
Prev_stats should be all the stats data we have until the most recent game. We want to update these stats and return a dictionary again

OUTPUT:
Outputs a dictionary with the updated stats
"""


def updateStats(match, prev_stats):
    from utils.common import getWinnerLoserIDS, get_match_value
    from utils.match_tracking import (
        generate_match_id,
        is_match_processed,
        add_processed_match,
    )
    import traceback

    match_id = match["match_id"]

    # Generate chronological sort key for this match
    sort_key = match["chronological_key"]

    # Parse string chronological_key to tuple if needed
    if isinstance(sort_key, str) and sort_key.startswith("("):
        import ast
        import pandas as pd

        try:
            # First try direct parsing
            parsed_key = ast.literal_eval(sort_key)
            # Ensure we got a tuple/list with 3 elements
            if isinstance(parsed_key, (tuple, list)) and len(parsed_key) >= 3:
                sort_key = parsed_key
                # Convert date string to datetime if needed
                if isinstance(sort_key[0], str):
                    try:
                        date_obj = pd.to_datetime(sort_key[0])
                        sort_key = (date_obj.to_pydatetime(), sort_key[1], sort_key[2])
                    except:
                        pass  # Keep original if parsing fails
            else:
                print(f"[WARNING] Parsed sort_key is not a valid tuple: {parsed_key}")
                raise ValueError("Invalid tuple format")
        except (ValueError, SyntaxError):
            # If that fails, extract manually
            try:
                content = sort_key.strip("()")
                parts = [part.strip().strip("'\"") for part in content.split(",")]
                if len(parts) == 3:
                    date_part, tourney_id, round_order = parts
                    date_obj = pd.to_datetime(date_part)
                    sort_key = (
                        date_obj.to_pydatetime(),
                        int(tourney_id),
                        int(round_order),
                    )
            except Exception as e:
                print(f"[WARNING] Could not parse sort_key: {sort_key}, error: {e}")
                sort_key = (
                    match.get("tourney_date"),
                    match.get("tourney_id", 0),
                    match.get("round_order", 999),
                )

    # Get Winner and Loser ID'S
    try:
        player1_id, player2_id, surface, result, rnd, level = (
            match["player1_id"],
            match["player2_id"],
            match["surface"],
            match["RESULT"],
            match["round"],
            match["tourney_level"],
        )
    except Exception as e:
        print(f"[ERROR] Failed to unpack match data: {e}")
        raise
    w_id, l_id = getWinnerLoserIDS(player1_id, player2_id, result)
    tourney_id = (
        match.get("tourney_id") if hasattr(match, "get") else match["tourney_id"]
    )

    # Note: Elo handling removed - now handled by utils/elo_manager.py

    # No. Matches Played - store individual match events
    prev_stats["matches_played"][w_id].append(sort_key, 1)
    prev_stats["matches_played"][l_id].append(sort_key, 1)

    # No. Matches Won
    try:
        prev_stats["last_k_matches"][w_id].append(sort_key, 1)
        prev_stats["last_k_matches"][l_id].append(sort_key, 0)
    except Exception as e:
        print(f"[ERROR] Failed to append to last_k_matches: {e}")
        raise

    # No. Championships Won - store individual championship events
    if rnd == "F":
        prev_stats["championships"][w_id].append(sort_key, 1)

    # No. Games Played/Won per Round - store individual round events
    prev_stats["round_history"][w_id][rnd].append(sort_key, {"wins": 1, "matches": 1})
    prev_stats["round_history"][l_id][rnd].append(sort_key, {"wins": 0, "matches": 1})

    # Player's Head to Head record - store individual H2H events
    try:
        prev_stats["h2h"][w_id][l_id].append(sort_key, {"wins": 1, "matches": 1})
        prev_stats["h2h"][l_id][w_id].append(sort_key, {"wins": 0, "matches": 1})
        prev_stats["h2h_surface"][surface][w_id][l_id].append(
            sort_key, {"wins": 1, "matches": 1}
        )
        prev_stats["h2h_surface"][surface][l_id][w_id].append(
            sort_key, {"wins": 0, "matches": 1}
        )
    except Exception as e:
        print(f"[ERROR] Failed to append to h2h: {e}")
        raise

    # Update Play Based Stats
    try:
        winner_check = getWinnerLoserIDS(player1_id, player2_id, result)[0]
    except Exception as e:
        print(f"[ERROR] Failed in getWinnerLoserIDS: {e}")
        raise

    if player1_id == winner_check:
        try:
            w_ace, l_ace = (
                get_match_value(match, "p1_ace"),
                get_match_value(match, "p2_ace"),
            )
        except Exception as e:
            print(f"[ERROR] Failed to unpack w_ace, l_ace: {e}")
            raise
        try:
            w_df, l_df = (
                get_match_value(match, "p1_df"),
                get_match_value(match, "p2_df"),
            )
        except Exception as e:
            print(f"[ERROR] Failed to unpack w_df, l_df: {e}")
            raise
        try:
            w_svpt, l_svpt = (
                get_match_value(match, "p1_svpt"),
                get_match_value(match, "p2_svpt"),
            )
        except Exception as e:
            print(f"[ERROR] Failed to unpack w_svpt, l_svpt: {e}")
            raise
        try:
            w_1stIn, l_1stIn = (
                get_match_value(match, "p1_1stIn"),
                get_match_value(match, "p2_1stIn"),
            )
        except Exception as e:
            print(f"[ERROR] Failed to unpack w_1stIn, l_1stIn: {e}")
            raise
        try:
            w_1stWon, l_1stWon = (
                get_match_value(match, "p1_1stWon"),
                get_match_value(match, "p2_1stWon"),
            )
        except Exception as e:
            print(f"[ERROR] Failed to unpack w_1stWon, l_1stWon: {e}")
            raise
        try:
            w_2ndWon, l_2ndWon = (
                get_match_value(match, "p1_2ndWon"),
                get_match_value(match, "p2_2ndWon"),
            )
        except Exception as e:
            print(f"[ERROR] Failed to unpack w_2ndWon, l_2ndWon: {e}")
            raise
        w_bpSaved, l_bpSaved = (
            get_match_value(match, "p1_bpSaved"),
            get_match_value(match, "p2_bpSaved"),
        )
        w_bpFaced, l_bpFaced = (
            get_match_value(match, "p1_bpFaced"),
            get_match_value(match, "p2_bpFaced"),
        )
        w_SvGms, l_SvGms = (
            get_match_value(match, "p1_SvGms"),
            get_match_value(match, "p2_SvGms"),
        )
        w_sw, l_sw = get_match_value(match, "p1_sw"), get_match_value(match, "p2_sw")
    else:
        w_ace, l_ace = (
            get_match_value(match, "p2_ace"),
            get_match_value(match, "p1_ace"),
        )
        w_df, l_df = get_match_value(match, "p2_df"), get_match_value(match, "p1_df")
        w_svpt, l_svpt = (
            get_match_value(match, "p2_svpt"),
            get_match_value(match, "p1_svpt"),
        )
        w_1stIn, l_1stIn = (
            get_match_value(match, "p2_1stIn"),
            get_match_value(match, "p1_1stIn"),
        )
        w_1stWon, l_1stWon = (
            get_match_value(match, "p2_1stWon"),
            get_match_value(match, "p1_1stWon"),
        )
        w_2ndWon, l_2ndWon = (
            get_match_value(match, "p2_2ndWon"),
            get_match_value(match, "p1_2ndWon"),
        )
        w_bpSaved, l_bpSaved = (
            get_match_value(match, "p2_bpSaved"),
            get_match_value(match, "p1_bpSaved"),
        )
        w_bpFaced, l_bpFaced = (
            get_match_value(match, "p2_bpFaced"),
            get_match_value(match, "p1_bpFaced"),
        )
        w_SvGms, l_SvGms = (
            get_match_value(match, "p2_SvGms"),
            get_match_value(match, "p1_SvGms"),
        )
        w_sw, l_sw = get_match_value(match, "p2_sw"), get_match_value(match, "p1_sw")

    if (w_svpt != 0) and (w_svpt != w_1stIn):
        prev_stats["last_k_matches_stats"][w_id]["p_spw"].append(
            sort_key, 100 * ((w_1stWon + w_2ndWon) / w_svpt)
        )

        prev_stats["last_k_matches_stats"][w_id]["p_ace"].append(
            sort_key, 100 * (w_ace / w_svpt)
        )

        prev_stats["last_k_matches_stats"][w_id]["p_df"].append(
            sort_key, 100 * (w_df / w_svpt)
        )

        prev_stats["last_k_matches_stats"][w_id]["p_ace_df"].append(
            sort_key, 100 * (w_ace / (max(w_df, 1)))
        )

        prev_stats["last_k_matches_stats"][w_id]["p_df_2nd"].append(
            sort_key, 100 * (w_df / (w_svpt - w_1stIn))
        )

        prev_stats["last_k_matches_stats"][w_id]["p_1stIn"].append(
            sort_key, 100 * (w_1stIn / w_svpt)
        )

        prev_stats["last_k_matches_stats"][w_id]["p_2ndWon"].append(
            sort_key, 100 * (w_2ndWon / (w_svpt - w_1stIn))
        )

    if l_svpt != 0 and (l_svpt != l_1stIn):
        prev_stats["last_k_matches_stats"][l_id]["p_spw"].append(
            sort_key, 100 * ((l_1stWon + l_2ndWon) / l_svpt)
        )
        prev_stats["last_k_matches_stats"][l_id]["p_ace"].append(
            sort_key, 100 * (l_ace / l_svpt)
        )
        prev_stats["last_k_matches_stats"][l_id]["p_df"].append(
            sort_key, 100 * (l_df / l_svpt)
        )
        prev_stats["last_k_matches_stats"][l_id]["p_df_2nd"].append(
            sort_key, 100 * (l_df / (l_svpt - l_1stIn))
        )
        prev_stats["last_k_matches_stats"][l_id]["p_1stIn"].append(
            sort_key, 100 * (l_1stIn / l_svpt)
        )
        prev_stats["last_k_matches_stats"][l_id]["p_2ndWon"].append(
            sort_key, 100 * (l_2ndWon / (l_svpt - l_1stIn))
        )

    if w_1stIn != 0:
        prev_stats["last_k_matches_stats"][w_id]["p_1stWon"].append(
            sort_key, 100 * (w_1stWon / w_1stIn)
        )
    if l_1stIn != 0:
        prev_stats["last_k_matches_stats"][l_id]["p_1stWon"].append(
            sort_key, 100 * (l_1stWon / l_1stIn)
        )

    if w_bpFaced != 0:
        prev_stats["last_k_matches_stats"][w_id]["p_bpSaved"].append(
            sort_key, 100 * (w_bpSaved / w_bpFaced)
        )
        prev_stats["last_k_matches_stats"][l_id]["p_bpconv"].append(
            sort_key, 100 * ((w_bpSaved - w_bpFaced) / w_bpFaced)
        )
    if l_bpFaced != 0:
        prev_stats["last_k_matches_stats"][l_id]["p_bpSaved"].append(
            sort_key, 100 * (l_bpSaved / l_bpFaced)
        )
        prev_stats["last_k_matches_stats"][w_id]["p_bpconv"].append(
            sort_key, 100 * ((l_bpSaved - l_bpFaced) / l_bpFaced)
        )

    if w_SvGms != 0:
        prev_stats["last_k_matches_stats"][w_id]["p_bp/rg"].append(
            sort_key, 100 * (w_bpFaced / w_SvGms)
        )
    if l_SvGms != 0:
        prev_stats["last_k_matches_stats"][l_id]["p_bp/rg"].append(
            sort_key, 100 * (l_bpFaced / l_SvGms)
        )

    if w_svpt + l_svpt > 0:
        prev_stats["last_k_matches_stats"][w_id]["p_tpw"].append(
            sort_key,
            100
            * (
                (w_1stWon + w_2ndWon)
                + (l_svpt - (l_1stWon + l_2ndWon)) / (w_svpt + l_svpt)
            ),
        )
        prev_stats["last_k_matches_stats"][l_id]["p_tpw"].append(
            sort_key,
            100
            * (
                (l_1stWon + l_2ndWon)
                + (w_svpt - (w_1stWon + w_2ndWon)) / (w_svpt + l_svpt)
            ),
        )

        prev_stats["last_k_matches_stats"][w_id]["p_tpw"].append(
            sort_key, 100 * ((w_1stWon + w_2ndWon) / (w_svpt + l_svpt))
        )
        prev_stats["last_k_matches_stats"][l_id]["p_tpw"].append(
            sort_key, 100 * ((l_1stWon + l_2ndWon) / (w_svpt + l_svpt))
        )

    if w_sw + l_sw > 0:
        prev_stats["last_k_matches_stats"][w_id]["p_setsWon"].append(
            sort_key, w_sw / (w_sw + l_sw) * 100
        )
        prev_stats["last_k_matches_stats"][l_id]["p_setsWon"].append(
            sort_key, l_sw / (w_sw + l_sw) * 100
        )

    # Dates Players Played - no longer needed, using matches_played chronological data

    # Player Tournament Record - store individual tournament events
    prev_stats["tourney_history"][w_id][tourney_id].append(
        sort_key, {"wins": 1, "matches": 1}
    )
    prev_stats["tourney_history"][l_id][tourney_id].append(
        sort_key, {"wins": 0, "matches": 1}
    )

    # Player Tournament Level Record - store individual level events
    prev_stats["level_history"][w_id][level].append(sort_key, {"wins": 1, "matches": 1})
    prev_stats["level_history"][l_id][level].append(sort_key, {"wins": 0, "matches": 1})

    # Player Set Score Difference
    sets_played = 0
    p1_game_diff = 0
    p2_game_diff = 0

    for i in range(1, 6):
        p1_games = match.get(f"p1_set{i}", None)
        p2_games = match.get(f"p2_set{i}", None)

        if p1_games is not None and p2_games is not None:
            p1_diff = p1_games - p2_games
            p2_diff = p2_games - p1_games
            p1_game_diff += p1_diff
            p2_game_diff += p2_diff
            sets_played += 1
        else:
            continue

    try:
        second_winner_check = getWinnerLoserIDS(player1_id, player2_id, result)[0]
    except Exception as e:
        print(f"[ERROR] Failed in second getWinnerLoserIDS: {e}")
        raise

    if player1_id == second_winner_check:
        w_game_diff = p1_game_diff / sets_played if sets_played != 0 else float("nan")
        l_game_diff = p2_game_diff / sets_played if sets_played != 0 else float("nan")
        prev_stats["games_diff"][w_id].append(sort_key, w_game_diff)
        prev_stats["games_diff"][l_id].append(sort_key, l_game_diff)
    else:
        w_game_diff = p2_game_diff / sets_played if sets_played != 0 else float("nan")
        l_game_diff = p1_game_diff / sets_played if sets_played != 0 else float("nan")
        prev_stats["games_diff"][l_id].append(sort_key, l_game_diff)
        prev_stats["games_diff"][w_id].append(sort_key, w_game_diff)

    # Mark match as processed
    add_processed_match(match_id, prev_stats["processed_matches"])

    return prev_stats


"""
INPUTS:
Player1 and Player2 should be a dictionaries with the following keys for each player: ID, ATP_POINTS, ATP_RANK, AGE, HEIGHT, 
Match should be a dict with common information about the game (like the number of BEST_OF, DRAW_SIZE, SURFACE). (cool thing is that in the future we can add more stuff here).
Prev_stats should be all the stats data we have until the most recent game. If we were predicting new data, we would just pass all the calculated stats in the dataset from 1991 to now.
before_time (optional): Chronological sort key tuple (datetime, tourney_id, round_order) - if provided, only considers matches before this cutoff.

OUTPUT:
Outputs a dictionary with all the stats calcualted
"""


def getStats(player1, player2, match, prev_stats, before_time=None, elo_ratings=None):
    from utils.common import (
        mean,
        nan_safe_diff,
        get_chronological_values,
        get_cumulative_count,
        get_cumulative_history_stats,
        deduce_match_date_from_sort_key,
        count_matches_in_date_range,
    )
    from utils.elo_manager import (
        create_elo_ratings,
        get_player_elo_before_match,
        calculate_elo_gradient,
        parse_chronological_key
    )
    from datetime import timedelta
    import numpy as np
    import pandas as pd

    # Parse string before_time to tuple if needed
    if isinstance(before_time, str) and before_time.startswith("("):
        import ast

        try:
            # First try direct parsing
            parsed_time = ast.literal_eval(before_time)
            # Ensure we got a tuple/list with 3 elements
            if isinstance(parsed_time, (tuple, list)) and len(parsed_time) >= 3:
                before_time = parsed_time
                # Convert date string to datetime if needed
                if isinstance(before_time[0], str):
                    try:
                        date_obj = pd.to_datetime(before_time[0])
                        before_time = (
                            date_obj.to_pydatetime(),
                            before_time[1],
                            before_time[2],
                        )
                    except:
                        pass  # Keep original if parsing fails
            else:
                print(
                    f"[WARNING] Parsed before_time is not a valid tuple: {parsed_time}"
                )
                raise ValueError("Invalid tuple format")
        except (ValueError, SyntaxError):
            # If that fails, extract manually
            try:
                content = before_time.strip("()")
                parts = [part.strip().strip("'\"") for part in content.split(",")]
                if len(parts) == 3:
                    date_part, tourney_id, round_order = parts
                    date_obj = pd.to_datetime(date_part)
                    before_time = (
                        date_obj.to_pydatetime(),
                        int(tourney_id),
                        int(round_order),
                    )
            except:
                print(f"[WARNING] Could not parse before_time: {before_time}")
                before_time = None

    output = {}
    PLAYER1_ID = player1["ID"]
    PLAYER2_ID = player2["ID"]
    SURFACE = match["SURFACE"]
    rnd = match["ROUND"]
    output["BEST_OF"] = match["BEST_OF"]
    output["DRAW_SIZE"] = match["DRAW_SIZE"]
    
    # Load Elo ratings if not provided
    if elo_ratings is None:
        elo_ratings = create_elo_ratings(load_existing=True)
    
    # Parse before_time to get current_sort_key for Elo calculations
    current_sort_key = None
    if before_time:
        current_sort_key = parse_chronological_key(before_time)
    
    # Get current Elo ratings for both players (before this match)
    if current_sort_key:
        # Pre-match Elo ratings (before current match)
        p1_general_elo = get_player_elo_before_match(PLAYER1_ID, elo_ratings, current_sort_key)
        p2_general_elo = get_player_elo_before_match(PLAYER2_ID, elo_ratings, current_sort_key)
        p1_surface_elo = get_player_elo_before_match(PLAYER1_ID, elo_ratings, current_sort_key, SURFACE)
        p2_surface_elo = get_player_elo_before_match(PLAYER2_ID, elo_ratings, current_sort_key, SURFACE)
    else:
        # Current Elo ratings (most recent available)
        from utils.elo_manager import get_player_current_elo
        p1_general_elo = get_player_current_elo(PLAYER1_ID, elo_ratings)
        p2_general_elo = get_player_current_elo(PLAYER2_ID, elo_ratings)
        p1_surface_elo = get_player_current_elo(PLAYER1_ID, elo_ratings, SURFACE)
        p2_surface_elo = get_player_current_elo(PLAYER2_ID, elo_ratings, SURFACE)
    
    # Add Elo ratings to output
    output["P1_ELO"] = p1_general_elo
    output["P2_ELO"] = p2_general_elo
    output["P1_SURFACE_ELO"] = p1_surface_elo
    output["P2_SURFACE_ELO"] = p2_surface_elo
    output["ELO_DIFF"] = p1_general_elo - p2_general_elo
    output["SURFACE_ELO_DIFF"] = p1_surface_elo - p2_surface_elo

    # Calculate Differences
    output["AGE_DIFF"] = player1["AGE"] - player2["AGE"]
    output["HEIGHT_DIFF"] = player1["HEIGHT"] - player2["HEIGHT"]
    output["ATP_RANK_DIFF"] = player1["ATP_RANK"] - player2["ATP_RANK"]
    output["ATP_POINTS_DIFF"] = player1["ATP_POINTS"] - player2["ATP_POINTS"]

    # Get Stats from Dictionary
    last_k_matches = prev_stats["last_k_matches"]
    last_k_matches_stats = prev_stats["last_k_matches_stats"]
    matches_played = prev_stats["matches_played"]
    h2h = prev_stats["h2h"]
    h2h_surface = prev_stats["h2h_surface"]
    games_diff = prev_stats["games_diff"]

    # Matches Played
    # Matches Played - calculate from individual match events
    p1_matches_played = get_cumulative_count(matches_played[PLAYER1_ID], before_time)
    p2_matches_played = get_cumulative_count(matches_played[PLAYER2_ID], before_time)
    output["MATCHES_PLAYED_P1"] = p1_matches_played
    output["MATCHES_PLAYED_P2"] = p2_matches_played
    output["N_GAMES_DIFF"] = p1_matches_played - p2_matches_played

    # Head to Head - calculate from individual H2H events with before_time support
    p1_vs_p2 = get_cumulative_history_stats(
        prev_stats["h2h"][PLAYER1_ID][PLAYER2_ID], before_time
    )
    p2_vs_p1 = get_cumulative_history_stats(
        prev_stats["h2h"][PLAYER2_ID][PLAYER1_ID], before_time
    )
    p1_vs_p2_surface = get_cumulative_history_stats(
        prev_stats["h2h_surface"][SURFACE][PLAYER1_ID][PLAYER2_ID], before_time
    )
    p2_vs_p1_surface = get_cumulative_history_stats(
        prev_stats["h2h_surface"][SURFACE][PLAYER2_ID][PLAYER1_ID], before_time
    )

    output["H2H_P1"] = p1_vs_p2["wins"]
    output["H2H_P2"] = p2_vs_p1["wins"]
    output["H2H_SURFACE_P1"] = p1_vs_p2_surface["wins"]
    output["H2H_SURFACE_P2"] = p2_vs_p1_surface["wins"]
    output["H2H_DIFF"] = p1_vs_p2["wins"] - p2_vs_p1["wins"]
    output["H2H_SURFACE_DIFF"] = p1_vs_p2_surface["wins"] - p2_vs_p1_surface["wins"]

    # Championships Won - calculate from individual championship events
    p1_championships = get_cumulative_count(
        prev_stats["championships"][PLAYER1_ID], before_time
    )
    p2_championships = get_cumulative_count(
        prev_stats["championships"][PLAYER2_ID], before_time
    )
    output["CHAMPIONSHIPS_P1"] = p1_championships
    output["CHAMPIONSHIPS_P2"] = p2_championships
    output["CHAMPIONSHIPS_DIFF"] = p1_championships - p2_championships

    for k in [5, 10, 25, 50, 100, 200]:
        # Wins in last K matches
        if (
            len(last_k_matches[PLAYER1_ID]) >= k
            and len(last_k_matches[PLAYER2_ID]) >= k
        ):
            p1_wins = get_chronological_values(
                last_k_matches[PLAYER1_ID], k, before_time
            )
            p2_wins = get_chronological_values(
                last_k_matches[PLAYER2_ID], k, before_time
            )
            output["WIN_LAST_" + str(k) + "_P1"] = sum(p1_wins)
            output["WIN_LAST_" + str(k) + "_P2"] = sum(p2_wins)
            output["WIN_LAST_" + str(k) + "_DIFF"] = sum(p1_wins) - sum(p2_wins)
        else:
            output["WIN_LAST_" + str(k) + "_DIFF"] = 0

        # Elo Gradient calculations using real Elo data
        if current_sort_key:
            # Calculate gradients using before_time to prevent data leakage
            p1_elo_gradient = calculate_elo_gradient(PLAYER1_ID, elo_ratings, k, current_sort_key)
            p2_elo_gradient = calculate_elo_gradient(PLAYER2_ID, elo_ratings, k, current_sort_key)
        else:
            # No before_time specified, cannot calculate gradients safely
            p1_elo_gradient = 0.0
            p2_elo_gradient = 0.0
        
        output[f"ELO_GRAD_LAST_{k}_P1"] = p1_elo_gradient
        output[f"ELO_GRAD_LAST_{k}_P2"] = p2_elo_gradient
        output[f"ELO_GRAD_LAST_{k}_DIFF"] = p1_elo_gradient - p2_elo_gradient

        # Average Set Win difference
        p1_recent = get_chronological_values(games_diff[PLAYER1_ID], k, before_time)
        p2_recent = get_chronological_values(games_diff[PLAYER2_ID], k, before_time)

        p1_avg = sum(p1_recent) / len(p1_recent) if p1_recent else 0
        p2_avg = sum(p2_recent) / len(p2_recent) if p2_recent else 0

        output["SET_WINDIFF_" + str(k) + "_P1"] = p1_avg
        output["SET_WINDIFF_" + str(k) + "_P2"] = p2_avg
        output["SET_WINDIFF_" + str(k) + "_DIFF"] = p1_avg - p2_avg

        # Play based Stats
        p1_bp_saved = get_chronological_values(
            last_k_matches_stats[PLAYER1_ID]["p_bpSaved"], k, before_time
        )
        p2_bp_saved = get_chronological_values(
            last_k_matches_stats[PLAYER2_ID]["p_bpSaved"], k, before_time
        )
        output["P_BP_SAVED_LAST_" + str(k) + "_DIFF"] = nan_safe_diff(
            p1_bp_saved, p2_bp_saved
        )

        p1_tpw = get_chronological_values(
            last_k_matches_stats[PLAYER1_ID]["p_tpw"], k, before_time
        )
        p2_tpw = get_chronological_values(
            last_k_matches_stats[PLAYER2_ID]["p_tpw"], k, before_time
        )
        output["P_TOTAL_PWON_LAST_" + str(k) + "_DIFF"] = nan_safe_diff(p1_tpw, p2_tpw)

    for k in [5, 10, 25, 50]:
        p1_spw = get_chronological_values(
            last_k_matches_stats[PLAYER1_ID]["p_spw"], k, before_time
        )
        p2_spw = get_chronological_values(
            last_k_matches_stats[PLAYER2_ID]["p_spw"], k, before_time
        )
        output["P_SPW_LAST_" + str(k) + "_DIFF"] = nan_safe_diff(p1_spw, p2_spw)

        p1_ace = get_chronological_values(
            last_k_matches_stats[PLAYER1_ID]["p_ace"], k, before_time
        )
        p2_ace = get_chronological_values(
            last_k_matches_stats[PLAYER2_ID]["p_ace"], k, before_time
        )
        output["P_ACE_LAST_" + str(k) + "_DIFF"] = nan_safe_diff(p1_ace, p2_ace)

        p1_df = get_chronological_values(
            last_k_matches_stats[PLAYER1_ID]["p_df"], k, before_time
        )
        p2_df = get_chronological_values(
            last_k_matches_stats[PLAYER2_ID]["p_df"], k, before_time
        )
        output["P_DF_LAST_" + str(k) + "_DIFF"] = nan_safe_diff(p1_df, p2_df)

        p1_ace_df = get_chronological_values(
            last_k_matches_stats[PLAYER1_ID]["p_ace_df"], k, before_time
        )
        p2_ace_df = get_chronological_values(
            last_k_matches_stats[PLAYER2_ID]["p_ace_df"], k, before_time
        )
        output["P_ACE_DF_LAST_" + str(k) + "_DIFF"] = nan_safe_diff(
            p1_ace_df, p2_ace_df
        )

        p1_df_2nd = get_chronological_values(
            last_k_matches_stats[PLAYER1_ID]["p_df_2nd"], k, before_time
        )
        p2_df_2nd = get_chronological_values(
            last_k_matches_stats[PLAYER2_ID]["p_df_2nd"], k, before_time
        )
        output["P_DF_2ND_LAST_" + str(k) + "_DIFF"] = nan_safe_diff(
            p1_df_2nd, p2_df_2nd
        )

        p1_1st_in = get_chronological_values(
            last_k_matches_stats[PLAYER1_ID]["p_1stIn"], k, before_time
        )
        p2_1st_in = get_chronological_values(
            last_k_matches_stats[PLAYER2_ID]["p_1stIn"], k, before_time
        )
        output["P_1ST_IN_LAST_" + str(k) + "_DIFF"] = nan_safe_diff(
            p1_1st_in, p2_1st_in
        )

        p1_1st_won = get_chronological_values(
            last_k_matches_stats[PLAYER1_ID]["p_1stWon"], k, before_time
        )
        p2_1st_won = get_chronological_values(
            last_k_matches_stats[PLAYER2_ID]["p_1stWon"], k, before_time
        )
        output["P_1ST_WON_LAST_" + str(k) + "_DIFF"] = nan_safe_diff(
            p1_1st_won, p2_1st_won
        )

        p1_2nd_won = get_chronological_values(
            last_k_matches_stats[PLAYER1_ID]["p_2ndWon"], k, before_time
        )
        p2_2nd_won = get_chronological_values(
            last_k_matches_stats[PLAYER2_ID]["p_2ndWon"], k, before_time
        )
        output["P_2ND_WON_LAST_" + str(k) + "_DIFF"] = nan_safe_diff(
            p1_2nd_won, p2_2nd_won
        )

        p1_bp_conv = get_chronological_values(
            last_k_matches_stats[PLAYER1_ID]["p_bpconv"], k, before_time
        )
        p2_bp_conv = get_chronological_values(
            last_k_matches_stats[PLAYER2_ID]["p_bpconv"], k, before_time
        )
        output["P_BP_CONVERTED_LAST_" + str(k) + "_DIFF"] = nan_safe_diff(
            p1_bp_conv, p2_bp_conv
        )

        p1_bp_rg = get_chronological_values(
            last_k_matches_stats[PLAYER1_ID]["p_bp/rg"], k, before_time
        )
        p2_bp_rg = get_chronological_values(
            last_k_matches_stats[PLAYER2_ID]["p_bp/rg"], k, before_time
        )
        output["P_BP/RECG_LAST_" + str(k) + "_DIFF"] = nan_safe_diff(p1_bp_rg, p2_bp_rg)

    # Percentage of Matches Won per Round - calculate from individual round events
    rh1 = get_cumulative_history_stats(
        prev_stats["round_history"][PLAYER1_ID][rnd], before_time
    )
    rh2 = get_cumulative_history_stats(
        prev_stats["round_history"][PLAYER2_ID][rnd], before_time
    )

    output["HAS_PLAYED_CUR_ROUND_P1"] = 1 if rh1["matches"] else 0
    output["PCT_WIN_CUR_ROUND_P1"] = (
        (rh1["wins"] / rh1["matches"] * 100) if rh1["matches"] else 0.0
    )

    output["HAS_PLAYED_CUR_ROUND_P2"] = 1 if rh2["matches"] else 0
    output["PCT_WIN_CUR_ROUND_P2"] = (
        (rh2["wins"] / rh2["matches"] * 100) if rh2["matches"] else 0.0
    )

    # Matches in the last K days - using chronological matches_played data
    today = match["TOURNEY_DATE"]

    windows = {
        "1YR": 365,
        "6MO": 183,
        "3MO": 90,
        "1MO": 30,
    }

    matches_played = prev_stats["matches_played"]

    for label, days in windows.items():
        # Determine reference date for the time window
        if before_time is not None:
            # When filtering, use the cutoff date as reference point
            if isinstance(before_time, str):
                # If before_time is a string chronological key, use match date
                reference_date = match.get("TOURNEY_DATE", today)
                if isinstance(reference_date, str):
                    reference_date = pd.to_datetime(reference_date)
            else:
                # If before_time is tuple, extract datetime
                reference_date = before_time[
                    0
                ]  # Extract datetime from (datetime, tourney_id, round_order)
        else:
            # When not filtering, use current match date as reference point
            reference_date = today

        cutoff = reference_date - timedelta(days=days)

        output[f"MATCHES_LAST_{label}_P1"] = count_matches_in_date_range(
            matches_played[PLAYER1_ID], cutoff, reference_date, before_time
        )
        output[f"MATCHES_LAST_{label}_P2"] = count_matches_in_date_range(
            matches_played[PLAYER2_ID], cutoff, reference_date, before_time
        )

    # Tournament History
    tid = match["TOURNEY_ID"]
    level = match["TOURNEY_LEVEL"]

    # Tournament History - calculate from individual tournament events
    hist1 = get_cumulative_history_stats(
        prev_stats["tourney_history"][PLAYER1_ID][tid], before_time
    )
    hist2 = get_cumulative_history_stats(
        prev_stats["tourney_history"][PLAYER2_ID][tid], before_time
    )

    output["PCT_WIN_CUR_TOURN_P1"] = (
        (hist1["wins"] / hist1["matches"]) if hist1["matches"] else 0
    )
    output["HAS_PLAYED_CUR_TOURN_P1"] = 1 if hist1["matches"] else 0
    output["PCT_WIN_CUR_TOURN_P2"] = (
        (hist2["wins"] / hist2["matches"]) if hist2["matches"] else 0
    )
    output["HAS_PLAYED_CUR_TOURN_P2"] = 1 if hist2["matches"] else 0

    # Tournament Level History - calculate from individual level events
    lvl1 = get_cumulative_history_stats(
        prev_stats["level_history"][PLAYER1_ID][level], before_time
    )
    lvl2 = get_cumulative_history_stats(
        prev_stats["level_history"][PLAYER2_ID][level], before_time
    )

    output["PCT_WIN_CUR_LEVEL_P1"] = (
        (lvl1["wins"] / lvl1["matches"]) if lvl1["matches"] else 0
    )
    output["HAS_PLAYED_CUR_LEVEL_P1"] = 1 if lvl1["matches"] else 0
    output["PCT_WIN_CUR_LEVEL_P2"] = (
        (lvl2["wins"] / lvl2["matches"]) if lvl2["matches"] else 0
    )
    output["HAS_PLAYED_CUR_LEVEL_P2"] = 1 if lvl2["matches"] else 0

    return output


if __name__ == "__main__":
    pass
