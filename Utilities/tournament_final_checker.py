#!/usr/bin/env python3
"""
Tournament Final Processing Checker

This module provides functionality to check if tournament finals have been
processed by updateStats.py, specifically for the PredictPipeline.ipynb
tournament discovery process.
"""

import pandas as pd
from datetime import datetime
from .match_tracking import load_processed_matches, generate_match_id


def get_unprocessed_tournament_finals(tournaments_df, current_year=None):
    """
    Filter tournaments to find those whose finals haven't been processed yet.

    Args:
        tournaments_df (pd.DataFrame): DataFrame of tournaments from ATP calendar scraping
        current_year (int, optional): Year to filter for. Defaults to current year.

    Returns:
        pd.DataFrame: Tournaments whose finals haven't been processed yet

    The function checks for final matches (round='F') that haven't been processed
    by looking at the match tracking system used by updateStats.py.
    """
    if tournaments_df.empty:
        print("❌ No tournaments provided for final processing check")
        return pd.DataFrame()

    if current_year is None:
        current_year = datetime.now().year

    # Load processed matches from the tracking system
    processed_matches = load_processed_matches()

    # Filter tournaments to current year only
    tournaments_current_year = tournaments_df[
        tournaments_df["start_date"].dt.year == current_year
    ].copy()

    print(
        f"Checking {len(tournaments_current_year)} tournaments from {current_year} for unprocessed finals..."
    )

    unprocessed_tournaments = []
    processed_finals_count = 0
    ongoing_tournaments_count = 0

    for idx, tournament in tournaments_current_year.iterrows():
        tournament_id = tournament.get("tournament_id", tournament.get("id", ""))
        tournament_name = tournament.get(
            "tournament", tournament.get("tournament_name", "Unknown")
        )
        start_date = tournament.get("start_date")
        end_date = tournament.get("end_date")

        # Skip tournaments without proper ID
        if not tournament_id:
            print(
                f"Warning: Skipping tournament '{tournament_name}' - no tournament ID"
            )
            continue

        # Check if tournament has finished (final should have been played)
        current_date = pd.to_datetime("today")
        tournament_started = start_date <= current_date
        tournament_finished = end_date < current_date

        # Check if final has been processed
        final_processed = is_tournament_final_processed(
            tournament_id, current_year, processed_matches
        )

        if not final_processed:
            if tournament_finished:
                # Tournament finished but final not processed
                unprocessed_tournaments.append(tournament)
            elif tournament_started and not tournament_finished:
                # Tournament ongoing - final hasn't happened yet
                unprocessed_tournaments.append(tournament)
                ongoing_tournaments_count += 1

        else:
            processed_finals_count += 1

    unprocessed_df = pd.DataFrame(unprocessed_tournaments)

    print(f"\\nTournament Final Processing Summary:")
    print(f"   Total tournaments checked: {len(tournaments_current_year)}")
    print(f"   Finals already processed: {processed_finals_count}")
    print(f"   Ongoing tournaments: {ongoing_tournaments_count}")
    print(f"   Tournaments needing processing: {len(unprocessed_df)}")

    return unprocessed_df


def is_tournament_final_processed(tournament_id, year, processed_matches):
    """
    Check if a specific tournament's final has been processed.

    Args:
        tournament_id (str): Tournament ID
        year (int): Tournament year
        processed_matches (dict): Processed matches dictionary from match_tracking

    Returns:
        bool: True if final has been processed, False otherwise
    """
    try:
        # Format tournament ID as 4-digit string
        tourney_key = f"{int(tournament_id):04d}"
        year_key = f"{year % 100:02d}"  # Last 2 digits of year
        final_round_code = "002"  # Finals round code from match_tracking.py

        # Check if any final matches exist for this tournament
        if (
            tourney_key in processed_matches
            and year_key in processed_matches[tourney_key]
            and final_round_code in processed_matches[tourney_key][year_key]
        ):
            final_matches = processed_matches[tourney_key][year_key][final_round_code]
            return len(final_matches) > 0

        return False

    except (ValueError, KeyError) as e:
        print(f"Warning: Error checking final for tournament {tournament_id}: {e}")
        return False


def get_detailed_tournament_status(tournaments_df, current_year=None):
    """
    Get detailed processing status for all tournaments.

    Args:
        tournaments_df (pd.DataFrame): DataFrame of tournaments
        current_year (int, optional): Year to filter for. Defaults to current year.

    Returns:
        pd.DataFrame: Tournaments with added processing status columns
    """
    if tournaments_df.empty:
        return pd.DataFrame()

    if current_year is None:
        current_year = datetime.now().year

    processed_matches = load_processed_matches()
    current_date = pd.to_datetime("today")

    tournaments_with_status = tournaments_df.copy()

    # Add status columns
    tournaments_with_status["final_processed"] = False
    tournaments_with_status["tournament_status"] = "Unknown"
    tournaments_with_status["processing_priority"] = "Low"

    for idx, tournament in tournaments_with_status.iterrows():
        tournament_id = tournament.get("tournament_id", tournament.get("id", ""))
        end_date = tournament.get("end_date")

        if not tournament_id or pd.isna(end_date):
            continue

        # Check processing status
        final_processed = is_tournament_final_processed(
            tournament_id, current_year, processed_matches
        )

        tournaments_with_status.loc[idx, "final_processed"] = final_processed

        # Determine tournament status
        if final_processed:
            tournaments_with_status.loc[idx, "tournament_status"] = "Processed"
            tournaments_with_status.loc[idx, "processing_priority"] = "None"
        elif end_date < current_date:
            tournaments_with_status.loc[idx, "tournament_status"] = (
                "Finished - Needs Processing"
            )
            tournaments_with_status.loc[idx, "processing_priority"] = "High"
        elif end_date >= current_date:
            tournaments_with_status.loc[idx, "tournament_status"] = "Ongoing"
            tournaments_with_status.loc[idx, "processing_priority"] = "Medium"

    return tournaments_with_status


if __name__ == "__main__":
    # Example usage and testing
    print("Tournament Final Checker - Testing")

    # Load processed matches for testing
    processed_matches = load_processed_matches()
    total_processed = sum(
        len(matches)
        for tourney in processed_matches.values()
        for year in tourney.values()
        for matches in year.values()
    )

    print(f"Total processed matches in system: {total_processed}")
