"""
Statistics Import/Export Module

This module handles all import/export functionality for tennis statistics data,
including chronological data serialization and file management operations.
"""

import json
import os
from collections import defaultdict
from datetime import datetime


def export_chronological_stats(prev_stats, filepath):
    """
    Export stats with chronological data preservation.
    
    Args:
        prev_stats (dict): The statistics dictionary
        filepath (str): Path to save the export file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Convert chronological deques to serializable format
    export_data = {
        "metadata": {
            "export_date": datetime.now().isoformat(),
            "version": "1.0",
            "format": "chronological_stats"
        }
    }
    
    serialized_stats = {}
    
    for key, value in prev_stats.items():
        if key in ["last_k_matches", "last_k_matches_stats", "last_150_round_results", "games_diff"]:
            # These use ChronologicalDeque structures
            serialized_stats[key] = {}
            for player_id, deque_data in value.items():
                if key == "last_k_matches_stats":
                    # Nested structure
                    serialized_stats[key][player_id] = {}
                    for stat_name, chrono_deque in deque_data.items():
                        serialized_stats[key][player_id][stat_name] = chrono_deque.to_dict()
                else:
                    # Single level structure
                    serialized_stats[key][player_id] = deque_data.to_dict()
        else:
            # Regular structures - handle tuple keys for h2h dictionaries
            if key in ["h2h", "h2h_surface"]:
                if key == "h2h":
                    # Convert tuple keys to strings for JSON serialization
                    h2h_data = {}
                    for k, v in value.items():
                        key_str = str(k) if isinstance(k, tuple) else k
                        h2h_data[key_str] = v
                    serialized_stats[key] = h2h_data
                elif key == "h2h_surface":
                    # Handle nested structure with tuple keys
                    surface_data = {}
                    for surface, surface_h2h in value.items():
                        surface_data[surface] = {}
                        for k, v in surface_h2h.items():
                            key_str = str(k) if isinstance(k, tuple) else k
                            surface_data[surface][key_str] = v
                    serialized_stats[key] = surface_data
            else:
                # Regular structures - copy as is
                serialized_stats[key] = dict(value) if hasattr(value, 'items') else value
    
    export_data["stats"] = serialized_stats
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)  # default=str handles datetime objects
    
    print(f"Chronological stats exported to {filepath}")


def import_chronological_stats(filepath):
    """
    Import stats and reconstruct chronological structures.
    
    Args:
        filepath (str): Path to the import file
        
    Returns:
        dict: Reconstructed statistics dictionary
    """
    from utils.chronological_storage import (
        ChronologicalDeque,
        create_chronological_defaultdict,
        create_nested_chronological_defaultdict
    )
    from utils.match_tracking import create_processed_matches_structure
    
    if not os.path.exists(filepath):
        print(f"Import file not found: {filepath}")
        return None
    
    try:
        with open(filepath, 'r') as f:
            export_data = json.load(f)
        
        if export_data.get("metadata", {}).get("format") != "chronological_stats":
            print("Warning: Import file format not recognized")
            return None
        
        # Reconstruct the statistics dictionary
        prev_stats = {}
        serialized_stats = export_data["stats"]
        
        # Reconstruct chronological structures
        for key in ["last_k_matches", "last_150_round_results", "games_diff"]:
            if key in serialized_stats:
                prev_stats[key] = create_chronological_defaultdict(maxlen=1000 if key != "last_150_round_results" else 150)
                for player_id, deque_data in serialized_stats[key].items():
                    # Convert string keys back to integers for player IDs
                    player_id_int = int(player_id) if isinstance(player_id, str) and player_id.isdigit() else player_id
                    prev_stats[key][player_id_int] = ChronologicalDeque.from_dict(deque_data)
        
        # Reconstruct nested chronological structure
        if "last_k_matches_stats" in serialized_stats:
            prev_stats["last_k_matches_stats"] = create_nested_chronological_defaultdict(maxlen=1000)
            for player_id, player_stats in serialized_stats["last_k_matches_stats"].items():
                # Convert string keys back to integers for player IDs
                player_id_int = int(player_id) if isinstance(player_id, str) and player_id.isdigit() else player_id
                for stat_name, deque_data in player_stats.items():
                    prev_stats["last_k_matches_stats"][player_id_int][stat_name] = ChronologicalDeque.from_dict(deque_data)
        
        # Reconstruct chronological and regular structures
        for key in ["matches_played", "championships", "tourney_history", "level_history", "round_history", "h2h", "h2h_surface"]:
            if key in serialized_stats:
                if key in ["matches_played", "championships"]:
                    # These are chronological defaultdicts - only handle ChronologicalDeque format
                    prev_stats[key] = create_chronological_defaultdict(maxlen=1000)
                    for player_id, deque_data in serialized_stats[key].items():
                        if isinstance(deque_data, dict) and '_deque' in deque_data:
                            # Chronological data - reconstruct ChronologicalDeque
                            prev_stats[key][int(player_id)] = ChronologicalDeque.from_dict(deque_data)
                        else:
                            # Unsupported data format - skip with warning
                            print(f"[WARNING] Unsupported data format for {key}[{player_id}]. Expected ChronologicalDeque format. Skipping.")
                elif key in ["tourney_history", "level_history", "round_history"]:
                    # These are nested chronological defaultdicts - only handle ChronologicalDeque format
                    prev_stats[key] = create_nested_chronological_defaultdict(maxlen=1000)
                    for player_id, player_data in serialized_stats[key].items():
                        for sub_key, sub_data in player_data.items():
                            if isinstance(sub_data, dict) and '_deque' in sub_data:
                                # Chronological data - reconstruct ChronologicalDeque
                                prev_stats[key][int(player_id)][sub_key] = ChronologicalDeque.from_dict(sub_data)
                            else:
                                # Unsupported data format - skip with warning
                                print(f"[WARNING] Unsupported data format for {key}[{player_id}][{sub_key}]. Expected ChronologicalDeque format. Skipping.")
                elif key == "h2h":
                    # Convert tuple keys to strings for JSON serialization
                    h2h_data = {}
                    for k, v in serialized_stats[key].items():
                        # Keys stored as strings like "(1, 2)", convert back to tuples
                        try:
                            key_tuple = eval(k) if isinstance(k, str) and k.startswith('(') else k
                            h2h_data[key_tuple] = v
                        except:
                            h2h_data[k] = v  # Keep original if conversion fails
                    prev_stats[key] = defaultdict(int, h2h_data)
                elif key == "h2h_surface":
                    prev_stats[key] = defaultdict(lambda: defaultdict(int))
                    for surface, surface_h2h in serialized_stats[key].items():
                        for k, v in surface_h2h.items():
                            try:
                                key_tuple = eval(k) if isinstance(k, str) and k.startswith('(') else k
                                prev_stats[key][surface][key_tuple] = v
                            except:
                                prev_stats[key][surface][k] = v  # Keep original if conversion fails
        
        # Handle processed_matches (match_dates removed - using matches_played chronological data)
        if "processed_matches" in serialized_stats:
            prev_stats["processed_matches"] = serialized_stats["processed_matches"]
        else:
            prev_stats["processed_matches"] = create_processed_matches_structure()
        
        print(f"Chronological stats imported from {filepath}")
        print(f"Export date: {export_data['metadata'].get('export_date', 'Unknown')}")
        
        return prev_stats
        
    except Exception as e:
        print(f"Error importing chronological stats: {e}")
        return None


def clear_all_stats():
    """
    Clear all existing statistics data and reset to fresh state.
    
    This method will:
    1. Clear all persistent storage files
    2. Reset processed matches tracking
    3. Return a fresh statistics dictionary
    
    Returns:
        dict: Fresh statistics dictionary
    """
    from utils.stats_persistence import get_stats_file_path
    from utils.match_tracking import get_processed_matches_file_path
    
    cleared_items = []
    
    try:
        # Clear persistent stats file
        stats_file = get_stats_file_path()
        if os.path.exists(stats_file):
            os.remove(stats_file)
            cleared_items.append(f"Stats file: {stats_file}")
            
        # Clear processed matches file
        matches_file = get_processed_matches_file_path()
        if os.path.exists(matches_file):
            os.remove(matches_file)
            cleared_items.append(f"Processed matches file: {matches_file}")
            
        # Clear any backup or export files in the stats directory
        stats_dir = os.path.dirname(stats_file)
        if os.path.exists(stats_dir):
            for filename in os.listdir(stats_dir):
                if filename.endswith('.json') or filename.endswith('.pkl'):
                    filepath = os.path.join(stats_dir, filename)
                    try:
                        os.remove(filepath)
                        cleared_items.append(f"Additional file: {filepath}")
                    except Exception as e:
                        print(f"Warning: Could not remove {filepath}: {e}")
        
        # Create fresh stats structure
        from utils.updateStats import createStats
        fresh_stats = createStats(load_existing=False)
        
        print("=== STATS RESET COMPLETE ===")
        print(f"Cleared {len(cleared_items)} items:")
        for item in cleared_items:
            print(f"  - {item}")
        print("Fresh statistics structure created.")
        print("All previous match data and statistics have been permanently deleted.")
        
        return fresh_stats
        
    except Exception as e:
        print(f"Error during stats reset: {e}")
        print("Creating fresh stats structure anyway...")
        from utils.updateStats import createStats
        return createStats(load_existing=False)


def clear_all_stats_with_confirmation():
    """
    Clear all existing statistics data with user confirmation.
    
    This is a safer version of clear_all_stats() that asks for confirmation
    before permanently deleting all data.
    
    Returns:
        dict or None: Fresh statistics dictionary if confirmed, None if cancelled
    """
    from utils.stats_persistence import get_stats_file_path
    
    # Check if there's existing data
    stats_file = get_stats_file_path()
    has_existing_data = os.path.exists(stats_file)
    
    if not has_existing_data:
        print("No existing statistics data found.")
        from utils.updateStats import createStats
        return createStats(load_existing=False)
    
    # Get info about existing data
    try:
        from utils.updateStats import createStats, getStatsInfo
        existing_stats = createStats(load_existing=True)
        stats_info = getStatsInfo(existing_stats)
        
        print("=== EXISTING STATISTICS DATA ===")
        print(f"Total players: {stats_info.get('total_players', 'Unknown')}")
        print(f"Processed matches: {stats_info.get('processed_match_ids', 'Unknown')}")
        print(f"Data file: {stats_file}")
        print("\nWARNING: This action will permanently delete ALL statistics data!")
        print("This cannot be undone.")
        
        # Ask for confirmation
        while True:
            response = input("\nAre you sure you want to clear all statistics? (yes/no): ").strip().lower()
            if response in ['yes', 'y']:
                print("Clearing all statistics data...")
                return clear_all_stats()
            elif response in ['no', 'n']:
                print("Operation cancelled. No data was deleted.")
                return None
            else:
                print("Please enter 'yes' or 'no'.")
                
    except Exception as e:
        print(f"Error checking existing data: {e}")
        print("Proceeding with clear operation...")
        return clear_all_stats()