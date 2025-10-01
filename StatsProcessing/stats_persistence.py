#!/usr/bin/env python3
"""
Statistics Persistence System for updateStats.py

Provides save/load functionality for all player statistics, allowing
the system to maintain state across program restarts.
"""

import json
import os
import pandas as pd
import gzip
import glob
from collections import defaultdict, deque
from datetime import datetime
from .match_tracking import create_processed_matches_structure, load_processed_matches, save_processed_matches


def serialize_stats(prev_stats):
    """
    Convert complex defaultdicts/deques to JSON-serializable format.
    
    Args:
        prev_stats (dict): The statistics dictionary from updateStats.py
        
    Returns:
        dict: JSON-serializable version of the statistics
    """
    serialized = {}
    
    for key, value in prev_stats.items():
        if key == "processed_matches":
            # Skip - this is handled separately by match_tracking.py
            continue
            
        elif key in ["elo_grad_players", "last_k_matches", "games_diff"]:
            # defaultdict with ChronologicalDeque objects
            serialized[key] = {
                "_type": "defaultdict_chronological_deque",
                "_maxlen": 1000 if key != "last_150_round_results" else 150,
                "_data": {str(player_id): deque_obj.to_dict() for player_id, deque_obj in value.items()}
            }
            
        elif key == "last_150_round_results":
            # defaultdict with ChronologicalDeque objects (maxlen=150)
            serialized[key] = {
                "_type": "defaultdict_chronological_deque", 
                "_maxlen": 150,
                "_data": {str(player_id): deque_obj.to_dict() for player_id, deque_obj in value.items()}
            }
            
        elif key == "last_k_matches_stats":
            # defaultdict(lambda: defaultdict(lambda: ChronologicalDeque(maxlen=1000)))
            serialized[key] = {
                "_type": "defaultdict_defaultdict_chronological_deque",
                "_maxlen": 1000,
                "_data": {}
            }
            for player_id, stat_dict in value.items():
                serialized[key]["_data"][str(player_id)] = {
                    stat_name: deque_obj.to_dict() for stat_name, deque_obj in stat_dict.items()
                }
                
        elif key in ["matches_played", "championships"]:
            # defaultdict with ChronologicalDeque objects (now)
            serialized[key] = {
                "_type": "defaultdict_chronological_deque",
                "_maxlen": 1000,
                "_data": {str(player_id): deque_obj.to_dict() for player_id, deque_obj in value.items()}
            }
            
        elif key == "match_dates":
            # defaultdict(list)
            serialized[key] = {
                "_type": "defaultdict_list",
                "_data": {str(player_id): [d.isoformat() if hasattr(d, 'isoformat') else str(d) 
                         for d in date_list] for player_id, date_list in value.items()}
            }
            
        elif key in ["tourney_history", "level_history", "round_history"]:
            # defaultdict(lambda: defaultdict(lambda: ChronologicalDeque))
            serialized[key] = {
                "_type": "defaultdict_defaultdict_chronological_deque",
                "_maxlen": 1000,
                "_data": {}
            }
            for player_id, inner_dict in value.items():
                serialized[key]["_data"][str(player_id)] = {
                    inner_key: deque_obj.to_dict() for inner_key, deque_obj in inner_dict.items()
                }
                
        elif key == "h2h":
            # Nested chronological defaultdict: {player1_id: {player2_id: ChronologicalDeque}}
            serialized[key] = {
                "_type": "nested_chronological_defaultdict",
                "_data": {}
            }
            for p1_id, inner_dict in value.items():
                serialized[key]["_data"][p1_id] = {}
                for p2_id, chrono_deque in inner_dict.items():
                    serialized[key]["_data"][p1_id][p2_id] = chrono_deque.to_dict()
            
        elif key == "h2h_surface":
            # defaultdict(lambda: nested_chronological_defaultdict)
            serialized[key] = {
                "_type": "defaultdict_nested_chronological",
                "_data": {}
            }
            for surface, nested_dict in value.items():
                serialized[key]["_data"][surface] = {}
                for p1_id, inner_dict in nested_dict.items():
                    serialized[key]["_data"][surface][p1_id] = {}
                    for p2_id, chrono_deque in inner_dict.items():
                        serialized[key]["_data"][surface][p1_id][p2_id] = chrono_deque.to_dict()
        else:
            # Regular data that doesn't need special handling
            serialized[key] = value
            
    return serialized


def deserialize_stats(serialized_data):
    """
    Reconstruct defaultdicts/deques from JSON data.
    
    Args:
        serialized_data (dict): JSON-loaded statistics data
        
    Returns:
        dict: Reconstructed statistics dictionary
    """
    from utils.chronological_storage import ChronologicalDeque, create_chronological_defaultdict, create_nested_chronological_defaultdict
    
    prev_stats = {}
    
    for key, value in serialized_data.items():
        if not isinstance(value, dict) or "_type" not in value:
            # Regular data
            prev_stats[key] = value
            continue
            
        data_type = value["_type"]
        data = value["_data"]
        
        if data_type == "defaultdict_chronological_deque":
            # ChronologicalDeque objects
            maxlen = value.get("_maxlen", 1000)
            prev_stats[key] = create_chronological_defaultdict(maxlen=maxlen)
            for player_id, deque_data in data.items():
                # Convert string keys back to integers for player IDs
                player_id_int = int(player_id) if isinstance(player_id, str) and player_id.isdigit() else player_id
                prev_stats[key][player_id_int] = ChronologicalDeque.from_dict(deque_data)
                
        elif data_type == "defaultdict_defaultdict_chronological_deque":
            # Nested ChronologicalDeque objects
            maxlen = value.get("_maxlen", 1000)
            prev_stats[key] = create_nested_chronological_defaultdict(maxlen=maxlen)
            for player_id, stat_dict in data.items():
                # Convert string keys back to integers for player IDs
                player_id_int = int(player_id) if isinstance(player_id, str) and player_id.isdigit() else player_id
                for stat_name, deque_data in stat_dict.items():
                    # Convert inner keys back to integers if they represent numeric IDs (like tournament IDs)
                    if key == "tourney_history" and isinstance(stat_name, str) and stat_name.isdigit():
                        # Tournament IDs should be integers
                        inner_key = int(stat_name)
                    else:
                        # Keep original key for non-numeric keys (like levels: "G", "A", rounds: "F", "SF")
                        inner_key = stat_name
                    prev_stats[key][player_id_int][inner_key] = ChronologicalDeque.from_dict(deque_data)
                    
        elif data_type == "defaultdict_deque":
            maxlen = value["_maxlen"]
            prev_stats[key] = defaultdict(lambda: deque(maxlen=maxlen))
            for player_id, deque_list in data.items():
                prev_stats[key][int(player_id)] = deque(deque_list, maxlen=maxlen)
                
        elif data_type == "defaultdict_defaultdict_deque":
            maxlen = value["_maxlen"]
            prev_stats[key] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=maxlen)))
            for player_id, stat_dict in data.items():
                for stat_name, deque_list in stat_dict.items():
                    prev_stats[key][int(player_id)][stat_name] = deque(deque_list, maxlen=maxlen)
                    
        elif data_type == "defaultdict_int":
            prev_stats[key] = defaultdict(int)
            for player_id, count in data.items():
                prev_stats[key][int(player_id)] = count
                
        elif data_type == "defaultdict_list":
            prev_stats[key] = defaultdict(list)
            for player_id, date_strings in data.items():
                # Convert ISO strings back to datetime objects
                prev_stats[key][int(player_id)] = [
                    pd.to_datetime(d) if isinstance(d, str) else d for d in date_strings
                ]
                
        elif data_type == "defaultdict_defaultdict_dict":
            prev_stats[key] = defaultdict(lambda: defaultdict(lambda: {"wins": 0, "matches": 0}))
            for player_id, inner_dict in data.items():
                for inner_key, inner_value in inner_dict.items():
                    prev_stats[key][int(player_id)][inner_key] = inner_value
                    
        elif data_type == "defaultdict_int_tuple_keys":
            prev_stats[key] = defaultdict(int)
            for key_str, count in data.items():
                p1_id, p2_id = map(int, key_str.split('_'))
                prev_stats[key][(p1_id, p2_id)] = count
                
        elif data_type == "defaultdict_defaultdict_int_tuple_keys":
            prev_stats[key] = defaultdict(lambda: defaultdict(int))
            for surface, h2h_dict in data.items():
                for key_str, count in h2h_dict.items():
                    p1_id, p2_id = map(int, key_str.split('_'))
                    prev_stats[key][surface][(p1_id, p2_id)] = count
                    
        elif data_type == "nested_chronological_defaultdict":
            # H2H: {player1_id: {player2_id: ChronologicalDeque}}
            maxlen = value.get("_maxlen", 1000)
            prev_stats[key] = create_nested_chronological_defaultdict(maxlen=maxlen)
            for p1_id, inner_dict in data.items():
                p1_id_int = int(p1_id) if isinstance(p1_id, str) and p1_id.isdigit() else p1_id
                for p2_id, deque_data in inner_dict.items():
                    p2_id_int = int(p2_id) if isinstance(p2_id, str) and p2_id.isdigit() else p2_id
                    prev_stats[key][p1_id_int][p2_id_int] = ChronologicalDeque.from_dict(deque_data)
                    
        elif data_type == "defaultdict_nested_chronological":
            # H2H_surface: {surface: {player1_id: {player2_id: ChronologicalDeque}}}
            maxlen = value.get("_maxlen", 1000)
            prev_stats[key] = defaultdict(lambda: create_nested_chronological_defaultdict(maxlen=maxlen))
            for surface, nested_dict in data.items():
                for p1_id, inner_dict in nested_dict.items():
                    p1_id_int = int(p1_id) if isinstance(p1_id, str) and p1_id.isdigit() else p1_id
                    for p2_id, deque_data in inner_dict.items():
                        p2_id_int = int(p2_id) if isinstance(p2_id, str) and p2_id.isdigit() else p2_id
                        prev_stats[key][surface][p1_id_int][p2_id_int] = ChronologicalDeque.from_dict(deque_data)
    
    return prev_stats


def save_player_stats(prev_stats, filepath="data/stats/player_stats.json", auto_backup=True, backup_threshold=100):
    """
    Save player statistics to JSON file with automatic backup support.
    
    Args:
        prev_stats (dict): The statistics dictionary from updateStats.py
        filepath (str): Path to save the statistics file
        auto_backup (bool): Whether to automatically create backups when threshold met
        backup_threshold (int): Number of new matches to trigger backup
    """
    # Check if backup should be created before saving
    backup_created = False
    if auto_backup and should_create_backup(prev_stats, backup_threshold):
        try:
            create_timestamped_backup(prev_stats, reason="auto")
            cleanup_old_backups(keep_count=5)  # Keep 5 most recent backups
            backup_created = True
        except Exception as e:
            print(f"Warning: Backup creation failed: {e}")
    
    # Create directory if needed
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Serialize the stats
    serialized = serialize_stats(prev_stats)
    
    # Get stats summary for metadata
    stats_summary = get_stats_summary(prev_stats)
    
    # Add metadata
    save_data = {
        "metadata": {
            "saved_at": datetime.now().isoformat(),
            "version": "1.0",
            "total_players": stats_summary["total_players"],
            "processed_match_ids": stats_summary["processed_match_ids"],
            "total_match_plays": stats_summary["total_match_plays"],
            "data_sections": list(serialized.keys()),
            "backup_created": backup_created
        },
        "player_stats": serialized
    }
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"Saved player statistics to {filepath}")
    if backup_created:
        print("  (Automatic backup was created)")


def load_player_stats(filepath="data/stats/player_stats.json"):
    """
    Load player statistics from JSON file.
    
    Args:
        filepath (str): Path to load the statistics file
        
    Returns:
        dict: Reconstructed statistics dictionary
    """
    if not os.path.exists(filepath):
        print(f"Stats file not found at {filepath}")
        return None
    
    try:
        with open(filepath, 'r') as f:
            save_data = json.load(f)
        
        # Extract and deserialize stats
        serialized_stats = save_data.get("player_stats", {})
        prev_stats = deserialize_stats(serialized_stats)
        
        # Load processed matches separately
        prev_stats["processed_matches"] = load_processed_matches()
        
        # Print load info
        metadata = save_data.get("metadata", {})
        print(f"Loaded player statistics from {filepath}")
        print(f"  Saved: {metadata.get('saved_at', 'Unknown')}")
        print(f"  Players: {metadata.get('total_players', 'Unknown')}")
        print(f"  Sections: {len(metadata.get('data_sections', []))}")
        
        return prev_stats
        
    except Exception as e:
        print(f"Error loading stats from {filepath}: {e}")
        return None


def get_stats_summary(prev_stats):
    """
    Get a summary of the current statistics.
    
    Args:
        prev_stats (dict): The statistics dictionary
        
    Returns:
        dict: Summary information
    """
    if not prev_stats:
        return {"total_players": 0, "total_matches": 0}
    
    # Count unique players
    all_player_ids = set()
    for key in ["matches_played", "elo_grad_players", "last_k_matches"]:
        if key in prev_stats:
            all_player_ids.update(prev_stats[key].keys())
    
    # Count total matches processed from ChronologicalDeque structure
    total_matches = 0
    if "matches_played" in prev_stats:
        for player_deque in prev_stats["matches_played"].values():
            if hasattr(player_deque, 'get_chronological_values'):
                # ChronologicalDeque format
                total_matches += sum(player_deque.get_chronological_values())
            else:
                # Legacy format (should not happen after fix)
                total_matches += player_deque if isinstance(player_deque, int) else 0
        total_matches = total_matches // 2  # Divide by 2 since both players increment
    
    # Count processed match IDs
    try:
        from .match_tracking import get_processed_count
    except ImportError:
        from match_tracking import get_processed_count
    processed_matches = get_processed_count(prev_stats.get("processed_matches", {}))
    
    return {
        "total_players": len(all_player_ids),
        "total_match_plays": total_matches * 2,  # Each match counted for both players
        "unique_matches": total_matches,
        "processed_match_ids": processed_matches
    }


def stats_file_exists(filepath="data/stats/player_stats.json"):
    """Check if statistics file exists."""
    return os.path.exists(filepath)


def get_stats_file_path():
    """Get the default path for the statistics file."""
    return "data/stats/player_stats.json"


# ============================================================================
# BACKUP SYSTEM FUNCTIONS
# ============================================================================

def should_create_backup(prev_stats, threshold=100, backup_dir="data/stats/backups"):
    """
    Determine if a backup should be created based on processed matches since last backup.
    
    Args:
        prev_stats (dict): Current statistics dictionary
        threshold (int): Minimum new matches to trigger backup
        backup_dir (str): Directory containing backups
        
    Returns:
        bool: True if backup should be created
    """
    # Get current processed match count
    current_count = get_stats_summary(prev_stats).get("processed_match_ids", 0)
    
    # Check if backup directory exists
    if not os.path.exists(backup_dir):
        return True  # Create first backup
    
    # Get most recent backup
    backups = list_available_backups(backup_dir)
    if not backups:
        return True  # No backups exist
    
    # Get the most recent backup info
    latest_backup = backups[0]  # Sorted by date, most recent first
    
    # Load the most recent backup metadata to check match count
    try:
        backup_path = os.path.join(backup_dir, latest_backup["filename"])
        with gzip.open(backup_path, 'rt', encoding='utf-8') as f:
            backup_data = json.load(f)
        
        last_backup_count = backup_data.get("metadata", {}).get("processed_match_ids", 0)
        matches_since_backup = current_count - last_backup_count
        
        return matches_since_backup >= threshold
        
    except Exception as e:
        print(f"Warning: Could not read backup metadata: {e}")
        return True  # Err on side of creating backup


def create_timestamped_backup(prev_stats, reason="auto", backup_dir="data/stats/backups"):
    """
    Create a timestamped backup of player statistics.
    
    Args:
        prev_stats (dict): The statistics dictionary to backup
        reason (str): Reason for backup creation ("auto", "manual")
        backup_dir (str): Directory to store backups
        
    Returns:
        str: Path to created backup file
    """
    # Create backup directory if needed
    os.makedirs(backup_dir, exist_ok=True)
    
    # Generate timestamp-based filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backup_filename = f"player_stats_backup_{timestamp}.json.gz"
    backup_path = os.path.join(backup_dir, backup_filename)
    
    # Serialize the stats
    serialized = serialize_stats(prev_stats)
    
    # Get stats summary for backup metadata
    stats_summary = get_stats_summary(prev_stats)
    
    # Add backup-specific metadata
    backup_data = {
        "backup_metadata": {
            "created_at": datetime.now().isoformat(),
            "backup_reason": reason,
            "backup_version": "1.0",
            "original_file_size": "N/A"  # Will be filled if backing up from file
        },
        "metadata": {
            "saved_at": datetime.now().isoformat(),
            "version": "1.0",
            "total_players": stats_summary["total_players"],
            "processed_match_ids": stats_summary["processed_match_ids"],
            "total_match_plays": stats_summary["total_match_plays"],
            "data_sections": list(serialized.keys())
        },
        "player_stats": serialized
    }
    
    # Save compressed backup
    with gzip.open(backup_path, 'wt', encoding='utf-8') as f:
        json.dump(backup_data, f, indent=2)
    
    print(f"Created backup: {backup_filename}")
    print(f"  Reason: {reason}")
    print(f"  Players: {stats_summary['total_players']}")
    print(f"  Matches: {stats_summary['processed_match_ids']}")
    
    return backup_path


def cleanup_old_backups(keep_count=5, backup_dir="data/stats/backups"):
    """
    Remove old backups, keeping only the most recent ones.
    
    Args:
        keep_count (int): Number of most recent backups to keep
        backup_dir (str): Directory containing backups
    """
    if not os.path.exists(backup_dir):
        return
    
    # Get all backup files sorted by modification time (newest first)
    backup_pattern = os.path.join(backup_dir, "player_stats_backup_*.json.gz")
    backup_files = glob.glob(backup_pattern)
    backup_files.sort(key=os.path.getmtime, reverse=True)
    
    # Remove old backups beyond keep_count
    removed_count = 0
    for backup_file in backup_files[keep_count:]:
        try:
            os.remove(backup_file)
            print(f"Removed old backup: {os.path.basename(backup_file)}")
            removed_count += 1
        except Exception as e:
            print(f"Warning: Could not remove backup {backup_file}: {e}")
    
    if removed_count > 0:
        print(f"Cleaned up {removed_count} old backup(s), keeping {min(len(backup_files), keep_count)} most recent")


def list_available_backups(backup_dir="data/stats/backups"):
    """
    List available backups with their metadata.
    
    Args:
        backup_dir (str): Directory containing backups
        
    Returns:
        list: List of backup info dictionaries sorted by date (newest first)
    """
    if not os.path.exists(backup_dir):
        return []
    
    backup_pattern = os.path.join(backup_dir, "player_stats_backup_*.json.gz")
    backup_files = glob.glob(backup_pattern)
    
    backups = []
    for backup_file in backup_files:
        try:
            # Get file info
            file_stat = os.stat(backup_file)
            file_size = file_stat.st_size / 1024  # KB
            
            # Try to read metadata
            metadata = {}
            try:
                with gzip.open(backup_file, 'rt', encoding='utf-8') as f:
                    backup_data = json.load(f)
                metadata = backup_data.get("metadata", {})
                backup_metadata = backup_data.get("backup_metadata", {})
            except:
                backup_metadata = {}
            
            backup_info = {
                "filename": os.path.basename(backup_file),
                "full_path": backup_file,
                "size_kb": round(file_size, 1),
                "created_at": backup_metadata.get("created_at", "Unknown"),
                "reason": backup_metadata.get("backup_reason", "Unknown"),
                "players": metadata.get("total_players", "Unknown"),
                "matches": metadata.get("processed_match_ids", "Unknown")
            }
            backups.append(backup_info)
            
        except Exception as e:
            print(f"Warning: Could not read backup {backup_file}: {e}")
    
    # Sort by creation time (newest first)
    backups.sort(key=lambda x: x["created_at"], reverse=True)
    return backups


def restore_from_backup(timestamp, backup_dir="data/stats/backups", target_file="data/stats/player_stats.json"):
    """
    Restore statistics from a specific timestamped backup.
    
    Args:
        timestamp (str): Timestamp part of backup filename (YYYY-MM-DD_HH-MM-SS)
        backup_dir (str): Directory containing backups
        target_file (str): Target file to restore to
        
    Returns:
        dict: Restored statistics dictionary or None if failed
    """
    backup_filename = f"player_stats_backup_{timestamp}.json.gz"
    backup_path = os.path.join(backup_dir, backup_filename)
    
    if not os.path.exists(backup_path):
        print(f"Backup not found: {backup_filename}")
        available = list_available_backups(backup_dir)
        if available:
            print("Available backups:")
            for backup in available[:5]:  # Show first 5
                print(f"  {backup['filename']} ({backup['created_at']})")
        return None
    
    try:
        # Load backup data
        with gzip.open(backup_path, 'rt', encoding='utf-8') as f:
            backup_data = json.load(f)
        
        # Extract and deserialize stats
        serialized_stats = backup_data.get("player_stats", {})
        prev_stats = deserialize_stats(serialized_stats)
        
        # Load processed matches separately
        prev_stats["processed_matches"] = load_processed_matches()
        
        # Save as current stats file
        os.makedirs(os.path.dirname(target_file), exist_ok=True)
        with open(target_file, 'w') as f:
            # Recreate the standard format
            save_data = {
                "metadata": backup_data.get("metadata", {}),
                "player_stats": serialized_stats
            }
            json.dump(save_data, f, indent=2)
        
        print(f"Restored from backup: {backup_filename}")
        backup_metadata = backup_data.get("backup_metadata", {})
        metadata = backup_data.get("metadata", {})
        print(f"  Created: {backup_metadata.get('created_at', 'Unknown')}")
        print(f"  Players: {metadata.get('total_players', 'Unknown')}")
        print(f"  Matches: {metadata.get('processed_match_ids', 'Unknown')}")
        
        return prev_stats
        
    except Exception as e:
        print(f"Error restoring from backup {backup_filename}: {e}")
        return None


if __name__ == "__main__":
    # Test the serialization/deserialization
    import sys
    sys.path.append('..')
    from utils.updateStats import createStats
    
    print("Testing stats persistence...")
    
    # Create test stats
    test_stats = createStats()
    
    # Add some test data
    test_stats["matches_played"][12345] = 10
    test_stats["elo_grad_players"][12345].extend([1500, 1520, 1535])
    test_stats["last_k_matches_stats"][12345]["p_ace"].extend([5.2, 6.1, 4.8])
    
    # Test save
    print("Testing save...")
    save_player_stats(test_stats, "test_stats.json")
    
    # Test load
    print("Testing load...")
    loaded_stats = load_player_stats("test_stats.json")
    
    if loaded_stats:
        print("Success! Stats loaded correctly")
        summary = get_stats_summary(loaded_stats)
        print(f"Summary: {summary}")
    else:
        print("Failed to load stats")
    
    # Cleanup
    if os.path.exists("test_stats.json"):
        os.remove("test_stats.json")
        print("Cleaned up test file")