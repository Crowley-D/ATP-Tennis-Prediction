"""
CRUD operations for the player matching database.
"""

import sqlite3
from typing import Optional, List, Dict, Any, Tuple
from datetime import date, datetime
import pandas as pd

from .connection import get_db_connection


def get_source_name(source_code: int) -> str:
    """
    Convert source code to human-readable name.
    
    Args:
        source_code: Integer source code
        
    Returns:
        Source name string
    """
    source_mapping = {
        0: 'main_dataset',
        1: 'infosys_api', 
        2: 'tennis_api'
    }
    return source_mapping.get(source_code, f'unknown_source_{source_code}')


def get_next_available_player_id(db_path: str = "databases/tennis_players.db") -> int:
    """
    Get the next available player ID starting from 100000.
    
    Args:
        db_path: Path to database file
        
    Returns:
        Next available player ID (6-digit integer)
    """
    db = get_db_connection(db_path)
    
    query = "SELECT MAX(player_id) FROM players"
    result = db.execute_query(query)
    
    max_id = result[0][0] if result and result[0][0] is not None else 99999
    return max_id + 1


def create_player(
    primary_name: str,
    player_id: Optional[int] = None,
    dob: Optional[str] = None,  # Accept string in YYYY-MM-DD format
    hand: Optional[str] = None,
    height: Optional[int] = None,
    db_path: str = "databases/tennis_players.db"
) -> int:
    """
    Create a new player in the database.
    
    Args:
        primary_name: Primary display name (preprocessed)
        player_id: Specific player ID to use (if None, auto-generated)
        dob: Date of birth in YYYY-MM-DD format (string)
        hand: Playing hand (L/R/U)
        height: Height in cm
        db_path: Path to database file
        
    Returns:
        Player ID of created player
    """
    db = get_db_connection(db_path)
    
    if player_id is None:
        player_id = get_next_available_player_id(db_path)
    
    query = """
        INSERT INTO players (player_id, primary_name, dob, hand, height)
        VALUES (?, ?, ?, ?, ?)
    """
    
    db.execute_update(query, (player_id, primary_name, dob, hand, height))
    return player_id


def get_player_by_id(
    player_id: int,
    db_path: str = "databases/tennis_players.db"
) -> Optional[Dict[str, Any]]:
    """
    Get player information by player ID.
    
    Args:
        player_id: Player ID to look up
        db_path: Path to database file
        
    Returns:
        Player information dict or None if not found
    """
    db = get_db_connection(db_path)
    
    query = """
        SELECT player_id, primary_name, dob, hand, height, created_date, last_updated
        FROM players
        WHERE player_id = ?
    """
    
    result = db.execute_query(query, (player_id,))
    
    if result:
        row = result[0]
        return {
            'player_id': row[0],
            'primary_name': row[1],
            'dob': row[2],
            'hand': row[3],
            'height': row[4],
            'created_date': row[5],
            'last_updated': row[6]
        }
    
    return None


def get_player_by_source_id(
    source_code: int,
    source_id: str,
    db_path: str = "databases/tennis_players.db"
) -> Optional[Dict[str, Any]]:
    """
    Get player information by source code and source ID.
    
    Args:
        source_code: Source code (0=main_dataset, 1=infosys_api, 2=tennis_api)
        source_id: Original ID from source
        db_path: Path to database file
        
    Returns:
        Player information dict or None if not found
    """
    db = get_db_connection(db_path)
    
    query = """
        SELECT p.player_id, p.primary_name, p.dob, p.hand, p.height,
               ps.source_id, ps.source_name_variant, ps.preprocessed_name
        FROM players p
        JOIN player_sources ps ON p.player_id = ps.player_id
        WHERE ps.source_code = ? AND ps.source_id = ?
    """
    
    result = db.execute_query(query, (source_code, source_id))
    
    if result:
        row = result[0]
        return {
            'player_id': row[0],
            'primary_name': row[1],
            'dob': row[2],
            'hand': row[3],
            'height': row[4],
            'source_id': row[5],
            'source_name_variant': row[6],
            'preprocessed_name': row[7]
        }
    
    return None


def add_source_id_to_player(
    player_id: int,
    source_code: int,
    source_id: str,
    source_name_variant: str,
    preprocessed_name: str,
    is_primary_name: bool = False,
    db_path: str = "databases/tennis_players.db"
) -> bool:
    """
    Add a source ID mapping to an existing player.
    
    Args:
        player_id: Player ID to link to
        source_code: Source code (0=main_dataset, 1=infosys_api, 2=tennis_api)
        source_id: Original ID from source
        source_name_variant: Original name from source
        preprocessed_name: Processed name for matching
        is_primary_name: Whether this is the primary display name
        db_path: Path to database file
        
    Returns:
        True if successful, False otherwise
    """
    db = get_db_connection(db_path)
    
    try:
        query = """
            INSERT INTO player_sources 
            (player_id, source_code, source_id, source_name_variant, preprocessed_name, is_primary_name)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        
        db.execute_update(query, (
            player_id, source_code, source_id, 
            source_name_variant, preprocessed_name, is_primary_name
        ))
        return True
        
    except sqlite3.IntegrityError:
        # Source ID already exists for this source
        return False


def search_players_by_name(
    preprocessed_name: str,
    source_code: Optional[int] = None,
    similarity_threshold: float = 0.3,
    limit: int = 20,
    db_path: str = "databases/tennis_players.db"
) -> List[Dict[str, Any]]:
    """
    Search for players by preprocessed name.
    
    Args:
        preprocessed_name: Name to search for
        source_code: Limit search to specific source code (0/1/2)
        similarity_threshold: Minimum similarity score
        limit: Maximum results to return
        db_path: Path to database file
        
    Returns:
        List of matching players with similarity scores
    """
    db = get_db_connection(db_path)
    
    # For now, implement simple exact and LIKE matching
    # TODO: Add fuzzy matching with similarity scores
    
    if source_code is not None:
        query = """
            SELECT DISTINCT p.player_id, p.primary_name, p.dob, p.hand, p.height,
                   ps.source_code, ps.preprocessed_name
            FROM players p
            JOIN player_sources ps ON p.player_id = ps.player_id
            WHERE ps.source_code = ? 
            AND (ps.preprocessed_name = ? OR ps.preprocessed_name LIKE ?)
            LIMIT ?
        """
        params = (source_code, preprocessed_name, f"%{preprocessed_name}%", limit)
    else:
        query = """
            SELECT DISTINCT p.player_id, p.primary_name, p.dob, p.hand, p.height,
                   ps.source_code, ps.preprocessed_name
            FROM players p
            JOIN player_sources ps ON p.player_id = ps.player_id
            WHERE ps.preprocessed_name = ? OR ps.preprocessed_name LIKE ?
            LIMIT ?
        """
        params = (preprocessed_name, f"%{preprocessed_name}%", limit)
    
    results = db.execute_query(query, params)
    
    # Group by player_id and calculate similarity (simplified for now)
    players = {}
    for row in results:
        player_id = row[0]
        if player_id not in players:
            # Simple similarity calculation (exact match = 1.0, partial = 0.7)
            similarity = 1.0 if row[6] == preprocessed_name else 0.7
            
            players[player_id] = {
                'player_id': player_id,
                'primary_name': row[1],
                'dob': row[2],
                'hand': row[3],
                'height': row[4],
                'sources': [row[5]],
                'similarity': similarity
            }
        else:
            if row[5] not in players[player_id]['sources']:
                players[player_id]['sources'].append(row[5])
    
    # Filter by similarity threshold and sort
    filtered_players = [
        player for player in players.values() 
        if player['similarity'] >= similarity_threshold
    ]
    
    return sorted(filtered_players, key=lambda x: x['similarity'], reverse=True)


def update_player_metadata(
    player_id: int,
    dob: Optional[date] = None,
    hand: Optional[str] = None,
    height: Optional[int] = None,
    db_path: str = "databases/tennis_players.db"
) -> bool:
    """
    Update player metadata fields.
    
    Args:
        player_id: Player ID to update
        dob: Date of birth
        hand: Playing hand
        height: Height in cm
        db_path: Path to database file
        
    Returns:
        True if successful, False otherwise
    """
    db = get_db_connection(db_path)
    
    updates = []
    params = []
    
    if dob is not None:
        updates.append("dob = ?")
        params.append(dob)
    if hand is not None:
        updates.append("hand = ?")
        params.append(hand)
    if height is not None:
        updates.append("height = ?")
        params.append(height)
    
    if not updates:
        return False
    
    updates.append("last_updated = ?")
    params.append(datetime.now().isoformat())
    params.append(player_id)
    
    query = f"UPDATE players SET {', '.join(updates)} WHERE player_id = ?"
    
    rows_affected = db.execute_update(query, tuple(params))
    return rows_affected > 0


def get_players_metadata_batch(
    player_ids: List[int], 
    db_path: str = "databases/tennis_players.db"
) -> Dict[int, Dict[str, Any]]:
    """
    Retrieve metadata for multiple players in a single query.
    
    Args:
        player_ids: List of player IDs to retrieve metadata for
        db_path: Path to database file
        
    Returns:
        Dict mapping player_id to metadata dict {dob, hand, height}
    """
    if not player_ids:
        return {}
    
    db = get_db_connection(db_path)
    
    # Create placeholders for IN clause
    placeholders = ','.join(['?'] * len(player_ids))
    
    query = f"""
        SELECT player_id, dob, hand, height
        FROM players 
        WHERE player_id IN ({placeholders})
    """
    
    results = db.execute_query(query, tuple(player_ids))
    
    # Convert results to dictionary
    metadata_dict = {}
    for row in results:
        player_id, dob, hand, height = row
        metadata_dict[player_id] = {
            'dob': dob,
            'hand': hand, 
            'height': height
        }
    
    return metadata_dict


def update_players_metadata_batch(
    metadata_updates: List[Dict[str, Any]], 
    db_path: str = "databases/tennis_players.db"
) -> int:
    """
    Update metadata for multiple players in batch.
    
    Args:
        metadata_updates: List of dicts with player_id and metadata fields
                         Each dict should have 'player_id' and optional 'dob', 'hand', 'height'
        db_path: Path to database file
        
    Returns:
        Number of players updated successfully
    """
    if not metadata_updates:
        return 0
    
    db = get_db_connection(db_path)
    updated_count = 0
    current_time = datetime.now().isoformat()
    
    for update_data in metadata_updates:
        player_id = update_data.get('player_id')
        if not player_id:
            continue
        
        updates = []
        params = []
        
        # Build update clause based on provided fields
        if 'dob' in update_data and update_data['dob'] is not None:
            updates.append("dob = ?")
            params.append(update_data['dob'])
        if 'hand' in update_data and update_data['hand'] is not None:
            updates.append("hand = ?")
            params.append(update_data['hand'])
        if 'height' in update_data and update_data['height'] is not None:
            updates.append("height = ?")
            params.append(update_data['height'])
        
        if not updates:
            continue  # Skip if no metadata to update
        
        # Add last_updated timestamp
        updates.append("last_updated = ?")
        params.append(current_time)
        params.append(player_id)
        
        query = f"UPDATE players SET {', '.join(updates)} WHERE player_id = ?"
        
        try:
            rows_affected = db.execute_update(query, tuple(params))
            if rows_affected > 0:
                updated_count += 1
        except Exception as e:
            print(f"Error updating player {player_id}: {e}")
            continue
    
    return updated_count


def get_players_with_missing_metadata(
    metadata_fields: Optional[List[str]] = None,
    limit: Optional[int] = None,
    db_path: str = "databases/tennis_players.db"
) -> List[Dict[str, Any]]:
    """
    Get players with missing metadata fields.
    
    Args:
        metadata_fields: List of fields to check for missing values ('dob', 'hand', 'height')
                        If None, checks all fields
        limit: Maximum number of results to return
        db_path: Path to database file
        
    Returns:
        List of player dicts with missing metadata
    """
    if metadata_fields is None:
        metadata_fields = ['dob', 'hand', 'height']
    
    db = get_db_connection(db_path)
    
    # Build WHERE clause for missing fields
    conditions = []
    for field in metadata_fields:
        if field in ['dob', 'hand', 'height']:
            conditions.append(f"{field} IS NULL OR {field} = ''")
    
    if not conditions:
        return []
    
    where_clause = " OR ".join(conditions)
    limit_clause = f"LIMIT {limit}" if limit else ""
    
    query = f"""
        SELECT player_id, primary_name, dob, hand, height
        FROM players 
        WHERE {where_clause}
        ORDER BY player_id
        {limit_clause}
    """
    
    results = db.execute_query(query)
    
    players = []
    for row in results:
        player_id, primary_name, dob, hand, height = row
        players.append({
            'player_id': player_id,
            'primary_name': primary_name,
            'dob': dob,
            'hand': hand,
            'height': height,
            'missing_fields': [
                field for field in metadata_fields 
                if (field == 'dob' and not dob) or 
                   (field == 'hand' and not hand) or 
                   (field == 'height' and not height)
            ]
        })
    
    return players


def get_metadata_coverage_stats(
    db_path: str = "databases/tennis_players.db"
) -> Dict[str, Any]:
    """
    Get statistics on metadata coverage in the database.
    
    Args:
        db_path: Path to database file
        
    Returns:
        Dict with coverage statistics for each metadata field
    """
    db = get_db_connection(db_path)
    
    # Get total player count
    total_query = "SELECT COUNT(*) FROM players"
    total_result = db.execute_query(total_query)
    total_players = total_result[0][0] if total_result else 0
    
    if total_players == 0:
        return {
            'total_players': 0,
            'dob_coverage': 0.0,
            'hand_coverage': 0.0,
            'height_coverage': 0.0
        }
    
    # Get coverage for each field
    coverage_stats = {'total_players': total_players}
    
    for field in ['dob', 'hand', 'height']:
        field_query = f"SELECT COUNT(*) FROM players WHERE {field} IS NOT NULL AND {field} != ''"
        field_result = db.execute_query(field_query)
        field_count = field_result[0][0] if field_result else 0
        coverage_stats[f'{field}_coverage'] = (field_count / total_players) * 100
        coverage_stats[f'{field}_count'] = field_count
    
    return coverage_stats


if __name__ == "__main__":
    # Test CRUD operations
    print("Testing CRUD operations...")
    
    try:
        from .connection import get_db_connection, close_all_connections
        
        # Test next available ID
        next_id = get_next_available_player_id()
        print(f"Next available player ID: {next_id}")
        
        # Test create player
        player_id = create_player(
            primary_name="test-player",
            dob="1990-01-01",  # Use string format
            hand="R",
            height=180
        )
        print(f"Created player with ID: {player_id}")
        
        # Test get player
        player = get_player_by_id(player_id)
        print(f"Retrieved player: {player}")
        
        # Test add source mapping
        success = add_source_id_to_player(
            player_id=player_id,
            source_code=0,  # main_dataset
            source_id="test_001",
            source_name_variant="Test Player",
            preprocessed_name="test-player",
            is_primary_name=True
        )
        print(f"Added source mapping: {success}")
        
        # Test search
        matches = search_players_by_name("test-player")
        print(f"Search results: {len(matches)} matches")
        
        # Test new batch metadata functions
        print("\n=== Testing Batch Metadata Functions ===")
        
        # Create a few more test players
        test_players = []
        for i in range(3):
            pid = create_player(
                primary_name=f"test-player-{i+2}",
                dob=f"199{i}-01-01",
                hand=['L', 'R', 'U'][i],
                height=175 + i * 5
            )
            test_players.append(pid)
        
        print(f"Created test players: {test_players}")
        
        # Test batch metadata retrieval
        all_test_ids = [player_id] + test_players
        metadata_batch = get_players_metadata_batch(all_test_ids)
        print(f"Batch metadata retrieval: {len(metadata_batch)} players")
        for pid, meta in metadata_batch.items():
            print(f"  Player {pid}: {meta}")
        
        # Test batch metadata update
        updates = [
            {'player_id': test_players[0], 'height': 190},
            {'player_id': test_players[1], 'hand': 'L', 'dob': '1985-06-15'},
            {'player_id': test_players[2], 'height': 185, 'hand': 'R'}
        ]
        updated_count = update_players_metadata_batch(updates)
        print(f"Batch update result: {updated_count} players updated")
        
        # Test metadata coverage stats
        coverage = get_metadata_coverage_stats()
        print(f"Metadata coverage: {coverage}")
        
        # Test missing metadata query
        missing = get_players_with_missing_metadata(limit=5)
        print(f"Players with missing metadata: {len(missing)}")
        
        print("CRUD operations test: SUCCESS")
        
    except Exception as e:
        print(f"CRUD operations test: FAILED - {e}")
        import traceback
        traceback.print_exc()
    finally:
        close_all_connections()