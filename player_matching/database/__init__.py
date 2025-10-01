"""
Database management for player ID matching system.
"""

from .connection import get_db_connection, DatabaseConnection, close_all_connections
from .schema import create_tables, drop_tables
from .crud_operations import (
    create_player,
    get_player_by_id,
    get_player_by_source_id,
    add_source_id_to_player,
    search_players_by_name,
    get_next_available_player_id
)

__all__ = [
    'get_db_connection',
    'DatabaseConnection',
    'close_all_connections',
    'create_tables',
    'drop_tables',
    'create_player',
    'get_player_by_id',
    'get_player_by_source_id', 
    'add_source_id_to_player',
    'search_players_by_name',
    'get_next_available_player_id'
]