"""
Database connection management for the tennis tournament matching system.
"""

import sqlite3
import os
from contextlib import contextmanager
from typing import Generator


class DatabaseConnection:
    """Manages SQLite database connections for the tennis matching system."""
    
    def __init__(self, db_path: str = "databases/tennis_tournaments.db"):
        """
        Initialize database connection manager.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        
        # Ensure the databases directory exists
        import os
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
        
    def get_connection(self) -> sqlite3.Connection:
        """
        Get a database connection with proper configuration.
        
        Returns:
            sqlite3.Connection: Configured database connection
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access to rows
        return conn
    
    @contextmanager
    def get_cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        """
        Context manager for database operations with automatic commit/rollback.
        
        Yields:
            sqlite3.Cursor: Database cursor for operations
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def initialize_database(self) -> None:
        """Initialize database schema if it doesn't exist."""
        from tennis_matching.database.schema import create_database_schema
        create_database_schema(self.db_path)
    
    def database_exists(self) -> bool:
        """Check if database file exists."""
        return os.path.exists(self.db_path)


# Global database connection instance
_db_connection = None


def get_db_connection() -> DatabaseConnection:
    """
    Get the global database connection instance.
    
    Returns:
        DatabaseConnection: Global database connection manager
    """
    global _db_connection
    if _db_connection is None:
        _db_connection = DatabaseConnection()
        if not _db_connection.database_exists():
            _db_connection.initialize_database()
    return _db_connection


def set_db_path(db_path: str) -> None:
    """
    Set a custom database path (useful for testing).
    
    Args:
        db_path: Path to the SQLite database file
    """
    global _db_connection
    _db_connection = DatabaseConnection(db_path)
    if not _db_connection.database_exists():
        _db_connection.initialize_database()