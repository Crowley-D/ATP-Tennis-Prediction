"""
Database connection management for the player matching system.
"""

import sqlite3
from contextlib import contextmanager
from typing import Optional, Generator
import os
from pathlib import Path


class DatabaseConnection:
    """
    Database connection manager for the player matching system.
    """
    
    def __init__(self, db_path: str = "databases/tennis_players.db"):
        """
        Initialize database connection manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.ensure_directory_exists()
    
    def ensure_directory_exists(self):
        """Ensure the database directory exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    @contextmanager
    def get_cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        """
        Get a database cursor with automatic connection management.
        
        Yields:
            sqlite3.Cursor: Database cursor
        """
        conn = None
        cursor = None
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row  # Enable column access by name
            cursor = conn.cursor()
            yield cursor
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            # Explicitly close cursor first, then connection
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    
    def execute_query(self, query: str, params: tuple = ()) -> list:
        """
        Execute a SELECT query and return results.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            List of query results
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchall()
    
    def execute_update(self, query: str, params: tuple = ()) -> int:
        """
        Execute an INSERT/UPDATE/DELETE query.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Number of affected rows
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            return cursor.rowcount
    
    def execute_insert(self, query: str, params: tuple = ()) -> int:
        """
        Execute an INSERT query and return the last row ID.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Last inserted row ID
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            return cursor.lastrowid
    
    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.
        
        Args:
            table_name: Name of the table to check
            
        Returns:
            True if table exists, False otherwise
        """
        query = """
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name=?;
        """
        result = self.execute_query(query, (table_name,))
        return len(result) > 0
    
    def get_table_info(self, table_name: str) -> list:
        """
        Get information about table structure.
        
        Args:
            table_name: Name of the table
            
        Returns:
            List of column information
        """
        query = f"PRAGMA table_info({table_name});"
        return self.execute_query(query)


# Global database connection instance
_db_connection: Optional[DatabaseConnection] = None


def get_db_connection(db_path: str = "databases/tennis_players.db") -> DatabaseConnection:
    """
    Get or create the global database connection instance.
    
    Args:
        db_path: Path to SQLite database file
        
    Returns:
        DatabaseConnection instance
    """
    global _db_connection
    
    if _db_connection is None or str(_db_connection.db_path) != db_path:
        _db_connection = DatabaseConnection(db_path)
    
    return _db_connection


def close_all_connections():
    """
    Close all cached database connections and clear the global connection.
    
    This helps prevent file locking issues on Windows.
    """
    global _db_connection
    _db_connection = None
    
    # Force garbage collection
    import gc
    gc.collect()


@contextmanager
def get_cursor(db_path: str = "databases/tennis_players.db") -> Generator[sqlite3.Cursor, None, None]:
    """
    Convenience function to get a database cursor.
    
    Args:
        db_path: Path to SQLite database file
        
    Yields:
        sqlite3.Cursor: Database cursor
    """
    # Create a new connection instance for each call to avoid connection caching issues
    db_connection = DatabaseConnection(db_path)
    with db_connection.get_cursor() as cursor:
        yield cursor


if __name__ == "__main__":
    # Test database connection
    print("Testing database connection...")
    
    try:
        db = get_db_connection()
        print(f"Database path: {db.db_path}")
        print(f"Directory exists: {db.db_path.parent.exists()}")
        
        # Test cursor
        with db.get_cursor() as cursor:
            cursor.execute("SELECT sqlite_version();")
            version = cursor.fetchone()[0]
            print(f"SQLite version: {version}")
        
        print("Database connection test: SUCCESS")
        
    except Exception as e:
        print(f"Database connection test: FAILED - {e}")