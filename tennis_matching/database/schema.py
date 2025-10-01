"""
Database schema definitions for the tennis tournament matching system.
"""

import sqlite3
from typing import Optional


def create_database_schema(db_path: str) -> None:
    """
    Create the database schema with all required tables and indexes.
    
    Args:
        db_path: Path to the SQLite database file
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Create tournaments table with tournament_id and tourney_level
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tournaments (
                tournament_id INTEGER PRIMARY KEY,
                primary_name TEXT NOT NULL,
                tourney_level TEXT NOT NULL,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create tournament_sources table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tournament_sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tournament_id INTEGER REFERENCES tournaments(tournament_id),
                source_name TEXT NOT NULL,
                source_id TEXT NOT NULL,
                source_name_variant TEXT NOT NULL,
                is_primary_name BOOLEAN DEFAULT FALSE,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for efficient lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_source_lookup 
            ON tournament_sources(source_name, source_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_name_lookup 
            ON tournament_sources(source_name, source_name_variant)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_tournament_source 
            ON tournament_sources(tournament_id, source_name)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_tournament_level 
            ON tournaments(tourney_level)
        """)
        
        conn.commit()
        
    finally:
        conn.close()


def drop_database_schema(db_path: str) -> None:
    """
    Drop all tables (useful for testing).
    
    Args:
        db_path: Path to the SQLite database file
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("DROP TABLE IF EXISTS tournament_sources")
        cursor.execute("DROP TABLE IF EXISTS tournaments")
        conn.commit()
        
    finally:
        conn.close()