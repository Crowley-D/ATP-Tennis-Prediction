"""
Database schema definitions for the player matching system.
"""

import sqlite3
from typing import Optional
import os


def create_tables(db_path: str = "databases/tennis_players.db") -> bool:
    """
    Create the database tables for player matching system.
    
    Args:
        db_path: Path to SQLite database file
        
    Returns:
        True if tables created successfully, False otherwise
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Create players table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS players (
                    player_id INTEGER PRIMARY KEY,              -- 6-digit player ID (100000+)
                    primary_name TEXT NOT NULL,                 -- Primary display name (preprocessed)
                    dob DATE,                                   -- Date of birth (YYYY-MM-DD)
                    hand TEXT,                                  -- L (left), R (right), U (unknown)
                    height INTEGER,                             -- Height in cm
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create player_sources table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS player_sources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_id INTEGER REFERENCES players(player_id),
                    source_code INTEGER NOT NULL,           -- 0=main_dataset, 1=infosys_api, 2=tennis_api
                    source_id TEXT NOT NULL,                -- Original ID from source
                    source_name_variant TEXT NOT NULL,      -- Original name from source (before preprocessing)
                    preprocessed_name TEXT NOT NULL,        -- Name after preprocessing
                    is_primary_name BOOLEAN DEFAULT FALSE,  -- Whether this is the primary display name
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(source_code, source_id)
                );
            """)
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_players_primary_name ON players(primary_name);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_players_dob ON players(dob);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_player_sources_player_id ON player_sources(player_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_player_sources_preprocessed ON player_sources(preprocessed_name);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_player_sources_source ON player_sources(source_code, source_id);")
            
            conn.commit()
            return True
            
    except Exception as e:
        print(f"Error creating database tables: {e}")
        return False


def drop_tables(db_path: str = "databases/tennis_players.db") -> bool:
    """
    Drop all tables in the database (for testing/reset purposes).
    
    Args:
        db_path: Path to SQLite database file
        
    Returns:
        True if tables dropped successfully, False otherwise
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("DROP TABLE IF EXISTS player_sources;")
            cursor.execute("DROP TABLE IF EXISTS players;")
            
            conn.commit()
            return True
            
    except Exception as e:
        print(f"Error dropping database tables: {e}")
        return False


def verify_schema(db_path: str = "databases/tennis_players.db") -> dict:
    """
    Verify that the database schema exists and is correct.
    
    Args:
        db_path: Path to SQLite database file
        
    Returns:
        Dict with verification results
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Check if tables exist
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name IN ('players', 'player_sources');
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            # Check table structures
            results = {
                'tables_exist': len(tables) == 2,
                'players_table': 'players' in tables,
                'player_sources_table': 'player_sources' in tables,
                'indexes': []
            }
            
            # Check indexes
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='index' AND name LIKE 'idx_%';
            """)
            results['indexes'] = [row[0] for row in cursor.fetchall()]
            
            return results
            
    except Exception as e:
        return {'error': str(e)}


def migrate_source_names_to_codes(db_path: str = "databases/tennis_players.db") -> bool:
    """
    Migrate existing string source names to numeric codes.
    
    Args:
        db_path: Path to SQLite database file
        
    Returns:
        True if migration successful, False otherwise
    """
    source_mapping = {
        'main_dataset': 0,
        'infosys_api': 1,
        'tennis_api': 2
    }
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Check if we need to migrate (if source_name column exists)
            cursor.execute("PRAGMA table_info(player_sources)")
            columns = [col[1] for col in cursor.fetchall()]
            
            if 'source_name' not in columns:
                print("No migration needed - schema already uses numeric codes")
                return True
            
            if 'source_code' in columns:
                print("Migration already in progress - source_code column exists")
                return True
            
            print("Starting migration from string source names to numeric codes...")
            
            # Add source_code column
            cursor.execute("ALTER TABLE player_sources ADD COLUMN source_code INTEGER")
            
            # Update source_code based on source_name
            for source_name, source_code in source_mapping.items():
                cursor.execute(
                    "UPDATE player_sources SET source_code = ? WHERE source_name = ?",
                    (source_code, source_name)
                )
                rows_updated = cursor.rowcount
                print(f"Updated {rows_updated} rows for {source_name} -> {source_code}")
            
            # Check for any unmapped source names
            cursor.execute("SELECT DISTINCT source_name FROM player_sources WHERE source_code IS NULL")
            unmapped = cursor.fetchall()
            if unmapped:
                print(f"Warning: Found unmapped source names: {[row[0] for row in unmapped]}")
                return False
            
            # Make source_code NOT NULL
            cursor.execute("""
                CREATE TABLE player_sources_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_id INTEGER REFERENCES players(player_id),
                    source_code INTEGER NOT NULL,
                    source_id TEXT NOT NULL,
                    source_name_variant TEXT NOT NULL,
                    preprocessed_name TEXT NOT NULL,
                    is_primary_name BOOLEAN DEFAULT FALSE,
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(source_code, source_id)
                )
            """)
            
            # Copy data to new table
            cursor.execute("""
                INSERT INTO player_sources_new 
                (id, player_id, source_code, source_id, source_name_variant, 
                 preprocessed_name, is_primary_name, created_date)
                SELECT id, player_id, source_code, source_id, source_name_variant, 
                       preprocessed_name, is_primary_name, created_date
                FROM player_sources
            """)
            
            # Drop old table and rename new one
            cursor.execute("DROP TABLE player_sources")
            cursor.execute("ALTER TABLE player_sources_new RENAME TO player_sources")
            
            # Recreate indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_player_sources_player_id ON player_sources(player_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_player_sources_preprocessed ON player_sources(preprocessed_name);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_player_sources_source ON player_sources(source_code, source_id);")
            
            conn.commit()
            print("Migration completed successfully!")
            return True
            
    except Exception as e:
        print(f"Error during migration: {e}")
        return False


if __name__ == "__main__":
    # Test schema creation
    print("Creating database schema...")
    success = create_tables()
    print(f"Schema creation: {'Success' if success else 'Failed'}")
    
    # Run migration if needed
    print("\nRunning migration...")
    migration_success = migrate_source_names_to_codes()
    print(f"Migration: {'Success' if migration_success else 'Failed'}")
    
    # Verify schema
    print("\nVerifying schema...")
    verification = verify_schema()
    print(f"Verification results: {verification}")