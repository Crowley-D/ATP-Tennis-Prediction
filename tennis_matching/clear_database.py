#!/usr/bin/env python3
"""
Script to clear the tennis tournament matching database.
"""

import os
import sys
from pathlib import Path

# Add current directory to path so we can import tennis_matching
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

from tennis_matching.database.schema import drop_database_schema, create_database_schema
from tennis_matching.database.connection import set_db_path


def clear_database(db_path='databases/tennis_tournaments.db'):
    """Clear the tournament matching database and start fresh."""
    
    print(f"Clearing database: {db_path}")
    
    # Method 1: Delete file approach (safest)
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
            print(f"Successfully deleted database file: {db_path}")
        except Exception as e:
            print(f"Could not delete file {db_path}: {e}")
            print("Trying schema drop approach...")
            
            # Method 2: Schema drop approach
            try:
                set_db_path(db_path)
                drop_database_schema(db_path)
                print("Successfully dropped all database tables")
            except Exception as e2:
                print(f"Could not drop schema: {e2}")
                return False
    
    # Ensure database directory exists
    db_dir = Path(db_path).parent
    db_dir.mkdir(exist_ok=True)
    
    # Create fresh database
    try:
        set_db_path(db_path)
        create_database_schema(db_path)
        print(f"Successfully created fresh database: {db_path}")
        print("\nDatabase is ready for new data!")
        return True
    except Exception as e:
        print(f"Could not create fresh database: {e}")
        return False


def clear_all_databases():
    """Clear all database files (main and test databases)."""
    
    database_files = [
        'databases/tennis_tournaments.db',
        'interactive_test_tennis_tournaments.db',
        'test_tennis_tournaments.db'
    ]
    
    print("Clearing all tournament databases...")
    print("=" * 50)
    
    success_count = 0
    for db_path in database_files:
        if clear_database(db_path):
            success_count += 1
        print("-" * 30)
    
    print(f"\nCleared {success_count}/{len(database_files)} databases successfully")
    
    if success_count == len(database_files):
        print("\nAll databases cleared! Ready for fresh data.")
    else:
        print("\nSome databases could not be cleared. Check error messages above.")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '--all':
            clear_all_databases()
        else:
            db_path = sys.argv[1]
            clear_database(db_path)
    else:
        # Clear main database by default
        clear_database()