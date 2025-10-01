"""
Database exploration utilities for the tennis tournament matching system.
"""

import pandas as pd
from typing import List, Dict, Any, Optional
from .connection import get_db_connection
from ..matching.text_processing import combined_similarity_score


def search_tournaments_by_name(
    search_term: str,
    source_name: Optional[str] = None,
    similarity_threshold: float = 0.3,
    limit: int = 20
) -> pd.DataFrame:
    """
    Search tournaments by name with fuzzy matching.
    
    Args:
        search_term: Name or partial name to search for
        source_name: Optional source to limit search to
        similarity_threshold: Minimum similarity score (0.0-1.0)
        limit: Maximum number of results
        
    Returns:
        pd.DataFrame: Search results with similarity scores
    """
    db_conn = get_db_connection()
    
    with db_conn.get_cursor() as cursor:
        if source_name:
            cursor.execute("""
                SELECT DISTINCT t.tournament_id, t.primary_name, t.tourney_level, ts.source_name, 
                       ts.source_id, ts.source_name_variant
                FROM tournaments t
                JOIN tournament_sources ts ON t.tournament_id = ts.tournament_id
                WHERE ts.source_name = ?
                ORDER BY t.primary_name
            """, (source_name,))
        else:
            cursor.execute("""
                SELECT DISTINCT t.tournament_id, t.primary_name, t.tourney_level, ts.source_name,
                       ts.source_id, ts.source_name_variant
                FROM tournaments t
                JOIN tournament_sources ts ON t.tournament_id = ts.tournament_id
                ORDER BY t.primary_name
            """)
        
        rows = cursor.fetchall()
    
    if not rows:
        return pd.DataFrame()
    
    # Calculate similarities and filter
    results = []
    for row in rows:
        # Check similarity with both primary name and source variant
        primary_sim = combined_similarity_score(search_term, row['primary_name'])
        variant_sim = combined_similarity_score(search_term, row['source_name_variant'])
        max_sim = max(primary_sim, variant_sim)
        
        if max_sim >= similarity_threshold:
            results.append({
                'tournament_id': row['tournament_id'],
                'primary_name': row['primary_name'],
                'tourney_level': row['tourney_level'],
                'source_name': row['source_name'],
                'source_id': row['source_id'],
                'source_name_variant': row['source_name_variant'],
                'similarity_score': round(max_sim, 3),
                'matched_field': 'primary_name' if primary_sim >= variant_sim else 'source_variant'
            })
    
    # Sort by similarity score descending
    results.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    # Limit results
    results = results[:limit]
    
    return pd.DataFrame(results)


def get_tournament_details(tournament_id: int) -> Dict[str, Any]:
    """
    Get complete details for a tournament ID.
    
    Args:
        tournament_id: ID of the tournament
        
    Returns:
        Dict: Complete tournament information
    """
    db_conn = get_db_connection()
    
    with db_conn.get_cursor() as cursor:
        # Get tournament basic info
        cursor.execute("""
            SELECT tournament_id, primary_name, tourney_level, created_date, last_updated
            FROM tournaments
            WHERE tournament_id = ?
        """, (tournament_id,))
        
        tournament_row = cursor.fetchone()
        if not tournament_row:
            return {}
        
        # Get all source mappings
        cursor.execute("""
            SELECT source_name, source_id, source_name_variant, is_primary_name, created_date
            FROM tournament_sources
            WHERE tournament_id = ?
            ORDER BY source_name, created_date
        """, (tournament_id,))
        
        source_mappings = []
        for row in cursor.fetchall():
            source_mappings.append({
                'source_name': row['source_name'],
                'source_id': row['source_id'],
                'source_name_variant': row['source_name_variant'],
                'is_primary_name': bool(row['is_primary_name']),
                'created_date': row['created_date']
            })
        
        return {
            'tournament_id': tournament_row['tournament_id'],
            'primary_name': tournament_row['primary_name'],
            'tourney_level': tournament_row['tourney_level'],
            'created_date': tournament_row['created_date'],
            'last_updated': tournament_row['last_updated'],
            'source_mappings': source_mappings
        }


def list_all_tournaments(source_name: Optional[str] = None, limit: int = 100) -> pd.DataFrame:
    """
    List all tournaments, optionally filtered by source.
    
    Args:
        source_name: Optional source to filter by
        limit: Maximum number of results
        
    Returns:
        pd.DataFrame: Tournament list
    """
    db_conn = get_db_connection()
    
    with db_conn.get_cursor() as cursor:
        if source_name:
            cursor.execute("""
                SELECT DISTINCT t.tournament_id, t.primary_name, t.tourney_level, t.created_date,
                       COUNT(ts.id) as source_count
                FROM tournaments t
                JOIN tournament_sources ts ON t.tournament_id = ts.tournament_id
                WHERE ts.source_name = ?
                GROUP BY t.tournament_id, t.primary_name, t.tourney_level, t.created_date
                ORDER BY t.primary_name
                LIMIT ?
            """, (source_name, limit))
        else:
            cursor.execute("""
                SELECT t.tournament_id, t.primary_name, t.tourney_level, t.created_date,
                       COUNT(ts.id) as source_count,
                       GROUP_CONCAT(DISTINCT ts.source_name) as sources
                FROM tournaments t
                JOIN tournament_sources ts ON t.tournament_id = ts.tournament_id
                GROUP BY t.tournament_id, t.primary_name, t.tourney_level, t.created_date
                ORDER BY t.primary_name
                LIMIT ?
            """, (limit,))
        
        rows = cursor.fetchall()
    
    if not rows:
        return pd.DataFrame()
    
    results = []
    for row in rows:
        result = {
            'tournament_id': row['tournament_id'],
            'primary_name': row['primary_name'],
            'tourney_level': row['tourney_level'],
            'created_date': row['created_date'],
            'source_count': row['source_count']
        }
        
        if 'sources' in row.keys():
            result['sources'] = row['sources']
            
        results.append(result)
    
    return pd.DataFrame(results)


def get_tournaments_by_source(source_name: str) -> pd.DataFrame:
    """
    Get all tournaments from a specific source with their IDs and names.
    
    Args:
        source_name: Name of the source
        
    Returns:
        pd.DataFrame: Tournament information for the source
    """
    db_conn = get_db_connection()
    
    with db_conn.get_cursor() as cursor:
        cursor.execute("""
            SELECT t.tournament_id, t.primary_name, t.tourney_level, ts.source_id, ts.source_name_variant,
                   ts.is_primary_name, ts.created_date
            FROM tournaments t
            JOIN tournament_sources ts ON t.tournament_id = ts.tournament_id
            WHERE ts.source_name = ?
            ORDER BY t.primary_name, ts.source_id
        """, (source_name,))
        
        rows = cursor.fetchall()
    
    if not rows:
        return pd.DataFrame()
    
    results = []
    for row in rows:
        results.append({
            'tournament_id': row['tournament_id'],
            'primary_name': row['primary_name'],
            'tourney_level': row['tourney_level'],
            'source_id': row['source_id'],
            'source_name_variant': row['source_name_variant'],
            'is_primary_name': bool(row['is_primary_name']),
            'created_date': row['created_date']
        })
    
    return pd.DataFrame(results)


def find_duplicate_tournament_names() -> pd.DataFrame:
    """
    Find potential duplicate tournaments (same name, different UUIDs).
    
    Returns:
        pd.DataFrame: Potential duplicates
    """
    db_conn = get_db_connection()
    
    with db_conn.get_cursor() as cursor:
        cursor.execute("""
            SELECT ts1.source_name_variant as name,
                   COUNT(DISTINCT ts1.tournament_id) as tournament_count,
                   GROUP_CONCAT(DISTINCT ts1.tournament_id) as tournament_ids,
                   GROUP_CONCAT(DISTINCT ts1.source_name) as sources
            FROM tournament_sources ts1
            GROUP BY ts1.source_name_variant
            HAVING COUNT(DISTINCT ts1.tournament_id) > 1
            ORDER BY tournament_count DESC
        """)
        
        rows = cursor.fetchall()
    
    if not rows:
        return pd.DataFrame()
    
    results = []
    for row in rows:
        results.append({
            'tournament_name': row['name'],
            'tournament_count': row['tournament_count'],
            'tournament_ids': row['tournament_ids'],
            'sources': row['sources']
        })
    
    return pd.DataFrame(results)


def get_database_stats() -> Dict[str, Any]:
    """
    Get database statistics.
    
    Returns:
        Dict: Database statistics
    """
    db_conn = get_db_connection()
    
    with db_conn.get_cursor() as cursor:
        # Count tournaments
        cursor.execute("SELECT COUNT(*) as count FROM tournaments")
        tournament_count = cursor.fetchone()['count']
        
        # Count source mappings
        cursor.execute("SELECT COUNT(*) as count FROM tournament_sources")
        mapping_count = cursor.fetchone()['count']
        
        # Count by source
        cursor.execute("""
            SELECT source_name, COUNT(*) as count
            FROM tournament_sources
            GROUP BY source_name
            ORDER BY source_name
        """)
        source_counts = {row['source_name']: row['count'] for row in cursor.fetchall()}
        
        # Count unique tournament IDs by source
        cursor.execute("""
            SELECT source_name, COUNT(DISTINCT source_id) as unique_ids
            FROM tournament_sources
            GROUP BY source_name
            ORDER BY source_name
        """)
        unique_id_counts = {row['source_name']: row['unique_ids'] for row in cursor.fetchall()}
        
        # Find cross-source tournaments
        cursor.execute("""
            SELECT tournament_id, COUNT(DISTINCT source_name) as source_count
            FROM tournament_sources
            GROUP BY tournament_id
            HAVING COUNT(DISTINCT source_name) > 1
        """)
        cross_source_tournaments = cursor.fetchall()
        cross_source_count = len(cross_source_tournaments)
        
    return {
        'total_tournaments': tournament_count,
        'total_source_mappings': mapping_count,
        'mappings_by_source': source_counts,
        'unique_ids_by_source': unique_id_counts,
        'cross_source_tournaments': cross_source_count
    }


def manually_link_tournaments(
    tournament_id: int,
    source_name: str,
    source_id: str,
    tournament_name: str
) -> bool:
    """
    Manually link a tournament ID to an existing tournament.
    
    Args:
        tournament_id: Existing tournament ID to link to
        source_name: Source name
        source_id: Source ID to link
        tournament_name: Tournament name variant
        
    Returns:
        bool: True if successfully linked
    """
    from .crud_operations import add_source_id_to_tournament
    
    success = add_source_id_to_tournament(
        tournament_id, source_name, source_id, tournament_name
    )
    
    if success:
        print(f"Successfully linked {source_name}:{source_id} '{tournament_name}' to {tournament_id}")
    else:
        print(f"Failed to link - mapping may already exist")
    
    return success


def export_database_to_csv(output_dir: str = ".") -> Dict[str, str]:
    """
    Export database tables to CSV files for inspection.
    
    Args:
        output_dir: Directory to save CSV files
        
    Returns:
        Dict: Paths to created files
    """
    import os
    
    files_created = {}
    
    # Export tournaments table
    tournaments_df = list_all_tournaments()
    tournaments_path = os.path.join(output_dir, "tournaments.csv")
    tournaments_df.to_csv(tournaments_path, index=False)
    files_created['tournaments'] = tournaments_path
    
    # Export source mappings by source
    for source_name in ['main_dataset', 'infosys_api', 'tennis_api']:
        source_df = get_tournaments_by_source(source_name)
        if not source_df.empty:
            source_path = os.path.join(output_dir, f"tournaments_{source_name}.csv")
            source_df.to_csv(source_path, index=False)
            files_created[source_name] = source_path
    
    # Export potential duplicates
    duplicates_df = find_duplicate_tournament_names()
    if not duplicates_df.empty:
        duplicates_path = os.path.join(output_dir, "potential_duplicates.csv")
        duplicates_df.to_csv(duplicates_path, index=False)
        files_created['duplicates'] = duplicates_path
    
    return files_created


def interactive_tournament_search():
    """
    Interactive command-line tournament search tool.
    """
    print("TOURNAMENT DATABASE SEARCH")
    print("=" * 40)
    print("Commands:")
    print("  search <name>     - Search by tournament name")
    print("  list [source]     - List all tournaments (optionally by source)")
    print("  details <id>      - Get tournament details")
    print("  stats             - Database statistics")
    print("  duplicates        - Find potential duplicates")
    print("  export            - Export database to CSV")
    print("  quit              - Exit")
    print()
    
    while True:
        try:
            command = input("db> ").strip()
            
            if command.lower() in ['quit', 'exit', 'q']:
                break
            elif command.startswith('search '):
                search_term = command[7:].strip()
                if search_term:
                    results = search_tournaments_by_name(search_term)
                    if results.empty:
                        print(f"No tournaments found matching '{search_term}'")
                    else:
                        print(f"\nFound {len(results)} matches:")
                        print(results.to_string(index=False))
                else:
                    print("Usage: search <tournament name>")
            
            elif command.startswith('list'):
                parts = command.split()
                source = parts[1] if len(parts) > 1 else None
                
                if source and source not in ['main_dataset', 'infosys_api', 'tennis_api']:
                    print("Invalid source. Use: main_dataset, infosys_api, or tennis_api")
                    continue
                
                results = list_all_tournaments(source)
                if results.empty:
                    print(f"No tournaments found" + (f" for source '{source}'" if source else ""))
                else:
                    print(f"\nTournaments" + (f" from {source}" if source else "") + f" ({len(results)} total):")
                    print(results.to_string(index=False))
            
            elif command.startswith('details '):
                tournament_id_str = command[8:].strip()
                if tournament_id_str:
                    try:
                        tournament_id = int(tournament_id_str)
                        details = get_tournament_details(tournament_id)
                        if not details:
                            print(f"Tournament ID '{tournament_id}' not found")
                        else:
                            print(f"\nTournament Details:")
                            print(f"Tournament ID: {details['tournament_id']}")
                            print(f"Primary Name: {details['primary_name']}")
                            print(f"Tournament Level: {details['tourney_level']}")
                            print(f"Created: {details['created_date']}")
                            print(f"Last Updated: {details['last_updated']}")
                            print(f"\nSource Mappings:")
                            for mapping in details['source_mappings']:
                                primary_marker = " (PRIMARY)" if mapping['is_primary_name'] else ""
                                print(f"  {mapping['source_name']}: ID '{mapping['source_id']}' - '{mapping['source_name_variant']}'{primary_marker}")
                    except ValueError:
                        print(f"Invalid tournament ID: '{tournament_id_str}'. Must be a number.")
                else:
                    print("Usage: details <tournament_id>")
            
            elif command == 'stats':
                stats = get_database_stats()
                print(f"\nDatabase Statistics:")
                print(f"Total tournaments: {stats['total_tournaments']}")
                print(f"Total source mappings: {stats['total_source_mappings']}")
                print(f"Cross-source tournaments: {stats['cross_source_tournaments']}")
                print(f"\nMappings by source:")
                for source, count in stats['mappings_by_source'].items():
                    unique_count = stats['unique_ids_by_source'].get(source, 0)
                    print(f"  {source}: {count} mappings, {unique_count} unique IDs")
            
            elif command == 'duplicates':
                duplicates = find_duplicate_tournament_names()
                if duplicates.empty:
                    print("No potential duplicate tournament names found")
                else:
                    print(f"\nPotential Duplicates ({len(duplicates)} found):")
                    print(duplicates.to_string(index=False))
            
            elif command == 'export':
                files = export_database_to_csv()
                print(f"\nDatabase exported to CSV files:")
                for table, path in files.items():
                    print(f"  {table}: {path}")
            
            elif command == 'help' or command == '?':
                print("Commands:")
                print("  search <name>     - Search by tournament name")
                print("  list [source]     - List all tournaments (optionally by source)")
                print("  details <id>      - Get tournament details")
                print("  stats             - Database statistics")
                print("  duplicates        - Find potential duplicates")
                print("  export            - Export database to CSV")
                print("  quit              - Exit")
            
            else:
                print(f"Unknown command: {command}. Type 'help' for available commands.")
        
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def quick_search(search_term: str, source_name: str = None) -> None:
    """
    Quick command-line search function.
    
    Args:
        search_term: Tournament name to search for
        source_name: Optional source to limit search to
    """
    print(f"Searching for '{search_term}'" + (f" in {source_name}" if source_name else ""))
    print("-" * 50)
    
    results = search_tournaments_by_name(search_term, source_name)
    
    if results.empty:
        print("No matches found")
    else:
        print(f"Found {len(results)} matches:")
        # Show condensed results
        for _, row in results.iterrows():
            print(f"[{row['similarity_score']:.1%}] {row['source_name']}: '{row['source_name_variant']}' (ID: {row['source_id']})")
            print(f"      Tournament ID: {row['tournament_id']} | Primary: '{row['primary_name']}' | Level: {row['tourney_level']}")
            print()


if __name__ == '__main__':
    # Command-line interface for database exploration
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'search' and len(sys.argv) > 2:
            search_term = ' '.join(sys.argv[2:])
            quick_search(search_term)
        elif sys.argv[1] == 'interactive':
            interactive_tournament_search()
        else:
            print("Usage:")
            print("  python explorer.py search <tournament name>")
            print("  python explorer.py interactive")
    else:
        interactive_tournament_search()