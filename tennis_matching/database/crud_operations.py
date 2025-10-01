"""
CRUD operations for the tennis tournament matching system.
"""

from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from tennis_matching.database.connection import get_db_connection


def get_level_range(tourney_level: str) -> Tuple[int, int]:
    """
    Returns (min_id, max_id) for tournament level

    Args:
        tourney_level: G, M, F, D, A, C, S

    Returns:
        tuple: (min_id, max_id) for the level

    Examples:
        get_level_range('G') -> (5000, 5999)
        get_level_range('A') -> (2000, 3999)
    """
    ranges = {
        "G": (5000, 5999),
        "O": (4800, 4999),
        "M": (4500, 4799),
        "F": (4500, 4999),
        "D": (4000, 4499),
        "A": (2000, 3999),
        "C": (1000, 1999),
        "S": (1000, 1999),
    }
    return ranges.get(tourney_level.upper(), (1000, 1999))


def validate_source_id_for_level(source_id: str, tourney_level: str) -> bool:
    """
    Check if source_id is 4-digit numeric and within level range

    Args:
        source_id: Original source tournament ID
        tourney_level: Tournament level classification

    Returns:
        bool: True if source_id is valid for the level
    """
    # Check if numeric and 4 digits
    if not source_id.isdigit() or len(source_id) != 4:
        return False

    # Check if within level range
    id_num = int(source_id)
    min_id, max_id = get_level_range(tourney_level)
    return min_id <= id_num <= max_id


def get_lowest_available_id(tourney_level: str) -> int:
    """
    Find the lowest available tournament ID in the level range

    Args:
        tourney_level: Tournament level classification

    Returns:
        int: Lowest available 4-digit ID in range

    Logic:
        - Query database for existing tournament_ids in level range
        - Find gaps and return lowest available number
        - May have gaps due to source_ids that fit requirements
    """
    min_id, max_id = get_level_range(tourney_level)
    db_conn = get_db_connection()

    with db_conn.get_cursor() as cursor:
        cursor.execute(
            """
            SELECT tournament_id 
            FROM tournaments 
            WHERE tournament_id BETWEEN ? AND ?
            ORDER BY tournament_id
        """,
            (min_id, max_id),
        )

        existing_ids = {row["tournament_id"] for row in cursor.fetchall()}

        # Find the lowest available ID
        for potential_id in range(min_id, max_id + 1):
            if potential_id not in existing_ids:
                return potential_id

        # If no gaps found, this should not happen in practice given the ranges
        raise ValueError(
            f"No available IDs in range {min_id}-{max_id} for level {tourney_level}"
        )


def suggest_tournament_id(source_id: str, tourney_level: str) -> int:
    """
    Suggest tournament ID based on source_id and level

    Args:
        source_id: Original source tournament ID
        tourney_level: Tournament level classification

    Returns:
        int: Suggested 4-digit tournament ID

    Logic:
        1. Validate source_id format and range
        2. Check if source_id is available (not used by different tournament)
        3. If valid and available -> return source_id as int
        4. Otherwise -> return get_lowest_available_id(tourney_level)
    """
    # Check if source_id is valid for the level
    if validate_source_id_for_level(source_id, tourney_level):
        source_id_int = int(source_id)

        # Check if this ID is already taken
        db_conn = get_db_connection()
        with db_conn.get_cursor() as cursor:
            cursor.execute(
                """
                SELECT tournament_id FROM tournaments WHERE tournament_id = ?
            """,
                (source_id_int,),
            )

            if not cursor.fetchone():
                # ID is available
                return source_id_int

    # Source ID not valid or not available, get lowest available
    return get_lowest_available_id(tourney_level)


def create_tournament(
    primary_name: str, tourney_level: str, suggested_id: Optional[int] = None
) -> int:
    """
    Create a new tournament with level-based ID

    Args:
        primary_name: Primary tournament name
        tourney_level: Tournament level classification
        suggested_id: Optional suggested ID (will be validated)

    Returns:
        int: The assigned tournament ID

    Process:
        1. If suggested_id provided, validate it's in range and available
        2. Otherwise use get_lowest_available_id(tourney_level)
        3. Create tournament record
        4. Return assigned ID
    """
    if suggested_id is not None:
        # Validate suggested ID is in correct range
        min_id, max_id = get_level_range(tourney_level)
        if not (min_id <= suggested_id <= max_id):
            raise ValueError(
                f"Suggested ID {suggested_id} not in range {min_id}-{max_id} for level {tourney_level}"
            )

        # Check if ID is available
        db_conn = get_db_connection()
        with db_conn.get_cursor() as cursor:
            cursor.execute(
                """
                SELECT tournament_id FROM tournaments WHERE tournament_id = ?
            """,
                (suggested_id,),
            )

            if cursor.fetchone():
                raise ValueError(f"Tournament ID {suggested_id} is already taken")

        tournament_id = suggested_id
    else:
        tournament_id = get_lowest_available_id(tourney_level)

    db_conn = get_db_connection()

    with db_conn.get_cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO tournaments (tournament_id, primary_name, tourney_level, created_date, last_updated)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                tournament_id,
                primary_name,
                tourney_level,
                datetime.now(),
                datetime.now(),
            ),
        )

    return tournament_id


def query_by_source_id(source_name: str, source_id: str) -> Optional[Dict[str, Any]]:
    """
    Query tournament by source name and source ID.

    Args:
        source_name: Name of the data source
        source_id: ID from the source

    Returns:
        Optional[Dict]: Tournament information if found, None otherwise
    """
    db_conn = get_db_connection()

    with db_conn.get_cursor() as cursor:
        cursor.execute(
            """
            SELECT t.tournament_id, t.primary_name, t.tourney_level, ts.source_name_variant
            FROM tournaments t
            JOIN tournament_sources ts ON t.tournament_id = ts.tournament_id
            WHERE ts.source_name = ? AND ts.source_id = ?
            LIMIT 1
        """,
            (source_name, source_id),
        )

        row = cursor.fetchone()
        if row:
            return {
                "tournament_id": row["tournament_id"],
                "primary_name": row["primary_name"],
                "tourney_level": row["tourney_level"],
                "source_name_variant": row["source_name_variant"],
            }

    return None


def check_strict_name_match(
    source_name: str, tournament_name: str
) -> Optional[Dict[str, Any]]:
    """
    Check for exact name match within the same source.

    Args:
        source_name: Name of the data source
        tournament_name: Tournament name to match

    Returns:
        Optional[Dict]: Tournament information if exact match found, None otherwise
    """
    db_conn = get_db_connection()

    with db_conn.get_cursor() as cursor:
        cursor.execute(
            """
            SELECT t.tournament_id, t.primary_name, t.tourney_level
            FROM tournaments t
            JOIN tournament_sources ts ON t.tournament_id = ts.tournament_id
            WHERE ts.source_name = ? AND ts.source_name_variant = ?
            LIMIT 1
        """,
            (source_name, tournament_name),
        )

        row = cursor.fetchone()
        if row:
            return {
                "tournament_id": row["tournament_id"],
                "primary_name": row["primary_name"],
                "tourney_level": row["tourney_level"],
            }

    return None


def add_source_id_to_tournament(
    tournament_id: int,
    source_name: str,
    source_id: str,
    name_variant: str,
    is_primary_name: bool = False,
) -> bool:
    """
    Add a new source ID and name variant to an existing tournament.

    Args:
        tournament_id: ID of the tournament
        source_name: Name of the data source
        source_id: ID from the source
        name_variant: Name variant from the source
        is_primary_name: Whether this should be the primary name

    Returns:
        bool: True if successfully added, False if already exists
    """
    db_conn = get_db_connection()

    with db_conn.get_cursor() as cursor:
        try:
            cursor.execute(
                """
                INSERT INTO tournament_sources 
                (tournament_id, source_name, source_id, source_name_variant, is_primary_name)
                VALUES (?, ?, ?, ?, ?)
            """,
                (tournament_id, source_name, source_id, name_variant, is_primary_name),
            )

            # Update last_updated timestamp for the tournament
            cursor.execute(
                """
                UPDATE tournaments 
                SET last_updated = ? 
                WHERE tournament_id = ?
            """,
                (datetime.now(), tournament_id),
            )

            return True

        except Exception:  # Unique constraint violation
            return False


def get_all_source_names_for_tournament(
    tournament_id: int, source_name: str
) -> List[str]:
    """
    Get all name variants for a tournament from a specific source.

    Args:
        tournament_id: ID of the tournament
        source_name: Name of the data source

    Returns:
        List[str]: List of name variants
    """
    db_conn = get_db_connection()

    with db_conn.get_cursor() as cursor:
        cursor.execute(
            """
            SELECT source_name_variant
            FROM tournament_sources
            WHERE tournament_id = ? AND source_name = ?
        """,
            (tournament_id, source_name),
        )

        return [row["source_name_variant"] for row in cursor.fetchall()]


def get_all_tournaments_for_fuzzy_matching(
    source_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Get all tournaments for fuzzy matching (cross-source or within source).
    Returns both primary names and source name variants as separate candidates.

    Args:
        source_name: If provided, only get tournaments from this source

    Returns:
        List[Dict]: List of tournament information for matching, including both
                   primary names and source variants as separate entries
    """
    db_conn = get_db_connection()
    candidates = []

    with db_conn.get_cursor() as cursor:
        if source_name:
            cursor.execute(
                """
                SELECT DISTINCT t.tournament_id, t.primary_name, t.tourney_level, ts.source_name_variant, ts.source_name
                FROM tournaments t
                JOIN tournament_sources ts ON t.tournament_id = ts.tournament_id
                WHERE ts.source_name = ?
            """,
                (source_name,),
            )
        else:
            cursor.execute("""
                SELECT DISTINCT t.tournament_id, t.primary_name, t.tourney_level, ts.source_name_variant, ts.source_name
                FROM tournaments t
                JOIN tournament_sources ts ON t.tournament_id = ts.tournament_id
            """)

        rows = cursor.fetchall()

        # Track processed tournaments to avoid duplicate primary names
        processed_tournaments = set()

        for row in rows:
            tournament_id = row["tournament_id"]
            primary_name = row["primary_name"]
            tourney_level = row["tourney_level"]
            source_name_variant = row["source_name_variant"]
            source_name_entry = row["source_name"]

            # Add primary name as candidate (once per tournament)
            if tournament_id not in processed_tournaments:
                candidates.append(
                    {
                        "tournament_id": tournament_id,
                        "primary_name": primary_name,
                        "tourney_level": tourney_level,
                        "source_name_variant": primary_name,  # Use primary name as candidate
                        "source_name": "primary_name",  # Special marker
                        "is_primary_name": True,
                    }
                )
                processed_tournaments.add(tournament_id)

            # Add source variant as candidate
            candidates.append(
                {
                    "tournament_id": tournament_id,
                    "primary_name": primary_name,
                    "tourney_level": tourney_level,
                    "source_name_variant": source_name_variant,
                    "source_name": source_name_entry,
                    "is_primary_name": False,
                }
            )

        return candidates


def update_tournament_primary_name(tournament_id: int, new_primary_name: str) -> bool:
    """
    Update the primary name of a tournament.

    Args:
        tournament_id: ID of the tournament
        new_primary_name: New primary name

    Returns:
        bool: True if updated successfully
    """
    db_conn = get_db_connection()

    with db_conn.get_cursor() as cursor:
        cursor.execute(
            """
            UPDATE tournaments 
            SET primary_name = ?, last_updated = ?
            WHERE tournament_id = ?
        """,
            (new_primary_name, datetime.now(), tournament_id),
        )

        return cursor.rowcount > 0


def get_tournament_info(tournament_id: int) -> Optional[Dict[str, Any]]:
    """
    Get complete tournament information including all source mappings.

    Args:
        tournament_id: ID of the tournament

    Returns:
        Optional[Dict]: Complete tournament information
    """
    db_conn = get_db_connection()

    with db_conn.get_cursor() as cursor:
        # Get tournament basic info
        cursor.execute(
            """
            SELECT tournament_id, primary_name, tourney_level, created_date, last_updated
            FROM tournaments
            WHERE tournament_id = ?
        """,
            (tournament_id,),
        )

        tournament_row = cursor.fetchone()
        if not tournament_row:
            return None

        # Get all source mappings
        cursor.execute(
            """
            SELECT source_name, source_id, source_name_variant, is_primary_name, created_date
            FROM tournament_sources
            WHERE tournament_id = ?
            ORDER BY source_name, created_date
        """,
            (tournament_id,),
        )

        source_mappings = [
            {
                "source_name": row["source_name"],
                "source_id": row["source_id"],
                "source_name_variant": row["source_name_variant"],
                "is_primary_name": bool(row["is_primary_name"]),
                "created_date": row["created_date"],
            }
            for row in cursor.fetchall()
        ]

        return {
            "tournament_id": tournament_row["tournament_id"],
            "primary_name": tournament_row["primary_name"],
            "tourney_level": tournament_row["tourney_level"],
            "created_date": tournament_row["created_date"],
            "last_updated": tournament_row["last_updated"],
            "source_mappings": source_mappings,
        }
