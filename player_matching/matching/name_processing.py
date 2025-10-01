"""
Name processing and similarity calculation functions.
"""

import re
import pandas as pd
from typing import Optional
from difflib import SequenceMatcher


def merge_player_names(first_name: str, last_name: str) -> str:
    """
    Merge first and last names into single string.
    
    Args:
        first_name: Player's first name
        last_name: Player's last name
        
    Returns:
        Combined full name string
    """
    if pd.isna(first_name) or pd.isna(last_name):
        return ""
    
    first_clean = str(first_name).strip()
    last_clean = str(last_name).strip()
    
    if not first_clean or not last_clean:
        return ""
    
    return f"{first_clean} {last_clean}"


def preprocess_player_name(name: str) -> str:
    """
    Standardize player name for matching:
    1. Convert to lowercase
    2. Replace all spaces with hyphens
    3. Handle special characters
    4. Remove multiple consecutive hyphens
    
    Args:
        name: Original player name
        
    Returns:
        Preprocessed name for matching
    """
    if not name or pd.isna(name):
        return ""
    
    # Convert to lowercase and strip
    name = str(name).lower().strip()
    
    # Remove special characters except spaces and hyphens
    name = re.sub(r'[^\w\s\-]', '', name)
    
    # Replace multiple spaces with single space
    name = re.sub(r'\s+', ' ', name)
    
    # Replace spaces with hyphens
    name = name.replace(' ', '-')
    
    # Remove multiple consecutive hyphens
    name = re.sub(r'-+', '-', name)
    
    # Remove leading/trailing hyphens
    name = name.strip('-')
    
    return name


def calculate_name_similarity(name1: str, name2: str) -> float:
    """
    Calculate similarity between two player names using multiple approaches.
    
    This uses a combination of:
    - Exact string matching (highest weight)
    - SequenceMatcher for fuzzy matching
    - Word-level matching for partial matches
    
    Args:
        name1: First name to compare
        name2: Second name to compare
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not name1 or not name2:
        return 0.0
    
    name1 = str(name1).lower().strip()
    name2 = str(name2).lower().strip()
    
    # Exact match
    if name1 == name2:
        return 1.0
    
    # Fuzzy string matching (primary method)
    fuzzy_score = SequenceMatcher(None, name1, name2).ratio()
    
    # Word-level matching for partial matches
    words1 = set(name1.split('-'))
    words2 = set(name2.split('-'))
    
    if words1 and words2:
        common_words = words1.intersection(words2)
        total_words = words1.union(words2)
        word_score = len(common_words) / len(total_words) if total_words else 0.0
    else:
        word_score = 0.0
    
    # Combined score: weighted average
    # Fuzzy matching gets 70% weight, word matching gets 30%
    combined_score = (fuzzy_score * 0.7) + (word_score * 0.3)
    
    return round(combined_score, 3)


def extract_name_variations(name: str) -> list:
    """
    Generate common name variations for better matching.
    
    Args:
        name: Original name
        
    Returns:
        List of name variations
    """
    if not name:
        return []
    
    variations = [name]
    
    # Split into parts
    parts = name.split('-')
    if len(parts) > 1:
        # Add individual parts
        variations.extend(parts)
        
        # Add reversed order
        reversed_name = '-'.join(reversed(parts))
        if reversed_name != name:
            variations.append(reversed_name)
        
        # Add first-last combinations
        if len(parts) == 2:
            # First name + last initial
            variations.append(f"{parts[0]}-{parts[1][0]}")
            # First initial + last name
            variations.append(f"{parts[0][0]}-{parts[1]}")
    
    return list(set(variations))  # Remove duplicates


def normalize_source_code(source_input) -> int:
    """
    Convert source input to standardized source code.
    
    Args:
        source_input: Either string name or integer code
        
    Returns:
        Standardized source code integer (0, 1, 2)
        
    Raises:
        ValueError: If source_input is not recognized
    """
    if isinstance(source_input, int):
        if source_input in [0, 1, 2]:
            return source_input
        else:
            raise ValueError(f"Invalid source code: {source_input}. Must be 0, 1, or 2")
    
    if isinstance(source_input, str):
        source_mapping = {
            'main_dataset': 0,
            'infosys_api': 1,
            'tennis_api': 2
        }
        if source_input in source_mapping:
            return source_mapping[source_input]
        else:
            raise ValueError(f"Invalid source name: {source_input}. Must be one of {list(source_mapping.keys())}")
    
    raise ValueError(f"Invalid source input type: {type(source_input)}. Must be int or str")


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


def validate_name_input(name: str) -> tuple[bool, str]:
    """
    Validate that a name input is acceptable for processing.
    
    Args:
        name: Name to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not name:
        return False, "Name cannot be empty"
    
    if pd.isna(name):
        return False, "Name cannot be NaN"
    
    name_str = str(name).strip()
    
    if len(name_str) < 2:
        return False, "Name must be at least 2 characters long"
    
    if len(name_str) > 100:
        return False, "Name cannot be longer than 100 characters"
    
    # Check for reasonable characters
    if not re.match(r'^[a-zA-Z\s\-\'\.]+$', name_str):
        return False, "Name contains invalid characters"
    
    return True, ""


if __name__ == "__main__":
    # Test name processing functions
    print("Testing name processing functions...")
    
    try:
        # Test merge names
        full_name = merge_player_names("Roger", "Federer")
        print(f"Merged name: '{full_name}'")
        
        # Test preprocessing
        processed = preprocess_player_name("Roger Federer")
        print(f"Preprocessed: '{processed}'")
        
        # Test similarity
        similarity = calculate_name_similarity("roger-federer", "r-federer")
        print(f"Similarity: {similarity}")
        
        # Test variations
        variations = extract_name_variations("roger-federer")
        print(f"Variations: {variations}")
        
        # Test validation
        valid, msg = validate_name_input("Roger Federer")
        print(f"Validation: {valid}, {msg}")
        
        print("Name processing test: SUCCESS")
        
    except Exception as e:
        print(f"Name processing test: FAILED - {e}")