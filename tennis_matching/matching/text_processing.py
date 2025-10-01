"""
Text processing utilities for tennis tournament name matching.
"""

import re
from typing import List, Set
from difflib import SequenceMatcher


# Common tournament words to remove during preprocessing
COMMON_WORDS = {
    'open', 'international', 'masters', 'atp', 'wta', 'championship',
    'championships', 'tournament', 'cup', 'series', 'classic', 'tour'
}


def preprocess_tournament_name(name: str) -> List[str]:
    """
    Preprocess tournament name for matching by removing common words,
    converting to lowercase, and removing special characters.
    
    Args:
        name: Tournament name to preprocess
        
    Returns:
        List[str]: List of cleaned words
    """
    if not name:
        return []
    
    # Convert to lowercase and remove special characters
    cleaned = re.sub(r'[^\w\s]', ' ', name.lower())
    
    # Split into words and remove common words
    words = cleaned.split()
    filtered_words = [word for word in words if word not in COMMON_WORDS and len(word) > 1]
    
    return filtered_words


def calculate_fuzzy_similarity(name1: str, name2: str) -> float:
    """
    Calculate fuzzy similarity between two tournament names using SequenceMatcher.
    
    Args:
        name1: First tournament name
        name2: Second tournament name
        
    Returns:
        float: Similarity score between 0.0 and 1.0
    """
    if not name1 or not name2:
        return 0.0
    
    # Use SequenceMatcher for fuzzy string matching
    matcher = SequenceMatcher(None, name1.lower(), name2.lower())
    return matcher.ratio()


def word_by_word_match(name1: str, name2: str) -> float:
    """
    Calculate word-by-word matching score between preprocessed tournament names.
    
    Args:
        name1: First tournament name
        name2: Second tournament name
        
    Returns:
        float: Word overlap score between 0.0 and 1.0
    """
    words1 = set(preprocess_tournament_name(name1))
    words2 = set(preprocess_tournament_name(name2))
    
    if not words1 or not words2:
        return 0.0
    
    # Calculate Jaccard similarity (intersection over union)
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    if union == 0:
        return 0.0
    
    return intersection / union


def combined_similarity_score(name1: str, name2: str) -> float:
    """
    Calculate combined similarity score using both fuzzy matching and word matching.
    
    Args:
        name1: First tournament name
        name2: Second tournament name
        
    Returns:
        float: Combined similarity score between 0.0 and 1.0
    """
    fuzzy_score = calculate_fuzzy_similarity(name1, name2)
    word_score = word_by_word_match(name1, name2)
    
    # Weight fuzzy matching slightly higher (60% fuzzy, 40% word-based)
    combined_score = (fuzzy_score * 0.6) + (word_score * 0.4)
    
    return combined_score


def get_similarity_category(similarity_score: float) -> str:
    """
    Categorize similarity score based on thresholds.
    
    Args:
        similarity_score: Similarity score between 0.0 and 1.0
        
    Returns:
        str: Category ('auto_accept', 'manual_review', 'reject')
    """
    if similarity_score >= 0.8:
        return 'auto_accept'
    elif similarity_score >= 0.6:
        return 'manual_review'
    else:
        return 'reject'


def find_best_matches(
    target_name: str,
    candidate_names: List[str],
    min_threshold: float = 0.6
) -> List[tuple]:
    """
    Find the best matches for a target name from a list of candidates.
    
    Args:
        target_name: Name to find matches for
        candidate_names: List of candidate names to match against
        min_threshold: Minimum similarity threshold
        
    Returns:
        List[tuple]: List of (candidate_name, similarity_score, category) sorted by score
    """
    matches = []
    
    for candidate in candidate_names:
        if candidate == target_name:
            # Exact match gets perfect score
            matches.append((candidate, 1.0, 'auto_accept'))
        else:
            score = combined_similarity_score(target_name, candidate)
            if score >= min_threshold:
                category = get_similarity_category(score)
                matches.append((candidate, score, category))
    
    # Sort by similarity score (descending)
    matches.sort(key=lambda x: x[1], reverse=True)
    
    return matches


def normalize_tournament_name(name: str) -> str:
    """
    Normalize tournament name for consistent storage.
    
    Args:
        name: Tournament name to normalize
        
    Returns:
        str: Normalized tournament name
    """
    if not name:
        return ""
    
    # Remove extra whitespace and capitalize properly
    normalized = ' '.join(name.split())
    
    # Title case with some exceptions for common abbreviations
    words = normalized.split()
    normalized_words = []
    
    for word in words:
        if word.upper() in ['ATP', 'WTA', 'ITF', 'US', 'UK']:
            normalized_words.append(word.upper())
        else:
            normalized_words.append(word.capitalize())
    
    return ' '.join(normalized_words)