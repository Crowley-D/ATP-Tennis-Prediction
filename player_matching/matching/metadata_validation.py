"""
Metadata validation functions for player data.
"""

import re
from datetime import datetime, date
from typing import Optional, Dict, Any, Union
import warnings


def validate_dob(dob_input: Union[str, date, None]) -> Optional[str]:
    """
    Validate and normalize date of birth to YYYY-MM-DD format.
    
    Args:
        dob_input: Date input in various formats
        
    Returns:
        Normalized date string in YYYY-MM-DD format or None if invalid
    """
    if dob_input is None or dob_input == '':
        return None
    
    # Handle date objects
    if isinstance(dob_input, date):
        return dob_input.strftime('%Y-%m-%d')
    
    # Convert to string and clean
    dob_str = str(dob_input).strip()
    
    # Skip empty, unknown, or clearly invalid values
    invalid_values = ['', '0', '00000000', 'unknown', 'n/a', 'na', 'null', 'none']
    if dob_str.lower() in invalid_values:
        return None
    
    # Try different date formats
    date_formats = [
        '%Y-%m-%d',     # 1981-08-08
        '%Y/%m/%d',     # 1981/08/08
        '%Y%m%d',       # 19810808
        '%m/%d/%Y',     # 08/08/1981
        '%d/%m/%Y',     # 08/08/1981 (ambiguous, but try)
        '%m-%d-%Y',     # 08-08-1981
        '%d-%m-%Y',     # 08-08-1981 (ambiguous, but try)
    ]
    
    parsed_date = None
    for fmt in date_formats:
        try:
            parsed_date = datetime.strptime(dob_str, fmt).date()
            break
        except ValueError:
            continue
    
    if parsed_date is None:
        warnings.warn(f"Could not parse date of birth: {dob_str}")
        return None
    
    # Validate reasonable date range for tennis players
    min_date = date(1940, 1, 1)  # Oldest reasonable birth year
    max_date = date.today().replace(year=date.today().year - 16)  # At least 16 years old
    
    if parsed_date < min_date or parsed_date > max_date:
        warnings.warn(f"Date of birth out of reasonable range: {parsed_date}")
        return None
    
    return parsed_date.strftime('%Y-%m-%d')


def validate_hand(hand_input: Union[str, None]) -> Optional[str]:
    """
    Validate and normalize playing hand value.
    
    Args:
        hand_input: Hand input in various formats
        
    Returns:
        Normalized hand value ('L', 'R', 'U') or None if invalid
    """
    if hand_input is None or hand_input == '':
        return None
    
    # Convert to string and clean
    hand_str = str(hand_input).strip().upper()
    
    # Skip empty or clearly invalid values  
    invalid_values = ['', '0', 'N/A', 'NA', 'NULL', 'NONE']
    if hand_str in invalid_values:
        return None
    
    # Direct mappings
    hand_mappings = {
        'L': 'L',
        'R': 'R', 
        'U': 'U',
        'LEFT': 'L',
        'RIGHT': 'R',
        'UNKNOWN': 'U',
        'AMBIDEXTROUS': 'U',
        'BOTH': 'U',
        'LEFTHANDED': 'L',
        'RIGHTHANDED': 'R',
        'LEFT-HANDED': 'L',
        'RIGHT-HANDED': 'R'
    }
    
    if hand_str in hand_mappings:
        return hand_mappings[hand_str]
    
    # Try first character if it's a valid hand indicator
    if len(hand_str) >= 1 and hand_str[0] in ['L', 'R', 'U']:
        return hand_str[0]
    
    warnings.warn(f"Could not parse hand value: {hand_input}")
    return None


def validate_height(height_input: Union[str, int, float, None]) -> Optional[int]:
    """
    Validate and normalize height value in centimeters.
    
    Args:
        height_input: Height input in various formats
        
    Returns:
        Height as integer in cm or None if invalid
    """
    if height_input is None or height_input == '':
        return None
    
    # Convert to string first for processing
    height_str = str(height_input).strip()
    
    # Skip empty or clearly invalid values
    invalid_values = ['', '0', '0.0', 'unknown', 'n/a', 'na', 'null', 'none']
    if height_str.lower() in invalid_values:
        return None
    
    # Try to extract numeric value
    height_value = None
    
    # Handle direct numeric input
    if isinstance(height_input, (int, float)):
        height_value = float(height_input)
    else:
        # Extract numbers from string (handles "185 cm", "185cm", "6'1\"", etc.)
        numbers = re.findall(r'\d+\.?\d*', height_str)
        if numbers:
            height_value = float(numbers[0])
        else:
            warnings.warn(f"Could not extract numeric height from: {height_input}")
            return None
    
    # Convert to cm if it looks like it might be in different units
    if height_value < 10:  # Likely meters (e.g., 1.85m)
        height_value *= 100
    elif 50 <= height_value <= 100:  # Likely feet/inches or invalid
        # Could be feet (6 feet = ~183cm) but more likely invalid
        warnings.warn(f"Ambiguous height value: {height_input}")
        return None
    
    # Convert to integer
    height_cm = int(round(height_value))
    
    # Validate reasonable range for professional tennis players
    min_height = 150  # Shortest reasonable height
    max_height = 220  # Tallest reasonable height
    
    if height_cm < min_height or height_cm > max_height:
        warnings.warn(f"Height out of reasonable range: {height_cm}cm")
        return None
    
    return height_cm


def validate_player_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate all player metadata fields.
    
    Args:
        metadata: Dict containing player metadata fields
        
    Returns:
        Dict with validated and normalized metadata fields
    """
    validated = {}
    
    # Validate each field if present
    if 'dob' in metadata:
        validated['dob'] = validate_dob(metadata['dob'])
    
    if 'hand' in metadata:
        validated['hand'] = validate_hand(metadata['hand'])
        
    if 'height' in metadata:
        validated['height'] = validate_height(metadata['height'])
    
    # Copy through any other fields unchanged
    for key, value in metadata.items():
        if key not in ['dob', 'hand', 'height']:
            validated[key] = value
    
    return validated


def get_metadata_validation_summary(
    original_metadata: Dict[str, Any], 
    validated_metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Get summary of validation changes for reporting.
    
    Args:
        original_metadata: Original metadata dict
        validated_metadata: Validated metadata dict
        
    Returns:
        Summary of validation results
    """
    summary = {
        'total_fields': 0,
        'valid_fields': 0,
        'invalid_fields': 0,
        'changes': {}
    }
    
    metadata_fields = ['dob', 'hand', 'height']
    
    for field in metadata_fields:
        if field in original_metadata:
            summary['total_fields'] += 1
            original_val = original_metadata[field]
            validated_val = validated_metadata.get(field)
            
            if validated_val is not None:
                summary['valid_fields'] += 1
                
                # Check if value was changed during validation
                if str(original_val) != str(validated_val):
                    summary['changes'][field] = {
                        'original': original_val,
                        'validated': validated_val
                    }
            else:
                summary['invalid_fields'] += 1
                summary['changes'][field] = {
                    'original': original_val,
                    'validated': None,
                    'status': 'invalid'
                }
    
    return summary


if __name__ == "__main__":
    # Test metadata validation functions
    print("Testing metadata validation functions...")
    
    # Test DOB validation
    print("\n=== DOB Validation Tests ===")
    dob_tests = [
        "1981-08-08",      # Standard format
        "1981/08/08",      # Slash format
        "19810808",        # Compact format
        "08/08/1981",      # US format
        "1990-02-29",      # Invalid leap year
        "2010-01-01",     # Too recent
        "1930-01-01",     # Too old
        "",               # Empty
        "unknown",        # Invalid string
        None              # None
    ]
    
    for test in dob_tests:
        result = validate_dob(test)
        print(f"  {test} -> {result}")
    
    # Test hand validation
    print("\n=== Hand Validation Tests ===")
    hand_tests = [
        "L", "R", "U",           # Standard values
        "Left", "Right",         # Full words
        "left-handed",           # Descriptive
        "unknown",               # Unknown
        "ambidextrous",          # Special case
        "",                      # Empty
        "invalid",               # Invalid
        None                     # None
    ]
    
    for test in hand_tests:
        result = validate_hand(test)
        print(f"  {test} -> {result}")
    
    # Test height validation
    print("\n=== Height Validation Tests ===")
    height_tests = [
        185,                # Integer cm
        1.85,              # Meters as float
        "185",             # String cm
        "185 cm",          # String with units
        "6'1\"",           # Feet/inches (should warn)
        50,                # Ambiguous value
        300,               # Too tall
        100,               # Too short
        "",                # Empty
        "unknown",         # Invalid
        None               # None
    ]
    
    for test in height_tests:
        result = validate_height(test)
        print(f"  {test} -> {result}")
    
    # Test combined validation
    print("\n=== Combined Metadata Validation Test ===")
    test_metadata = {
        'dob': '1981-08-08',
        'hand': 'Right',
        'height': '185 cm',
        'extra_field': 'should be preserved'
    }
    
    validated = validate_player_metadata(test_metadata)
    print(f"Original: {test_metadata}")
    print(f"Validated: {validated}")
    
    # Test validation summary
    summary = get_metadata_validation_summary(test_metadata, validated)
    print(f"Validation summary: {summary}")
    
    print("\nMetadata validation tests completed!")