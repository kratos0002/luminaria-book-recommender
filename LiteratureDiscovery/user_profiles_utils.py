"""
User profile utilities for the Luminaria application.
This module provides functions for managing user profiles and interactions.
"""

import json
import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from LiteratureDiscovery.models import UserProfile, LiteratureItem

# Setup logging
logger = logging.getLogger(__name__)

# Directory to store user profiles
PROFILES_DIR = os.path.join(os.path.dirname(__file__), 'data', 'user_profiles')

# Ensure the directory exists
os.makedirs(PROFILES_DIR, exist_ok=True)

# In-memory cache of user profiles
_profile_cache: Dict[str, UserProfile] = {}

def get_profile_path(session_id: str) -> str:
    """Get the file path for a user profile"""
    return os.path.join(PROFILES_DIR, f"{session_id}.json")

def load_user_profile(session_id: str) -> Optional[UserProfile]:
    """
    Load a user profile from disk or cache.
    
    Args:
        session_id: The session ID of the user
        
    Returns:
        UserProfile if found, None otherwise
    """
    # Check cache first
    if session_id in _profile_cache:
        return _profile_cache[session_id]
    
    # Try to load from disk
    profile_path = get_profile_path(session_id)
    if os.path.exists(profile_path):
        try:
            with open(profile_path, 'r') as f:
                data = json.load(f)
                profile = UserProfile.from_dict(data)
                _profile_cache[session_id] = profile
                return profile
        except Exception as e:
            logger.error(f"Error loading profile for {session_id}: {e}")
    
    # Create a new profile if none exists
    profile = UserProfile(session_id)
    _profile_cache[session_id] = profile
    return profile

def save_user_profile(profile: UserProfile) -> bool:
    """
    Save a user profile to disk.
    
    Args:
        profile: The UserProfile to save
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Update the cache
        _profile_cache[profile.user_id] = profile
        
        # Save to disk
        with open(get_profile_path(profile.user_id), 'w') as f:
            json.dump(profile.to_dict(), f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving profile: {e}")
        return False

def update_user_interaction(session_id: str, item: LiteratureItem, interaction_type: str) -> bool:
    """
    Update a user's profile based on an interaction with a literature item.
    
    Args:
        session_id: The session ID of the user
        item: The literature item the user interacted with
        interaction_type: Type of interaction (view, save, rate, finish)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load the user profile
        profile = load_user_profile(session_id)
        
        # Update based on interaction type
        if interaction_type == 'view':
            profile.add_view(item)
        elif interaction_type == 'save':
            profile.add_to_reading_list(item)
        elif interaction_type == 'rate':
            profile.add_rating(item, 5)  # Default rating of 5
        elif interaction_type == 'finish':
            profile.add_to_reading_history(item)
        else:
            logger.warning(f"Unknown interaction type: {interaction_type}")
            return False
        
        # Save the updated profile
        return save_user_profile(profile)
    except Exception as e:
        logger.error(f"Error updating user interaction: {e}")
        return False

def get_user_preferences(session_id: str) -> Dict[str, Any]:
    """
    Get a user's preferences for use in recommendations.
    
    Args:
        session_id: The session ID of the user
        
    Returns:
        dict: User preferences including top genres and authors
    """
    profile = load_user_profile(session_id)
    if not profile:
        return {
            'top_genres': [],
            'top_authors': [],
            'reading_history': []
        }
    
    return {
        'top_genres': profile.get_top_genres(5),
        'top_authors': profile.get_top_authors(5),
        'reading_history': profile.reading_history
    }

def clear_profile_cache() -> None:
    """Clear the in-memory profile cache"""
    _profile_cache.clear()
