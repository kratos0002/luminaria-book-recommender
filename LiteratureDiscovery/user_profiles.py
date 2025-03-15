import json
import os
from typing import Dict, Optional
from datetime import datetime
from .models import UserProfile, LiteratureItem

# Directory to store user profiles
PROFILES_DIR = os.path.join(os.path.dirname(__file__), 'data', 'user_profiles')

# Ensure the directory exists
os.makedirs(PROFILES_DIR, exist_ok=True)

# In-memory cache of user profiles
_profile_cache: Dict[str, UserProfile] = {}

def get_profile_path(user_id: str) -> str:
    """Get the file path for a user profile"""
    return os.path.join(PROFILES_DIR, f"{user_id}.json")

def save_profile(profile: UserProfile) -> bool:
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
        print(f"Error saving profile: {e}")
        return False

def load_profile(user_id: str) -> Optional[UserProfile]:
    """
    Load a user profile from disk or cache.
    
    Args:
        user_id: The ID of the user to load
        
    Returns:
        UserProfile if found, None otherwise
    """
    # Check cache first
    if user_id in _profile_cache:
        return _profile_cache[user_id]
    
    # Try to load from disk
    profile_path = get_profile_path(user_id)
    if os.path.exists(profile_path):
        try:
            with open(profile_path, 'r') as f:
                data = json.load(f)
                profile = UserProfile.from_dict(data)
                _profile_cache[user_id] = profile
                return profile
        except Exception as e:
            print(f"Error loading profile: {e}")
    
    # Create a new profile if not found
    profile = UserProfile(user_id)
    _profile_cache[user_id] = profile
    save_profile(profile)
    return profile

def update_user_interaction(user_id: str, item: LiteratureItem, interaction_type: str) -> bool:
    """
    Update a user's profile based on an interaction with a literature item.
    
    Args:
        user_id: The ID of the user
        item: The literature item the user interacted with
        interaction_type: Type of interaction (view, save, rate, finish)
        
    Returns:
        bool: True if successful, False otherwise
    """
    profile = load_profile(user_id)
    profile.update_preferences(item, interaction_type)
    return save_profile(profile)

def get_user_preferences(user_id: str) -> dict:
    """
    Get a user's preferences for use in recommendations.
    
    Args:
        user_id: The ID of the user
        
    Returns:
        dict: User preferences including top genres and authors
    """
    profile = load_profile(user_id)
    return {
        'top_genres': profile.get_top_genres(),
        'top_authors': profile.get_top_authors(),
        'reading_history': profile.reading_history
    }

def clear_profile_cache():
    """Clear the in-memory profile cache"""
    _profile_cache.clear()
