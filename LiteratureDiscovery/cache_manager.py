"""
Cache manager for the Luminaria Book Recommender application.
Provides multiple levels of caching to improve performance.
"""

import logging
import pickle
import sqlite3
import time
from functools import wraps
from typing import Any, Dict, Optional, Tuple, Union

from cachetools import TTLCache

logger = logging.getLogger(__name__)

# In-memory cache for frequently accessed data
# Different TTLs for different types of data
RECOMMENDATION_CACHE = TTLCache(maxsize=500, ttl=604800)  # 1 week
BOOK_INFO_CACHE = TTLCache(maxsize=1000, ttl=2592000)  # 1 month
AUTHOR_INFO_CACHE = TTLCache(maxsize=500, ttl=2592000)  # 1 month
GENRE_INFO_CACHE = TTLCache(maxsize=100, ttl=2592000)  # 1 month
NEWS_CACHE = TTLCache(maxsize=300, ttl=21600)  # 6 hours

# Cache keys
def recommendation_key(input_term: str) -> str:
    """Generate a cache key for recommendations."""
    return f"rec:{input_term.lower().strip()}"

def book_info_key(book_title: str) -> str:
    """Generate a cache key for book info."""
    return f"book:{book_title.lower().strip()}"

def author_info_key(author_name: str) -> str:
    """Generate a cache key for author info."""
    return f"author:{author_name.lower().strip()}"

def genre_info_key(genre: str) -> str:
    """Generate a cache key for genre info."""
    return f"genre:{genre.lower().strip()}"

def news_key(search_term: str) -> str:
    """Generate a cache key for news and social updates."""
    return f"news:{search_term.lower().strip()}"

# Cache decorator with customizable TTL
def cache_result(cache: TTLCache, key_func=None):
    """
    Decorator to cache function results.
    
    Args:
        cache: The TTLCache instance to use
        key_func: Optional function to generate a key from the arguments
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Skip cache if force_refresh is True
            if kwargs.get('force_refresh', False):
                kwargs.pop('force_refresh', None)
                return func(*args, **kwargs)
                
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
                
            # Check if result is in cache
            if key in cache:
                logger.debug(f"Cache hit for {key}")
                return cache[key]
                
            # Call function and cache result
            result = func(*args, **kwargs)
            if result is not None:
                logger.debug(f"Caching result for {key}")
                cache[key] = result
            return result
        return wrapper
    return decorator

# Database cache for persistence across restarts
class DBCache:
    """Database-backed cache for persistence across restarts."""
    
    def __init__(self, db_path: str):
        """
        Initialize the database cache.
        
        Args:
            db_path: Path to the SQLite database
        """
        self.db_path = db_path
        self._create_tables()
    
    def _create_tables(self):
        """Create the cache tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS cache (
            key TEXT PRIMARY KEY,
            value BLOB,
            expires_at INTEGER
        )
        ''')
        # Create index on expires_at for faster cleanup
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_cache_expires_at ON cache(expires_at)')
        conn.commit()
        conn.close()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            default: Default value if key not found
            
        Returns:
            The cached value or default
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = int(time.time())
        
        # Get value and check expiration
        cursor.execute(
            'SELECT value FROM cache WHERE key = ? AND (expires_at IS NULL OR expires_at > ?)',
            (key, now)
        )
        result = cursor.fetchone()
        conn.close()
        
        if result:
            try:
                return pickle.loads(result[0])
            except Exception as e:
                logger.error(f"Error unpickling cache value: {e}")
                return default
        return default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Calculate expiration time
        expires_at = int(time.time()) + ttl if ttl else None
        
        try:
            # Pickle the value
            pickled_value = pickle.dumps(value)
            
            # Insert or update
            cursor.execute(
                'INSERT OR REPLACE INTO cache (key, value, expires_at) VALUES (?, ?, ?)',
                (key, pickled_value, expires_at)
            )
            conn.commit()
        except Exception as e:
            logger.error(f"Error setting cache value: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def delete(self, key: str) -> None:
        """
        Delete a key from the cache.
        
        Args:
            key: Cache key
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM cache WHERE key = ?', (key,))
        conn.commit()
        conn.close()
    
    def clear_expired(self) -> int:
        """
        Clear expired cache entries.
        
        Returns:
            Number of entries removed
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = int(time.time())
        cursor.execute('DELETE FROM cache WHERE expires_at IS NOT NULL AND expires_at <= ?', (now,))
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        return deleted_count

# Function to warm the cache with popular searches
def warm_cache(popular_searches, get_recommendations_func, get_search_info_func):
    """
    Warm the cache with popular searches.
    
    Args:
        popular_searches: List of popular search terms
        get_recommendations_func: Function to get recommendations
        get_search_info_func: Function to get search info
    """
    logger.info(f"Warming cache with {len(popular_searches)} popular searches")
    for search in popular_searches:
        try:
            # Pre-fetch and cache results
            logger.debug(f"Warming cache for search: {search}")
            get_recommendations_func(search)
            get_search_info_func(search)
        except Exception as e:
            logger.error(f"Error warming cache for {search}: {str(e)}")
