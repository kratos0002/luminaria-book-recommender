import sqlite3
import os
import logging
from datetime import datetime
from typing import List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database file path
DB_PATH = os.path.join(os.path.dirname(__file__), 'user_history.db')

def init_db():
    """Initialize the SQLite database with required tables."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Create user_inputs table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_inputs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            input TEXT NOT NULL,
            timestamp DATETIME NOT NULL
        )
        ''')
        
        conn.commit()
        logger.info(f"Database initialized at {DB_PATH}")
        return True
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        return False
    finally:
        if conn:
            conn.close()

def store_user_input(session_id: str, literature_input: str) -> bool:
    """
    Store a user's literature input in the database.
    
    Args:
        session_id: Unique identifier for the user session
        literature_input: The literature input provided by the user
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not session_id or not literature_input:
        logger.warning("Cannot store empty session_id or literature_input")
        return False
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Insert the user input with current timestamp
        current_time = datetime.now().isoformat()
        cursor.execute(
            "INSERT INTO user_inputs (session_id, input, timestamp) VALUES (?, ?, ?)",
            (session_id, literature_input.strip(), current_time)
        )
        
        conn.commit()
        logger.info(f"Stored input for session {session_id}: {literature_input[:30]}...")
        return True
    except Exception as e:
        logger.error(f"Failed to store user input: {str(e)}")
        return False
    finally:
        if conn:
            conn.close()

def get_user_history(session_id: str, limit: int = 5) -> List[str]:
    """
    Retrieve the user's recent literature inputs.
    
    Args:
        session_id: Unique identifier for the user session
        limit: Maximum number of history items to retrieve
        
    Returns:
        List of the user's recent inputs, ordered by most recent first
    """
    if not session_id:
        logger.warning("Cannot retrieve history for empty session_id")
        return []
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get the most recent inputs for this session, excluding the current one
        cursor.execute(
            "SELECT input FROM user_inputs WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
            (session_id, limit)
        )
        
        # Extract the inputs from the result
        history = [row[0] for row in cursor.fetchall()]
        logger.info(f"Retrieved {len(history)} history items for session {session_id}")
        return history
    except Exception as e:
        logger.error(f"Failed to retrieve user history: {str(e)}")
        return []
    finally:
        if conn:
            conn.close()

# Initialize the database when this module is imported
init_db()
