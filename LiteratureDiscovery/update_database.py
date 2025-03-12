"""
Script to update the SQLite database schema and add feedback functions
"""
import re
import os
import logging
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def backup_file(file_path):
    """Create a backup of the file before modifying it."""
    backup_path = f"{file_path}.bak"
    if not os.path.exists(backup_path):
        shutil.copy2(file_path, backup_path)
        logger.info(f"Created backup at {backup_path}")

def update_database_schema():
    """Update the init_db function to add user_feedback table."""
    file_path = "literature_logic.py"
    
    # Create a backup
    backup_file(file_path)
    
    # Read the current content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the init_db function
    init_db_pattern = r'def init_db\(\):.*?CREATE TABLE IF NOT EXISTS user_inputs \(.*?UNIQUE\(session_id, input_text\).*?\).*?conn\.commit\(\)'
    init_db_match = re.search(init_db_pattern, content, re.DOTALL)
    
    if not init_db_match:
        logger.error("Could not find init_db function")
        return False
    
    # Replace with updated init_db function
    updated_init_db = '''def init_db():
    """Initialize the SQLite database for user history tracking."""
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_history.db")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create user_inputs table if it doesn't exist
    cursor.execute(\'\'\'
    CREATE TABLE IF NOT EXISTS user_inputs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        input_text TEXT,
        timestamp DATETIME,
        UNIQUE(session_id, input_text)
    )
    \'\'\')
    
    # Create user_feedback table if it doesn't exist
    cursor.execute(\'\'\'
    CREATE TABLE IF NOT EXISTS user_feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        title TEXT,
        feedback INTEGER,
        timestamp DATETIME,
        UNIQUE(session_id, title)
    )
    \'\'\')
    
    conn.commit()'''
    
    content = content.replace(init_db_match.group(0), updated_init_db)
    
    # Write the updated content back to the file
    with open(file_path, 'w') as f:
        f.write(content)
    
    logger.info("Updated database schema with user_feedback table")
    return True

def add_feedback_functions():
    """Add feedback functions after get_user_history."""
    file_path = "literature_logic.py"
    
    # Read the current content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the get_user_history function
    user_history_pattern = r'def get_user_history\(.*?return \[\].*?\n'
    user_history_match = re.search(user_history_pattern, content, re.DOTALL)
    
    if not user_history_match:
        logger.error("Could not find get_user_history function")
        return False
    
    # Feedback functions to add
    feedback_functions = '''
def store_feedback(session_id: str, title: str, feedback: int):
    """
    Store user feedback (thumbs up/down) for a recommendation.
    
    Args:
        session_id: User's session ID
        title: Title of the literature item
        feedback: 1 for thumbs up, -1 for thumbs down
    
    Returns:
        Boolean indicating success
    """
    if not session_id or not title:
        logger.warning("Missing session_id or title, not storing feedback")
        return False
    
    if feedback not in [1, -1]:
        logger.warning(f"Invalid feedback value: {feedback}, must be 1 or -1")
        return False
    
    try:
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_history.db")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Insert or replace the feedback
        cursor.execute(\'\'\'
        INSERT OR REPLACE INTO user_feedback (session_id, title, feedback, timestamp)
        VALUES (?, ?, ?, ?)
        \'\'\', (session_id, title, feedback, datetime.now()))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Feedback stored for session {session_id}, title: {title}, feedback: {feedback}")
        return True
    except Exception as e:
        logger.error(f"Error storing feedback: {e}")
        traceback.print_exc()
        return False

def get_user_feedback(session_id: str) -> Dict[str, int]:
    """
    Retrieve user feedback for recommendations.
    
    Args:
        session_id: User's session ID
        
    Returns:
        Dictionary mapping title to feedback value (1 or -1)
    """
    if not session_id:
        logger.warning("Missing session_id, not retrieving feedback")
        return {}
    
    try:
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_history.db")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all feedback for this session
        cursor.execute(\'\'\'
        SELECT title, feedback FROM user_feedback
        WHERE session_id = ?
        \'\'\', (session_id,))
        
        feedback_dict = {row[0]: row[1] for row in cursor.fetchall()}
        
        conn.close()
        
        logger.info(f"Retrieved {len(feedback_dict)} feedback items for session {session_id}")
        return feedback_dict
    except Exception as e:
        logger.error(f"Error retrieving user feedback: {e}")
        traceback.print_exc()
        return {}
'''
    
    # Insert feedback functions after get_user_history
    content = content.replace(user_history_match.group(0), user_history_match.group(0) + feedback_functions)
    
    # Write the updated content back to the file
    with open(file_path, 'w') as f:
        f.write(content)
    
    logger.info("Added feedback functions")
    return True

if __name__ == "__main__":
    update_database_schema()
    add_feedback_functions()
    print("Database schema updated and feedback functions added successfully!")
