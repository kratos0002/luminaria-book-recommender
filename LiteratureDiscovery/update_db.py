"""
Script to update the init_db function in literature_logic.py
"""

import re
import os

def update_init_db():
    """Update the init_db function"""
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "literature_logic.py")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the function definition
    pattern = r'def init_db\(\).*?conn\.close\(\)'
    
    # New function code
    new_code = '''def init_db():
    """Initialize the SQLite database for user history tracking."""
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_history.db")
    logger.info(f"Initializing database at {db_path}")
    
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_inputs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            input TEXT NOT NULL UNIQUE,
            timestamp DATETIME NOT NULL
        )
        """)
        conn.commit()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
    finally:
        conn.close()'''
    
    # Replace the function
    updated_content = re.sub(pattern, new_code, content, flags=re.DOTALL)
    
    # Write back to the file
    with open(file_path, 'w') as f:
        f.write(updated_content)
    
    print("Updated init_db function")

if __name__ == "__main__":
    update_init_db()
