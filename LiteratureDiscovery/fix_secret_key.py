"""
Script to add a secret key to the Flask application
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
    backup_path = f"{file_path}.bak2"
    if not os.path.exists(backup_path):
        shutil.copy2(file_path, backup_path)
        logger.info(f"Created backup at {backup_path}")

def add_secret_key():
    """Add a secret key to the Flask application."""
    file_path = "app.py"
    
    # Create a backup
    backup_file(file_path)
    
    # Read the current content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the app configuration section
    app_config_pattern = r'# Configure the application\napp = Flask\(__name__\)\napp\.logger\.setLevel\(logging\.INFO\)'
    
    if app_config_pattern not in content:
        logger.error("Could not find app configuration section")
        return False
    
    # Add secret key
    updated_config = '''# Configure the application
app = Flask(__name__)
app.logger.setLevel(logging.INFO)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24).hex())  # Add secret key for session management'''
    
    content = content.replace(app_config_pattern, updated_config)
    
    # Write the updated content back to the file
    with open(file_path, 'w') as f:
        f.write(content)
    
    logger.info("Added secret key to Flask application")
    return True

if __name__ == "__main__":
    add_secret_key()
    print("Secret key added to Flask application successfully!")
