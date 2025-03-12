"""
Script to fix the recommend_literature function to accept session_id parameter
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
    backup_path = f"{file_path}.bak4"
    if not os.path.exists(backup_path):
        shutil.copy2(file_path, backup_path)
        logger.info(f"Created backup at {backup_path}")

def fix_recommend_literature():
    """Fix the recommend_literature function to accept session_id parameter."""
    file_path = "literature_logic.py"
    
    # Create a backup
    backup_file(file_path)
    
    # Read the current content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the recommend_literature function definition
    old_def = "def recommend_literature(trending_items: List[LiteratureItem], user_terms: List[str], literature_input: str = None) -> List[Tuple[LiteratureItem, float, List[str]]]:"
    
    if old_def not in content:
        logger.error("Could not find recommend_literature function definition")
        return False
    
    # Update the function definition to include session_id parameter
    new_def = "def recommend_literature(trending_items: List[LiteratureItem], user_terms: List[str], literature_input: str = None, session_id: str = None) -> List[Tuple[LiteratureItem, float, List[str]]]:"
    
    content = content.replace(old_def, new_def)
    
    # Find the function docstring
    old_docstring = '''    """
    Score and recommend literature items based on user terms.
    
    Args:
        trending_items: List of LiteratureItem objects to score
        user_terms: List of user preference terms
        literature_input: Original user input to avoid self-recommendation
        
    Returns:
        List of tuples (LiteratureItem, score, matched_terms)
    """'''
    
    if old_docstring not in content:
        logger.error("Could not find recommend_literature function docstring")
        return False
    
    # Update the function docstring to include session_id parameter
    new_docstring = '''    """
    Score and recommend literature items based on user terms.
    
    Args:
        trending_items: List of LiteratureItem objects to score
        user_terms: List of user preference terms
        literature_input: Original user input to avoid self-recommendation
        session_id: Optional session ID for retrieving user feedback
        
    Returns:
        List of tuples (LiteratureItem, score, matched_terms)
    """'''
    
    content = content.replace(old_docstring, new_docstring)
    
    # Write the updated content back to the file
    with open(file_path, 'w') as f:
        f.write(content)
    
    logger.info("Fixed recommend_literature function to accept session_id parameter")
    return True

if __name__ == "__main__":
    fix_recommend_literature()
    print("recommend_literature function fixed successfully!")
