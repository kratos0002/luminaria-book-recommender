"""
Script to update the LiteratureItem class in literature_logic.py
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
    shutil.copy2(file_path, backup_path)
    logger.info(f"Created backup at {backup_path}")

def update_literature_item():
    """Update the LiteratureItem class to include summary and match_score fields."""
    file_path = "literature_logic.py"
    
    # Create a backup
    backup_file(file_path)
    
    # Read the current content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the LiteratureItem class definition
    class_pattern = r'class LiteratureItem:.*?def __init__\(self, title: str, author: str, publication_date: str = "", .*?genre: str = "", description: str = "", item_type: str = "book"\):.*?self\.matched_terms = set\(\).*?\n'
    class_match = re.search(class_pattern, content, re.DOTALL)
    
    if not class_match:
        logger.error("Could not find LiteratureItem class definition")
        return False
    
    # Replace with updated class definition
    updated_class = '''class LiteratureItem:
    """Class representing a literature item (book, poem, essay, etc.)
    with its metadata."""
    
    def __init__(self, title: str, author: str, publication_date: str = "", 
                 genre: str = "", description: str = "", item_type: str = "book", 
                 summary: str = ""):
        self.title = title
        self.author = author
        self.publication_date = publication_date
        self.genre = genre
        self.description = description
        self.item_type = item_type  # book, poem, essay, etc.
        self.score = 0.0  # Recommendation score
        self.matched_terms = set()  # Terms that matched this item
        self.summary = summary  # 2-3 sentence summary of the work
        self.match_score = 0  # Match score (0-100) indicating how well it matches user input
    
'''
    
    content = content.replace(class_match.group(0), updated_class)
    
    # Find the to_dict method
    to_dict_pattern = r'def to_dict\(self\):.*?return \{.*?"matched_terms": list\(self\.matched_terms\).*?\}'
    to_dict_match = re.search(to_dict_pattern, content, re.DOTALL)
    
    if not to_dict_match:
        logger.error("Could not find to_dict method")
        return False
    
    # Replace with updated to_dict method
    updated_to_dict = '''def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "title": self.title,
            "author": self.author,
            "publication_date": self.publication_date,
            "genre": self.genre,
            "description": self.description,
            "item_type": self.item_type,
            "score": self.score,
            "matched_terms": list(self.matched_terms),
            "summary": self.summary,
            "match_score": self.match_score
        }'''
    
    content = content.replace(to_dict_match.group(0), updated_to_dict)
    
    # Write the updated content back to the file
    with open(file_path, 'w') as f:
        f.write(content)
    
    logger.info("Updated LiteratureItem class with summary and match_score fields")
    return True

if __name__ == "__main__":
    update_literature_item()
    print("LiteratureItem class updated successfully!")
