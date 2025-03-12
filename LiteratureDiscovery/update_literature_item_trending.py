"""
Script to update the LiteratureItem class and related functions to include is_trending flag
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
    backup_path = f"{file_path}.bak5"
    if not os.path.exists(backup_path):
        shutil.copy2(file_path, backup_path)
        logger.info(f"Created backup at {backup_path}")

def update_literature_item_class():
    """Update the LiteratureItem class to include is_trending field."""
    file_path = "literature_logic.py"
    
    # Create a backup
    backup_file(file_path)
    
    # Read the current content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Update the LiteratureItem class init method
    init_pattern = r'def __init__\(self, title: str, author: str, publication_date: str = "", \n.*?genre: str = "", description: str = "", item_type: str = "book", \n.*?summary: str = ""\):'
    init_match = re.search(init_pattern, content, re.DOTALL)
    
    if not init_match:
        logger.error("Could not find LiteratureItem __init__ method")
        return False
    
    updated_init = '''def __init__(self, title: str, author: str, publication_date: str = "", 
                 genre: str = "", description: str = "", item_type: str = "book", 
                 summary: str = "", is_trending: bool = False):'''
    
    content = content.replace(init_match.group(0), updated_init)
    
    # Update the instance variables in __init__
    vars_pattern = r'self\.match_score = 0  # Match score \(0-100\) indicating how well it matches user input'
    
    if vars_pattern not in content:
        logger.error("Could not find match_score variable in LiteratureItem")
        return False
    
    updated_vars = '''self.match_score = 0  # Match score (0-100) indicating how well it matches user input
        self.is_trending = is_trending  # Flag indicating if this is a trending item'''
    
    content = content.replace(vars_pattern, updated_vars)
    
    # Update the to_dict method
    to_dict_pattern = r'"match_score": self\.match_score'
    
    if to_dict_pattern not in content:
        logger.error("Could not find match_score in to_dict method")
        return False
    
    updated_to_dict = '''"match_score": self.match_score,
            "is_trending": self.is_trending'''
    
    content = content.replace(to_dict_pattern, updated_to_dict)
    
    # Write the updated content back to the file
    with open(file_path, 'w') as f:
        f.write(content)
    
    logger.info("Updated LiteratureItem class with is_trending field")
    return True

def update_get_trending_literature():
    """Update the get_trending_literature function to set is_trending=False."""
    file_path = "literature_logic.py"
    
    # Read the current content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the parse_literature_items call in get_trending_literature
    parse_call_pattern = r'items = parse_literature_items\(response_text\)'
    
    if parse_call_pattern not in content:
        logger.error("Could not find parse_literature_items call in get_trending_literature")
        return False
    
    updated_parse_call = 'items = parse_literature_items(response_text, is_trending=False)'
    
    content = content.replace(parse_call_pattern, updated_parse_call)
    
    # Write the updated content back to the file
    with open(file_path, 'w') as f:
        f.write(content)
    
    logger.info("Updated get_trending_literature function to set is_trending=False")
    return True

def update_get_literary_trends():
    """Update the get_literary_trends function to set is_trending=True."""
    file_path = "literature_logic.py"
    
    # Read the current content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the parse_literature_items call in get_literary_trends
    parse_call_pattern = r'items = parse_literature_items\(response_text\)'
    
    # Count occurrences to make sure we're updating the right one
    occurrences = content.count(parse_call_pattern)
    if occurrences != 2:
        logger.warning(f"Found {occurrences} occurrences of parse_literature_items call, expected 2")
    
    # Find the specific occurrence in get_literary_trends
    trends_function = re.search(r'def get_literary_trends.*?return items', content, re.DOTALL)
    if not trends_function or parse_call_pattern not in trends_function.group(0):
        logger.error("Could not find parse_literature_items call in get_literary_trends")
        return False
    
    updated_parse_call = 'items = parse_literature_items(response_text, is_trending=True)'
    
    # Replace only in the get_literary_trends function
    updated_trends_function = trends_function.group(0).replace(parse_call_pattern, updated_parse_call)
    content = content.replace(trends_function.group(0), updated_trends_function)
    
    # Write the updated content back to the file
    with open(file_path, 'w') as f:
        f.write(content)
    
    logger.info("Updated get_literary_trends function to set is_trending=True")
    return True

def update_parse_literature_items():
    """Update the parse_literature_items function to accept is_trending parameter."""
    file_path = "literature_logic.py"
    
    # Read the current content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Update the function signature
    old_signature = "def parse_literature_items(text: str) -> List[LiteratureItem]:"
    
    if old_signature not in content:
        logger.error("Could not find parse_literature_items function signature")
        return False
    
    new_signature = "def parse_literature_items(text: str, is_trending: bool = False) -> List[LiteratureItem]:"
    
    content = content.replace(old_signature, new_signature)
    
    # Update the function docstring
    old_docstring = '''    """
    Parse the response from Perplexity API into LiteratureItem objects.
    
    Args:
        text: The text response from Perplexity
        
    Returns:
        List of LiteratureItem objects
    """'''
    
    if old_docstring not in content:
        logger.error("Could not find parse_literature_items function docstring")
        return False
    
    new_docstring = '''    """
    Parse the response from Perplexity API into LiteratureItem objects.
    
    Args:
        text: The text response from Perplexity
        is_trending: Flag indicating if these are trending items
        
    Returns:
        List of LiteratureItem objects
    """'''
    
    content = content.replace(old_docstring, new_docstring)
    
    # Update the item creation
    item_creation_pattern = r'item = LiteratureItem\(\s*title=title,\s*author=author,\s*item_type=item_type,\s*description=description,\s*summary=summary\s*\)'
    
    item_creation_match = re.search(item_creation_pattern, content, re.DOTALL)
    if not item_creation_match:
        logger.error("Could not find item creation in parse_literature_items")
        return False
    
    updated_item_creation = '''item = LiteratureItem(
                title=title,
                author=author,
                item_type=item_type,
                description=description,
                summary=summary,
                is_trending=is_trending
            )'''
    
    content = content.replace(item_creation_match.group(0), updated_item_creation)
    
    # Write the updated content back to the file
    with open(file_path, 'w') as f:
        f.write(content)
    
    logger.info("Updated parse_literature_items function to accept is_trending parameter")
    return True

def update_get_recommendations():
    """Update the get_recommendations function to include input and history in the return value."""
    file_path = "literature_logic.py"
    
    # Read the current content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the return statement in get_recommendations
    return_pattern = r'return \{\s*"core": core_recommendations\[:5\],  # Limit to top 5\s*"trending": trending_recommendations\[:5\],  # Limit to top 5\s*"terms": user_terms,\s*"context_description": context_desc,\s*"history": history_used\s*\}'
    
    return_match = re.search(return_pattern, content, re.DOTALL)
    if not return_match:
        logger.error("Could not find return statement in get_recommendations")
        return False
    
    updated_return = '''return {
        "core": core_recommendations[:5],  # Limit to top 5
        "trending": trending_recommendations[:5],  # Limit to top 5
        "terms": user_terms,
        "context_description": context_desc,
        "history": history_used,
        "input": literature_input  # Include the original input
    }'''
    
    content = content.replace(return_match.group(0), updated_return)
    
    # Write the updated content back to the file
    with open(file_path, 'w') as f:
        f.write(content)
    
    logger.info("Updated get_recommendations function to include input in the return value")
    return True

if __name__ == "__main__":
    update_literature_item_class()
    update_parse_literature_items()
    update_get_trending_literature()
    update_get_literary_trends()
    update_get_recommendations()
    print("Updated LiteratureItem class and related functions successfully!")
