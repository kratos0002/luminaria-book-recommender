"""
Script to update the LiteratureItem class and related functions to support the is_trending flag
"""
import re
import os
import shutil

def backup_file(file_path):
    """Create a backup of the file before modifying it."""
    backup_path = f"{file_path}.bak_trending"
    shutil.copy2(file_path, backup_path)
    print(f"Created backup at {backup_path}")

def update_literature_item():
    """Update the LiteratureItem class to include is_trending field."""
    file_path = "literature_logic.py"
    
    # Create a backup
    backup_file(file_path)
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Update the LiteratureItem class init method
    class_pattern = r'class LiteratureItem:.*?def __init__\(self, title: str, author: str, publication_date: str = "", \n.*?genre: str = "", description: str = "", item_type: str = "book", \n.*?summary: str = ""\):.*?self\.match_score = 0.*?\n'
    
    replacement = '''class LiteratureItem:
    """Class representing a literature item (book, poem, essay, etc.)
    with its metadata."""
    
    def __init__(self, title: str, author: str, publication_date: str = "", 
                 genre: str = "", description: str = "", item_type: str = "book", 
                 summary: str = "", is_trending: bool = False):
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
        self.is_trending = is_trending  # Flag indicating if this is a trending item
    
'''
    
    updated_content = re.sub(class_pattern, replacement, content, flags=re.DOTALL)
    
    # Update the to_dict method
    to_dict_pattern = r'def to_dict\(self\):.*?return \{.*?"match_score": self\.match_score\n.*?\}'
    
    to_dict_replacement = '''def to_dict(self):
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
            "match_score": self.match_score,
            "is_trending": self.is_trending
        }'''
    
    updated_content = re.sub(to_dict_pattern, to_dict_replacement, updated_content, flags=re.DOTALL)
    
    with open(file_path, 'w') as f:
        f.write(updated_content)
    
    print("Updated LiteratureItem class with is_trending field")

def update_get_trending_literature():
    """Update get_trending_literature to set is_trending=False."""
    file_path = "literature_logic.py"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the parse_literature_items call in get_trending_literature
    trending_function = re.search(r'def get_trending_literature.*?return literature_items', content, re.DOTALL)
    if not trending_function:
        print("Could not find get_trending_literature function")
        return
    
    # Find the line with parse_literature_items
    parse_line_match = re.search(r'(\s+literature_items = parse_literature_items\(content\))', trending_function.group(0))
    if not parse_line_match:
        print("Could not find parse_literature_items call in get_trending_literature")
        return
    
    # Replace with is_trending=False
    updated_line = parse_line_match.group(1).replace('parse_literature_items(content)', 'parse_literature_items(content, is_trending=False)')
    
    # Update the content
    updated_content = content.replace(parse_line_match.group(1), updated_line)
    
    with open(file_path, 'w') as f:
        f.write(updated_content)
    
    print("Updated get_trending_literature function to set is_trending=False")

def update_get_literary_trends():
    """Update get_literary_trends to set is_trending=True."""
    file_path = "literature_logic.py"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the parse_literature_items call in get_literary_trends
    trends_function = re.search(r'def get_literary_trends.*?return literature_items', content, re.DOTALL)
    if not trends_function:
        print("Could not find get_literary_trends function")
        return
    
    # Find the line with parse_literature_items
    parse_line_match = re.search(r'(\s+literature_items = parse_literature_items\(content\))', trends_function.group(0))
    if not parse_line_match:
        print("Could not find parse_literature_items call in get_literary_trends")
        return
    
    # Replace with is_trending=True
    updated_line = parse_line_match.group(1).replace('parse_literature_items(content)', 'parse_literature_items(content, is_trending=True)')
    
    # Update the content
    updated_content = content.replace(parse_line_match.group(1), updated_line)
    
    with open(file_path, 'w') as f:
        f.write(updated_content)
    
    print("Updated get_literary_trends function to set is_trending=True")

def update_parse_literature_items():
    """Update parse_literature_items to accept is_trending parameter."""
    file_path = "literature_logic.py"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the parse_literature_items function
    parse_function_match = re.search(r'def parse_literature_items\(text: str\).*?return items', content, re.DOTALL)
    if not parse_function_match:
        print("Could not find parse_literature_items function")
        return
    
    # Update function signature
    updated_function = parse_function_match.group(0).replace(
        'def parse_literature_items(text: str)',
        'def parse_literature_items(text: str, is_trending: bool = False)'
    )
    
    # Update docstring to include is_trending parameter
    docstring_pattern = r'""".*?Args:.*?text: The text response from Perplexity.*?Returns:.*?List of LiteratureItem objects.*?"""'
    docstring_match = re.search(docstring_pattern, updated_function, re.DOTALL)
    
    if docstring_match:
        updated_docstring = docstring_match.group(0).replace(
            'Args:\n        text: The text response from Perplexity',
            'Args:\n        text: The text response from Perplexity\n        is_trending: Flag indicating if these are trending items'
        )
        updated_function = updated_function.replace(docstring_match.group(0), updated_docstring)
    
    # Update item creation to include is_trending
    item_pattern = r'item = LiteratureItem\(\s*title=title,\s*author=author,\s*item_type=item_type,\s*description=description,\s*summary=summary\s*\)'
    item_match = re.search(item_pattern, updated_function, re.DOTALL)
    
    if item_match:
        updated_item = '''item = LiteratureItem(
                title=title,
                author=author,
                item_type=item_type,
                description=description,
                summary=summary,
                is_trending=is_trending
            )'''
        updated_function = updated_function.replace(item_match.group(0), updated_item)
    
    # Update the content
    updated_content = content.replace(parse_function_match.group(0), updated_function)
    
    with open(file_path, 'w') as f:
        f.write(updated_content)
    
    print("Updated parse_literature_items function to accept is_trending parameter")

def update_get_recommendations():
    """Update get_recommendations to include input in the return value."""
    file_path = "literature_logic.py"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the return statement in get_recommendations
    return_pattern = r'return \{\s*"core": core_recommendations\[:5\],.*?"trending": trending_recommendations\[:5\],.*?"terms": user_terms,.*?"context_description": context_desc,.*?"history": history_used.*?\}'
    return_match = re.search(return_pattern, content, re.DOTALL)
    
    if not return_match:
        print("Could not find return statement in get_recommendations")
        return
    
    # Update the return statement to include input
    updated_return = '''return {
        "core": core_recommendations[:5],  # Limit to top 5
        "trending": trending_recommendations[:5],  # Limit to top 5
        "terms": user_terms,
        "context_description": context_desc,
        "history": history_used,
        "input": literature_input  # Include the original input
    }'''
    
    # Update the content
    updated_content = content.replace(return_match.group(0), updated_return)
    
    with open(file_path, 'w') as f:
        f.write(updated_content)
    
    print("Updated get_recommendations function to include input in the return value")

if __name__ == "__main__":
    update_literature_item()
    update_parse_literature_items()
    update_get_trending_literature()
    update_get_literary_trends()
    update_get_recommendations()
    print("All updates completed successfully!")
