"""
Script to add the get_recommendations function to literature_logic.py
"""
import os

def read_file(file_path):
    with open(file_path, 'r') as f:
        return f.read()

def write_file(file_path, content):
    with open(file_path, 'w') as f:
        f.write(content)

def add_function():
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "literature_logic.py")
    content = read_file(file_path)
    
    # Check if function already exists
    if "def get_recommendations" in content:
        print("get_recommendations function already exists")
        return
    
    # Define the new function
    new_function = """\
def get_recommendations(literature_input: str, session_id: str = None) -> Dict:
    """
    Get both core and trending recommendations for a literature input.
    
    Args:
        literature_input: The literature input from the user
        session_id: Optional session ID for retrieving user history
        
    Returns:
        Dictionary with core recommendations, trending recommendations, terms, and history
    """
    # Store user input if session_id is provided
    if session_id:
        store_user_input(session_id, literature_input)
    
    # Get user preferences
    terms, context, history = get_user_preferences(literature_input, session_id)
    logger.info(f"Extracted terms: {terms}")
    
    # Get core recommendations (classic literature)
    core_items = get_trending_literature(terms, literature_input)
    logger.info(f"Found {len(core_items)} core literature items")
    
    # Score and recommend core items
    core_recs = recommend_literature(core_items, terms, literature_input)
    logger.info(f"Generated {len(core_recs)} core recommendations")
    
    # Get trending recommendations (recent literature)
    trending_items = get_literary_trends(terms)
    logger.info(f"Found {len(trending_items)} trending literature items")
    
    # Score and recommend trending items
    trending_recs = recommend_literature(trending_items, terms, literature_input)
    logger.info(f"Generated {len(trending_recs)} trending recommendations")
    
    # Return results
    return {
        "core": core_recs,
        "trending": trending_recs,
        "terms": terms,
        "history": history if session_id else []
    }
"""
    
    # Add at the end of the file
    updated_content = content + new_function
    
    # Write the updated content
    write_file(file_path, updated_content)
    print("Added get_recommendations function to literature_logic.py")

if __name__ == "__main__":
    add_function()
