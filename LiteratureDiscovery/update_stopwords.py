"""
Script to update the STOPWORDS set in literature_logic.py
"""

import re
import os

def update_stopwords():
    """Update the STOPWORDS set"""
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "literature_logic.py")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the STOPWORDS definition
    pattern = r'STOPWORDS = \{[^}]*\}'
    
    # New STOPWORDS definition with added terms
    new_stopwords = '''STOPWORDS = {
    "the", "and", "book", "novel", "also", "prominent", "story", "literature", "literary", 
    "fiction", "nonfiction", "read", "reading", "author", "writer", "books", "novels", 
    "stories", "poem", "poetry", "essay", "articles", "text", "publication", "publish", 
    "published", "pursue", "character", "theme", "plot", "narrative", "chapter", "page", 
    "write", "written", "work", "reader", "this", "that", "with", "for", "from", "its",
    "themes", "elements", "style", "about", "genre", "genres", "psychological", "philosophical"
}'''
    
    # Replace the STOPWORDS
    updated_content = re.sub(pattern, new_stopwords, content, flags=re.DOTALL)
    
    # Write back to the file
    with open(file_path, 'w') as f:
        f.write(updated_content)
    
    print("Updated STOPWORDS set")

def add_special_case_for_idiot():
    """Add special case for 'The Idiot'"""
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "literature_logic.py")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the SPECIAL_CASES definition
    special_cases_pattern = r'SPECIAL_CASES = \{[^}]*\}'
    special_cases_match = re.search(special_cases_pattern, content, flags=re.DOTALL)
    
    if special_cases_match:
        # Get the current SPECIAL_CASES content
        current_special_cases = special_cases_match.group(0)
        
        # Check if 'the idiot' is already in the special cases
        if 'the idiot' not in current_special_cases.lower():
            # Add 'The Idiot' case
            new_special_cases = current_special_cases[:-1] + ',\n    "the idiot": {\n        "terms": ["existentialism", "moral ambiguity", "russian literature", "19th century", "psychological novel", "dostoevsky"],\n        "context": "Dostoevsky\'s novel exploring themes of innocence, good vs. evil, and human nature through Prince Myshkin\'s experiences in Russian society."\n    }\n}'
            
            # Replace the SPECIAL_CASES
            updated_content = content.replace(current_special_cases, new_special_cases)
            
            # Write back to the file
            with open(file_path, 'w') as f:
                f.write(updated_content)
            
            print("Added special case for 'The Idiot'")
            return True
        else:
            print("Special case for 'The Idiot' already exists")
            return False
    else:
        print("SPECIAL_CASES not found")
        return False

def add_test_function():
    """Add a test function at the end of the file"""
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "literature_logic.py")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if the test function already exists
    if "def test_recommendations" in content:
        print("Test function already exists")
        return False
    
    # Add the test function at the end of the file
    test_function = '''

def test_recommendations(input_text="the brothers karamazov", session_id="test"):
    """
    Test function to check recommendation quality for a given input.
    
    Args:
        input_text: Text to test (e.g., 'the brothers karamazov')
        session_id: Session ID to use for testing
    """
    print(f"\\nTesting recommendations for: '{input_text}'")
    
    # Store the input
    store_user_input(session_id, input_text)
    
    # Get user preferences
    user_terms, context, history = get_user_preferences(input_text, session_id)
    
    print(f"\\nExtracted terms: {user_terms}")
    print(f"Context: {context}")
    print(f"History: {history}")
    
    # Get trending literature
    trending_items = get_trending_literature(user_terms, input_text)
    
    print(f"\\nFound {len(trending_items)} trending items")
    
    # Recommend literature
    recommendations = recommend_literature(trending_items, user_terms, input_text)
    
    # Print recommendations
    print(f"\\nTop recommendations:")
    for i, (item, score, matched_terms) in enumerate(recommendations, 1):
        print(f"{i}. {item.title} by {item.author} (Score: {score})")
        print(f"   Type: {item.item_type}")
        print(f"   Matched terms: {', '.join(matched_terms)}")
        print(f"   Description: {item.description[:100]}...")
        print()

if __name__ == "__main__":
    # Initialize the database
    init_db()
    
    # Test recommendations
    test_recommendations("the brothers karamazov", "test_session")
    
    # Test with another input to show history blending
    test_recommendations("the idiot", "test_session")
'''
    
    # Append the test function
    with open(file_path, 'a') as f:
        f.write(test_function)
    
    print("Added test function")
    return True

if __name__ == "__main__":
    update_stopwords()
    add_special_case_for_idiot()
    add_test_function()
