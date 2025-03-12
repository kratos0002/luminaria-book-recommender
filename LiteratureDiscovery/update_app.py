"""
Script to update app.py with imports and initialization for literature_logic.

This script modifies app.py to import the improved functions from literature_logic.py
and adds database initialization before running the Flask app.
"""

import re
import os

def update_app_file():
    """Update the app.py file to use literature_logic.py functions"""
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    
    with open(app_path, 'r') as f:
        content = f.read()
    
    # Add imports if they don't exist
    import_statement = """
# Import from our new literature_logic module
from literature_logic import (
    get_user_preferences, 
    get_trending_literature, 
    recommend_literature, 
    store_user_input, 
    init_db
)
"""
    
    if "from literature_logic import" not in content:
        # Add after existing imports
        import_pattern = r"from database import .*?\n"
        if re.search(import_pattern, content):
            content = re.sub(import_pattern, "from database import get_user_history\n\n" + import_statement, content)
        else:
            # Fallback - add after dotenv import
            import_pattern = r"from dotenv import load_dotenv\n"
            content = re.sub(import_pattern, "from dotenv import load_dotenv\n\n" + import_statement, content)
    
    # Update main block to initialize database
    if "__name__ == \"__main__\"" in content and "init_db()" not in content:
        main_pattern = r"if __name__ == \"__main__\":\s*\n\s*app\.run\("
        replacement = """if __name__ == "__main__":
    # Initialize the database
    init_db()
    app.run("""
        content = re.sub(main_pattern, replacement, content)
    
    # Update recommendations endpoint
    recommendations_pattern = r"@app\.route\('/recommendations', methods=\['POST'\]\)\ndef recommendations\(\):.*?return response"
    if recommendations_pattern in content:
        recommendations_replacement = """@app.route('/recommendations', methods=['POST'])
def recommendations():
    \"\"\"
    Process user input and return literature recommendations.
    \"\"\"
    app.logger.info(f"Received recommendations request")
    
    # Process user input - accept both JSON and form data
    if request.is_json:
        data = request.get_json()
    else:
        # Handle form data
        data = {'literature_input': request.form.get('literature_input', '')}
    
    literature_input = data.get('literature_input', '').strip()
    
    if not literature_input:
        app.logger.warning("Empty literature input received")
        return jsonify({"error": "No literature input provided"}), 400
    
    app.logger.info(f"Processing input: '{literature_input}'")
    
    # Get or create a session ID
    session_id = request.cookies.get('session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        app.logger.info(f"Created new session ID: {session_id}")
    
    # Store user input in the database
    store_user_input(session_id, literature_input)
    
    # Get user preferences based on input and history
    user_terms, context, history = get_user_preferences(literature_input, session_id)
    
    if not user_terms:
        app.logger.warning("No user preferences extracted")
        return jsonify({"error": "Could not determine preferences from input"}), 400
    
    app.logger.info(f"Extracted user terms: {user_terms}")
    
    # Get trending literature for the user's preferences
    trending_items = get_trending_literature(user_terms, literature_input)
    
    if not trending_items:
        app.logger.warning("No trending items found")
        return jsonify({"error": "Could not find trending literature for your preferences"}), 400
    
    # Recommend literature based on user terms and trending items
    recommendations = recommend_literature(trending_items, user_terms, literature_input)
    
    # Prepare the response
    result = {
        "user_input": literature_input,
        "user_terms": user_terms,
        "context": context,
        "history": history,
        "recommendations": [
            {
                "title": item.title,
                "author": item.author,
                "publication_date": item.publication_date,
                "genre": item.genre,
                "description": item.description,
                "item_type": item.item_type,
                "score": score,
                "matched_terms": matched_terms
            } for item, score, matched_terms in recommendations
        ]
    }
    
    # Set the session ID cookie in the response
    response = jsonify(result)
    response.set_cookie('session_id', session_id, max_age=60*60*24*30)  # 30 days
    
    return response"""
        
        content = re.sub(recommendations_pattern, recommendations_replacement, content, flags=re.DOTALL)
    
    # Write the updated content back to app.py
    with open(app_path, 'w') as f:
        f.write(content)
    
    print("Successfully updated app.py to use the improved functions from literature_logic.py")

if __name__ == "__main__":
    update_app_file()
