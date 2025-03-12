"""
Script to add the feedback endpoint to app.py
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

def add_feedback_endpoint():
    """Add the feedback endpoint to app.py."""
    file_path = "app.py"
    
    # Create a backup
    backup_file(file_path)
    
    # Read the current content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Add the feedback endpoint after the api_trending endpoint
    api_trending_pattern = r'@app\.route\("/api/trending"\).*?return jsonify\(response\).*?\n'
    api_trending_match = re.search(api_trending_pattern, content, re.DOTALL)
    
    if not api_trending_match:
        logger.error("Could not find api_trending endpoint")
        return False
    
    feedback_endpoint = '''
@app.route("/feedback", methods=["POST"])
def submit_feedback():
    """
    Endpoint for submitting user feedback on recommendations.
    
    Expects JSON with:
    - title: Title of the literature item
    - feedback: 1 for thumbs up, -1 for thumbs down
    
    Returns:
    - JSON with success status and message
    """
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({"success": False, "message": "No data provided"}), 400
        
        title = data.get("title")
        feedback = data.get("feedback")
        
        if not title or feedback not in [1, -1]:
            return jsonify({"success": False, "message": "Invalid data provided"}), 400
        
        # Get session ID from cookie
        session_id = request.cookies.get("session_id")
        if not session_id:
            # Generate a new session ID if none exists
            session_id = str(uuid.uuid4())
        
        # Store the feedback
        success = literature_logic.submit_feedback(session_id, title, feedback)
        
        if success:
            response = jsonify({"success": True, "message": "Feedback submitted successfully"})
            # Set session cookie if it doesn't exist
            if not request.cookies.get("session_id"):
                response.set_cookie("session_id", session_id, max_age=60*60*24*30)  # 30 days
            return response
        else:
            return jsonify({"success": False, "message": "Failed to store feedback"}), 500
            
    except Exception as e:
        app.logger.error(f"Error in feedback endpoint: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "message": str(e)}), 500

'''
    
    content = content.replace(api_trending_match.group(0), api_trending_match.group(0) + feedback_endpoint)
    
    # Update the get_recommendations_route function to pass session_id to get_recommendations
    get_rec_pattern = r'recommendations = literature_logic\.get_recommendations\(.*?\)'
    get_rec_match = re.search(get_rec_pattern, content)
    
    if not get_rec_match:
        logger.error("Could not find get_recommendations call in get_recommendations_route")
        return False
    
    updated_get_rec = "recommendations = literature_logic.get_recommendations(literature_input, user_terms, context_description, session_id)"
    
    content = content.replace(get_rec_match.group(0), updated_get_rec)
    
    # Write the updated content back to the file
    with open(file_path, 'w') as f:
        f.write(content)
    
    logger.info("Added feedback endpoint to app.py")
    return True

if __name__ == "__main__":
    add_feedback_endpoint()
    print("Feedback endpoint added to app.py successfully!")
