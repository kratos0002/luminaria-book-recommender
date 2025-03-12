"""
Script to update the app.py file to use the new get_recommendations function
"""
import os
import re

def read_file(file_path):
    with open(file_path, 'r') as f:
        return f.read()

def write_file(file_path, content):
    with open(file_path, 'w') as f:
        f.write(content)

def update_imports(content):
    """Update imports to include get_recommendations"""
    # Check if we need to add the import
    if "from literature_logic import" not in content:
        # Add the import at the beginning of the file
        import_line = "from literature_logic import get_recommendations, init_db\n"
        # Find a good position to insert the import
        flask_import = content.find("from flask import")
        if flask_import != -1:
            next_line = content.find("\n", flask_import)
            if next_line != -1:
                return content[:next_line+1] + import_line + content[next_line+1:]
        return import_line + content
    
    # Find the literature_logic import line
    import_match = re.search(r"from literature_logic import .*", content)
    if not import_match:
        print("literature_logic import not found")
        return content
    
    # Check if get_recommendations is already imported
    if "get_recommendations" in import_match.group(0):
        print("get_recommendations already imported")
        return content
    
    # Update the import line to include get_recommendations
    old_import = import_match.group(0)
    imports = old_import.split("import ")[1].strip()
    imports_list = [imp.strip() for imp in imports.split(",")]
    
    if "get_recommendations" not in imports_list:
        imports_list.append("get_recommendations")
    
    new_import = f"from literature_logic import {', '.join(imports_list)}"
    return content.replace(old_import, new_import)

def update_recommendations_endpoint(content):
    """Update the /recommendations endpoint to use get_recommendations"""
    # Find the /recommendations route
    route_start = content.find("@app.route('/recommendations', methods=['POST'])")
    if route_start == -1:
        print("/recommendations route not found")
        return content
    
    # Find the end of the route function
    next_route = content.find("@app.route", route_start + 10)
    if next_route == -1:
        next_route = len(content)
    
    # Extract the route function
    old_route = content[route_start:next_route]
    
    # Create the updated route function
    new_route = """@app.route('/recommendations', methods=['POST'])
def get_recommendations_route():
    try:
        # Process user input - accept both JSON and form data
        if request.is_json:
            data = request.get_json()
        else:
            # Handle form data
            data = {'literature_input': request.form.get('literature_input', '')}
            
        literature_input = data.get('literature_input', '')
        
        if not literature_input:
            return jsonify({"error": "No literature input provided"}), 400
            
        # Get or create session ID
        session_id = request.cookies.get('session_id')
        if not session_id:
            session_id = str(uuid.uuid4())
            
        # Get recommendations using the new function
        recommendations = get_recommendations(literature_input, session_id)
        
        # Prepare response
        response = make_response(render_template('recommendations.html', 
                                               literature_input=literature_input,
                                               recommendations=recommendations))
                                               
        # Set session cookie
        if not request.cookies.get('session_id'):
            response.set_cookie('session_id', session_id, max_age=60*60*24*30)  # 30 days
            
        return response
        
    except Exception as e:
        app.logger.error(f"Error in recommendations: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
"""
    
    # Replace the old route with the new one
    return content.replace(old_route, new_route)

def add_db_init(content):
    """Add database initialization before app.run()"""
    # Find app.run()
    run_pos = content.find("app.run(")
    if run_pos == -1:
        print("app.run() not found")
        return content
    
    # Check if init_db() is already called before app.run()
    init_check = content[:run_pos].rfind("init_db()")
    if init_check != -1:
        print("init_db() already called before app.run()")
        return content
    
    # Find the line before app.run()
    prev_line = content.rfind("\n", 0, run_pos)
    if prev_line == -1:
        prev_line = 0
    
    # Add init_db() call before app.run()
    db_init = "\n# Initialize the database\ninit_db()\n\n"
    return content[:prev_line] + db_init + content[prev_line:]

def update_app():
    """Update app.py to use the new get_recommendations function"""
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    
    # Read the current content
    content = read_file(file_path)
    
    # Make updates
    content = update_imports(content)
    content = update_recommendations_endpoint(content)
    content = add_db_init(content)
    
    # Write the updated content
    write_file(file_path, content)
    
    print("Updated app.py to use the new get_recommendations function")

if __name__ == "__main__":
    update_app()
