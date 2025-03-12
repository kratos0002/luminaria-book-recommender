"""
Script to change the port in app.py from 5002 to 5003
"""

import re
import os

def change_port():
    """Change the port in app.py from 5002 to 5003"""
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace port 5002 with 5003
    updated_content = re.sub(r'port=5002', 'port=5003', content)
    
    # If not found, try to add it to the app.run line
    if content == updated_content:
        updated_content = re.sub(r'app\.run\(debug=True\)', 'app.run(debug=True, port=5003)', content)
    
    # If still not found, modify the app.run line however it appears
    if content == updated_content:
        updated_content = re.sub(r'app\.run\((.*?)\)', r'app.run(\1, port=5003)', content)
    
    # Write back to the file
    with open(file_path, 'w') as f:
        f.write(updated_content)
    
    print("Changed port to 5003 in app.py")

if __name__ == "__main__":
    change_port()
