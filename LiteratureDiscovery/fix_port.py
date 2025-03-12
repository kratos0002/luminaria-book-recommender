"""
Script to update the app.py file to support command-line arguments for port
"""
import re

def add_port_argument():
    file_path = "app.py"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the app.run line
    run_pattern = r'app\.run\(debug=True, port=5003\)'
    
    # Add command-line argument support
    replacement = '''# Parse command line arguments for port
    import argparse
    parser = argparse.ArgumentParser(description='Run the LiteratureDiscovery Flask application')
    parser.add_argument('--port', type=int, default=5003, help='Port to run the server on')
    args = parser.parse_args()

    # Run the Flask application
    app.run(debug=True, port=args.port)'''
    
    # Replace the app.run line
    updated_content = re.sub(run_pattern, replacement, content)
    
    with open(file_path, 'w') as f:
        f.write(updated_content)
    
    print("Updated app.py to support command-line arguments for port")

if __name__ == "__main__":
    add_port_argument()
