"""
Script to fix and run all update scripts in the correct order
"""
import os
import sys
import importlib.util
import subprocess

def run_script(script_path):
    """Run a Python script using subprocess"""
    print(f"Running {os.path.basename(script_path)}...")
    try:
        result = subprocess.run([sys.executable, script_path], 
                               capture_output=True, text=True, check=True)
        print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {os.path.basename(script_path)}: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def fix_docstrings(script_path):
    """Fix docstring syntax in a script file"""
    if not os.path.exists(script_path):
        print(f"File not found: {script_path}")
        return False
        
    print(f"Fixing docstrings in {os.path.basename(script_path)}")
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Fix common docstring syntax issues
    fixed_content = content.replace('"""Get ', '"""\n    Get ')
    fixed_content = fixed_content.replace('"""Use ', '"""\n    Use ')
    fixed_content = fixed_content.replace('"""Score ', '"""\n    Score ')
    fixed_content = fixed_content.replace('"""Store ', '"""\n    Store ')
    
    # Make sure triple quotes are properly escaped in string literals
    fixed_content = fixed_content.replace('new_function = """', 'new_function = """\\')
    
    with open(script_path, 'w') as f:
        f.write(fixed_content)
    
    return True

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # List of script files to fix and run in order
    script_files = [
        "add_author_cache.py",
        "add_literary_trends.py", 
        "update_trending_literature.py",
        "update_recommend_literature.py",
        "add_get_recommendations.py",
        "update_recommendations_template.py",
        "update_app_endpoint.py"
    ]
    
    # Fix and run each script
    for script in script_files:
        script_path = os.path.join(base_dir, script)
        if fix_docstrings(script_path):
            run_script(script_path)

if __name__ == "__main__":
    main()
