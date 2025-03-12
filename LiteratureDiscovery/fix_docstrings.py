"""
Script to fix docstring syntax in update_literature_logic.py
"""
import os
import re

def read_file(file_path):
    with open(file_path, 'r') as f:
        return f.read()

def write_file(file_path, content):
    with open(file_path, 'w') as f:
        f.write(content)

def fix_docstrings(content):
    """Fix docstring syntax in the content"""
    # Replace problematic docstring patterns
    patterns = [
        (r'def ([^\(]+)\([^\)]*\)[^:]*:[^"\']*"""([^"]+)"""', r'def \1(...): """\2"""'),
        (r'def ([^\(]+)\([^\)]*\)[^:]*:\s*"""([^"]+)', r'def \1(...): """\2'),
    ]
    
    fixed_content = content
    for pattern, replacement in patterns:
        fixed_content = re.sub(pattern, replacement, fixed_content)
    
    return fixed_content

def main():
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "update_literature_logic.py")
    
    # Read the current content
    content = read_file(file_path)
    
    # Fix docstrings
    fixed_content = fix_docstrings(content)
    
    # Write the fixed content
    write_file(file_path, fixed_content)
    
    print("Fixed docstring syntax in update_literature_logic.py")

if __name__ == "__main__":
    main()
