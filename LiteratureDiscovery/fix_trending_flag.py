"""
Script to fix the is_trending flag in get_literary_trends function
"""
import re

def fix_trending_flag():
    file_path = "literature_logic.py"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the specific line in get_literary_trends function
    pattern = r'(literature_items = parse_literature_items\(content, is_trending=)False(\))'
    
    # Replace False with True
    updated_content = re.sub(pattern, r'\1True\2', content)
    
    with open(file_path, 'w') as f:
        f.write(updated_content)
    
    print("Fixed is_trending flag in get_literary_trends function")

if __name__ == "__main__":
    fix_trending_flag()
