"""
Script to fix the OpenAI client reference in the get_user_preferences function
"""

import re
import os

def fix_openai_client():
    """Fix the openai_client reference in get_user_preferences"""
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "literature_logic.py")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace openai_client.ChatCompletion.create with openai.ChatCompletion.create
    updated_content = content.replace("openai_client.ChatCompletion.create", "openai.ChatCompletion.create")
    
    # Write back to the file
    with open(file_path, 'w') as f:
        f.write(updated_content)
    
    print("Fixed OpenAI client reference")

if __name__ == "__main__":
    fix_openai_client()
