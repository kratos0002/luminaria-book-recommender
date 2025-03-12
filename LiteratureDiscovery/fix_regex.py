"""
Script to fix the regex syntax error in literature_logic.py
"""
import re

# Get the file path
file_path = "literature_logic.py"

# Read the file content
with open(file_path, 'r') as f:
    content = f.read()

# Fix the regex pattern
content = content.replace("r'^[\"']|[\"']$'", "r'^[\"\']|[\"\']$'")

# Write the fixed content back to the file
with open(file_path, 'w') as f:
    f.write(content)

print("Fixed regex pattern in literature_logic.py")
