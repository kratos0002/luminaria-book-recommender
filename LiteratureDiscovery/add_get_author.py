"""
Script to add the get_author function to literature_logic.py
"""
import os

def read_file(file_path):
    with open(file_path, 'r') as f:
        return f.read()

def write_file(file_path, content):
    with open(file_path, 'w') as f:
        f.write(content)

def add_function():
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "literature_logic.py")
    content = read_file(file_path)
    
    # Add the author cache if not already present
    if "author_cache" not in content:
        # Find the caches section
        cache_section = content.find("# Create caches with TTL")
        if cache_section != -1:
            # Find the end of the cache definitions
            cache_end = content.find("\n\n", cache_section)
            if cache_end == -1:
                cache_end = content.find("\n# Define stopwords", cache_section)
            
            if cache_end != -1:
                # Insert the author cache
                author_cache_def = "\nauthor_cache = TTLCache(maxsize=50, ttl=24*3600)  # 24 hours for author lookups"
                content = content[:cache_end] + author_cache_def + content[cache_end:]
    
    # Check if the function already exists
    if "def get_author" in content:
        print("get_author function already exists")
        return

    # Find recommend_literature function to insert our new function before it
    rec_pos = content.find("def recommend_literature")
    if rec_pos == -1:
        # If not found, add to the end of the file
        rec_pos = len(content)
    
    # Define the new function
    new_function = """
def get_author(literature_input: str) -> Optional[str]:
    \"\"\"
    Get the author of a literary work using Perplexity API.
    
    Args:
        literature_input: The title of the literary work
        
    Returns:
        The author's name or None if not found
    \"\"\"
    if not literature_input or not PERPLEXITY_API_KEY:
        return None
    
    # Check cache first
    literature_lower = literature_input.lower().strip()
    if literature_lower in author_cache:
        logger.info(f"Author cache hit for: {literature_input}")
        return author_cache[literature_lower]
    
    try:
        # Prepare the prompt for Perplexity
        prompt = f\"\"\"Who is the author of '{literature_input}'? If this is not a known literary work or you're not sure, say "Unknown". Respond only with the author's name or "Unknown".\"\"\"
        
        logger.info(f"Querying Perplexity for author of: {literature_input}")
        
        # IMPORTANT: DO NOT CHANGE THIS API CONFIGURATION WITHOUT EXPLICIT PERMISSION
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "sonar",  # DO NOT CHANGE THIS MODEL NAME
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are a literary expert. Answer only with the author's name or 'Unknown'."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 50
            }
        )
        
        if response.status_code == 200:
            response_data = response.json()
            
            # Extract the content from the response
            if "choices" in response_data and len(response_data["choices"]) > 0:
                author = response_data["choices"][0]["message"]["content"].strip()
                
                # Clean up the author name
                if author.lower() == "unknown":
                    logger.info(f"Unknown author for: {literature_input}")
                    return None
                
                # Remove quotes and extra info
                author = re.sub(r'^["']|["']$', '', author)
                author = re.sub(r'\(.*?\)', '', author).strip()
                
                logger.info(f"Found author for '{literature_input}': {author}")
                
                # Cache the result
                author_cache[literature_lower] = author
                
                return author
            else:
                logger.warning(f"Unexpected response from Perplexity for author lookup: {response_data}")
        else:
            logger.warning(f"Failed to query Perplexity for author: {response.status_code} - {response.text}")
        
        return None
    except Exception as e:
        logger.error(f"Error querying Perplexity for author: {str(e)}")
        return None

"""
    
    # Insert the new function
    content = content[:rec_pos] + new_function + content[rec_pos:]
    
    # Write the updated content
    write_file(file_path, content)
    print("Added get_author function to literature_logic.py")

if __name__ == "__main__":
    add_function()
