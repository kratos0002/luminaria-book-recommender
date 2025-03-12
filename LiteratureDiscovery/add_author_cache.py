"""
Script to add author cache and get_author function to literature_logic.py
"""
import os

def read_file(file_path):
    with open(file_path, 'r') as f:
        return f.read()

def write_file(file_path, content):
    with open(file_path, 'w') as f:
        f.write(content)

def add_author_cache():
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "literature_logic.py")
    content = read_file(file_path)
    
    # Check if author_cache is already defined
    if "author_cache = TTLCache" in content:
        print("author_cache already exists")
    else:
        # Find where the other caches are defined
        cache_pos = content.find("trends_cache = TTLCache")
        if cache_pos != -1:
            end_line = content.find("\n", cache_pos)
            
            # Add author_cache after trends_cache
            cache_def = "\n# Cache for author lookups (24 hour TTL)\nauthor_cache = TTLCache(maxsize=50, ttl=24*3600)\n"
            content = content[:end_line+1] + cache_def + content[end_line+1:]
            print("Added author_cache to literature_logic.py")
    
    # Check if get_author function already exists
    if "def get_author" in content:
        print("get_author function already exists")
        return content
    
    # Find a good position to insert the function (before recommend_literature)
    function_pos = content.find("def recommend_literature")
    if function_pos == -1:
        # If recommend_literature doesn't exist, add at the end
        function_pos = len(content)
    
    # Define the new function
    new_function = """\
def get_author(literature_input: str) -> Optional[str]:
    """
    Get the author of a literary work using Perplexity API.
    
    Args:
        literature_input: The title of the literary work
        
    Returns:
        The author's name or None if not found
    """
    if not literature_input or not PERPLEXITY_API_KEY:
        return None
    
    # Check cache first
    cache_key_val = f"author_{literature_input.lower().strip()}"
    if cache_key_val in author_cache:
        logger.info(f"Using cached author for: {literature_input}")
        return author_cache[cache_key_val]
    
    try:
        # Prepare the prompt for Perplexity
        prompt = f"Who wrote '{literature_input}'? Return only the author's name, nothing else."
        
        logger.info(f"Querying Perplexity for author of: {literature_input}")
        
        # Query Perplexity API
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "sonar",
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are a literary expert. Respond only with the author's name, nothing else."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 50
            }
        )
        
        if response.status_code == 200:
            response_data = response.json()
            
            # Extract the content from the response
            if "choices" in response_data and len(response_data["choices"]) > 0:
                author = response_data["choices"][0]["message"]["content"].strip()
                logger.info(f"Found author for '{literature_input}': {author}")
                
                # Cache the result
                author_cache[cache_key_val] = author
                
                return author
        
        logger.warning(f"Failed to get author for: {literature_input}")
        return None
    except Exception as e:
        logger.error(f"Error getting author: {str(e)}")
        return None
"""
    
    # Insert the new function
    updated_content = content[:function_pos] + new_function + content[function_pos:]
    
    # Write the updated content
    write_file(file_path, updated_content)
    print("Added get_author function to literature_logic.py")

if __name__ == "__main__":
    add_author_cache()
