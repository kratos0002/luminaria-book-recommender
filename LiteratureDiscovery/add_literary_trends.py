"""
Script to add get_literary_trends function to literature_logic.py
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
    
    # Check if function already exists
    if "def get_literary_trends" in content:
        print("get_literary_trends function already exists")
        return
    
    # Find a good position to insert the function (after get_trending_literature)
    function_pos = content.find("def get_trending_literature")
    if function_pos == -1:
        # If get_trending_literature doesn't exist, add after parse_literature_items
        function_pos = content.find("def parse_literature_items")
    
    # Find the end of the function
    next_def = content.find("def ", function_pos + 10)
    if next_def == -1:
        next_def = len(content)
    
    # Define the new function
    new_function = """\
def get_literary_trends(user_terms: List[str] = None) -> List[LiteratureItem]:
    """
    Use Perplexity API to search for trending recent literature across categories.
    
    Args:
        user_terms: Optional list of terms to focus the search
        
    Returns:
        List of LiteratureItem objects representing trending recent literature
    """
    if not PERPLEXITY_API_KEY:
        logger.error("Perplexity API key not configured")
        return []
    
    # Generate cache key if caching is enabled
    if user_terms:
        cache_key_val = f"trending_{'_'.join(user_terms)}"
        if cache_key_val in trends_cache:
            logger.info(f"Using cached trending literature items: {len(trends_cache[cache_key_val])} items")
            return trends_cache[cache_key_val]
    
    try:
        # Prepare the prompt for Perplexity
        if user_terms and len(user_terms) > 0:
            terms_text = ", ".join(user_terms)
            
            prompt = f\"\"\"List 5 trending narrative books or short stories (no plays, nonfiction, essays, poetry) from recent years matching themes: {terms_text}.

For each item, provide the following information in this exact format:

Title: [Full title]
Author: [Author's full name]
Type: [book, short story, novella, etc.]
Description: [Brief description highlighting themes related to: {terms_text}]

Focus only on narrative fiction (novels, short stories, novellas) from the past 5-10 years.
Please ensure each entry follows this exact format with clear labels for each field.\"\"\"
        else:
            prompt = \"\"\"List 5 trending narrative books or short stories from recent years (past 5-10 years). No plays, nonfiction, or essays.

For each item, provide the following information in this exact format:

Title: [Full title]
Author: [Author's full name]
Type: [book, short story, novella, etc.]
Description: [Brief description highlighting key themes]

Please ensure each entry follows this exact format with clear labels for each field.\"\"\"
        
        logger.info(f"Querying Perplexity for trending recent literature with terms: {user_terms}")
        
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
                        "content": "You are a literary expert specializing in trending contemporary book recommendations."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 1000
            }
        )
        
        if response.status_code == 200:
            response_data = response.json()
            logger.info(f"Received response from Perplexity API for trending literature")
            
            # Extract the content from the response
            if "choices" in response_data and len(response_data["choices"]) > 0:
                content = response_data["choices"][0]["message"]["content"].strip()
                logger.info(f"Perplexity content preview for trends: {content[:100]}...")
                
                # Parse the content to extract literature items
                literature_items = parse_literature_items(content)
                logger.info(f"Parsed {len(literature_items)} trending literature items from Perplexity response")
                
                # Cache the results
                if user_terms:
                    trends_cache[cache_key_val] = literature_items
                    logger.info(f"Cached {len(literature_items)} trending literature items for terms: {user_terms}")
                
                return literature_items
            else:
                logger.warning(f"Unexpected response structure from Perplexity for trends: {response_data}")
        else:
            logger.warning(f"Failed to query Perplexity for trends: {response.status_code} - {response.text}")
        
        # If we reach here, there was an error, so return an empty list
        return []
    except Exception as e:
        logger.error(f"Error querying Perplexity for trending literature: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return []
"""
    
    # Insert the new function
    updated_content = content[:next_def] + new_function + content[next_def:]
    
    # Write the updated content
    write_file(file_path, updated_content)
    print("Added get_literary_trends function to literature_logic.py")

if __name__ == "__main__":
    add_function()
