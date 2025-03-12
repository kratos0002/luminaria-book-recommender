"""
Script to update the get_trending_literature function in literature_logic.py
"""
import os
import re

def read_file(file_path):
    with open(file_path, 'r') as f:
        return f.read()

def write_file(file_path, content):
    with open(file_path, 'w') as f:
        f.write(content)

def update_function():
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "literature_logic.py")
    content = read_file(file_path)
    
    # Find the function
    function_start = content.find("def get_trending_literature")
    if function_start == -1:
        print("get_trending_literature function not found")
        return
    
    # Find the end of the function
    next_def = content.find("def ", function_start + 10)
    if next_def == -1:
        next_def = len(content)
    
    # Extract the function
    old_function = content[function_start:next_def]
    
    # Create the updated function
    new_function = """\def get_trending_literature(user_terms: List[str] = None, literature_input: str = None) -> List[LiteratureItem]:
    """
    Use Perplexity API to search for classic literature across categories.
    
    Args:
        user_terms: Optional list of terms to focus the search
        literature_input: Original user input to exclude from recommendations
        
    Returns:
        List of LiteratureItem objects
    """
    if not PERPLEXITY_API_KEY:
        logger.error("Perplexity API key not configured")
        return []
    
    # Generate cache key if caching is enabled
    if user_terms:
        cache_key_val = cache_key("classics", user_terms)
        if cache_key_val in trends_cache:
            logger.info(f"Using cached classic literature items: {len(trends_cache[cache_key_val])} items")
            return trends_cache[cache_key_val]
    
    try:
        # Prepare the prompt for Perplexity
        if user_terms and len(user_terms) > 0:
            terms_text = ", ".join(user_terms)
            exclusion_text = f" Exclude {literature_input}." if literature_input else ""
            
            prompt = f\"\"\"List 10 narrative books or short stories (no plays, nonfiction, essays, poetry) from 19th-century or classic literature matching themes: {terms_text}.{exclusion_text}

For each item, provide the following information in this exact format:

Title: [Full title]
Author: [Author's full name]
Type: [book, short story, novella, etc.]
Description: [Brief description highlighting themes related to: {terms_text}]

Focus only on narrative fiction (novels, short stories, novellas) from classic literature.
Please ensure each entry follows this exact format with clear labels for each field.\"\"\"
        else:
            prompt = \"\"\"List 10 diverse narrative books or short stories from classic literature. No plays, nonfiction, or essays.

For each item, provide the following information in this exact format:

Title: [Full title]
Author: [Author's full name]
Type: [book, short story, novella, etc.]
Description: [Brief description highlighting key themes]

Please ensure each entry follows this exact format with clear labels for each field.\"\"\"
        
        logger.info(f"Querying Perplexity for classic literature with terms: {user_terms}")
        
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
                        "content": "You are a literary expert specializing in classic literature recommendations."
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
            logger.info(f"Received response from Perplexity API for classic literature")
            
            # Extract the content from the response
            if "choices" in response_data and len(response_data["choices"]) > 0:
                content = response_data["choices"][0]["message"]["content"].strip()
                logger.info(f"Perplexity content preview for classics: {content[:100]}...")
                
                # Parse the content to extract literature items
                literature_items = parse_literature_items(content)
                logger.info(f"Parsed {len(literature_items)} classic literature items from Perplexity response")
                
                # Cache the results
                if user_terms:
                    trends_cache[cache_key_val] = literature_items
                    logger.info(f"Cached {len(literature_items)} classic literature items for terms: {user_terms}")
                
                return literature_items
            else:
                logger.warning(f"Unexpected response structure from Perplexity for classics: {response_data}")
        else:
            logger.warning(f"Failed to query Perplexity for classics: {response.status_code} - {response.text}")
        
        # If we reach here, there was an error, so return an empty list
        return []
    except Exception as e:
        logger.error(f"Error querying Perplexity for classic literature: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return []
"""
    
    # Replace the old function with the new one
    updated_content = content.replace(old_function, new_function)
    
    # Write the updated content
    write_file(file_path, updated_content)
    print("Updated get_trending_literature function in literature_logic.py")

if __name__ == "__main__":
    update_function()
