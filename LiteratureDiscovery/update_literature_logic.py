"""
Script to update literature_logic.py with new functions and modifications
"""
import re
import os

def read_file(file_path):
    with open(file_path, 'r') as f:
        return f.read()

def write_file(file_path, content):
    with open(file_path, 'w') as f:
        f.write(content)

def add_imports_and_setup(content):
    """Ensure all necessary imports are present"""
    imports = """import os
import sqlite3
import uuid
import logging
import requests
import traceback
import openai
import json
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
from cachetools import TTLCache
from dotenv import load_dotenv
"""
    
    # Check if imports are already present
    if all(imp in content for imp in ["import sqlite3", "import uuid", "from cachetools import TTLCache"]):
        print("Imports already present")
        return content
    
    # Add imports at the beginning of the file
    return imports + content

def add_caches(content):
    """Add or update cache definitions"""
    caches = """
# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

# Configure OpenAI
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
    # Using module-level approach as required
    openai_client = openai
else:
    logger.warning("OpenAI API key not set in environment variables")

# Cache for preferences (1 hour TTL)
prefs_cache = TTLCache(maxsize=100, ttl=3600)

# Cache for trending literature (1 hour TTL)
trends_cache = TTLCache(maxsize=100, ttl=3600)

# Cache for author lookups (24 hour TTL)
author_cache = TTLCache(maxsize=50, ttl=24*3600)
"""
    
    # Check if caches are already defined
    if "prefs_cache = TTLCache" in content and "trends_cache = TTLCache" in content:
        # Update to include author_cache if not present
        if "author_cache = TTLCache" not in content:
            # Find where the other caches are defined
            cache_pos = content.find("trends_cache = TTLCache")
            end_line = content.find("\n", cache_pos)
            
            # Add author_cache after trends_cache
            return content[:end_line+1] + "\n# Cache for author lookups (24 hour TTL)\nauthor_cache = TTLCache(maxsize=50, ttl=24*3600)\n" + content[end_line+1:]
        return content
    
    # Find a good position to insert caches (after imports)
    import_end = max(content.find("\n\n", content.find("import ")), content.find("\n\n", content.find("from ")))
    if import_end == -1:
        import_end = 0
    
    return content[:import_end] + caches + content[import_end:]

def add_get_literary_trends(content):
    """Add the get_literary_trends function"""
    function_code = """
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
            
            prompt = f"""List 5 trending narrative books or short stories (no plays, nonfiction, essays, poetry) from recent years matching themes: {terms_text}.

For each item, provide the following information in this exact format:

Title: [Full title]
Author: [Author's full name]
Type: [book, short story, novella, etc.]
Description: [Brief description highlighting themes related to: {terms_text}]

Focus only on narrative fiction (novels, short stories, novellas) from the past 5-10 years.
Please ensure each entry follows this exact format with clear labels for each field."""
        else:
            prompt = """List 5 trending narrative books or short stories from recent years (past 5-10 years). No plays, nonfiction, or essays.

For each item, provide the following information in this exact format:

Title: [Full title]
Author: [Author's full name]
Type: [book, short story, novella, etc.]
Description: [Brief description highlighting key themes]

Please ensure each entry follows this exact format with clear labels for each field."""
        
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
    
    # Check if function already exists
    if "def get_literary_trends" in content:
        print("get_literary_trends function already exists")
        return content
    
    # Find a good position to insert the function (after get_trending_literature)
    function_pos = content.find("def get_trending_literature")
    if function_pos == -1:
        # If get_trending_literature doesn't exist, add after parse_literature_items
        function_pos = content.find("def parse_literature_items")
    
    # Find the end of the function
    next_def = content.find("def ", function_pos + 10)
    if next_def == -1:
        next_def = len(content)
    
    return content[:next_def] + function_code + content[next_def:]

def update_get_trending_literature(content):
    """Update the get_trending_literature function to focus on classic literature"""
    # Find the function
    function_start = content.find("def get_trending_literature")
    if function_start == -1:
        print("get_trending_literature function not found")
        return content
    
    # Find the end of the function
    next_def = content.find("def ", function_start + 10)
    if next_def == -1:
        next_def = len(content)
    
    # Extract the function
    old_function = content[function_start:next_def]
    
    # Create the updated function
    new_function = """def get_trending_literature(user_terms: List[str] = None, literature_input: str = None) -> List[LiteratureItem]:
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
            
            prompt = f"""List 10 narrative books or short stories (no plays, nonfiction, essays, poetry) from 19th-century or classic literature matching themes: {terms_text}.{exclusion_text}

For each item, provide the following information in this exact format:

Title: [Full title]
Author: [Author's full name]
Type: [book, short story, novella, etc.]
Description: [Brief description highlighting themes related to: {terms_text}]

Focus only on narrative fiction (novels, short stories, novellas) from classic literature.
Please ensure each entry follows this exact format with clear labels for each field."""
        else:
            prompt = """List 10 diverse narrative books or short stories from classic literature. No plays, nonfiction, or essays.

For each item, provide the following information in this exact format:

Title: [Full title]
Author: [Author's full name]
Type: [book, short story, novella, etc.]
Description: [Brief description highlighting key themes]

Please ensure each entry follows this exact format with clear labels for each field."""
        
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
    return content.replace(old_function, new_function)

def add_get_author(content):
    """Add the get_author function"""
    function_code = """
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
    
    # Check if function already exists
    if "def get_author" in content:
        print("get_author function already exists")
        return content
    
    # Find a good position to insert the function (before recommend_literature)
    function_pos = content.find("def recommend_literature")
    if function_pos == -1:
        # If recommend_literature doesn't exist, add at the end
        return content + function_code
    
    return content[:function_pos] + function_code + content[function_pos:]

def update_recommend_literature(content):
    """Update the recommend_literature function"""
    # Find the function
    function_start = content.find("def recommend_literature")
    if function_start == -1:
        print("recommend_literature function not found")
        return content
    
    # Find the end of the function
    next_def = content.find("def ", function_start + 10)
    if next_def == -1:
        next_def = len(content)
    
    # Extract the function
    old_function = content[function_start:next_def]
    
    # Create the updated function
    new_function = """def recommend_literature(trending_items: List[LiteratureItem], user_terms: List[str], literature_input: str = None) -> List[Tuple[LiteratureItem, float, List[str]]]:
    """
    Score and recommend literature items based on user terms.
    
    Args:
        trending_items: List of LiteratureItem objects to score
        user_terms: List of user preference terms
        literature_input: Original user input to avoid self-recommendation
        
    Returns:
        List of tuples (LiteratureItem, score, matched_terms)
    """
    logger.info(f"Starting recommendation process with {len(trending_items)} items and {len(user_terms)} terms")
    
    if not trending_items or not user_terms:
        logger.warning(f"No trending items ({len(trending_items)}) or user terms ({len(user_terms)})")
        return []
    
    scored_items = []
    
    # Get the author of the input literature if available
    input_author = None
    if literature_input:
        input_author = get_author(literature_input)
        if input_author:
            logger.info(f"Found author for input: {input_author}")
    
    # Normalize input for comparison
    literature_input_normalized = literature_input.lower().strip() if literature_input else None
    
    for item in trending_items:
        # Skip self-recommendations (if the item title matches the user input)
        if literature_input_normalized and item.title.lower().strip() == literature_input_normalized:
            logger.info(f"Skipping self-recommendation: {item.title}")
            continue
            
        score = 0.0
        matched_terms = set()
        
        # Convert item fields to lowercase for case-insensitive matching
        title_lower = item.title.lower()
        author_lower = item.author.lower()
        description_lower = item.description.lower()
        item_type_lower = item.item_type.lower()
        
        # Score each term
        for term in user_terms:
            term_lower = term.lower()
            
            # Check for exact matches in different fields
            if term_lower in title_lower:
                score += 1.0
                matched_terms.add(term)
            
            if term_lower in author_lower:
                score += 0.5
                matched_terms.add(term)
                
            if term_lower in item_type_lower:
                score += 1.0
                matched_terms.add(term)
            
            # Higher score for matches in description
            if term_lower in description_lower:
                score += 3.0
                matched_terms.add(term)
        
        # Thematic depth bonus: if 3 or more terms match, add bonus points
        if len(matched_terms) >= 3:
            score += 5.0
            logger.info(f"Applied thematic depth bonus to: {item.title} (matched {len(matched_terms)} terms)")
        
        # Author bonus: if the author matches the input author, add bonus points
        if input_author and input_author.lower() in author_lower:
            score += 4.0
            logger.info(f"Applied author bonus to: {item.title} (author: {item.author})")
        
        # Add to scored items if there's at least one match
        if matched_terms:
            # Store matched terms and score in the item
            item.matched_terms = list(matched_terms)
            item.score = score
            
            scored_items.append((item, score, list(matched_terms)))
    
    logger.info(f"Scored {len(scored_items)} items with at least one match")
    
    # Sort by score in descending order
    scored_items.sort(key=lambda x: x[1], reverse=True)
    
    # Get top 5 items
    top_items = scored_items[:5]
    
    # If we have fewer than 5 items with matches, pad with highest non-zero scores
    if len(top_items) < 5 and len(scored_items) > len(top_items):
        remaining_items = scored_items[len(top_items):]
        # Sort remaining items by score
        remaining_items.sort(key=lambda x: x[1], reverse=True)
        # Add highest scoring remaining items until we have 5 or run out
        for item in remaining_items:
            if item[1] > 0 and len(top_items) < 5:
                top_items.append(item)
    
    # Log top scores for debugging
    if top_items:
        top_scores = [f"{item[0].title[:20]}... ({item[1]})" for item in top_items[:3]]
        logger.info(f"Top scores: {', '.join(top_scores)}")
    
    return top_items
"""
    
    # Replace the old function with the new one
    return content.replace(old_function, new_function)

def add_get_recommendations(content):
    """Add the get_recommendations function"""
    function_code = """
def get_recommendations(literature_input: str, session_id: str = None) -> Dict:
    """
    Get both core and trending recommendations for a literature input.
    
    Args:
        literature_input: The literature input from the user
        session_id: Optional session ID for retrieving user history
        
    Returns:
        Dictionary with core recommendations, trending recommendations, terms, and history
    """
    # Store user input if session_id is provided
    if session_id:
        store_user_input(session_id, literature_input)
    
    # Get user preferences
    terms, context, history = get_user_preferences(literature_input, session_id)
    logger.info(f"Extracted terms: {terms}")
    
    # Get core recommendations (classic literature)
    core_items = get_trending_literature(terms, literature_input)
    logger.info(f"Found {len(core_items)} core literature items")
    
    # Score and recommend core items
    core_recs = recommend_literature(core_items, terms, literature_input)
    logger.info(f"Generated {len(core_recs)} core recommendations")
    
    # Get trending recommendations (recent literature)
    trending_items = get_literary_trends(terms)
    logger.info(f"Found {len(trending_items)} trending literature items")
    
    # Score and recommend trending items
    trending_recs = recommend_literature(trending_items, terms, literature_input)
    logger.info(f"Generated {len(trending_recs)} trending recommendations")
    
    # Return results
    return {
        "core": core_recs,
        "trending": trending_recs,
        "terms": terms,
        "history": history if session_id else []
    }
"""
    
    # Check if function already exists
    if "def get_recommendations" in content:
        print("get_recommendations function already exists")
        return content
    
    # Add at the end of the file
    return content + function_code

def update_literature_logic():
    """Update the literature_logic.py file"""
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "literature_logic.py")
    
    # Read the current content
    content = read_file(file_path)
    
    # Make updates
    content = add_imports_and_setup(content)
    content = add_caches(content)
    content = update_get_trending_literature(content)
    content = add_get_literary_trends(content)
    content = add_get_author(content)
    content = update_recommend_literature(content)
    content = add_get_recommendations(content)
    
    # Write the updated content
    write_file(file_path, content)
    
    print("Updated literature_logic.py with new functions and modifications")

if __name__ == "__main__":
    update_literature_logic()
