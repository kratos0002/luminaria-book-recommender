"""
Script to update specific functions in literature_logic.py
"""

import re
import os

def update_file(function_name, new_code):
    """
    Update a specific function in literature_logic.py
    
    Args:
        function_name: Name of the function to update
        new_code: New function code (including def line)
    """
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "literature_logic.py")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the function definition
    pattern = r'def {}\(.*?\n(?:.*?\n)*?(?=\n\w|$)'.format(function_name)
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        # Replace the function
        updated_content = content[:match.start()] + new_code + content[match.end():]
        
        # Write back to the file
        with open(file_path, 'w') as f:
            f.write(updated_content)
        
        print(f"Updated function: {function_name}")
        return True
    else:
        print(f"Function {function_name} not found")
        return False

def update_init_db():
    """Update the init_db function"""
    new_code = '''def init_db():
    """Initialize the SQLite database for user history tracking."""
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_history.db")
    logger.info(f"Initializing database at {db_path}")
    
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_inputs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            input TEXT NOT NULL UNIQUE,
            timestamp DATETIME NOT NULL
        )
        ''')
        conn.commit()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
    finally:
        conn.close()
'''
    update_file('init_db', new_code)

def update_store_user_input():
    """Update the store_user_input function"""
    new_code = '''def store_user_input(session_id: str, literature_input: str):
    """
    Store a user input in the database with timestamp.
    
    Args:
        session_id: User's session ID
        literature_input: The literature input from the user
    """
    if not session_id or not literature_input:
        logger.warning("Cannot store user input: missing session_id or literature_input")
        return
    
    # Strip input to avoid duplicates with extra whitespace
    literature_input = literature_input.strip()
    
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_history.db")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Store the input with current timestamp (replace if exists)
        cursor.execute(
            "INSERT OR REPLACE INTO user_inputs (session_id, input, timestamp) VALUES (?, ?, ?)",
            (session_id, literature_input, datetime.now())
        )
        conn.commit()
        logger.info(f"Stored user input for session {session_id}: '{literature_input}'")
    except Exception as e:
        logger.error(f"Error storing user input: {str(e)}")
    finally:
        if conn:
            conn.close()
'''
    update_file('store_user_input', new_code)

def update_get_user_history():
    """Update the get_user_history function"""
    new_code = '''def get_user_history(session_id: str, limit: int = 5) -> List[str]:
    """
    Retrieve the user's recent inputs from the database.
    
    Args:
        session_id: User's session ID
        limit: Maximum number of history items to retrieve
        
    Returns:
        List of the user's recent unique inputs
    """
    if not session_id:
        logger.warning("Cannot get user history: missing session_id")
        return []
    
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_history.db")
    history = []
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get the most recent unique inputs
        cursor.execute(
            "SELECT DISTINCT input FROM user_inputs WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
            (session_id, limit)
        )
        
        # Extract the inputs
        results = cursor.fetchall()
        history = [result[0] for result in results]
        logger.info(f"Retrieved {len(history)} unique history items for session {session_id}")
    except Exception as e:
        logger.error(f"Error retrieving user history: {str(e)}")
    finally:
        if conn:
            conn.close()
    
    return history
'''
    update_file('get_user_history', new_code)

def update_get_user_preferences():
    """Update the get_user_preferences function"""
    new_code = '''def get_user_preferences(literature_input: str, session_id: str = None) -> Tuple[List[str], Optional[str], List[str]]:
    """
    Extract user preferences from input data and session history.
    Uses OpenAI GPT-3.5 to understand the query and extract specific themes.
    
    Args:
        literature_input: The literature input from the user
        session_id: Optional session ID for retrieving user history
        
    Returns:
        Tuple of (list of preference terms, optional context description, history used)
    """
    if not literature_input:
        return [], None, []
    
    # Strip input to ensure consistent matching
    literature_input = literature_input.strip()
    
    # Get user history if session_id is provided
    history = []
    if session_id:
        history = get_user_history(session_id)
    
    # Combine current input with history
    combined_input = literature_input
    if history:
        combined_input = f"{literature_input}, {', '.join(history)}"
    
    # Check cache first
    cache_key_val = cache_key("preferences", combined_input)
    if cache_key_val in prefs_cache:
        logger.info(f"Using cached preferences for input: {literature_input[:30]}...")
        cached_result = prefs_cache[cache_key_val]
        return cached_result[0], cached_result[1], history
    
    context_description = None
    
    # Check for special cases
    literature_input_lower = literature_input.lower()
    for key, value in SPECIAL_CASES.items():
        if key in literature_input_lower:
            logger.info(f"Detected special case: '{key}', adding relevant literary terms")
            terms = value["terms"]
            context_description = value["context"]
            logger.info(f"Added specific terms for {key}: {terms}")
            
            # Cache the result
            prefs_cache[cache_key_val] = (terms, context_description)
            
            return terms, context_description, history
    
    # Try to use OpenAI for other queries
    if OPENAI_API_KEY:
        try:
            logger.info(f"Querying OpenAI for themes from: '{combined_input}'")
            
            # Create a prompt that requests literary themes
            prompt = f"""Analyze: {combined_input}

Return 5-7 unique literary themes, genres, or styles (e.g., 'moral dilemma', 'existentialism') as a comma-separated list. 

Focus on:
- Specific literary genres (e.g., 'magical realism', 'dystopian fiction')
- Thematic elements (e.g., 'moral ambiguity', 'coming of age')
- Writing styles (e.g., 'stream of consciousness', 'unreliable narrator')
- Time periods or movements (e.g., 'victorian era', 'beat generation')

Avoid duplicates (e.g., 'psychological' if 'psychological complexity' exists) and generic terms ('book', 'novel', 'also', 'psychological', 'philosophical').

Return ONLY a comma-separated list with no additional text."""
            
            # IMPORTANT: Using the module-level approach for OpenAI API as per requirements
            completion = openai_client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a literary expert specializing in book recommendations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=150
            )
            
            # Extract the response content
            response_content = completion.choices[0].message.content.strip()
            logger.info(f"OpenAI API response: {response_content}")
            
            # Parse terms from response (comma-separated list)
            terms = [term.strip().lower() for term in response_content.split(',')]
            
            # Filter out stopwords and short terms
            cleaned_terms = []
            for term in terms:
                term = term.strip().lower()
                # Remove quotes if present
                term = term.strip('"\'')
                
                # Check if any word in the term is a stopword
                term_words = term.split()
                if all(word not in STOPWORDS for word in term_words) and len(term) > 2:
                    cleaned_terms.append(term)
            
            # Remove duplicates (e.g., if we have both "psychological" and "psychological complexity")
            deduplicated_terms = deduplicate_terms(cleaned_terms)
            
            # Limit to 5-7 terms
            if len(deduplicated_terms) > 7:
                deduplicated_terms = deduplicated_terms[:7]
            
            logger.info(f"Extracted literary terms: {deduplicated_terms}")
            
            # Try to get additional context from Perplexity
            perplexity_response = query_perplexity_about_literature(combined_input, deduplicated_terms)
            if perplexity_response:
                context_description = perplexity_response
                
                # Extract additional terms from Perplexity response
                additional_terms = extract_terms_from_text(perplexity_response)
                
                # Add new terms that aren't already in deduplicated_terms
                for term in additional_terms:
                    if term not in deduplicated_terms and len(deduplicated_terms) < 7:
                        deduplicated_terms.append(term)
            
            # Cache the result
            prefs_cache[cache_key_val] = (deduplicated_terms, context_description)
            
            if deduplicated_terms:
                return deduplicated_terms, context_description, history
            
        except Exception as e:
            logger.error(f"Error querying OpenAI API: {str(e)}")
            logger.error(traceback.format_exc())
    
    # Fallback: Basic term extraction from combined input
    logger.info("Using fallback term extraction from combined input")
    terms = extract_terms_from_text(combined_input)
    logger.info(f"Extracted basic terms: {terms}")
    
    # Cache the result
    prefs_cache[cache_key_val] = (terms, None)
    
    return terms, None, history
'''
    update_file('get_user_preferences', new_code)

def update_get_trending_literature():
    """Update the get_trending_literature function"""
    new_code = '''def get_trending_literature(user_terms: List[str] = None, literature_input: str = None) -> List[LiteratureItem]:
    """
    Use Perplexity API to search for trending literature across categories.
    
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
        cache_key_val = cache_key("trending", user_terms)
        if cache_key_val in trends_cache:
            logger.info(f"Using cached trending literature items: {len(trends_cache[cache_key_val])} items")
            return trends_cache[cache_key_val]
    
    try:
        # Prepare the prompt for Perplexity
        if user_terms and len(user_terms) > 0:
            terms_text = ", ".join(user_terms)
            exclusion_text = f" Exclude {literature_input}." if literature_input else ""
            
            prompt = f"""List 10 narrative books or short stories (no plays, nonfiction, essays) matching these themes: {terms_text}.{exclusion_text}

For each item, provide the following information in this exact format:

Title: [Full title]
Author: [Author's full name]
Type: [book, short story, novella, etc.]
Description: [Brief description highlighting themes related to: {terms_text}]

Focus only on narrative fiction (novels, short stories, novellas).
Please ensure each entry follows this exact format with clear labels for each field."""
        else:
            prompt = """List 10 diverse narrative books or short stories from various genres and time periods. No plays, nonfiction, or essays.

For each item, provide the following information in this exact format:

Title: [Full title]
Author: [Author's full name]
Type: [book, short story, novella, etc.]
Description: [Brief description highlighting key themes]

Please ensure each entry follows this exact format with clear labels for each field."""
        
        logger.info(f"Querying Perplexity for trending literature with terms: {user_terms}")
        
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
                        "content": "You are a literary expert specializing in book recommendations."
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
            logger.info(f"Received response from Perplexity API")
            
            # Extract the content from the response
            if "choices" in response_data and len(response_data["choices"]) > 0:
                content = response_data["choices"][0]["message"]["content"].strip()
                logger.info(f"Perplexity content preview: {content[:100]}...")
                
                # Parse the content to extract literature items
                literature_items = parse_literature_items(content)
                logger.info(f"Parsed {len(literature_items)} literature items from Perplexity response")
                
                # Cache the results
                if user_terms:
                    trends_cache[cache_key_val] = literature_items
                    logger.info(f"Cached {len(literature_items)} literature items for terms: {user_terms}")
                
                return literature_items
            else:
                logger.warning(f"Unexpected response structure from Perplexity: {response_data}")
        else:
            logger.warning(f"Failed to query Perplexity: {response.status_code} - {response.text}")
        
        # If we reach here, there was an error, so return an empty list
        return []
    except Exception as e:
        logger.error(f"Error querying Perplexity for trending literature: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return []
'''
    update_file('get_trending_literature', new_code)

def update_recommend_literature():
    """Update the recommend_literature function"""
    new_code = '''def recommend_literature(trending_items: List[LiteratureItem], user_terms: List[str], literature_input: str = None) -> List[Tuple[LiteratureItem, float, List[str]]]:
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
    
    # Extract potential author from literature_input
    author_name = None
    if literature_input:
        # Check if we have Dostoevsky in the input
        if "dostoevsky" in literature_input.lower() or any(name in literature_input.lower() for name in ["karamazov", "crime and punishment", "idiot"]):
            author_name = "dostoevsky"
        # Add other author detections as needed
    
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
        if author_name and author_name in author_lower:
            score += 3.0  # Increased from 2.0 to 3.0
            logger.info(f"Applied author bonus to: {item.title} (author: {item.author})")
        
        # Add to scored items if there's at least one match
        if matched_terms:
            # Store matched terms and score in the item
            item.matched_terms = matched_terms
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
'''
    update_file('recommend_literature', new_code)

def add_test_function():
    """Add a test function at the end of the file"""
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "literature_logic.py")
    
    with open(file_path, 'a') as f:
        f.write('''
def test_recommendations(input_text="the brothers karamazov", session_id="test"):
    """
    Test function to check recommendation quality for a given input.
    
    Args:
        input_text: Text to test (e.g., 'the brothers karamazov')
        session_id: Session ID to use for testing
    """
    print(f"\\nTesting recommendations for: '{input_text}'")
    
    # Store the input
    store_user_input(session_id, input_text)
    
    # Get user preferences
    user_terms, context, history = get_user_preferences(input_text, session_id)
    
    print(f"\\nExtracted terms: {user_terms}")
    print(f"Context: {context}")
    print(f"History: {history}")
    
    # Get trending literature
    trending_items = get_trending_literature(user_terms, input_text)
    
    print(f"\\nFound {len(trending_items)} trending items")
    
    # Recommend literature
    recommendations = recommend_literature(trending_items, user_terms, input_text)
    
    # Print recommendations
    print(f"\\nTop recommendations:")
    for i, (item, score, matched_terms) in enumerate(recommendations, 1):
        print(f"{i}. {item.title} by {item.author} (Score: {score})")
        print(f"   Type: {item.item_type}")
        print(f"   Matched terms: {', '.join(matched_terms)}")
        print(f"   Description: {item.description[:100]}...")
        print()

if __name__ == "__main__":
    # Initialize the database
    init_db()
    
    # Test recommendations
    test_recommendations("the brothers karamazov", "test_session")
    
    # Test with another input to show history blending
    test_recommendations("the idiot", "test_session")
''')
        print("Added test function")

def update_stopwords():
    """Update the STOPWORDS set"""
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "literature_logic.py")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Look for STOPWORDS definition
    stopwords_pattern = r'STOPWORDS = \{[^}]*\}'
    stopwords_match = re.search(stopwords_pattern, content)
    
    if stopwords_match:
        new_stopwords = '''STOPWORDS = {
    "the", "and", "book", "novel", "also", "prominent", "story", "literature", "literary", 
    "fiction", "nonfiction", "read", "reading", "author", "writer", "books", "novels", 
    "stories", "poem", "poetry", "essay", "articles", "text", "publication", "publish", 
    "published", "pursue", "character", "theme", "plot", "narrative", "chapter", "page", 
    "write", "written", "work", "reader", "this", "that", "with", "for", "from", "its",
    "themes", "elements", "style", "about", "genre", "genres", "psychological", "philosophical"
}'''
        
        # Replace the stopwords
        updated_content = content[:stopwords_match.start()] + new_stopwords + content[stopwords_match.end():]
        
        # Write back to the file
        with open(file_path, 'w') as f:
            f.write(updated_content)
        
        print("Updated STOPWORDS set")
        return True
    else:
        print("STOPWORDS set not found")
        return False

# Run all updates
if __name__ == "__main__":
    print("Updating functions in literature_logic.py...")
    update_init_db()
    update_store_user_input()
    update_get_user_history()
    update_get_user_preferences()
    update_get_trending_literature()
    update_recommend_literature()
    update_stopwords()
    add_test_function()
    print("All updates completed")
