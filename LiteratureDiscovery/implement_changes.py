"""
Script to implement all required changes to the LiteratureDiscovery app
"""
import os
import re

# Get the base directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# 1. First, add author_cache to literature_logic.py
print("Adding author cache to literature_logic.py...")
logic_path = os.path.join(base_dir, "literature_logic.py")

with open(logic_path, 'r') as f:
    content = f.read()

# Add author_cache
if "author_cache = TTLCache" not in content:
    cache_pos = content.find("trends_cache = TTLCache")
    if cache_pos != -1:
        end_line = content.find("\n", cache_pos)
        
        # Add author_cache after trends_cache
        cache_def = "\n# Cache for author lookups (24 hour TTL)\nauthor_cache = TTLCache(maxsize=50, ttl=24*3600)\n"
        content = content[:end_line+1] + cache_def + content[end_line+1:]
        print("Added author_cache to literature_logic.py")

# 2. Add get_author function
if "def get_author" not in content:
    print("Adding get_author function...")
    function_pos = content.find("def recommend_literature")
    if function_pos != -1:
        author_function = '''
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
    literature_lower = literature_input.lower().strip()
    if literature_lower in author_cache:
        logger.info(f"Author cache hit for: {literature_input}")
        return author_cache[literature_lower]
    
    try:
        # Prepare the prompt for Perplexity
        prompt = f"Who is the author of '{literature_input}'? If this is not a known literary work or you're not sure, say 'Unknown'. Respond only with the author's name or 'Unknown'."
        
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
                author = re.sub(r'\\(.*?\\)', '', author).strip()
                
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
'''
        content = content[:function_pos] + author_function + content[function_pos:]
        print("Added get_author function")

# 3. Add get_literary_trends function
if "def get_literary_trends" not in content:
    print("Adding get_literary_trends function...")
    function_pos = content.find("def parse_literature_items")
    if function_pos != -1:
        next_def = content.find("def ", function_pos + 10)
        if next_def == -1:
            next_def = len(content)
            
        trends_function = '''
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
            
            prompt = f"List 5 trending narrative books or short stories (no plays, nonfiction, essays, poetry) from recent years matching themes: {terms_text}. For each item, provide the following information in this exact format: Title: [Full title] Author: [Author's full name] Type: [book, short story, novella, etc.] Description: [Brief description highlighting themes related to: {terms_text}]. Focus only on narrative fiction (novels, short stories, novellas) from the past 5-10 years. Please ensure each entry follows this exact format with clear labels for each field."
        else:
            prompt = "List 5 trending narrative books or short stories from recent years (past 5-10 years). No plays, nonfiction, or essays. For each item, provide the following information in this exact format: Title: [Full title] Author: [Author's full name] Type: [book, short story, novella, etc.] Description: [Brief description highlighting key themes]. Please ensure each entry follows this exact format with clear labels for each field."
        
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
'''
        content = content[:next_def] + trends_function + content[next_def:]
        print("Added get_literary_trends function")

# 4. Update get_trending_literature function
if "def get_trending_literature" in content:
    print("Updating get_trending_literature function...")
    start_pos = content.find("def get_trending_literature")
    next_def = content.find("def ", start_pos + 10)
    if next_def == -1:
        next_def = len(content)
        
    updated_function = '''
def get_trending_literature(user_terms: List[str] = None, literature_input: str = None) -> List[LiteratureItem]:
    """
    Use Perplexity API to search for classic literature across categories.
    
    Args:
        user_terms: Optional list of terms to focus the search
        literature_input: Original user input to exclude from recommendations
        
    Returns:
        List of LiteratureItem objects representing classic literature
    """
    if not PERPLEXITY_API_KEY:
        logger.error("Perplexity API key not configured")
        return []
    
    # Generate cache key if caching is enabled
    if user_terms:
        cache_key_val = f"classic_{'_'.join(user_terms)}"
        if cache_key_val in trends_cache:
            logger.info(f"Using cached classic literature items: {len(trends_cache[cache_key_val])} items")
            return trends_cache[cache_key_val]
    
    try:
        # Prepare the prompt for Perplexity
        if user_terms and len(user_terms) > 0:
            terms_text = ", ".join(user_terms)
            
            prompt = f"List 5 classic literary works (books, novels, short stories) that match these themes: {terms_text}. Choose diverse works from different time periods and authors, focusing on established literary classics. For each item, provide the following information in this exact format: Title: [Full title] Author: [Author's full name] Type: [novel, short story, novella, etc.] Description: [Brief description highlighting themes related to: {terms_text}]. Please ensure each entry follows this exact format with clear labels for each field."
        else:
            prompt = "List 5 diverse classic literary works (books, novels, short stories) from different time periods and authors. For each item, provide the following information in this exact format: Title: [Full title] Author: [Author's full name] Type: [novel, short story, novella, etc.] Description: [Brief description highlighting key themes]. Please ensure each entry follows this exact format with clear labels for each field."
        
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
                
                # Filter out items that match the user's input (if provided)
                if literature_input:
                    literature_input_lower = literature_input.lower()
                    filtered_items = []
                    for item in literature_items:
                        if literature_input_lower not in item.title.lower() and literature_input_lower not in item.description.lower():
                            filtered_items.append(item)
                        else:
                            logger.info(f"Filtered out literature item that matched user input: {item.title}")
                    literature_items = filtered_items
                
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
'''
    content = content[:start_pos] + updated_function + content[next_def:]
    print("Updated get_trending_literature function")

# 5. Update recommend_literature function
if "def recommend_literature" in content:
    print("Updating recommend_literature function...")
    start_pos = content.find("def recommend_literature")
    next_def = content.find("def ", start_pos + 10)
    if next_def == -1:
        next_def = len(content)
        
    updated_function = '''
def recommend_literature(trending_items: List[LiteratureItem], user_terms: List[str], literature_input: str = None) -> List[Tuple[LiteratureItem, float, List[str]]]:
    """
    Score and recommend literature items based on user terms.
    
    Args:
        trending_items: List of LiteratureItem objects to score
        user_terms: List of user preference terms
        literature_input: Original user input to avoid self-recommendation
        
    Returns:
        List of tuples (LiteratureItem, score, matched_terms)
    """
    if not trending_items or not user_terms:
        return []
    
    recommendations = []
    literature_input_lower = literature_input.lower() if literature_input else ""
    
    # Extract potential author from literature input
    input_author = None
    if literature_input:
        input_author = get_author(literature_input)
        if input_author:
            logger.info(f"Detected author for input '{literature_input}': {input_author}")
    
    for item in trending_items:
        # Skip if the item matches the user's input too closely
        if literature_input_lower and (
            literature_input_lower in item.title.lower() or
            (item.author and literature_input_lower in item.author.lower())
        ):
            logger.info(f"Skipping item that matches input too closely: {item.title}")
            continue
        
        score = 0.0
        matched_terms = []
        
        # Score by matching user terms to item description and title
        for term in user_terms:
            term_lower = term.lower()
            
            # Thematic depth: Check if term appears in description (higher weight)
            if term_lower in item.description.lower():
                score += 1.0
                matched_terms.append(term)
            
            # Title relevance: Check if term appears in title (medium weight)
            if term_lower in item.title.lower():
                score += 0.5
                if term not in matched_terms:
                    matched_terms.append(term)
            
            # Genre match: Check if term appears in genre (lower weight)
            if item.genre and term_lower in item.genre.lower():
                score += 0.3
                if term not in matched_terms:
                    matched_terms.append(term)
        
        # Author matching: Boost score if the author matches the input author
        if input_author and item.author and input_author.lower() in item.author.lower():
            score += 1.0  # Significant boost for same author
            matched_terms.append(f"same author: {input_author}")
        
        # Normalize score based on number of terms
        if len(user_terms) > 0:
            normalized_score = score / len(user_terms)
        else:
            normalized_score = score
        
        # Only include items with at least some relevance
        if normalized_score > 0:
            item.score = normalized_score  # Set the score on the item
            recommendations.append((item, normalized_score, matched_terms))
    
    # Sort by score (descending)
    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    return recommendations
'''
    content = content[:start_pos] + updated_function + content[next_def:]
    print("Updated recommend_literature function")

# 6. Add get_recommendations function
if "def get_recommendations" not in content:
    print("Adding get_recommendations function...")
    function_pos = content.find("def test_recommendations")
    if function_pos == -1:
        function_pos = len(content)
        
    new_function = '''
def get_recommendations(literature_input: str, session_id: str = None) -> Dict:
    """
    Get both core and trending recommendations for a literature input.
    
    Args:
        literature_input: The literature input from the user
        session_id: Optional user session ID for history tracking
        
    Returns:
        Dictionary with core and trending recommendations
    """
    if not literature_input:
        return {"core": [], "trending": []}
    
    # Track user input if session_id is provided
    if session_id:
        store_user_input(session_id, literature_input)
        logger.info(f"Stored user input for session {session_id}: {literature_input}")
    
    # Get user preferences
    user_terms, context_desc, history_used = get_user_preferences(literature_input, session_id)
    logger.info(f"User terms extracted: {user_terms}")
    
    # Get classic literature recommendations (core)
    classic_items = get_trending_literature(user_terms, literature_input)
    logger.info(f"Retrieved {len(classic_items)} classic literature items")
    
    # Get trending recent literature
    trending_items = get_literary_trends(user_terms)
    logger.info(f"Retrieved {len(trending_items)} trending recent literature items")
    
    # Score and rank the recommendations
    core_recommendations = recommend_literature(classic_items, user_terms, literature_input)
    trending_recommendations = recommend_literature(trending_items, user_terms, literature_input)
    
    logger.info(f"Generated {len(core_recommendations)} core recommendations")
    logger.info(f"Generated {len(trending_recommendations)} trending recommendations")
    
    # Return both sets of recommendations
    return {
        "core": core_recommendations[:5],  # Limit to top 5
        "trending": trending_recommendations[:5],  # Limit to top 5
        "terms": user_terms,
        "context_description": context_desc,
        "history": history_used
    }
'''
    content = content[:function_pos] + new_function + content[function_pos:]
    print("Added get_recommendations function")

# Write the updated content back to the file
with open(logic_path, 'w') as f:
    f.write(content)
print("Successfully updated literature_logic.py")

# 7. Update app.py imports and endpoint
print("Updating app.py...")
app_path = os.path.join(base_dir, "app.py")

with open(app_path, 'r') as f:
    app_content = f.read()

# Update imports
if "from literature_logic import get_recommendations" not in app_content:
    import_section = app_content.find("from literature_logic import")
    if import_section != -1:
        end_import = app_content.find(")", import_section)
        if end_import != -1:
            updated_import = "from literature_logic import (\n    get_user_preferences, \n    get_trending_literature, \n    recommend_literature, \n    store_user_input, \n    init_db,\n    get_recommendations\n)"
            app_content = app_content[:import_section] + updated_import + app_content[end_import+1:]
            print("Updated imports in app.py")

# Update the /recommendations endpoint
start_pos = app_content.find("@app.route('/recommendations', methods=['POST'])")
if start_pos != -1:
    next_route = app_content.find("@app.route", start_pos + 10)
    if next_route == -1:
        next_route = len(app_content)
        
    updated_function = '''
@app.route('/recommendations', methods=['POST'])
def get_recommendations_route():
    """
    Endpoint for getting literature recommendations based on user input.
    Accepts both JSON and form data.
    """
    try:
        # Get the user's session ID (create one if it doesn't exist)
        session_id = request.cookies.get('session_id')
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Process user input - accept both JSON and form data
        if request.is_json:
            data = request.get_json()
            literature_input = data.get('literature_input', '')
        else:
            # Handle form data
            literature_input = request.form.get('literature_input', '')
        
        if not literature_input:
            flash("Please enter some literature information")
            return redirect(url_for('home'))
        
        app.logger.info(f"Processing recommendation request for: {literature_input}")
        
        # Get recommendations using the new combined function
        result = get_recommendations(literature_input, session_id)
        
        # Check if we got any recommendations
        if (not result.get('core') and not result.get('trending')) or (len(result.get('core', [])) == 0 and len(result.get('trending', [])) == 0):
            app.logger.warning(f"No recommendations found for: {literature_input}")
            if request.is_json:
                return jsonify({
                    "error": "No recommendations found",
                    "input": literature_input
                }), 404
            
            flash("Sorry, we couldn't find any recommendations for your input. Please try again with different literature.")
            return redirect(url_for('home'))
        
        # Return the recommendations
        if request.is_json:
            return jsonify({
                "recommendations": {
                    "core": [(item.to_dict(), score, terms) for item, score, terms in result.get('core', [])],
                    "trending": [(item.to_dict(), score, terms) for item, score, terms in result.get('trending', [])]
                },
                "terms": result.get('terms', []),
                "context_description": result.get('context_description'),
                "history": result.get('history', [])
            })
        
        # For web interface, render the template
        response = make_response(render_template(
            'recommendations.html', 
            recommendations={
                "core": result.get('core', []),
                "trending": result.get('trending', [])
            },
            terms=result.get('terms', []),
            context_description=result.get('context_description'),
            history=result.get('history', []),
            user_input=literature_input,
            cached=False
        ))
        
        # Set the session cookie
        response.set_cookie('session_id', session_id, max_age=30*24*60*60)  # 30 days
        
        return response
        
    except Exception as e:
        app.logger.error(f"Error processing recommendation request: {str(e)}")
        app.logger.error(traceback.format_exc())
        
        if request.is_json:
            return jsonify({"error": str(e)}), 500
        
        flash("An error occurred while processing your request. Please try again.")
        return redirect(url_for('home'))
'''
    app_content = app_content[:start_pos] + updated_function + app_content[next_route:]
    print("Updated /recommendations endpoint in app.py")

# Write the updated content back to the file
with open(app_path, 'w') as f:
    f.write(app_content)
print("Successfully updated app.py")

print("All changes have been implemented successfully!")
