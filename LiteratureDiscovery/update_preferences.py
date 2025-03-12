"""
Script to update the get_user_preferences function in literature_logic.py
"""

import re
import os

def update_get_user_preferences():
    """Update the get_user_preferences function"""
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "literature_logic.py")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the function definition
    pattern = r'def get_user_preferences\(literature_input:.*?\).*?return terms, None, history'
    
    # New function code
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
                term = term.strip('"\\\'')
                
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
    
    return terms, None, history'''
    
    # Replace the function
    updated_content = re.sub(pattern, new_code, content, flags=re.DOTALL)
    
    # Write back to the file
    with open(file_path, 'w') as f:
        f.write(updated_content)
    
    print("Updated get_user_preferences function")

if __name__ == "__main__":
    update_get_user_preferences()
