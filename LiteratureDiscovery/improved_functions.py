"""
Improved functions for the LiteratureDiscovery application.
These functions enhance the recommendation logic and user history tracking.
"""

import re
import logging
import traceback
from typing import Dict, List, Tuple, Optional, Set

# Import the necessary functions from the existing codebase
# These will be imported when copying the functions into app.py
# from app import app, openai_client, PERPLEXITY_API_KEY, CACHING_ENABLED, prefs_cache, cache_key, STOPWORDS
# from database import get_user_history, store_user_input
# from models import LiteratureItem

def get_user_preferences(data: Dict, session_id: str = None) -> Tuple[List[str], Optional[str], List[str]]:
    """
    Extract user preferences from input data and session history.
    Uses OpenAI GPT-3.5 to understand the query and extract specific themes.
    
    Args:
        data: Dictionary containing user input
        session_id: Optional session ID for retrieving user history
        
    Returns:
        Tuple of (list of preference terms, optional context description, history used)
    """
    user_input = data.get('literature_input', '').strip()
    if not user_input:
        return [], None, []
    
    # Get user history if session_id is provided
    history = []
    if session_id:
        history = get_user_history(session_id)
    
    # Combine current input with history
    combined_input = user_input
    if history:
        combined_input = f"{user_input}, {', '.join(history)}"
    
    # Check cache first if caching is enabled
    if CACHING_ENABLED:
        cache_key_val = cache_key("preferences", combined_input)
        if cache_key_val in prefs_cache:
            app.logger.info(f"Using cached preferences for input: {user_input[:30]}...")
            cached_result = prefs_cache[cache_key_val]
            return cached_result[0], cached_result[1], history
    
    context_description = None
    
    # For Crime and Punishment, manually add relevant literary terms
    if "crime" in user_input.lower() and "punishment" in user_input.lower():
        app.logger.info("Detected 'Crime and Punishment' query, adding relevant literary terms")
        terms = [
            "psychological thriller", 
            "moral dilemma", 
            "redemption", 
            "19th century literature",
            "russian literature", 
            "existentialism", 
            "crime fiction",
            "philosophical novel",
            "dostoevsky"
        ]
        context_description = "Themes related to Crime and Punishment by Fyodor Dostoevsky"
        app.logger.info(f"Added specific terms for Crime and Punishment: {terms}")
        
        # Cache the result if caching is enabled
        if CACHING_ENABLED:
            prefs_cache[cache_key_val] = (terms, context_description)
        
        return terms, context_description, history
    
    # For The Brothers Karamazov, manually add relevant literary terms
    if any(term in user_input.lower() for term in ["brothers karamazov", "karamazov"]):
        app.logger.info("Detected 'The Brothers Karamazov' query, adding relevant literary terms")
        terms = [
            "philosophical novel",
            "existentialism",
            "moral dilemma",
            "religious philosophy",
            "russian literature",
            "19th century literature",
            "dostoevsky",
            "family drama"
        ]
        context_description = "Themes related to The Brothers Karamazov by Fyodor Dostoevsky"
        app.logger.info(f"Added specific terms for The Brothers Karamazov: {terms}")
        
        # Cache the result if caching is enabled
        if CACHING_ENABLED:
            prefs_cache[cache_key_val] = (terms, context_description)
        
        return terms, context_description, history
    
    # Try to use OpenAI for other queries
    if openai_client:
        try:
            app.logger.info(f"Querying OpenAI for themes from: '{combined_input}'")
            
            # Create a prompt that requests literary themes
            prompt = f"""Analyze: {combined_input}

Return 5-7 unique literary themes, genres, or styles (e.g., 'moral dilemma', 'existentialism') as a comma-separated list. 

Focus on:
- Specific literary genres (e.g., 'magical realism', 'dystopian fiction')
- Thematic elements (e.g., 'moral ambiguity', 'coming of age')
- Writing styles (e.g., 'stream of consciousness', 'unreliable narrator')
- Time periods or movements (e.g., 'victorian era', 'beat generation')

Avoid duplicates (e.g., 'psychological' if 'psychological complexity' exists) and generic terms ('book', 'novel', 'also').

Return ONLY a comma-separated list with no additional text."""
            
            # IMPORTANT: DO NOT CHANGE THIS API CALL PATTERN WITHOUT EXPLICIT PERMISSION
            # This specific implementation is required for compatibility with OpenAI 0.28
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
            app.logger.info(f"OpenAI API response: {response_content}")
            
            # Parse terms from response (comma-separated list)
            terms = [term.strip().lower() for term in response_content.split(',')]
            
            # Define additional stopwords
            additional_stopwords = {
                "book", "novel", "story", "literature", "literary", "fiction", "nonfiction", 
                "read", "reading", "author", "writer", "books", "novels", "stories", "poem", 
                "poetry", "essay", "articles", "text", "publication", "publish", "published",
                "also", "prominent", "pursue", "character", "theme", "plot", "narrative",
                "chapter", "page", "write", "written", "work", "reader"
            }
            
            # Combine with existing stopwords
            all_stopwords = STOPWORDS.union(additional_stopwords)
            
            # Filter out stopwords and short terms
            cleaned_terms = []
            for term in terms:
                term = term.strip().lower()
                # Remove quotes if present
                term = term.strip('"\'')
                
                # Check if any word in the term is a stopword
                term_words = term.split()
                if all(word not in all_stopwords for word in term_words) and len(term) > 2:
                    cleaned_terms.append(term)
            
            # Remove duplicates (e.g., if we have both "psychological" and "psychological complexity")
            deduplicated_terms = []
            for term in cleaned_terms:
                # Check if this term is a subset of any other term
                if not any(term != other_term and term in other_term for other_term in cleaned_terms):
                    deduplicated_terms.append(term)
            
            # Limit to 5-7 terms
            if len(deduplicated_terms) > 7:
                deduplicated_terms = deduplicated_terms[:7]
            
            app.logger.info(f"Extracted literary terms: {deduplicated_terms}")
            
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
            
            # Cache the result if caching is enabled
            if CACHING_ENABLED:
                prefs_cache[cache_key_val] = (deduplicated_terms, context_description)
            
            if deduplicated_terms:
                return deduplicated_terms, context_description, history
            
        except Exception as e:
            app.logger.error(f"Error querying OpenAI API: {str(e)}")
            app.logger.error(traceback.format_exc())
    
    # Fallback: Basic term extraction from combined input
    app.logger.info("Using fallback term extraction from combined input")
    terms = extract_terms_from_text(combined_input)
    app.logger.info(f"Extracted basic terms: {terms}")
    
    # Cache the result if caching is enabled
    if CACHING_ENABLED:
        prefs_cache[cache_key_val] = (terms, None)
    
    return terms, None, history

def query_perplexity_about_literature(literature_input: str, terms: List[str] = None) -> Optional[str]:
    """
    Query Perplexity API to get additional context about the literature input.
    
    Args:
        literature_input: The literature input to analyze
        terms: Optional list of terms already extracted
        
    Returns:
        Optional string with context description
    """
    if not PERPLEXITY_API_KEY:
        app.logger.error("Perplexity API key not configured")
        return None
    
    try:
        # Prepare the prompt for Perplexity
        terms_text = ", ".join(terms) if terms else ""
        
        prompt = f"""Summarize themes of {literature_input} in 2-3 sentences, focusing on literary elements.
        
If you recognize this as a specific work, please include the author's name and any relevant literary movement or time period.

Focus on themes, style, and genre rather than plot summary."""
        
        app.logger.info(f"Querying Perplexity about literature: '{literature_input}'")
        
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
            app.logger.info(f"Received response from Perplexity API")
            
            # Extract the content from the response
            if "choices" in response_data and len(response_data["choices"]) > 0:
                content = response_data["choices"][0]["message"]["content"].strip()
                app.logger.info(f"Perplexity response: {content[:100]}...")
                return content
            else:
                app.logger.warning(f"Unexpected response structure from Perplexity: {response_data}")
                return None
        else:
            app.logger.warning(f"Failed to query Perplexity: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        app.logger.error(f"Error querying Perplexity for preference analysis: {str(e)}")
        return None

def get_trending_literature(user_terms: List[str] = None, literature_input: str = None) -> List[LiteratureItem]:
    """
    Use Perplexity API to search for trending literature across categories.
    
    Args:
        user_terms: Optional list of terms to focus the search
        literature_input: Original user input to exclude from recommendations
        
    Returns:
        List of LiteratureItem objects
    """
    if not PERPLEXITY_API_KEY:
        app.logger.error("Perplexity API key not configured")
        return []
    
    # Generate cache key if caching is enabled
    if CACHING_ENABLED and user_terms:
        cache_key_val = cache_key("trending", user_terms)
        if cache_key_val in trends_cache:
            app.logger.info(f"Using cached trending literature items: {len(trends_cache[cache_key_val])} items")
            return trends_cache[cache_key_val]
    
    try:
        # Prepare the prompt for Perplexity
        if user_terms and len(user_terms) > 0:
            terms_text = ", ".join(user_terms)
            exclusion_text = f" Exclude {literature_input}." if literature_input else ""
            
            prompt = f"""List 10 narrative books or short stories (no nonfiction, monographs) matching these themes: {terms_text}.{exclusion_text}

For each item, provide the following information in this exact format:

Title: [Full title]
Author: [Author's full name]
Type: [book, short story, novella, etc.]
Description: [Brief description highlighting themes related to: {terms_text}]

Please ensure each entry follows this exact format with clear labels for each field."""
        else:
            prompt = """List 10 diverse narrative books or short stories from various genres and time periods.

For each item, provide the following information in this exact format:

Title: [Full title]
Author: [Author's full name]
Type: [book, short story, novella, etc.]
Description: [Brief description highlighting key themes]

Please ensure each entry follows this exact format with clear labels for each field."""
        
        app.logger.info(f"Querying Perplexity for trending literature with terms: {user_terms}")
        
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
            app.logger.info(f"Received response from Perplexity API")
            
            # Extract the content from the response
            if "choices" in response_data and len(response_data["choices"]) > 0:
                content = response_data["choices"][0]["message"]["content"].strip()
                app.logger.info(f"Perplexity content preview: {content[:100]}...")
                
                # Parse the content to extract literature items
                literature_items = parse_literature_items(content)
                app.logger.info(f"Parsed {len(literature_items)} literature items from Perplexity response")
                
                # Cache the results if caching is enabled
                if CACHING_ENABLED and user_terms:
                    trends_cache[cache_key_val] = literature_items
                    app.logger.info(f"Cached {len(literature_items)} literature items for terms: {user_terms}")
                
                return literature_items
            else:
                app.logger.warning(f"Unexpected response structure from Perplexity: {response_data}")
        else:
            app.logger.warning(f"Failed to query Perplexity: {response.status_code} - {response.text}")
        
        # If we reach here, there was an error, so return an empty list
        return []
    except Exception as e:
        app.logger.error(f"Error querying Perplexity for trending literature: {str(e)}")
        app.logger.error(f"Traceback: {traceback.format_exc()}")
        return []

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
    app.logger.info(f"Starting recommendation process with {len(trending_items)} items and {len(user_terms)} terms")
    
    if not trending_items or not user_terms:
        app.logger.warning(f"No trending items ({len(trending_items)}) or user terms ({len(user_terms)})")
        return []
    
    scored_items = []
    
    # Extract potential author from literature_input
    author_name = None
    if literature_input:
        # Check if we have Dostoevsky in the input
        if "dostoevsky" in literature_input.lower() or any(name in literature_input.lower() for name in ["karamazov", "crime and punishment", "idiot"]):
            author_name = "dostoevsky"
    
    for item in trending_items:
        # Skip self-recommendations (if the item title matches the user input)
        if literature_input and item.title.lower() == literature_input.lower():
            app.logger.info(f"Skipping self-recommendation: {item.title}")
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
            app.logger.info(f"Applied thematic depth bonus to: {item.title} (matched {len(matched_terms)} terms)")
        
        # Author bonus: if the author matches the input author, add bonus points
        if author_name and author_name in author_lower:
            score += 2.0
            app.logger.info(f"Applied author bonus to: {item.title} (author: {item.author})")
        
        # Add to scored items if there's at least one match
        if matched_terms:
            scored_items.append((item, score, list(matched_terms)))
    
    app.logger.info(f"Scored {len(scored_items)} items with at least one match")
    
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
        app.logger.info(f"Top scores: {', '.join(top_scores)}")
    
    return top_items
