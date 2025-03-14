import os
import json
import time
import logging
import random
import re
import hashlib
import uuid
import sqlite3
import traceback
from datetime import datetime, timedelta
from functools import wraps
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional

from flask import (
    Flask, request, jsonify, render_template,
    redirect, url_for, session, make_response, flash
)
from cachetools import TTLCache
import requests
from dotenv import load_dotenv

# Import local modules directly
from LiteratureDiscovery.literature_logic import (
    get_user_preferences, 
    get_trending_literature,
    get_literary_trends, 
    recommend_literature, 
    store_user_input, 
    init_db,
    get_recommendations,
    get_book_by_goodreads_id,
    save_book,
    manage_bookshelves,
    update_book_status,
    remove_saved_book,
    get_my_books,
    sync_saved_books_to_reading_list,
    recommendations_cache
)

# Import book details functionality
from LiteratureDiscovery.book_routes import register_book_routes
from LiteratureDiscovery.book_details import extend_db_schema, book_cache, recs_cache

# Import models and database
from LiteratureDiscovery.models import LiteratureItem
from LiteratureDiscovery.database import get_user_history

# Load environment variables
load_dotenv()

# Configure the application
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24).hex())  # Add secret key for session management
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

# Initialize API clients
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = None

# Initialize OpenAI client
try:
    # IMPORTANT: DO NOT CHANGE THIS CONFIGURATION WITHOUT EXPLICIT PERMISSION
    # This specific implementation is required for compatibility with OpenAI 0.28
    import openai
    openai.api_key = OPENAI_API_KEY
    openai_client = openai
    app.logger.info("OpenAI client initialized successfully using module-level approach")
except Exception as e:
    app.logger.error(f"Error initializing OpenAI client: {str(e)}")
    openai_client = None

# Initialize Perplexity API client
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

# Global variables
GROK_API_KEY = os.getenv("GROK_API_KEY")
CACHING_ENABLED = bool(os.getenv("ENABLE_CACHING", "True").lower() == "true")
FRONTEND_ENABLED = bool(os.getenv("ENABLE_FRONTEND", "True").lower() == "true")

# Debug logging for API keys
if PERPLEXITY_API_KEY:
    app.logger.info("Perplexity API Key available")
else:
    app.logger.warning("Perplexity API Key not found - recommendations will be limited")
if OPENAI_API_KEY:
    app.logger.info("OpenAI API Key available")
else:
    app.logger.warning("OpenAI API Key not found - theme extraction will be limited")

# Initialize caches if caching is enabled
if CACHING_ENABLED:
    # Cache for trending literature (1 hour TTL)
    trends_cache = TTLCache(maxsize=100, ttl=3600)
    
    # Cache for user preferences (24 hours TTL)
    prefs_cache = TTLCache(maxsize=50, ttl=86400)
    
    app.logger.info("Caching enabled for Luminaria")
else:
    app.logger.info("Caching disabled for Luminaria")

# Common stopwords to filter out from user input
STOPWORDS = {
    "a", "an", "the", "and", "but", "or", "for", "nor", "on", "at", "to", "from", 
    "by", "about", "like", "through", "over", "before", "between", "after", "since", 
    "without", "under", "within", "along", "following", "across", "behind", "beyond", 
    "plus", "except", "but", "up", "out", "around", "down", "off", "above", "near",
    "book", "novel", "story", "literature", "literary", "fiction", "nonfiction", 
    "read", "reading", "author", "writer", "books", "novels", "stories", "poem", 
    "poetry", "essay", "articles", "text", "publication", "publish", "published"
}

def cache_key(prefix, *args):
    """Generate a cache key from the prefix and arguments."""
    key_parts = [prefix]
    for arg in args:
        if isinstance(arg, (list, tuple, set)):
            key_parts.extend(sorted(str(x).lower() for x in arg))
        else:
            key_parts.append(str(arg).lower())
    
    # Create a hash of the key parts
    key_string = "_".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()

def query_openai_for_themes(text: str) -> List[str]:
    """
    Use OpenAI to extract themes from user input.
    
    Args:
        text: The text to extract themes from
        
    Returns:
        List of extracted themes
    """
    if not openai_client:
        app.logger.error("OpenAI client not initialized")
        return []
    
    try:
        app.logger.info(f"Querying OpenAI for themes from: '{text}'")
        
        # IMPORTANT: DO NOT CHANGE THIS API CALL PATTERN WITHOUT EXPLICIT PERMISSION
        # This specific implementation is required for compatibility with OpenAI 0.28
        response = openai_client.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a literary expert helping extract themes from user input."},
                {"role": "user", "content": f"""
                Extract 3-5 key literary themes or interests from this text: "{text}"
                
                Return ONLY a comma-separated list of single words or short phrases (2-3 words max).
                Example: mystery, psychological thriller, redemption
                
                DO NOT include any explanations, headers, or additional text.
                """}
            ],
            temperature=0.3,
            max_tokens=100
        )
        
        # Extract the content from the response
        if hasattr(response, 'choices') and len(response.choices) > 0:
            content = response.choices[0].message.content.strip()
            app.logger.info(f"OpenAI returned themes: {content}")
            
            # Split by commas and clean up
            themes = [theme.strip() for theme in content.split(',')]
            return themes
        else:
            app.logger.error("Unexpected response structure from OpenAI")
            return []
    except Exception as e:
        app.logger.error(f"Error querying OpenAI API: {str(e)}")
        return []

def query_perplexity_about_literature(input_text: str, user_terms: List[str] = None) -> Optional[str]:
    """
    Query the Perplexity API to get more context about the literature input.
    
    Args:
        input_text: The user's literature input
        user_terms: Optional list of terms already extracted from the input
        
    Returns:
        The Perplexity API response text, or None if there was an error
    """
    if not PERPLEXITY_API_KEY:
        app.logger.warning("Perplexity API key not configured for preference analysis")
        return None
    
    try:
        # Prepare the prompt for Perplexity
        terms_text = ", ".join(user_terms) if user_terms else ""
        prompt = f"""What are key themes and elements of "{input_text}"? 
        {f"Consider these themes: {terms_text}. " if terms_text else ""}
        Summarize in 2-3 sentences focusing on specific literary themes, genres, and elements."""
        
        app.logger.info(f"Querying Perplexity about: '{input_text}'")
        
        # Prepare the headers
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Prepare the data
        data = {
            "model": "sonar",  # Use the correct model name
            "messages": [
                {"role": "system", "content": "You are a literary expert specializing in book recommendations."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }
        
        # Make the API call
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=data
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

def get_trending_literature(user_terms: List[str] = None) -> List[LiteratureItem]:
    """
    Use Perplexity API to search for trending literature across categories.
    
    Args:
        user_terms: Optional list of terms to focus the search
        
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
            prompt = f"""List 5 books, papers, or poems matching these themes: {terms_text}.
            For each item, provide the following information in this exact format:
            
            Title: [Full title]
            Author: [Author's full name]
            Type: [book, poem, essay, etc.]
            Description: [Brief description highlighting themes related to: {terms_text}]
            
            Please ensure each entry follows this exact format with clear labels for each field.
            """
        else:
            prompt = """List 5 diverse trending literary works from various genres and time periods.
            For each item, provide the following information in this exact format:
            
            Title: [Full title]
            Author: [Author's full name]
            Type: [book, poem, essay, etc.]
            Description: [Brief description highlighting key themes]
            
            Please ensure each entry follows this exact format with clear labels for each field.
            """
        
        app.logger.info(f"Querying Perplexity for trending literature with terms: {user_terms}")
        
        # IMPORTANT: DO NOT CHANGE THIS API CONFIGURATION WITHOUT EXPLICIT PERMISSION
        # The model name "sonar" has been tested and confirmed working
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

def parse_literature_items(text: str) -> List[LiteratureItem]:
    """
    Parse the response from Perplexity API into LiteratureItem objects.
    
    Args:
        text: The text response from Perplexity
        
    Returns:
        List of LiteratureItem objects
    """
    app.logger.info("Parsing literature items from text")
    
    # Clean up markdown formatting that might interfere with parsing
    text = text.replace('**', '')
    
    items = []
    current_item = None
    
    # Look for items with clear Title: Author: Type: Description: format
    title_pattern = re.compile(r'Title:\s*(.+?)(?:\n|$)', re.IGNORECASE)
    author_pattern = re.compile(r'Author:\s*(.+?)(?:\n|$)', re.IGNORECASE)
    type_pattern = re.compile(r'Type:\s*(.+?)(?:\n|$)', re.IGNORECASE)
    desc_pattern = re.compile(r'Description:\s*(.+?)(?:\n\n|$)', re.IGNORECASE | re.DOTALL)
    
    # Split text by numbered items or double newlines
    sections = re.split(r'\n\s*\d+\.|\n\n+', text)
    
    for section in sections:
        if not section.strip():
            continue
            
        # Extract fields using regex patterns
        title_match = title_pattern.search(section)
        author_match = author_pattern.search(section)
        type_match = type_pattern.search(section)
        desc_match = desc_pattern.search(section)
        
        # If we found a title, create a new item
        if title_match:
            title = title_match.group(1).strip()
            
            # Create a new literature item
            item = LiteratureItem(
                title=title,
                author=author_match.group(1).strip() if author_match else "Unknown Author",
                description=desc_match.group(1).strip() if desc_match else section.strip(),
                item_type=type_match.group(1).strip().lower() if type_match else "book"
            )
            
            items.append(item)
            app.logger.info(f"Parsed item: {item.title}")
    
    # If we couldn't parse any items with the structured approach, try a fallback approach
    if not items:
        app.logger.warning("Structured parsing failed, trying fallback approach")
        
        # Try to find numbered items (1., 2., etc.)
        numbered_items = re.split(r'\n\s*\d+\.', text)
        
        for item_text in numbered_items:
            if not item_text.strip():
                continue
                
            # Try to extract a title from the first line
            lines = item_text.strip().split('\n')
            title = lines[0].strip()
            
            if title:
                # Create a new literature item
                item = LiteratureItem(
                    title=title,
                    author="Unknown Author",
                    description=item_text.strip(),
                    item_type="book"
                )
                
                items.append(item)
                app.logger.info(f"Parsed item using fallback: {item.title}")
    
    # If we still couldn't parse any items, create a dummy item with the entire text
    if not items and text.strip():
        app.logger.warning("All parsing approaches failed, creating dummy item")
        
        dummy_item = LiteratureItem(
            title="Literature Recommendations",
            author="AI Assistant",
            description=text,
            item_type="recommendation"
        )
        
        items.append(dummy_item)
        app.logger.info("Created dummy item with full text")
    
    app.logger.info(f"Parsed {len(items)} literature items")
    return items

def extract_terms_from_text(text: str) -> List[str]:
    """
    Extract meaningful terms from text by splitting and filtering stopwords.
    
    Args:
        text: The text to extract terms from
        
    Returns:
        List of extracted terms
    """
    # Define additional stopwords
    additional_stopwords = {
        "also", "prominent", "pursue", "character", "theme", "plot", "narrative",
        "chapter", "page", "write", "written", "work", "reader"
    }
    
    # Combine with existing stopwords
    all_stopwords = STOPWORDS.union(additional_stopwords)
    
    # Clean the text
    cleaned_text = re.sub(r'[^\w\s-]', ' ', text.lower())
    
    # Extract words and simple phrases
    words = re.findall(r'\b\w+\b', cleaned_text)
    
    # Filter out stopwords and short words
    filtered_words = [word for word in words if word not in all_stopwords and len(word) > 2]
    
    # Get unique terms
    unique_terms = list(set(filtered_words))
    
    # Sort by length (prefer longer, more specific terms)
    unique_terms.sort(key=len, reverse=True)
    
    return unique_terms[:10]  # Limit to 10 terms

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
            prompt = f"""
            Analyze this literature input: {combined_input}
            
            Extract 5-7 specific literary themes, genres, or styles (e.g., 'moral dilemma', 'psychological depth'). 
            Avoid generic terms like 'book', 'novel', 'literary'.
            
            Return as a comma-separated list.
            """
            
            # Call the OpenAI API using the module-level approach
            # This works with OpenAI 1.12.0
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
            
            # Filter out stopwords and short terms
            cleaned_terms = []
            for term in terms:
                term = term.strip().lower()
                # Check if any word in the term is a stopword
                term_words = term.split()
                if all(word not in STOPWORDS for word in term_words) and len(term) > 2:
                    cleaned_terms.append(term)
            
            # Limit to 5-7 terms
            if len(cleaned_terms) > 7:
                cleaned_terms = cleaned_terms[:7]
            
            app.logger.info(f"Extracted literary terms: {cleaned_terms}")
            
            # Try to get additional context from Perplexity
            perplexity_response = query_perplexity_about_literature(combined_input, cleaned_terms)
            if perplexity_response:
                context_description = perplexity_response
                
                # Extract additional terms from Perplexity response
                additional_terms = extract_terms_from_text(perplexity_response)
                
                # Add new terms that aren't already in cleaned_terms
                for term in additional_terms:
                    if term not in cleaned_terms and len(cleaned_terms) < 7:
                        cleaned_terms.append(term)
            
            # Cache the result if caching is enabled
            if CACHING_ENABLED:
                prefs_cache[cache_key_val] = (cleaned_terms, context_description)
            
            if cleaned_terms:
                return cleaned_terms, context_description, history
            
        except Exception as e:
            app.logger.error(f"Error querying OpenAI API: {str(e)}")
    
    # Fallback: Basic term extraction from combined input
    app.logger.info("Using fallback term extraction from combined input")
    terms = extract_terms_from_text(combined_input)
    app.logger.info(f"Extracted basic terms: {terms}")
    
    # Cache the result if caching is enabled
    if CACHING_ENABLED:
        prefs_cache[cache_key_val] = (terms, None)
    
    return terms, None, history

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
        # Add other author detections as needed
    
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
    
    # Return top 5 items
    top_items = scored_items[:5]
    
    # Log top scores for debugging
    if top_items:
        top_scores = [f"{item[0].title[:20]}... ({item[1]})" for item in top_items[:3]]
        app.logger.info(f"Top scores: {', '.join(top_scores)}")
    
    return top_items

@app.route('/', methods=['GET'])
def home():
    """
    Homepage route that renders the index.html template with the form for user input.
    This serves as the entry point for the web interface.
    """
    return render_template('index.html')


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
        
        # Check if we have cached results for this input
        normalized_input = literature_input.lower().strip()
        memory_cache_key = f"{normalized_input}_{session_id or 'anonymous'}"
        is_cached = memory_cache_key in recs_cache
        
        start_time = time.time()
        
        # Get recommendations using the new combined function
        result = get_recommendations(literature_input, session_id)
        
        processing_time = time.time() - start_time
        app.logger.info(f"Recommendation processing time: {processing_time:.2f} seconds (cached: {is_cached})")
        
        # Check if we got any recommendations
        has_core = len(result.get('core', [])) > 0
        has_trending = len(result.get('trending', [])) > 0
        has_segmented = any(len(items) > 0 for items in result.get('segmented_recommendations', {}).values())
        has_news = len(result.get('news_and_social', [])) > 0
        
        if not (has_core or has_trending or has_segmented or has_news):
            app.logger.warning(f"No recommendations found for: {literature_input}")
            if request.is_json:
                return jsonify({
                    "error": "No recommendations found",
                    "input": literature_input
                }), 404
            
            flash("Sorry, we couldn't find any recommendations for your input. Please try again with different literature.")
            return redirect('/')
        
        # Return the recommendations
        if request.is_json:
            return jsonify({
                "recommendations": {
                    "core": [(item.to_dict(), score, terms) for item, score, terms in result.get('core', [])],
                    "trending": [(item.to_dict(), score, terms) for item, score, terms in result.get('trending', [])],
                    "news_and_social": result.get('news_and_social', []),
                    "segmented_recommendations": {
                        category: [(item.to_dict(), score, terms) for item, score, terms in items]
                        for category, items in result.get('segmented_recommendations', {}).items()
                    }
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
                "trending": result.get('trending', []),
                "terms": result.get('terms', []),
                "context_description": result.get('context_description'),
                "history": result.get('history', []),
                "news_and_social": result.get('news_and_social', []),
                "segmented_recommendations": result.get('segmented_recommendations', {}),
                "input": literature_input
            },
            cached=is_cached,
            processing_time=processing_time
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
@app.route('/api/trending', methods=['GET'])
def api_trending():
    """
    API endpoint to get trending literature items.
    """
    # Get trending literature items
    trending_items = get_trending_literature()
    
    # Return as JSON
    return jsonify({
        "items": [item.to_dict() for item in trending_items]
    })

@app.route('/book/<goodreads_id>')
def book_details(goodreads_id):
    """
    Display detailed information about a book.
    """
    try:
        # Get session ID for tracking
        session_id = session.get('session_id', str(uuid.uuid4()))
        session['session_id'] = session_id
        
        # Get book information from the database
        conn = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_history.db"))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Check if book is in reading list
        cursor.execute("PRAGMA table_info(user_reading_list)")
        columns = [column[1] for column in cursor.fetchall()]
        
        is_saved = False
        if 'id' in columns:
            cursor.execute("SELECT id FROM user_reading_list WHERE session_id = ? AND goodreads_id = ?",
                        (session_id, goodreads_id))
        else:
            cursor.execute("SELECT rowid FROM user_reading_list WHERE session_id = ? AND goodreads_id = ?",
                        (session_id, goodreads_id))
        
        if cursor.fetchone():
            is_saved = True
        
        # Try to get book details
        title = None
        author = None
        image_url = None
        
        # 1. First try book_covers table
        cursor.execute("SELECT * FROM book_covers WHERE goodreads_id = ?", (goodreads_id,))
        book_data = cursor.fetchone()
        
        if book_data:
            title = book_data['title']
            author = book_data['author']
            image_url = book_data['image_url']
        else:
            # 2. Try book_images table
            cursor.execute("SELECT * FROM book_images WHERE goodreads_id = ?", (goodreads_id,))
            book_images_data = cursor.fetchone()
            
            if book_images_data:
                title = book_images_data['title']
                image_url = book_images_data['image_url']
                author = "Unknown Author"  # Default value
                
                # Store in book_covers for future reference
                cursor.execute(
                    "INSERT OR IGNORE INTO book_covers (goodreads_id, title, author, image_url) VALUES (?, ?, ?, ?)",
                    (goodreads_id, title, author, image_url)
                )
                conn.commit()
        
        # 3. If still no data, try fetching from the module directly (if we have imported the module)
        if not title and 'literature_logic' in globals():
            book_from_logic = literature_logic.get_book_by_goodreads_id(goodreads_id)
            if book_from_logic:
                title = book_from_logic.title
                author = book_from_logic.author
                image_url = book_from_logic.image_url
                
                # Store in book_covers for future reference
                cursor.execute(
                    "INSERT OR IGNORE INTO book_covers (goodreads_id, title, author, image_url) VALUES (?, ?, ?, ?)",
                    (goodreads_id, title, author or "Unknown Author", image_url or "")
                )
                conn.commit()
        
        # 4. Last resort: create placeholder
        if not title:
            title = f"Book {goodreads_id}"
            author = "Unknown Author"
            image_url = url_for('static', filename='images/placeholder-cover.svg')
            
            app.logger.warning(f"Book with goodreads_id {goodreads_id} not found anywhere - using placeholder")
        
        # Get additional book details from Perplexity API
        book_info = get_book_details(title, author, goodreads_id)
        
        # Prepare data for the template
        book = {
            'title': title,
            'author': author,
            'image_url': image_url or url_for('static', filename='images/placeholder-cover.svg'),
            'goodreads_id': goodreads_id,
            'has_valid_goodreads_id': goodreads_id is not None and goodreads_id.isdigit(),
            'description': book_info.get('description', 'No description available.'),
            'publication_date': book_info.get('publication_date', 'Unknown'),
            'genre': book_info.get('genre', 'Unknown'),
            'themes': book_info.get('themes', []),
            'quotes': book_info.get('quotes', []),
            'similar_books': book_info.get('similar_books', []),
            'is_saved': is_saved,
            'match_score': 75  # Add default match score to prevent template error
        }
        
        conn.close()
        
        # Render the book details template
        return render_template('book.html', book=book)
    except Exception as e:
        app.logger.error(f"Error displaying book details: {str(e)}")
        app.logger.error(traceback.format_exc())
        # Still render the template with default info rather than redirecting
        try:
            default_book = {
                'title': f"Book {goodreads_id}",
                'author': "Unknown Author",
                'image_url': url_for('static', filename='images/placeholder-cover.svg'),
                'goodreads_id': goodreads_id,
                'has_valid_goodreads_id': goodreads_id is not None and goodreads_id.isdigit(),
                'description': "Could not retrieve book details due to an error.",
                'publication_date': "Unknown",
                'genre': "Unknown",
                'themes': [],
                'quotes': [],
                'similar_books': [],
                'is_saved': False,
                'match_score': 50  # Add default match score to prevent template error
            }
            return render_template('book.html', book=default_book)
        except Exception as e2:
            app.logger.error(f"Error rendering default book template: {str(e2)}")
            # Last resort
            return redirect('/')

def get_book_details(title, author, goodreads_id=None):
    """
    Get detailed information about a book using the Perplexity API.
    """
    if not PERPLEXITY_API_KEY:
        app.logger.error("Perplexity API key not configured")
        return {}
    
    try:
        # Prepare the prompt for Perplexity
        prompt = f"""Provide detailed information about the book "{title}" by {author}. Include the following information in JSON format:
        {{
            "description": "A detailed description of the book's plot and themes (2-3 paragraphs)",
            "publication_date": "Year of publication",
            "genre": "Primary genre of the book",
            "themes": ["List of 3-5 major themes in the book"],
            "quotes": ["3-4 notable quotes from the book with attribution"],
            "similar_books": [
                {{
                    "title": "Similar Book Title 1",
                    "author": "Author Name",
                    "reason": "Brief reason for recommendation"
                }},
                {{
                    "title": "Similar Book Title 2",
                    "author": "Author Name",
                    "reason": "Brief reason for recommendation"
                }},
                {{
                    "title": "Similar Book Title 3",
                    "author": "Author Name",
                    "reason": "Brief reason for recommendation"
                }}
            ]
        }}
        
        Ensure your response is ONLY valid JSON with no additional text or explanations."""
        
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
                        "content": "You are a literary expert providing detailed information about books in JSON format."
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
        
        if response.status_code != 200:
            app.logger.error(f"Perplexity API error: {response.status_code} - {response.text}")
            return {}
        
        response_data = response.json()
        content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        
        app.logger.info("Received response from Perplexity API for book details")
        app.logger.info(f"Perplexity content preview for book details: {content[:200]}...")
        
        # Clean the content to ensure it's valid JSON
        # Sometimes the API returns markdown code blocks with ```json
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]  # Remove ```json
        if content.endswith("```"):
            content = content[:-3]  # Remove trailing ```
        
        content = content.strip()
        
        try:
            book_info = json.loads(content)
            return book_info
        except json.JSONDecodeError as e:
            app.logger.error(f"Failed to parse JSON response from Perplexity: {content[:200]}...")
            app.logger.error(f"JSON decode error: {str(e)}")
            return {}
            
    except Exception as e:
        app.logger.error(f"Error fetching book details: {str(e)}")
        app.logger.error(traceback.format_exc())
        return {}

@app.route('/reading-list/add', methods=['POST'])
def add_to_reading_list():
    """
    Add a book to the user's reading list.
    """
    try:
        data = request.get_json()
        goodreads_id = data.get('goodreads_id')
        title = data.get('title')
        
        if not goodreads_id or not title:
            return jsonify({'success': False, 'message': 'Missing required parameters'}), 400
        
        # Get session ID
        session_id = session.get('session_id', str(uuid.uuid4()))
        session['session_id'] = session_id
        
        # Connect to database
        conn = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_history.db"))
        cursor = conn.cursor()
        
        # Check if the book is already in the reading list
        cursor.execute("SELECT * FROM user_reading_list WHERE session_id = ? AND goodreads_id = ?", 
                      (session_id, goodreads_id))
        existing = cursor.fetchone()
        
        if existing:
            # Remove from reading list
            cursor.execute("DELETE FROM user_reading_list WHERE session_id = ? AND goodreads_id = ?", 
                          (session_id, goodreads_id))
            conn.commit()
            conn.close()
            return jsonify({'success': True, 'added': False, 'message': 'Book removed from reading list'})
        else:
            # Add to reading list
            cursor.execute("INSERT INTO user_reading_list (session_id, goodreads_id, title, added_at) VALUES (?, ?, ?, ?)", 
                          (session_id, goodreads_id, title, datetime.now().isoformat()))
            conn.commit()
            conn.close()
            return jsonify({'success': True, 'added': True, 'message': 'Book added to reading list'})
    
    except Exception as e:
        app.logger.error(f"Error adding to reading list: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'success': False, 'message': 'An error occurred'}), 500

@app.route('/reading-list', methods=['GET'])
def view_reading_list():
    """
    View the user's reading list.
    """
    try:
        # Get session ID
        session_id = session.get('session_id', str(uuid.uuid4()))
        session['session_id'] = session_id
        
        # Connect to database
        conn = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_history.db"))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get books from reading list
        cursor.execute("SELECT * FROM user_reading_list WHERE session_id = ? ORDER BY added_at DESC", (session_id,))
        reading_list_items = cursor.fetchall()
        
        books = []
        for item in reading_list_items:
            goodreads_id = item['goodreads_id']
            
            # Get book details
            cursor.execute("SELECT * FROM book_covers WHERE goodreads_id = ?", (goodreads_id,))
            book_data = cursor.fetchone()
            
            if book_data:
                books.append({
                    'title': item['title'],
                    'author': book_data['author'] if book_data else 'Unknown',
                    'image_url': book_data['image_url'] if book_data else url_for('static', filename='images/placeholder-cover.svg'),
                    'goodreads_id': goodreads_id,
                    'added_at': item['added_at']
                })
        
        conn.close()
        
        return render_template('reading_list.html', books=books)
    
    except Exception as e:
        app.logger.error(f"Error viewing reading list: {str(e)}")
        app.logger.error(traceback.format_exc())
        flash("An error occurred while retrieving your reading list. Please try again.")
        return redirect('/')

# Initialize the extended database schema for book functionality
def extend_db_schema():
    """
    Extend the database schema to support book functionality.
    """
    try:
        conn = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_history.db"))
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS book_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            goodreads_id TEXT UNIQUE,
            title TEXT,
            image_url TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS book_covers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            goodreads_id TEXT UNIQUE,
            title TEXT,
            author TEXT,
            image_url TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Add user_reading_list table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_reading_list (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            goodreads_id TEXT,
            title TEXT,
            added_at TIMESTAMP,
            UNIQUE(session_id, goodreads_id)
        )
        ''')
        
        # Check if the user_reading_list table exists but has the wrong schema
        cursor.execute("PRAGMA table_info(user_reading_list)")
        columns = [column[1] for column in cursor.fetchall()]
        
        # If the table exists but doesn't have the 'id' column, drop and recreate it
        if columns and 'id' not in columns:
            app.logger.info("Recreating user_reading_list table with correct schema")
            cursor.execute("DROP TABLE user_reading_list")
            cursor.execute('''
            CREATE TABLE user_reading_list (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                goodreads_id TEXT,
                title TEXT,
                added_at TIMESTAMP,
                UNIQUE(session_id, goodreads_id)
            )
            ''')
        
        conn.commit()
        conn.close()
        app.logger.info("Extended database schema with user_reading_list table")
    except Exception as e:
        app.logger.error(f"Error extending database schema: {str(e)}")

# Call the function to extend the database schema
extend_db_schema()

# Register book routes
app = register_book_routes(app)

# Test route for debugging image issues
@app.route('/test_images')
def test_images():
    return render_template('test_images.html')

# My Books feature routes
@app.route('/my_books', methods=['GET'])
def my_books():
    """
    Display the user's saved books, bookshelves, and personalized recommendations.
    """
    try:
        # Get the user's session ID (create one if it doesn't exist)
        session_id = request.cookies.get('session_id')
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Log the session ID for debugging
        app.logger.info(f"My Books route - Using session ID: {session_id}")
        
        # Copy books from saved_books to user_reading_list for backward compatibility
        sync_saved_books_to_reading_list(session_id)
        
        # Get filter parameters
        shelf_filter = request.args.get('shelf')
        status_filter = request.args.get('status')
        
        # Get the user's books and bookshelves
        books, bookshelves, recommendations = get_my_books(session_id, shelf_filter, status_filter)
        
        # Debug log
        app.logger.info(f"Retrieved {len(books)} books for session {session_id}")
        
        # Render the template with the data
        response = make_response(render_template(
            'my_books.html',
            books=books,
            bookshelves=bookshelves,
            recommendations=recommendations,
            current_shelf=shelf_filter,
            current_status=status_filter
        ))
        
        # Set the session ID cookie if it doesn't exist
        if not request.cookies.get('session_id'):
            response.set_cookie('session_id', session_id, max_age=60*60*24*365)  # 1 year
        
        return response
    except Exception as e:
        app.logger.error(f"Error getting my books: {str(e)}")
        app.logger.error(traceback.format_exc())
        flash("An error occurred. Please try again later.")
        return redirect('/')

@app.route('/save_book', methods=['POST'])
def save_book_route():
    """
    Save a book to the user's "My Books" collection.
    """
    try:
        # Get the user's session ID
        session_id = request.cookies.get('session_id')
        if not session_id:
            session_id = str(uuid.uuid4())
            app.logger.info(f"Save Book route - Created new session ID: {session_id}")
        else:
            app.logger.info(f"Save Book route - Using existing session ID: {session_id}")
        
        # Parse the request data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form
        
        # Get the book details
        goodreads_id = data.get('goodreads_id')
        title = data.get('title')
        author = data.get('author')
        image_url = data.get('image_url', '')
        
        if not goodreads_id or not title or not author:
            return jsonify({"error": "Missing required book information"}), 400
        
        # Save the book
        saved_book_id = save_book(session_id, goodreads_id, title, author, image_url)
        
        if saved_book_id:
            response = jsonify({"success": True, "saved_book_id": saved_book_id, "action": "added"})
            
            # Set the session ID cookie if it was newly created
            if not request.cookies.get('session_id'):
                response.set_cookie('session_id', session_id, max_age=60*60*24*365)  # 1 year
                
            return response
        else:
            return jsonify({"error": "Failed to save book"}), 500
    except Exception as e:
        app.logger.error(f"Error saving book: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/update_book_status', methods=['POST'])
def update_book_status_route():
    """
    Update the reading status and/or progress of a saved book.
    """
    try:
        # Get the user's session ID
        session_id = request.cookies.get('session_id')
        if not session_id:
            return jsonify({"error": "Session ID not found"}), 400
        
        # Parse the request data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form
        
        # Get the book details
        saved_book_id = data.get('saved_book_id')
        status = data.get('status')
        progress = data.get('progress')
        
        if not saved_book_id:
            return jsonify({"error": "Missing saved book ID"}), 400
        
        # Convert saved_book_id to int
        try:
            saved_book_id = int(saved_book_id)
        except ValueError:
            return jsonify({"error": "Invalid saved book ID"}), 400
        
        # Convert progress to int if provided
        if progress is not None:
            try:
                progress = int(progress)
            except ValueError:
                return jsonify({"error": "Invalid progress value"}), 400
        
        # Update the book status
        success = update_book_status(session_id, saved_book_id, status, progress)
        
        if success:
            return jsonify({"success": True})
        else:
            return jsonify({"error": "Failed to update book status"}), 500
    except Exception as e:
        app.logger.error(f"Error updating book status: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/manage_bookshelf', methods=['POST'])
def manage_bookshelf_route():
    """
    Add or remove a book from a bookshelf.
    """
    try:
        # Get the user's session ID
        session_id = request.cookies.get('session_id')
        if not session_id:
            return jsonify({"error": "Session ID not found"}), 400
        
        # Parse the request data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form
        
        # Get the request details
        saved_book_id = data.get('saved_book_id')
        shelf_name = data.get('shelf_name')
        action = data.get('action', 'add')  # Default to 'add'
        
        if not saved_book_id or not shelf_name:
            return jsonify({"error": "Missing required parameters"}), 400
        
        # Convert saved_book_id to int
        try:
            saved_book_id = int(saved_book_id)
        except ValueError:
            return jsonify({"error": "Invalid saved book ID"}), 400
        
        # Manage the bookshelf
        success = manage_bookshelves(session_id, saved_book_id, shelf_name, action)
        
        if success:
            return jsonify({"success": True})
        else:
            return jsonify({"error": "Failed to manage bookshelf"}), 500
    except Exception as e:
        app.logger.error(f"Error managing bookshelf: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/remove_book', methods=['POST'])
def remove_book_route():
    """
    Remove a book from the user's saved collection.
    """
    try:
        # Get the user's session ID
        session_id = request.cookies.get('session_id')
        if not session_id:
            return jsonify({"error": "Session ID not found"}), 400
        
        # Parse the request data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form
        
        # Get the book ID
        saved_book_id = data.get('saved_book_id')
        
        if not saved_book_id:
            return jsonify({"error": "Missing saved book ID"}), 400
        
        # Convert saved_book_id to int
        try:
            saved_book_id = int(saved_book_id)
        except ValueError:
            return jsonify({"error": "Invalid saved book ID"}), 400
        
        # Remove the book
        success = remove_saved_book(session_id, saved_book_id)
        
        if success:
            return jsonify({"success": True})
        else:
            return jsonify({"error": "Failed to remove book"}), 500
    except Exception as e:
        app.logger.error(f"Error removing book: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/debug_saved_books', methods=['GET'])
def debug_saved_books():
    """
    Debug route to directly display saved books from the database.
    """
    try:
        # Get the user's session ID
        session_id = request.cookies.get('session_id')
        if not session_id:
            return "No session ID found. Please save a book first."
        
        # Connect to the database
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_history.db")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # This enables column access by name
        cursor = conn.cursor()
        
        # Query saved_books table
        cursor.execute(
            "SELECT * FROM saved_books WHERE session_id = ?",
            (session_id,)
        )
        saved_books = cursor.fetchall()
        
        # Query user_reading_list table
        cursor.execute(
            "SELECT * FROM user_reading_list WHERE session_id = ?",
            (session_id,)
        )
        reading_list = cursor.fetchall()
        
        # Close the connection
        conn.close()
        
        # Prepare HTML output
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Debug Saved Books</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <h1>Debug Saved Books</h1>
            <p>Session ID: {session_id}</p>
            
            <h2>Saved Books Table ({len(saved_books)} books)</h2>
            <table>
                <tr>
                    <th>ID</th>
                    <th>Goodreads ID</th>
                    <th>Title</th>
                    <th>Author</th>
                    <th>Status</th>
                    <th>Progress</th>
                    <th>Added Date</th>
                </tr>
        """
        
        for book in saved_books:
            html += f"""
                <tr>
                    <td>{book['id']}</td>
                    <td>{book['goodreads_id']}</td>
                    <td>{book['title']}</td>
                    <td>{book['author']}</td>
                    <td>{book['status']}</td>
                    <td>{book['progress']}</td>
                    <td>{book['added_date']}</td>
                </tr>
            """
        
        html += f"""
            </table>
            
            <h2>Reading List Table ({len(reading_list)} books)</h2>
            <table>
                <tr>
                    <th>ID</th>
                    <th>Goodreads ID</th>
                    <th>Title</th>
                    <th>Added At</th>
                </tr>
        """
        
        for book in reading_list:
            html += f"""
                <tr>
                    <td>{book['id']}</td>
                    <td>{book['goodreads_id']}</td>
                    <td>{book['title']}</td>
                    <td>{book['added_at']}</td>
                </tr>
            """
        
        html += f"""
            </table>
            
            <p><a href="/my_books">Go to My Books Page</a></p>
        </body>
        </html>
        """
        
        return html
        
    except Exception as e:
        app.logger.error(f"Error in debug_saved_books: {str(e)}")
        app.logger.error(traceback.format_exc())
        return f"Error: {str(e)}"

if __name__ == "__main__":
    # Print startup message
    print("Caching enabled for Luminaria") if CACHING_ENABLED else print("Caching disabled for Luminaria")
    print("Frontend enabled for Luminaria") if FRONTEND_ENABLED else print("Frontend disabled for Luminaria")
    print("Single-input recommendation system active")
    print("Enhanced recommendation quality active")
    print("User history tracking active")
    print("Book details functionality active")
    print("Reading list functionality active")
    print("Luminaria recommendation engine ready!")
    
    # Initialize the database
    init_db()

    # Run the Flask application
    # Parse command line arguments for port
    import argparse
    parser = argparse.ArgumentParser(description='Run the Luminaria Flask application')
    parser.add_argument('--port', type=int, default=5003, help='Port to run the server on')
    args = parser.parse_args()

    # Run the Flask application
    app.run(debug=True, host='0.0.0.0', port=args.port)
