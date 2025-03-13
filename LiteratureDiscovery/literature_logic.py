"""
Literature Logic Module for the LiteratureDiscovery application.

This module contains improved functions for:
1. User history tracking via SQLite
2. Literary preference extraction using OpenAI GPT-3.5
3. Trending literature retrieval using Perplexity API
4. Enhanced recommendation scoring
"""

import os
import re
import uuid
import sqlite3
import logging
import requests
import traceback
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime
from cachetools import TTLCache
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from urllib.parse import quote_plus

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize caches
author_cache = TTLCache(maxsize=100, ttl=3600)  # Cache author lookups for 1 hour

# Database path
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_history.db")

def get_db_connection():
    """Get a connection to the SQLite database."""
    return sqlite3.connect(DB_PATH)

# Configure OpenAI
import openai
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
else:
    logger.warning("OpenAI API key not set in environment variables")

# Configure Perplexity
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
if not PERPLEXITY_API_KEY:
    logger.warning("Perplexity API key not set in environment variables")

# Create caches with TTL (Time To Live)
prefs_cache = TTLCache(maxsize=50, ttl=24*3600)  # 24 hours
trends_cache = TTLCache(maxsize=100, ttl=3600)   # 1 hour

# Define stopwords for filtering
STOPWORDS = {
    "the", "and", "book", "novel", "also", "prominent", "story", "literature", "literary", 
    "fiction", "nonfiction", "read", "reading", "author", "writer", "books", "novels", 
    "stories", "poem", "poetry", "essay", "articles", "text", "publication", "publish", 
    "published", "pursue", "character", "theme", "plot", "narrative", "chapter", "page", 
    "write", "written", "work", "reader", "this", "that", "with", "for", "from", "its",
    "themes", "elements", "style", "about", "genre", "genres", "psychological", "philosophical"
}

# Special cases for known literary works
SPECIAL_CASES = {
    "the brothers karamazov": {
        "terms": [
            "philosophical novel",
            "moral dilemmas",
            "faith and doubt",
            "russian literature",
            "existentialism",
            "family drama"
        ],
        "context": "Themes related to The Brothers Karamazov by Fyodor Dostoevsky"
    },
    "the idiot": {
        "terms": ["existentialism", "moral ambiguity", "russian literature", "19th century", "psychological novel", "dostoevsky"],
        "context": "Dostoevsky's novel exploring themes of innocence, good vs. evil, and human nature through Prince Myshkin's experiences in Russian society."
    },
    "karamazov": {
        "terms": [
            "philosophical novel",
            "existentialism",
            "moral dilemma",
            "religious philosophy",
            "russian literature",
            "19th century literature",
            "dostoevsky",
            "family drama"
        ],
        "context": "Themes related to The Brothers Karamazov by Fyodor Dostoevsky"
    },
    "crime and punishment": {
        "terms": [
            "psychological thriller", 
            "moral dilemma", 
            "redemption", 
            "19th century literature",
            "russian literature", 
            "existentialism", 
            "crime fiction",
            "philosophical novel",
            "dostoevsky"
        ],
        "context": "Themes related to Crime and Punishment by Fyodor Dostoevsky"
    }
}

# Class definition for literature items
class LiteratureItem:
    """Class representing a literature item (book, poem, essay, etc.)
    with its metadata."""
    
    def __init__(self, title: str, author: str, publication_date: str = "", 
                 genre: str = "", description: str = "", item_type: str = "book", 
                 summary: str = "", is_trending: bool = False, image_url: str = "", goodreads_id: str = ""):
        self.title = title
        self.author = author
        self.publication_date = publication_date
        self.genre = genre
        self.description = description
        self.item_type = item_type  # book, poem, essay, etc.
        self.score = 0.0  # Recommendation score
        self.matched_terms = set()  # Terms that matched this item
        self.summary = summary  # 2-3 sentence summary of the work
        self.match_score = 0  # Match score (0-100) indicating how well it matches user input
        self.is_trending = is_trending  # Flag indicating if this is a trending item
        self.image_url = image_url  # URL to the book cover image
        self.goodreads_id = goodreads_id  # Goodreads ID for the book
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "title": self.title,
            "author": self.author,
            "publication_date": self.publication_date,
            "genre": self.genre,
            "description": self.description,
            "item_type": self.item_type,
            "score": self.score,
            "matched_terms": list(self.matched_terms),
            "summary": self.summary,
            "match_score": self.match_score,
            "is_trending": self.is_trending,
            "image_url": self.image_url,
            "goodreads_id": self.goodreads_id
        }

# SQLite Database Functions
def init_db():
    """Initialize the SQLite database for user history tracking."""
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_history.db")
    logger.info(f"Initializing database at {db_path}")
    
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        
        # Create user_history table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            input_text TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create book_covers table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS book_covers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            author TEXT,
            image_url TEXT,
            goodreads_id TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(title, author)
        )
        ''')
        
        conn.commit()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
    finally:
        conn.close()

def store_user_input(session_id: str, literature_input: str):
    """
    Store a user input in the database with timestamp.
    
    Args:
        session_id: User's session ID
        literature_input: The literature input from the user
    """
    if not session_id or not literature_input:
        logger.warning("Cannot store user input: missing session_id or literature_input")
        return
    
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_history.db")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Store the input with current timestamp
        cursor.execute(
            "INSERT INTO user_history (session_id, input_text, timestamp) VALUES (?, ?, ?)",
            (session_id, literature_input, datetime.now())
        )
        conn.commit()
        logger.info(f"Stored user input for session {session_id}: '{literature_input}'")
    except Exception as e:
        logger.error(f"Error storing user input: {str(e)}")
    finally:
        if conn:
            conn.close()

def get_user_history(session_id: str, limit: int = 5) -> List[str]:
    """
    Retrieve the user's recent inputs from the database.
    
    Args:
        session_id: User's session ID
        limit: Maximum number of history items to retrieve
        
    Returns:
        List of the user's recent inputs
    """
    if not session_id:
        logger.warning("Cannot get user history: missing session_id")
        return []

def store_feedback(session_id: str, title: str, feedback: int):
    """
    Store user feedback (thumbs up/down) for a recommendation.
    
    Args:
        session_id: User's session ID
        title: Title of the literature item
        feedback: 1 for thumbs up, -1 for thumbs down
    
    Returns:
        Boolean indicating success
    """
    if not session_id or not title:
        logger.warning("Missing session_id or title, not storing feedback")
        return False
    
    if feedback not in [1, -1]:
        logger.warning(f"Invalid feedback value: {feedback}, must be 1 or -1")
        return False
    
    try:
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_history.db")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Insert or replace the feedback
        cursor.execute('''
        INSERT OR REPLACE INTO user_feedback (session_id, title, feedback, timestamp)
        VALUES (?, ?, ?, ?)
        ''', (session_id, title, feedback, datetime.now()))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Feedback stored for session {session_id}, title: {title}, feedback: {feedback}")
        return True
    except Exception as e:
        logger.error(f"Error storing feedback: {e}")
        traceback.print_exc()
        return False

def get_user_feedback(session_id: str) -> Dict[str, int]:
    """
    Retrieve user feedback for recommendations.
    
    Args:
        session_id: User's session ID
        
    Returns:
        Dictionary mapping title to feedback value (1 or -1)
    """
    if not session_id:
        logger.warning("Missing session_id, not retrieving feedback")
        return {}
    
    try:
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_history.db")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all feedback for this session
        cursor.execute('''
        SELECT title, feedback FROM user_feedback
        WHERE session_id = ?
        ''', (session_id,))
        
        feedback_dict = {row[0]: row[1] for row in cursor.fetchall()}
        
        conn.close()
        
        logger.info(f"Retrieved {len(feedback_dict)} feedback items for session {session_id}")
        return feedback_dict
    except Exception as e:
        logger.error(f"Error retrieving user feedback: {e}")
        traceback.print_exc()
        return {}
    
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_history.db")
    history = []
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get the most recent inputs excluding the current one
        cursor.execute(
            "SELECT input_text FROM user_history WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
            (session_id, limit)
        )
        
        # Extract the inputs
        results = cursor.fetchall()
        history = [result[0] for result in results]
        logger.info(f"Retrieved {len(history)} history items for session {session_id}")
    except Exception as e:
        logger.error(f"Error retrieving user history: {str(e)}")
    finally:
        if conn:
            conn.close()
    
    return history

def cache_key(prefix: str, data) -> str:
    """Generate a cache key from data."""
    if isinstance(data, str):
        key = data
    elif isinstance(data, list):
        key = ",".join(sorted(data))
    else:
        key = str(data)
    
    # Limit key length and normalize
    key = key.lower()[:100]
    return f"{prefix}:{key}"

def extract_terms_from_text(text: str) -> List[str]:
    """
    Extract meaningful terms from text by filtering out stopwords and short words.
    
    Args:
        text: Text to extract terms from
        
    Returns:
        List of unique terms
    """
    if not text:
        return []
    
    # Convert to lowercase and split into words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Filter out stopwords
    terms = [word for word in words if word not in STOPWORDS and len(word) > 2]
    
    # Return unique terms
    return list(set(terms))

def deduplicate_terms(terms: List[str]) -> List[str]:
    """
    Remove duplicate terms and subsets of other terms.
    For example, if we have both "psychological" and "psychological complexity",
    we'll keep only "psychological complexity".
    
    Args:
        terms: List of terms to deduplicate
        
    Returns:
        List of deduplicated terms
    """
    deduplicated_terms = []
    for term in terms:
        # Check if this term is a subset of any other term
        if not any(term != other_term and term in other_term for other_term in terms):
            deduplicated_terms.append(term)
    return deduplicated_terms

def get_user_preferences(literature_input: str, session_id: str = None) -> Tuple[List[str], Optional[str], List[str]]:
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
            completion = openai.ChatCompletion.create(
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
        logger.error("Perplexity API key not configured")
        return None
    
    try:
        # Prepare the prompt for Perplexity
        terms_text = ", ".join(terms) if terms else ""
        
        prompt = f"""Summarize themes of {literature_input} in 2-3 sentences, focusing on literary elements.
        
If you recognize this as a specific work, please include the author's name and any relevant literary movement or time period.

Focus on themes, style, and genre rather than plot summary."""
        
        logger.info(f"Querying Perplexity about literature: '{literature_input}'")
        
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
                logger.info(f"Perplexity response: {content[:100]}...")
                return content
            else:
                logger.warning(f"Unexpected response structure from Perplexity: {response_data}")
                return None
        else:
            logger.warning(f"Failed to query Perplexity: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"Error querying Perplexity for preference analysis: {str(e)}")
        return None


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
            
            prompt = f"""List 5 classic literary works (books, novels, short stories) that match these themes: {terms_text}. Choose diverse works from different time periods and authors, focusing on established literary classics. For each item, provide the following information in this exact format: Title: [Full title] Author: [Author's full name] Type: [novel, short story, novella, etc.] Description: [Brief description highlighting themes related to: {terms_text}]. Please ensure each entry follows this exact format with clear labels for each field."""
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
                literature_items = parse_literature_items(content, is_trending=True)
                logger.info(f"Parsed {len(literature_items)} classic literature items from Perplexity response")
                
                # Filter out items that match the user's input (if provided)
                if literature_input:
                    literature_input_lower = literature_input.lower()
                    filtered_items = []
                    for item in literature_items:
                        if (literature_input_lower != item.title.lower() and 
                            literature_input_lower not in item.title.lower() and 
                            literature_input_lower not in item.description.lower()):
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

def get_literary_trends(user_terms: List[str] = None, literature_input: str = None) -> List[LiteratureItem]:
    """
    Use Perplexity API to search for trending recent literature across categories.
    
    Args:
        user_terms: Optional list of terms to focus the search
        literature_input: Original user input to exclude from recommendations
        
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
                literature_items = parse_literature_items(content, is_trending=True)
                logger.info(f"Parsed {len(literature_items)} trending literature items from Perplexity response")
                
                # Filter out items that match the user's input (if provided)
                if literature_input:
                    literature_input_lower = literature_input.lower()
                    filtered_items = []
                    for item in literature_items:
                        if (literature_input_lower != item.title.lower() and
                            literature_input_lower not in item.title.lower()):
                            filtered_items.append(item)
                        else:
                            logger.info(f"Filtered out trending literature item that matched user input: {item.title}")
                    literature_items = filtered_items
                
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

def parse_literature_items(text: str, is_trending: bool = False) -> List[LiteratureItem]:
    """
    Parse literature items from text response.
    
    Args:
        text: Text response from the API
        is_trending: Flag indicating if these are trending items
        
    Returns:
        List of LiteratureItem objects
    """
    if not text:
        return []
    
    items = []
    
    # Try to extract structured data
    sections = re.split(r'\n\s*\d+\.\s+', text)
    if len(sections) > 1:
        # Remove the introduction text
        sections = sections[1:]
        
        for section in sections:
            # Extract title
            title_match = re.search(r'(?:Title|Book):\s*([^\n]+)', section, re.IGNORECASE)
            title = title_match.group(1).strip() if title_match else "Unknown Title"
            
            # Extract author
            author_match = re.search(r'Author:\s*([^\n]+)', section, re.IGNORECASE)
            author = author_match.group(1).strip() if author_match else "Unknown Author"
            
            # Extract description
            desc_match = re.search(r'(?:Description|Summary):\s*([^\n]+(?:\n[^\n]+)*)', section, re.IGNORECASE)
            description = desc_match.group(1).strip() if desc_match else section.strip()
            
            # Extract type
            type_match = re.search(r'Type:\s*([^\n]+)', section, re.IGNORECASE)
            item_type = type_match.group(1).strip().lower() if type_match else "book"
            
            # Look for Goodreads ID in the description
            goodreads_id = None
            goodreads_match = re.search(r'Goodreads ID:\s*(\d+)', section, re.IGNORECASE)
            if goodreads_match:
                goodreads_id = goodreads_match.group(1)
            
            # Get book cover image URL
            image_url, goodreads_id = get_book_cover(title, author, goodreads_id)
            
            literature_item = LiteratureItem(
                title=title,
                author=author,
                description=description,
                item_type=item_type,
                is_trending=is_trending,
                image_url=image_url,
                goodreads_id=goodreads_id
            )
            
            items.append(literature_item)
    else:
        # Try to extract items from unstructured text
        # Split by newlines and look for patterns
        lines = text.split('\n')
        current_item_text = ""
        
        for line in lines:
            line = line.strip()
            
            # Check if this line starts a new item
            if re.match(r'^\d+\.\s+', line):
                # Process the previous item if it exists
                if current_item_text:
                    # Extract title from the first line
                    title_match = re.search(r'^\d+\.\s+([^:]+)', current_item_text)
                    title = title_match.group(1).strip() if title_match else "Unknown Title"
                    
                    # Get book cover image URL
                    image_url, goodreads_id = get_book_cover(title)
                    
                    # Create a new literature item
                    item = LiteratureItem(
                        title=title,
                        author="Unknown Author",
                        description=current_item_text.strip(),
                        item_type="book",
                        is_trending=is_trending,
                        image_url=image_url,
                        goodreads_id=goodreads_id
                    )
                    
                    items.append(item)
                
                # Start a new item
                current_item_text = line
            else:
                # Continue the current item
                current_item_text += "\n" + line
        
        # Process the last item
        if current_item_text:
            title_match = re.search(r'^\d+\.\s+([^:]+)', current_item_text)
            title = title_match.group(1).strip() if title_match else "Unknown Title"
            
            # Get book cover image URL
            image_url, goodreads_id = get_book_cover(title)
            
            item = LiteratureItem(
                title=title,
                author="Unknown Author",
                description=current_item_text.strip(),
                item_type="book",
                is_trending=is_trending,
                image_url=image_url,
                goodreads_id=goodreads_id
            )
            
            items.append(item)
    
    return items



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
                author = re.sub(r'^["\']|["\']$', '', author)
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

def recommend_literature(trending_items: List[LiteratureItem], user_terms: List[str], literature_input: str = None, session_id: str = None) -> List[Tuple[LiteratureItem, float, List[str]]]:
    """
    Score and recommend literature items based on user terms.
    
    Args:
        trending_items: List of LiteratureItem objects to score
        user_terms: List of user preference terms
        literature_input: Original user input to avoid self-recommendation
        session_id: Optional session ID for retrieving user feedback
        
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
            literature_input_lower == item.title.lower() or 
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
            # Convert normalized score to a 0-100 scale for display
            item.match_score = int(min(normalized_score * 100, 100))
            recommendations.append((item, normalized_score, matched_terms))
    
    # Sort by score (descending)
    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    return recommendations

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
    trending_items = get_literary_trends(user_terms, literature_input)
    logger.info(f"Retrieved {len(trending_items)} trending recent literature items")
    
    # Score and rank the recommendations
    core_recommendations = recommend_literature(classic_items, user_terms, literature_input, session_id)
    trending_recommendations = recommend_literature(trending_items, user_terms, literature_input, session_id)
    
    logger.info(f"Generated {len(core_recommendations)} core recommendations")
    logger.info(f"Generated {len(trending_recommendations)} trending recommendations")
    
    # Return both sets of recommendations
    return {
        "core": core_recommendations[:5],  # Limit to top 5
        "trending": trending_recommendations[:5],  # Limit to top 5
        "terms": user_terms,
        "context_description": context_desc,
        "history": history_used,
        "input": literature_input  # Include the original input
    }

def get_book_cover(title: str, author: str = "", goodreads_id: str = None) -> Tuple[str, str]:
    """
    Get the book cover image URL for a given book title.
    
    Args:
        title: The title of the book
        author: Optional author name
        goodreads_id: Optional Goodreads ID
        
    Returns:
        Tuple of (image_url, goodreads_id)
    """
    # Clean up the title
    title = title.strip()
    if not title:
        return "", ""
    
    # Check if we already have this book cover in the database
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # If we have a Goodreads ID, use it to look up the book
    if goodreads_id:
        cursor.execute("SELECT image_url FROM book_covers WHERE goodreads_id = ?", (goodreads_id,))
        result = cursor.fetchone()
        if result and result[0]:
            logger.info(f"Found image URL in database for Goodreads ID: {goodreads_id}")
            conn.close()
            return result[0], goodreads_id
    
    # Try to find by title and author
    if author:
        cursor.execute("SELECT image_url, goodreads_id FROM book_covers WHERE title = ? AND author = ?", (title, author))
    else:
        cursor.execute("SELECT image_url, goodreads_id FROM book_covers WHERE title = ?", (title,))
    
    result = cursor.fetchone()
    if result and result[0]:
        logger.info(f"Found image URL in legacy table for '{title}'")
        conn.close()
        return result[0], result[1] if len(result) > 1 and result[1] else ""
    
    conn.close()
    
    # If we have a Goodreads ID but no image URL, construct a high-quality image URL
    if goodreads_id:
        image_url = f"https://i.gr-assets.com/images/S/compressed.photo.goodreads.com/books//{goodreads_id}._SX318_.jpg"
        
        # Store in database for future use
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO book_covers (title, author, image_url, goodreads_id) VALUES (?, ?, ?, ?)",
            (title, author, image_url, goodreads_id)
        )
        conn.commit()
        conn.close()
        
        return image_url, goodreads_id
    
    # If we don't have the cover, try to scrape it from Goodreads
    logger.info(f"Scraping data for '{title}' from Goodreads")
    
    try:
        # Construct a search URL for Goodreads
        search_query = f"{title} {author}".strip()
        search_url = f"https://www.goodreads.com/search?q={quote_plus(search_query)}"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(search_url, headers=headers)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the first book result
            book_item = soup.select_one('tr.bookTitle')
            
            if book_item:
                # Extract the book cover image
                img_tag = book_item.select_one('img.bookCover')
                if img_tag and 'src' in img_tag.attrs:
                    image_url = img_tag['src']
                    
                    # Extract the Goodreads ID from the URL if possible
                    book_link = book_item.select_one('a.bookTitle')
                    if book_link and 'href' in book_link.attrs:
                        book_url = book_link['href']
                        id_match = re.search(r'/show/(\d+)', book_url)
                        if id_match:
                            extracted_goodreads_id = id_match.group(1)
                            
                            # Store in database for future use
                            conn = get_db_connection()
                            cursor = conn.cursor()
                            cursor.execute(
                                "INSERT OR REPLACE INTO book_covers (title, author, image_url, goodreads_id) VALUES (?, ?, ?, ?)",
                                (title, author, image_url, extracted_goodreads_id)
                            )
                            conn.commit()
                            conn.close()
                            
                            logger.info(f"Saved book data for '{title}' with Goodreads ID: {extracted_goodreads_id}")
                            
                            return image_url, extracted_goodreads_id
                    
                    # If we couldn't extract a Goodreads ID, just save the image URL
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT OR REPLACE INTO book_covers (title, author, image_url) VALUES (?, ?, ?)",
                        (title, author, image_url)
                    )
                    conn.commit()
                    conn.close()
                    
                    return image_url, ""
    
    except Exception as e:
        logger.error(f"Error scraping Goodreads for '{title}': {str(e)}")
    
    # If all else fails, return empty strings
    return "", ""

def store_book_metadata(title: str, author: str, goodreads_id: str = None, image_url: str = None) -> bool:
    """
    Store book metadata in the database.
    
    Args:
        title: Book title
        author: Book author
        goodreads_id: Optional Goodreads ID
        image_url: Optional image URL
        
    Returns:
        True if successful, False otherwise
    """
    if not title:
        return False
        
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if we already have this book
        if goodreads_id:
            cursor.execute("SELECT id FROM book_covers WHERE goodreads_id = ?", (goodreads_id,))
        else:
            cursor.execute("SELECT id FROM book_covers WHERE title = ? AND author = ?", (title, author))
            
        result = cursor.fetchone()
        
        if result:
            # Update existing record
            if goodreads_id and image_url:
                cursor.execute(
                    "UPDATE book_covers SET image_url = ?, goodreads_id = ? WHERE id = ?",
                    (image_url, goodreads_id, result[0])
                )
            elif goodreads_id:
                cursor.execute(
                    "UPDATE book_covers SET goodreads_id = ? WHERE id = ?",
                    (goodreads_id, result[0])
                )
            elif image_url:
                cursor.execute(
                    "UPDATE book_covers SET image_url = ? WHERE id = ?",
                    (image_url, result[0])
                )
        else:
            # Insert new record
            cursor.execute(
                "INSERT INTO book_covers (title, author, image_url, goodreads_id) VALUES (?, ?, ?, ?)",
                (title, author, image_url, goodreads_id)
            )
            
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error storing book metadata: {str(e)}")
        return False

def get_book_by_goodreads_id(goodreads_id: str) -> Optional[LiteratureItem]:
    """
    Get a book by its Goodreads ID.
    
    Args:
        goodreads_id: Goodreads ID
        
    Returns:
        LiteratureItem or None if not found
    """
    if not goodreads_id:
        return None
        
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT title, author, image_url FROM book_covers WHERE goodreads_id = ?",
            (goodreads_id,)
        )
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return LiteratureItem(
                title=result[0],
                author=result[1],
                image_url=result[2],
                goodreads_id=goodreads_id
            )
        return None
    except Exception as e:
        logger.error(f"Error getting book by Goodreads ID: {str(e)}")
        return None

def get_book_by_title(title: str, author: str = None) -> Optional[LiteratureItem]:
    """
    Get a book by its title and optional author.
    
    Args:
        title: Book title
        author: Optional author name
        
    Returns:
        LiteratureItem or None if not found
    """
    if not title:
        return None
        
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if author:
            cursor.execute(
                "SELECT title, author, image_url, goodreads_id FROM book_covers WHERE title = ? AND author = ?",
                (title, author)
            )
        else:
            cursor.execute(
                "SELECT title, author, image_url, goodreads_id FROM book_covers WHERE title = ?",
                (title,)
            )
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return LiteratureItem(
                title=result[0],
                author=result[1],
                image_url=result[2],
                goodreads_id=result[3] if len(result) > 3 else None
            )
        return None
    except Exception as e:
        logger.error(f"Error getting book by title: {str(e)}")
        return None

def test_recommendations(input_text="the brothers karamazov", session_id="test"):
    """
    Test function to check recommendation quality for a given input.
    
    Args:
        input_text: Text to test (e.g., 'the brothers karamazov')
        session_id: Session ID to use for testing
    """
    print(f"\nTesting recommendations for: '{input_text}'")
    
    # Store the input
    store_user_input(session_id, input_text)
    
    # Get user preferences
    user_terms, context, history = get_user_preferences(input_text, session_id)
    
    print(f"\nExtracted terms: {user_terms}")
    print(f"Context: {context}")
    print(f"History: {history}")
    
    # Get trending literature
    trending_items = get_trending_literature(user_terms, input_text)
    
    print(f"\nFound {len(trending_items)} trending items")
    
    # Recommend literature
    recommendations = recommend_literature(trending_items, user_terms, input_text)
    
    # Print recommendations
    print(f"\nTop recommendations:")
    for i, (item, score, matched_terms) in enumerate(recommendations, 1):
        print(f"{i}. {item.title} by {item.author} (Score: {score})")
        print(f"   Type: {item.item_type}")
        print(f"   Matched terms: {', '.join(matched_terms)}")
        print(f"   Description: {item.description[:100]}...")
        print()


def submit_feedback(session_id: str, title: str, feedback: int):
    """
    Submit user feedback for a recommendation.
    
    Args:
        session_id: User's session ID
        title: Title of the literature item
        feedback: 1 for thumbs up, -1 for thumbs down
        
    Returns:
        Boolean indicating success
    """
    return store_feedback(session_id, title, feedback)

if __name__ == "__main__":
    # Initialize the database
    init_db()
    
    # Test recommendations
    test_recommendations("the brothers karamazov", "test_session")
    
    # Test with another input to show history blending
    test_recommendations("the idiot", "test_session")
