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
import threading
from datetime import timedelta

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize caches
author_cache = TTLCache(maxsize=100, ttl=3600)  # Cache author lookups for 1 hour
recommendations_cache = TTLCache(maxsize=100, ttl=3600*24*7)  # Cache recommendations for 7 days instead of 1 hour

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
                 summary: str = "", is_trending: bool = False, image_url: str = "", goodreads_id: str = "",
                 status: str = "to_read", progress: int = 0, shelves: list = None, saved_id: int = None):
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
        # Add a property to check if goodreads_id is valid for use in actual Goodreads links
        self.has_valid_goodreads_id = goodreads_id is not None and goodreads_id.isdigit()
        self.status = status
        self.progress = progress
        self.shelves = shelves
        self.saved_id = saved_id
    
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
            "goodreads_id": self.goodreads_id,
            "has_valid_goodreads_id": self.has_valid_goodreads_id,
            "status": self.status,
            "progress": self.progress,
            "shelves": self.shelves,
            "saved_id": self.saved_id
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
        
        # Create book_covers table if it doesn't exist (legacy table)
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
        
        # Drop existing book_images table to recreate with optimized schema
        cursor.execute('DROP TABLE IF EXISTS book_images')
        
        # Create new optimized book_images table with goodreads_id as PRIMARY KEY
        # This improves lookup performance and ensures unique entries per book
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS book_images (
            goodreads_id TEXT PRIMARY KEY,
            title TEXT,
            image_url TEXT NOT NULL,
            local_path TEXT,
            timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Add index on goodreads_id for faster lookups
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_book_images_goodreads_id ON book_images(goodreads_id)')
        
        # Create table for persistent recommendation caching
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS recommendation_cache (
            input_hash TEXT PRIMARY KEY,
            literature_input TEXT NOT NULL,
            results TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Add index for faster lookups
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_recommendation_cache_input_hash ON recommendation_cache(input_hash)')
        
        # Create tables for "My Books" feature - allows users to save, organize, and track reading progress
        
        # Table for storing saved books with reading status and progress
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS saved_books (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            goodreads_id TEXT NOT NULL,
            title TEXT NOT NULL,
            author TEXT NOT NULL,
            status TEXT DEFAULT "to_read",  -- Options: "to_read", "reading", "finished"
            progress INTEGER DEFAULT 0,     -- Percentage read (0-100)
            added_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(session_id, goodreads_id)
        )
        ''')
        
        # Table for custom user-defined bookshelves
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS bookshelves (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            shelf_name TEXT NOT NULL,
            UNIQUE(session_id, shelf_name)
        )
        ''')
        
        # Table for mapping books to bookshelves (many-to-many relationship)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS bookshelf_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            shelf_id INTEGER NOT NULL,
            saved_book_id INTEGER NOT NULL,
            FOREIGN KEY(shelf_id) REFERENCES bookshelves(id) ON DELETE CASCADE,
            FOREIGN KEY(saved_book_id) REFERENCES saved_books(id) ON DELETE CASCADE,
            UNIQUE(shelf_id, saved_book_id)
        )
        ''')
        
        # Add indexes for faster lookups on frequently queried fields
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_saved_books_session_id ON saved_books(session_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_bookshelves_session_id ON bookshelves(session_id)')
        
        # Create directory for local book cover images if it doesn't exist
        book_covers_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "book_covers")
        if not os.path.exists(book_covers_dir):
            os.makedirs(book_covers_dir)
            logger.info(f"Created directory for book covers: {book_covers_dir}")
        
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
            
            # Clean up item_type - remove special characters and asterisks
            item_type = re.sub(r'[*]', '', item_type).strip()
            
            # If type_match is not available, try to infer item_type from description
            if not type_match and desc_match:
                description_lower = desc_match.group(1).lower() if desc_match else ""
                # Check for keywords in the description to determine item type
                if any(keyword in description_lower for keyword in ["poem", "poetry", "verse", "stanza"]):
                    item_type = "poem"
                elif any(keyword in description_lower for keyword in ["paper", "research", "study", "academic", "journal", "article"]):
                    item_type = "paper"
                elif any(keyword in description_lower for keyword in ["novel", "fiction", "story", "narrative"]):
                    item_type = "novel"
            
            # Look for Goodreads ID in the description
            goodreads_id = None
            goodreads_match = re.search(r'Goodreads ID:\s*(\d+)', section, re.IGNORECASE)
            if goodreads_match:
                goodreads_id = goodreads_match.group(1)
            
            # Get book cover image URL - pass goodreads_id first if available
            image_url, goodreads_id = get_book_cover(goodreads_id, title, author)
            
            # Debug logging
            logger.info(f"Book cover for '{title}': image_url={image_url}, goodreads_id={goodreads_id}")
            
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
    
    # Track authors and genres for diversity filtering
    author_counts = {}
    genre_counts = {}
    
    for item in trending_items:
        # Skip if the item matches the user's input too closely (exact title match or contains input author)
        # This prevents recommending the book the user is searching for
        if literature_input_lower and (
            literature_input_lower == item.title.lower() or 
            (input_author and item.author and input_author.lower() in item.author.lower())
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
                # Enhanced scoring with genre boost
                score += 0.5  # Increased from 0.3 to 0.5 to give more weight to genre matches
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
    
    # Apply diversity filter: limit to a maximum of 4 items per author or genre
    filtered_recommendations = []
    for item, score, terms in recommendations:
        author_key = item.author.lower() if item.author else "unknown"
        genre_key = item.genre.lower() if item.genre else "unknown"
        
        # Track counts
        author_counts[author_key] = author_counts.get(author_key, 0) + 1
        genre_counts[genre_key] = genre_counts.get(genre_key, 0) + 1
        
        # Only include if we haven't exceeded the diversity limits
        if author_counts[author_key] <= 4 and genre_counts[genre_key] <= 4:
            filtered_recommendations.append((item, score, terms))
    
    logger.info(f"Applied diversity filter: {len(recommendations)} → {len(filtered_recommendations)} recommendations")
    
    return filtered_recommendations

def get_recommendations(literature_input: str, session_id: str = None) -> Dict:
    """
    Get both core and trending recommendations for a literature input.
    
    Args:
        literature_input: The literature input from the user
        session_id: Optional user session ID for history tracking
        
    Returns:
        Dictionary with core and trending recommendations, news and social updates,
        and segmented recommendations by category
    """
    if not literature_input:
        return {"core": [], "trending": [], "news_and_social": [], "segmented_recommendations": {}}
    
    # Check for cached recommendations (in memory and database)
    cached_results = get_cached_recommendations(literature_input, session_id)
    if cached_results:
        return cached_results
    
    # Track user input if session_id is provided
    if session_id:
        store_user_input(session_id, literature_input)
        logger.info(f"Stored user input for session {session_id}: {literature_input}")
    
    # Get user preferences
    user_terms, context_desc, history_used = get_user_preferences(literature_input, session_id)
    logger.info(f"User terms extracted: {user_terms}")
    
    # Use threading to parallelize the API calls for classic and trending literature
    classic_items = []
    trending_items = []
    news_and_social = []
    classic_error = None
    trending_error = None
    news_error = None
    
    # Define thread functions to get literature items
    def fetch_classic_items():
        nonlocal classic_items, classic_error
        try:
            classic_items = get_classical_literature()
            logger.info(f"Retrieved {len(classic_items)} classic literature items")
        except Exception as e:
            classic_error = str(e)
            logger.error(f"Error retrieving classic literature: {str(e)}")
            classic_items = []
    
    def fetch_trending_items():
        nonlocal trending_items, trending_error
        try:
            trending_items = get_literary_trends(user_terms, literature_input)
            logger.info(f"Retrieved {len(trending_items)} trending recent literature items")
        except Exception as e:
            trending_error = str(e)
            logger.error(f"Error retrieving trending literature: {str(e)}")
            trending_items = []
    
    def fetch_news_and_social():
        nonlocal news_and_social, news_error
        try:
            news_and_social = fetch_search_updates(literature_input)
            logger.info(f"Retrieved {len(news_and_social)} news and social media updates")
        except Exception as e:
            news_error = str(e)
            logger.error(f"Error retrieving news and social updates: {str(e)}")
            news_and_social = []
    
    # Create and start threads
    classic_thread = threading.Thread(target=fetch_classic_items)
    trending_thread = threading.Thread(target=fetch_trending_items)
    news_thread = threading.Thread(target=fetch_news_and_social)
    
    classic_thread.start()
    trending_thread.start()
    news_thread.start()
    
    # Wait for all threads to complete
    classic_thread.join()
    trending_thread.join()
    news_thread.join()
    
    # Log any errors that occurred
    if classic_error:
        logger.error(f"Error in classic literature thread: {classic_error}")
    if trending_error:
        logger.error(f"Error in trending literature thread: {trending_error}")
    if news_error:
        logger.error(f"Error in news and social updates thread: {news_error}")
    
    # Generate recommendations for both literature types
    core_recommendations = recommend_literature(classic_items, user_terms, literature_input, session_id)
    trending_recommendations = recommend_literature(trending_items, user_terms, literature_input, session_id)
    
    logger.info(f"Generated {len(core_recommendations)} core recommendations")
    logger.info(f"Generated {len(trending_recommendations)} trending recommendations")
    
    # Deduplicate recommendations by combining core and trending into a single list
    # keeping only the first occurrence of each goodreads_id
    all_recommendations = []
    seen_ids = set()  # Track both goodreads_ids and title|author combinations
    
    # Process core recommendations first (higher priority)
    for item, score, terms in core_recommendations:
        # Create unique identifiers for deduplication
        goodreads_id = item.goodreads_id if item.goodreads_id else None
        item_key = f"{item.title}|{item.author}".lower()
        
        # Skip if we've seen this item before
        if (goodreads_id and goodreads_id in seen_ids) or (item_key in seen_ids):
            logger.info(f"Skipping duplicate core item: {item.title}")
            continue
        
        # Add identifiers to seen set
        if goodreads_id:
            seen_ids.add(goodreads_id)
        seen_ids.add(item_key)
        
        # Add to all recommendations
        all_recommendations.append((item, score, terms))
    
    # Process trending recommendations
    for item, score, terms in trending_recommendations:
        # Create unique identifiers for deduplication
        goodreads_id = item.goodreads_id if item.goodreads_id else None
        item_key = f"{item.title}|{item.author}".lower()
        
        # Skip if we've seen this item before
        if (goodreads_id and goodreads_id in seen_ids) or (item_key in seen_ids):
            logger.info(f"Skipping duplicate trending item: {item.title}")
            continue
        
        # Add identifiers to seen set
        if goodreads_id:
            seen_ids.add(goodreads_id)
        seen_ids.add(item_key)
        
        # Add to all recommendations
        all_recommendations.append((item, score, terms))
    
    # Segment recommendations by category
    segmented_recommendations = {
        "novels": [],
        "papers": [],
        "poems": [],
        "other": []
    }
    
    # Process all recommendations for segmentation
    for item, score, terms in all_recommendations:
        # Manually determine category based on item_type
        category = "other"
        if item.item_type:
            item_type_lower = item.item_type.lower()
            if "novel" in item_type_lower:
                category = "novels"
            elif any(keyword in item_type_lower for keyword in ["paper", "article", "research"]):
                category = "papers"
            elif any(keyword in item_type_lower for keyword in ["poem", "poetry", "verse"]):
                category = "poems"
        
        logger.info(f"Categorizing item '{item.title}' as '{category}' (type: '{item.item_type}')")
        segmented_recommendations[category].append((item, score, terms))
    
    # Sort each category by score
    for category in segmented_recommendations:
        segmented_recommendations[category].sort(key=lambda x: x[1], reverse=True)
    
    # Ensure we have news and social updates (use dummy data if empty)
    if not news_and_social:
        logger.warning("No news and social updates found, adding dummy data for testing")
        normalized_input = literature_input.strip()
        news_and_social = [
            {
                "title": f"Recent discussions about {normalized_input}",
                "source": "Social Media",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "url": "",
                "summary": f"Readers have been discussing themes and character development in {normalized_input} across various platforms.",
                "type": "social"
            },
            {
                "title": f"Literary analysis of {normalized_input}",
                "source": "Literary Blog",
                "date": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
                "url": "",
                "summary": f"A recent analysis explores the cultural impact and enduring themes of {normalized_input}.",
                "type": "review"
            }
        ]
    
    # Create the result dictionary
    result = {
        "core": core_recommendations,  # Keep original core recommendations for API compatibility
        "trending": trending_recommendations,  # Keep original trending recommendations for API compatibility
        "terms": user_terms,
        "context_description": context_desc,
        "history": history_used,
        "input": literature_input,  # Include the original input
        "news_and_social": news_and_social,  # Add news and social media updates
        "segmented_recommendations": segmented_recommendations  # Add segmented recommendations
    }
    
    # Store results in both memory cache and database
    store_recommendations_cache(literature_input, result, session_id)
    
    return result

def get_book_cover(goodreads_id: str = None, title: str = "", author: str = "") -> Tuple[str, str]:
    """
    Get the book cover image URL for a given book by its Goodreads ID or title/author.
    
    Args:
        goodreads_id: The Goodreads ID of the book (optional)
        title: The title of the book (optional)
        author: The author name (optional)
        
    Returns:
        Tuple of (image_url, goodreads_id)
    """
    # Validate inputs
    if not goodreads_id and not title:
        logger.warning("No goodreads_id or title provided for book cover lookup")
        return "/static/images/placeholder-cover.svg", ""
    
    # If we have a goodreads_id, check if it's a valid numeric ID for Goodreads
    valid_goodreads_id = False
    if goodreads_id:
        # Check if it's a numeric ID that can be used with Goodreads
        if goodreads_id.isdigit():
            valid_goodreads_id = True
        else:
            logger.warning(f"Non-numeric goodreads_id provided: {goodreads_id}, will generate a new one")
            goodreads_id = None
    
    # If no valid goodreads_id, generate one from title and author
    if not valid_goodreads_id and title:
        # Generate a deterministic ID based on title and author
        combined = (title + author).lower().replace(" ", "")
        import hashlib
        hash_object = hashlib.md5(combined.encode())
        goodreads_id = str(int(hash_object.hexdigest(), 16) % 10000000)
        logger.info(f"Generated pseudo-goodreads_id {goodreads_id} for '{title}' by '{author}'")
    
    # Check if we already have this book cover in the book_images table
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT image_url, local_path FROM book_images WHERE goodreads_id = ?", (goodreads_id,))
    result = cursor.fetchone()
    conn.close()
    
    if result:
        # If we have a local path, use it
        if result[1] and os.path.exists(result[1]):
            logger.info(f"Using local image file for goodreads_id: {goodreads_id}")
            # Convert absolute path to relative URL path for Flask static files
            static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
            if result[1].startswith(static_dir):
                # Extract just the part after /static/ for use with url_for in templates
                local_path = result[1].replace(static_dir, "").lstrip(os.path.sep)
                return f"/static/{local_path}", goodreads_id
        
        # Otherwise use the stored image URL
        if result[0]:
            logger.info(f"Using cached image URL for goodreads_id: {goodreads_id}")
            # Make sure the URL is absolute
            if result[0].startswith("http"):
                return result[0], goodreads_id
            else:
                # If it's a relative path, make sure it's properly formatted
                return result[0] if result[0].startswith("/") else f"/{result[0]}", goodreads_id
    
    # Try Open Library Covers API only if we have a valid numeric goodreads_id
    image_url = None
    if valid_goodreads_id:
        # Use a direct Goodreads URL pattern
        url = f"https://i.gr-assets.com/images/S/compressed.photo.goodreads.com/books/{goodreads_id}._SX318_.jpg"
        logger.info(f"Trying Goodreads cover URL: {url}")
        
        try:
            response = requests.head(url, timeout=3)
            
            if response.status_code == 200 and int(response.headers.get('Content-Length', 0)) > 1000:
                image_url = url
                logger.info(f"Found valid Goodreads cover for goodreads_id: {goodreads_id}")
        except Exception as e:
            logger.error(f"Error checking Goodreads URL: {str(e)}")
    
    if image_url:
        # Download and cache the image locally
        try:
            # Create the book_covers directory if it doesn't exist
            book_covers_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "book_covers")
            if not os.path.exists(book_covers_dir):
                os.makedirs(book_covers_dir)
            
            # Download the image
            local_path = os.path.join(book_covers_dir, f"{goodreads_id}.jpg")
            response = requests.get(image_url, timeout=5)
            
            if response.status_code == 200 and len(response.content) > 1000:
                with open(local_path, 'wb') as f:
                    f.write(response.content)
                
                # Compress the image to reduce file size
                local_path = compress_image(local_path)
                
                # Store in the book_images table
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT OR REPLACE INTO book_images (goodreads_id, title, image_url, local_path) VALUES (?, ?, ?, ?)",
                    (goodreads_id, title, image_url, local_path)
                )
                conn.commit()
                conn.close()
                
                logger.info(f"Downloaded and compressed image for goodreads_id: {goodreads_id} to {local_path}")
                
                # Convert absolute path to relative URL path for Flask static files
                static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
                if local_path.startswith(static_dir):
                    # Extract just the part after /static/ for use with url_for in templates
                    relative_path = local_path.replace(static_dir, "").lstrip(os.path.sep)
                    return f"/static/{relative_path}", goodreads_id
            else:
                logger.warning(f"Failed to download image from {image_url}: status={response.status_code}, size={len(response.content)}")
        except Exception as e:
            logger.error(f"Error downloading image for goodreads_id {goodreads_id}: {str(e)}")
        
        # If local caching fails, still return the API URL
        # Store in the book_images table without local_path
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO book_images (goodreads_id, title, image_url) VALUES (?, ?, ?)",
            (goodreads_id, title, image_url)
        )
        conn.commit()
        logger.info(f"Stored recommendations in persistent cache for: {title[:50]}...")
        conn.close()
        
        return image_url, goodreads_id
    
    # If all else fails, return the path to the placeholder image
    logger.warning(f"Using placeholder image for goodreads_id: {goodreads_id}")
    
    # Store the placeholder in the database to avoid repeated lookups
    conn = get_db_connection()
    cursor = conn.cursor()
    placeholder_url = "/static/images/placeholder-cover.svg"
    cursor.execute(
        "INSERT OR REPLACE INTO book_images (goodreads_id, title, image_url) VALUES (?, ?, ?)",
        (goodreads_id, title, placeholder_url)
    )
    conn.commit()
    logger.info(f"Stored placeholder image for goodreads_id: {goodreads_id}")
    conn.close()
    
    return placeholder_url, goodreads_id

def compress_image(image_path: str) -> str:
    """
    Compress an image to reduce file size while maintaining reasonable quality.
    
    Args:
        image_path: Path to the image file to compress
        
    Returns:
        Path to the compressed image (same as input path)
    """
    try:
        from PIL import Image
        
        # Open the image
        img = Image.open(image_path)
        
        # Save with reduced quality (80%)
        img.save(image_path, optimize=True, quality=80)
        
        logger.info(f"Compressed image at {image_path}")
        return image_path
    except ImportError:
        logger.warning("PIL/Pillow not installed - image compression skipped")
        return image_path
    except Exception as e:
        logger.error(f"Error compressing image: {str(e)}")
        return image_path

def fetch_book_cover_api(goodreads_id: str) -> Tuple[str, bool]:
    """
    Fetch a book cover image from the Open Library Covers API using goodreads_id.
    
    Args:
        goodreads_id: Goodreads ID for the book
        
    Returns:
        Tuple of (image_url, success_flag)
    
    Note: This function assumes a mapping between Goodreads IDs and Open Library IDs.
    In a production environment, you might need a more robust mapping service or database.
    The Open Library API is rate-limited to 100 requests per 5 minutes per IP.
    """
    if not goodreads_id:
        return "", False
    
    # First, try to use the goodreads_id directly with the Open Library API
    # This assumes some books might have the same ID in both systems
    try:
        # Try large size image first - use GET request instead of HEAD to follow redirects
        url = f"https://covers.openlibrary.org/b/id/{goodreads_id}-L.jpg"
        logger.info(f"Trying Open Library cover URL: {url}")
        
        response = requests.get(url, timeout=5, stream=True, allow_redirects=True)
        
        # Check if we got a valid image (status code 200 and content type is image)
        if response.status_code == 200 and response.headers.get('Content-Type', '').startswith('image/'):
            # Read a small part of the image to verify it's valid
            chunk = next(response.iter_content(1024), None)
            if chunk and len(chunk) > 100:  # If we got some image data
                logger.info(f"Found valid Open Library cover for goodreads_id: {goodreads_id} (large size)")
                return url, True
        
        # Try medium size as fallback
        url = f"https://covers.openlibrary.org/b/id/{goodreads_id}-M.jpg"
        logger.info(f"Trying Open Library cover URL: {url}")
        
        response = requests.get(url, timeout=5, stream=True, allow_redirects=True)
        
        if response.status_code == 200 and response.headers.get('Content-Type', '').startswith('image/'):
            chunk = next(response.iter_content(1024), None)
            if chunk and len(chunk) > 100:
                logger.info(f"Found valid Open Library cover for goodreads_id: {goodreads_id} (medium size)")
                return url, True
    except Exception as e:
        logger.error(f"Error checking Open Library direct ID match: {str(e)}")
    
    # Try using OLID (Open Library ID) format
    try:
        # Try OLID format
        olid = f"OL{goodreads_id}M"
        url = f"https://covers.openlibrary.org/b/olid/{olid}-L.jpg"
        logger.info(f"Trying Open Library OLID cover URL: {url}")
        
        response = requests.get(url, timeout=5, stream=True, allow_redirects=True)
        
        if response.status_code == 200 and response.headers.get('Content-Type', '').startswith('image/'):
            chunk = next(response.iter_content(1024), None)
            if chunk and len(chunk) > 100:
                logger.info(f"Found valid Open Library cover for OLID: {olid} (large size)")
                return url, True
    except Exception as e:
        logger.error(f"Error checking Open Library OLID lookup: {str(e)}")
    
    # If direct ID mapping fails, try using ISBN lookup
    # In a real implementation, you'd have a mapping service or database
    # For this implementation, we'll use a simple pattern to generate a fake ISBN
    # This is just a placeholder - in production, you would use a real ISBN lookup
    try:
        # Generate a fake ISBN based on goodreads_id for demonstration
        # In production, you would query a database or API for the actual ISBN
        isbn = f"978{goodreads_id.zfill(10)}"[:13]
        
        # Try ISBN-13 lookup
        url = f"https://covers.openlibrary.org/b/isbn/{isbn}-L.jpg"
        logger.info(f"Trying Open Library ISBN cover URL: {url}")
        
        response = requests.get(url, timeout=5, stream=True, allow_redirects=True)
        
        if response.status_code == 200 and response.headers.get('Content-Type', '').startswith('image/'):
            chunk = next(response.iter_content(1024), None)
            if chunk and len(chunk) > 100:
                logger.info(f"Found valid Open Library cover for ISBN: {isbn} (large size)")
                return url, True
    except Exception as e:
        logger.error(f"Error checking Open Library ISBN lookup: {str(e)}")
    
    # Try Goodreads URL as a fallback
    try:
        # Use a direct Goodreads URL pattern
        url = f"https://i.gr-assets.com/images/S/compressed.photo.goodreads.com/books/{goodreads_id}._SX318_.jpg"
        logger.info(f"Trying Goodreads cover URL: {url}")
        
        response = requests.get(url, timeout=5, stream=True, allow_redirects=True)
        
        if response.status_code == 200 and response.headers.get('Content-Type', '').startswith('image/'):
            chunk = next(response.iter_content(1024), None)
            if chunk and len(chunk) > 100:
                logger.info(f"Found valid Goodreads cover for goodreads_id: {goodreads_id}")
                return url, True
    except Exception as e:
        logger.error(f"Error checking Goodreads URL: {str(e)}")
    
    # If all attempts fail, return empty string and False
    logger.warning(f"Could not find any valid cover image for goodreads_id: {goodreads_id}")
    return "", False

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

def save_book(session_id: str, goodreads_id: str, title: str, author: str, image_url: str = "") -> int:
    """
    Save a book to the user's "My Books" collection.
    
    Args:
        session_id: User's session ID
        goodreads_id: Goodreads ID for the book
        title: Book title
        author: Book author
        image_url: URL to the book cover image (optional)
        
    Returns:
        ID of the saved book record, or None if failed
    """
    if not session_id or not title or not author:
        logger.warning("Cannot save book: missing required fields")
        return None
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if book already exists for this user
        cursor.execute(
            "SELECT id FROM saved_books WHERE session_id = ? AND goodreads_id = ?",
            (session_id, goodreads_id)
        )
        existing = cursor.fetchone()
        
        if existing:
            logger.info(f"Book '{title}' already saved for session {session_id}")
            return existing[0]
        
        # Insert new book
        cursor.execute(
            """
            INSERT INTO saved_books 
            (session_id, goodreads_id, title, author, status, progress, added_date) 
            VALUES (?, ?, ?, ?, 'to_read', 0, CURRENT_TIMESTAMP)
            """,
            (session_id, goodreads_id, title, author)
        )
        
        # Get the ID of the inserted book
        saved_book_id = cursor.lastrowid
        
        # Also ensure the book cover is saved
        if goodreads_id and image_url:
            try:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO book_images 
                    (goodreads_id, title, image_url) 
                    VALUES (?, ?, ?)
                    """,
                    (goodreads_id, title, image_url)
                )
            except Exception as e:
                logger.warning(f"Could not save book cover image: {e}")
        
        conn.commit()
        logger.info(f"Saved book '{title}' by '{author}' for session {session_id}")
        return saved_book_id
        
    except Exception as e:
        logger.error(f"Error saving book: {e}")
        return None
    finally:
        if 'conn' in locals():
            conn.close()

def update_book_status(session_id: str, saved_book_id: int, status: str = None, progress: int = None) -> bool:
    """
    Update the reading status and/or progress of a saved book.
    
    Args:
        session_id: User's session ID
        saved_book_id: ID of the saved book in the saved_books table
        status: New reading status ("to_read", "reading", "finished")
        progress: New reading progress (0-100)
        
    Returns:
        True if successful, False otherwise
    """
    if not session_id or not saved_book_id:
        logger.warning("Cannot update book status: missing required fields")
        return False
    
    # Validate status
    valid_statuses = ["to_read", "reading", "finished"]
    if status and status not in valid_statuses:
        logger.warning(f"Invalid status: {status}")
        return False
    
    # Validate progress
    if progress is not None:
        try:
            progress = int(progress)
            if progress < 0 or progress > 100:
                logger.warning(f"Invalid progress value: {progress}")
                return False
        except ValueError:
            logger.warning(f"Invalid progress value: {progress}")
            return False
    
    # If status is "finished", automatically set progress to 100%
    if status == "finished":
        progress = 100
    
    # Build the SQL update statement
    update_fields = []
    params = []
    
    if status:
        update_fields.append("status = ?")
        params.append(status)
    
    if progress is not None:
        update_fields.append("progress = ?")
        params.append(progress)
    
    if not update_fields:
        logger.warning("No fields to update")
        return False
    
    sql = f"UPDATE saved_books SET {', '.join(update_fields)} WHERE id = ? AND session_id = ?"
    params.extend([saved_book_id, session_id])
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(sql, params)
        
        if cursor.rowcount == 0:
            logger.warning(f"No book updated: book with ID {saved_book_id} not found for session {session_id}")
            return False
        
        conn.commit()
        logger.info(f"Updated book {saved_book_id} for session {session_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error updating book status: {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

def remove_saved_book(session_id: str, saved_book_id: int) -> bool:
    """
    Remove a book from the user's saved collection.
    
    Args:
        session_id: User's session ID
        saved_book_id: ID of the saved book in the saved_books table
        
    Returns:
        True if successful, False otherwise
    """
    if not session_id or not saved_book_id:
        logger.warning("Cannot remove book: missing required fields")
        return False
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Delete the book from saved_books - the foreign key constraints with CASCADE will handle bookshelf_items
        cursor.execute(
            "DELETE FROM saved_books WHERE id = ? AND session_id = ?",
            (saved_book_id, session_id)
        )
        
        if cursor.rowcount == 0:
            logger.warning(f"Book with ID {saved_book_id} not found for session {session_id}")
            return False
        
        conn.commit()
        logger.info(f"Removed book {saved_book_id} for session {session_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error removing book: {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

def get_my_books(session_id: str, shelf_filter: str = None, status_filter: str = None) -> tuple:
    """
    Retrieve the user's saved books, optionally filtered by shelf or status.
    Also returns a list of the user's custom bookshelves and personalized recommendations.
    
    Args:
        session_id: User's session ID
        shelf_filter: Optional name of a bookshelf to filter by
        status_filter: Optional status to filter by ("to_read", "reading", "finished")
        
    Returns:
        Tuple of (list of LiteratureItem objects, list of bookshelf names, list of recommended LiteratureItem objects)
    """
    if not session_id:
        logger.warning("Cannot get books: missing session_id")
        return [], [], []
    
    books = []
    bookshelves = []
    recommendations = []
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get all bookshelves for this user
        cursor.execute(
            "SELECT id, shelf_name FROM bookshelves WHERE session_id = ? ORDER BY shelf_name",
            (session_id,)
        )
        shelf_rows = cursor.fetchall()
        bookshelves = [row[1] for row in shelf_rows]
        shelf_id_map = {row[0]: row[1] for row in shelf_rows}
        
        # Build SQL query for books based on filters
        sql = """
            SELECT b.id, b.goodreads_id, b.title, b.author, b.status, b.progress, b.added_date,
                   GROUP_CONCAT(bs.shelf_name) as shelves
            FROM saved_books b
            LEFT JOIN bookshelf_items bi ON b.id = bi.saved_book_id
            LEFT JOIN bookshelves bs ON bi.shelf_id = bs.id AND bs.session_id = ?
        """
        params = [session_id]
        
        # Add WHERE clauses for filters
        where_clauses = ["b.session_id = ?"]
        params.append(session_id)
        
        if shelf_filter:
            where_clauses.append("bs.shelf_name = ?")
            params.append(shelf_filter)
        
        if status_filter:
            where_clauses.append("b.status = ?")
            params.append(status_filter)
        
        sql += " WHERE " + " AND ".join(where_clauses)
        sql += " GROUP BY b.id ORDER BY b.added_date DESC"
        
        # Debug log the SQL query and parameters
        logger.info(f"My Books SQL query: {sql}")
        logger.info(f"My Books SQL params: {params}")
        
        cursor.execute(sql, params)
        book_rows = cursor.fetchall()
        
        # Debug log the number of books found
        logger.info(f"Found {len(book_rows)} books for session {session_id}")
        
        # Convert rows to LiteratureItem objects
        for row in book_rows:
            book_id, goodreads_id, title, author, status, progress, added_date, shelves_str = row
            shelves_list = shelves_str.split(",") if shelves_str else []
            
            # Get book cover image if available
            image_url = ""
            if goodreads_id:
                cursor.execute(
                    "SELECT image_url FROM book_images WHERE goodreads_id = ?",
                    (goodreads_id,)
                )
                img_row = cursor.fetchone()
                if img_row:
                    image_url = img_row[0]
            
            # Create LiteratureItem object
            book = LiteratureItem(
                title=title,
                author=author,
                goodreads_id=goodreads_id,
                image_url=image_url,
                status=status,
                progress=progress,
                shelves=shelves_list,
                saved_id=book_id
            )
            books.append(book)
        
        # Generate personalized recommendations based on saved books
        if len(books) > 0:
            # Construct a prompt for recommendations based on saved books
            titles_authors = [f"{b.title} by {b.author}" for b in books[:5]]  # Use up to 5 books
            recommendation_input = f"Books similar to: {', '.join(titles_authors)}"
            
            # Use existing recommendation function with cache
            cache_key = generate_cache_key(recommendation_input)
            cached_results = get_cached_recommendations(cache_key)
            
            if cached_results:
                recommendations = cached_results
                logger.info(f"Using cached personalized recommendations for session {session_id}")
            else:
                # Generate new recommendations
                try:
                    recommendations = get_recommendations(recommendation_input, session_id)
                    # Cache the recommendations
                    store_recommendations_cache(cache_key, recommendations)
                    logger.info(f"Generated new personalized recommendations for session {session_id}")
                except Exception as e:
                    logger.error(f"Error generating personalized recommendations: {e}")
        
        return books, bookshelves, recommendations
        
    except Exception as e:
        logger.error(f"Error retrieving saved books: {e}")
        return [], [], []
    finally:
        if 'conn' in locals():
            conn.close()

def test_recommendations(input_text="the brothers karamazov", session_id="test"):
    """
    Test function to check recommendation quality for a given input.
    
    Args:
        input_text: Text to test (e.g., 'the brothers karamazov')
        session_id: Session ID to use for testing
    """
    print(f"\nTesting recommendations for: '{input_text}'")
    
    # Get full recommendations
    results = get_recommendations(input_text, session_id)
    
    # Print terms and context
    print(f"\nExtracted terms: {results['terms']}")
    print(f"Context: {results['context_description']}")
    print(f"History: {results['history']}")
    
    # Print news and social updates
    print(f"\nNews and Social Media Updates:")
    if results['news_and_social']:
        for i, update in enumerate(results['news_and_social'], 1):
            print(f"{i}. {update['title']} ({update['type']})")
            print(f"   Source: {update['source']} | Date: {update.get('date', 'N/A')}")
            print(f"   Summary: {update['summary']}")
            if 'url' in update and update['url']:
                print(f"   URL: {update['url']}")
            print()
    else:
        print("   No news or social updates found")
    
    # Print segmented recommendations
    print(f"\nSegmented Recommendations:")
    for category, recs in results['segmented_recommendations'].items():
        if recs:
            print(f"\n{category.upper()} ({len(recs)}):")
            for i, (item, score, matched_terms) in enumerate(recs[:3], 1):  # Show top 3 per category
                print(f"{i}. {item.title} by {item.author} (Score: {score:.2f})")
                print(f"   Type: {item.item_type} | Category: {item.category}")
                print(f"   Matched terms: {', '.join(matched_terms)}")
    
    # Print traditional recommendations
    print(f"\nTop Core Recommendations:")
    for i, (item, score, matched_terms) in enumerate(results['core'][:5], 1):  # Show top 5
        print(f"{i}. {item.title} by {item.author} (Score: {score:.2f})")
        print(f"   Type: {item.item_type} | Category: {item.category}")
        print(f"   Matched terms: {', '.join(matched_terms)}")
    
    print(f"\nTop Trending Recommendations:")
    for i, (item, score, matched_terms) in enumerate(results['trending'][:5], 1):  # Show top 5
        print(f"{i}. {item.title} by {item.author} (Score: {score:.2f})")
        print(f"   Type: {item.item_type} | Category: {item.category}")
        print(f"   Matched terms: {', '.join(matched_terms)}")
    
    return results

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

def generate_cache_key(literature_input, session_id=None):
    """Generate a consistent, deterministic key for caching based on input."""
    # Combine input and session ID (if provided)
    key_base = f"{literature_input}_{session_id or 'anonymous'}"
    # Generate hash for database key
    import hashlib
    hash_obj = hashlib.md5(key_base.encode())
    return hash_obj.hexdigest()

def get_cached_recommendations(literature_input, session_id=None):
    """
    Check for cached recommendations in both memory and database.
    Returns None if no cached result is found.
    """
    # Normalize input for more consistent caching
    normalized_input = literature_input.lower().strip()
    
    # Generate a cache key based on input and session
    memory_cache_key = f"{normalized_input}_{session_id or 'anonymous'}"
    db_cache_key = generate_cache_key(normalized_input, session_id)
    
    # First, check memory cache (faster)
    if memory_cache_key in recommendations_cache:
        logger.info(f"Using in-memory cached recommendations for: {normalized_input[:50]}...")
        cached_result = recommendations_cache[memory_cache_key]
        return cached_result
    
    # If not in memory, check database cache
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Query the recommendation_cache table
        cursor.execute(
            "SELECT results FROM recommendation_cache WHERE input_hash = ?", 
            (db_cache_key,)
        )
        result = cursor.fetchone()
        
        if result:
            # Deserialize the JSON results
            import json
            cached_results = json.loads(result[0])
            
            # Also update the memory cache for faster future access
            recommendations_cache[memory_cache_key] = cached_results
            
            logger.info(f"Using database cached recommendations for: {normalized_input[:50]}...")
            return cached_results
        
        # Try a fuzzy match if exact match not found
        cursor.execute(
            "SELECT literature_input, results FROM recommendation_cache"
        )
        all_results = cursor.fetchall()
        
        for stored_input, stored_results in all_results:
            # Check if the stored input is similar to our current input
            if stored_input and normalized_input and (
                normalized_input in stored_input.lower() or 
                stored_input.lower() in normalized_input or
                (len(normalized_input) > 5 and normalized_input[:5] in stored_input.lower())
            ):
                # Found a fuzzy match
                cached_results = json.loads(stored_results)
                
                # Update the memory cache
                recommendations_cache[memory_cache_key] = cached_results
                
                logger.info(f"Using fuzzy-matched cached recommendations for: {normalized_input[:50]}...")
                return cached_results
        
    except Exception as e:
        logger.error(f"Error retrieving cached recommendations: {e}")
    finally:
        if 'conn' in locals():
            conn.close()
    
    return None

def store_recommendations_cache(literature_input, results, session_id=None):
    """Store recommendations in both memory and database cache."""
    try:
        # Normalize input for more consistent caching
        normalized_input = literature_input.lower().strip()
        
        # Store in memory cache
        memory_cache_key = f"{normalized_input}_{session_id or 'anonymous'}"
        recommendations_cache[memory_cache_key] = results
        
        # Store in database
        db_cache_key = generate_cache_key(normalized_input, session_id)
        
        # Serialize the results to JSON
        import json
        serialized_results = json.dumps(results)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Insert or replace the cache entry
        cursor.execute(
            "INSERT OR REPLACE INTO recommendation_cache (input_hash, literature_input, results) VALUES (?, ?, ?)",
            (db_cache_key, normalized_input, serialized_results)
        )
        conn.commit()
        logger.info(f"Stored recommendations in persistent cache for: {normalized_input[:50]}...")
        
    except Exception as e:
        logger.error(f"Error storing recommendations cache: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

def manage_bookshelves(session_id: str, saved_book_id: int, shelf_name: str, action: str = "add") -> bool:
    """
    Add or remove a book from a bookshelf. Creates the shelf if it doesn't exist.
    
    Args:
        session_id: User's session ID
        saved_book_id: ID of the saved book in the saved_books table
        shelf_name: Name of the bookshelf
        action: "add" to add the book to the shelf, "remove" to remove it
        
    Returns:
        True if successful, False otherwise
    """
    if not session_id or not saved_book_id or not shelf_name:
        logger.warning("Cannot manage bookshelf: missing required fields")
        return False
    
    # Clean up shelf name (no special characters, max 50 chars)
    shelf_name = re.sub(r'[^\w\s-]', '', shelf_name).strip()[:50]
    if not shelf_name:
        logger.warning("Invalid shelf name")
        return False
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # First, verify the book exists and belongs to this user
        cursor.execute(
            "SELECT id FROM saved_books WHERE id = ? AND session_id = ?",
            (saved_book_id, session_id)
        )
        book = cursor.fetchone()
        if not book:
            logger.warning(f"Book with ID {saved_book_id} not found for session {session_id}")
            return False
        
        # Check if the shelf exists for this user
        cursor.execute(
            "SELECT id FROM bookshelves WHERE session_id = ? AND shelf_name = ?",
            (session_id, shelf_name)
        )
        shelf = cursor.fetchone()
        
        if not shelf:
            # Create the shelf if it doesn't exist
            cursor.execute(
                "INSERT INTO bookshelves (session_id, shelf_name) VALUES (?, ?)",
                (session_id, shelf_name)
            )
            shelf_id = cursor.lastrowid
        else:
            shelf_id = shelf[0]
        
        if action == "add":
            # Add the book to the shelf
            cursor.execute(
                "INSERT OR IGNORE INTO bookshelf_items (shelf_id, saved_book_id) VALUES (?, ?)",
                (shelf_id, saved_book_id)
            )
        elif action == "remove":
            # Remove the book from the shelf
            cursor.execute(
                "DELETE FROM bookshelf_items WHERE shelf_id = ? AND saved_book_id = ?",
                (shelf_id, saved_book_id)
            )
        else:
            logger.warning(f"Invalid action: {action}")
            return False
        
        conn.commit()
        logger.info(f"Managed bookshelf for session {session_id}: {action} book {saved_book_id} to shelf '{shelf_name}'")
        return True
    except Exception as e:
        logger.error(f"Error managing bookshelf: {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

def sync_saved_books_to_reading_list(session_id: str) -> None:
    """
    Synchronize books from the saved_books table to the user_reading_list table
    for backward compatibility.
    
    Args:
        session_id: User's session ID
    """
    if not session_id:
        logger.warning("Cannot sync books: missing session_id")
        return
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get all books from saved_books table
        cursor.execute(
            "SELECT goodreads_id, title, author, image_url FROM saved_books WHERE session_id = ?",
            (session_id,)
        )
        saved_books = cursor.fetchall()
        
        # For each saved book, check if it exists in user_reading_list
        for goodreads_id, title, author, image_url in saved_books:
            cursor.execute(
                "SELECT 1 FROM user_reading_list WHERE session_id = ? AND goodreads_id = ?",
                (session_id, goodreads_id)
            )
            exists = cursor.fetchone()
            
            # If not exists, add it to user_reading_list
            if not exists:
                cursor.execute(
                    """
                    INSERT INTO user_reading_list (session_id, goodreads_id, title, added_at)
                    VALUES (?, ?, ?, datetime('now'))
                    """,
                    (session_id, goodreads_id, title)
                )
                logger.info(f"Synced book '{title}' to reading list for session {session_id}")
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error syncing saved books to reading list: {str(e)}")
        logger.error(traceback.format_exc())

def fetch_search_updates(literature_input: str) -> List[Dict]:
    """
    Fetch recent news and social media updates about a literature item using the Perplexity API.
    
    Args:
        literature_input: The literature input from the user (book title, author, etc.)
        
    Returns:
        List of dictionaries containing news and social media updates
    """
    if not literature_input:
        logger.warning("Cannot fetch news and social updates: missing input")
        return []
    
    try:
        # Normalize the input to improve search results
        normalized_input = normalize_literature_input(literature_input)
        
        # Check if Perplexity API key is available
        if not PERPLEXITY_API_KEY:
            logger.warning("Perplexity API key not configured, returning dummy data for testing")
            # Return dummy data for testing when API key is not available
            return [
                {
                    "title": f"Recent discussions about {normalized_input}",
                    "source": "Social Media",
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "url": "",
                    "summary": f"Readers have been discussing themes and character development in {normalized_input} across various platforms.",
                    "type": "social"
                },
                {
                    "title": f"Literary analysis of {normalized_input}",
                    "source": "Literary Blog",
                    "date": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
                    "url": "",
                    "summary": f"A recent analysis explores the cultural impact and enduring themes of {normalized_input}.",
                    "type": "review"
                }
            ]
        
        # Prepare the prompt for Perplexity
        prompt = f"""
        Find the latest news, social media discussions, and recent updates about the book or literary work "{normalized_input}".
        Focus on:
        1. Recent news articles about the book or author
        2. Social media discussions or trending topics
        3. Upcoming adaptations (film, TV, etc.)
        4. Recent critical reception or reviews
        5. Author interviews or appearances
        
        Format each item as a JSON object with:
        - title: Title of the update
        - source: Source of the information (publication, social media platform)
        - date: Approximate date (YYYY-MM-DD format if available)
        - url: Link to the source (if available)
        - summary: Brief 1-2 sentence summary
        - type: Type of update (news, social, review, adaptation, interview)
        
        Return the results as a JSON array with at most 5 items, sorted by recency.
        """
        
        logger.info(f"Querying Perplexity for news and social updates about: '{normalized_input}'")
        
        # Query Perplexity API
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "sonar-medium-online",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that finds the latest news and social media updates about books and literary works."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.2,
                "max_tokens": 1024
            },
            timeout=30
        )
        
        if response.status_code == 200:
            response_data = response.json()
            if "choices" in response_data and len(response_data["choices"]) > 0:
                content = response_data["choices"][0]["message"]["content"]
                logger.info(f"Received news and social updates from Perplexity API")
                
                # Parse the JSON response
                try:
                    # Extract JSON array from the response (handling potential text before/after the JSON)
                    json_match = re.search(r'\[\s*{.*}\s*\]', content, re.DOTALL)
                    if json_match:
                        json_content = json_match.group(0)
                        updates = json.loads(json_content)
                        logger.info(f"Parsed {len(updates)} news and social updates")
                        return updates
                    else:
                        # Try to parse the entire content as JSON
                        updates = json.loads(content)
                        if isinstance(updates, list):
                            logger.info(f"Parsed {len(updates)} news and social updates")
                            return updates
                        else:
                            logger.warning(f"Unexpected JSON structure in Perplexity response")
                            return []
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from Perplexity response: {str(e)}")
                    # Return an empty list if parsing fails
                    return []
            else:
                logger.warning(f"Unexpected response structure from Perplexity for news updates")
                return []
        else:
            logger.warning(f"Failed to query Perplexity for news updates: {response.status_code} - {response.text}")
            return []
            
    except Exception as e:
        logger.error(f"Error querying Perplexity for news and social updates: {str(e)}")
        return []

if __name__ == "__main__":
    # Initialize the database
    init_db()
    
    # Test recommendations
    test_recommendations("the brothers karamazov", "test_session")
    
    # Test with another input to show history blending
    test_recommendations("the idiot", "test_session")
