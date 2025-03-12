"""
Book details and reading list functionality for the Luminaria application.
This module extends the literature_logic.py with book-specific features.
"""

import os
import sqlite3
import uuid
import logging
import requests
import traceback
import re
from typing import Dict, List, Optional
from datetime import datetime
from cachetools import TTLCache
from dotenv import load_dotenv

# Import from literature_logic
from literature_logic import LiteratureItem, init_db, logger

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API keys
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

# Book details cache (24 hour TTL)
book_cache = TTLCache(maxsize=500, ttl=24*3600)

# Recommendations cache reference (will be populated by app.py)
recs_cache = {}

def extend_db_schema():
    """
    Extend the database schema to include user_reading_list table.
    """
    try:
        # Connect to the database
        conn = sqlite3.connect("user_history.db")
        cursor = conn.cursor()
        
        # Create the user_reading_list table if it doesn't exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_reading_list (
            session_id TEXT,
            title TEXT,
            added_at DATETIME,
            UNIQUE(session_id, title)
        )
        """)
        
        # Commit changes and close connection
        conn.commit()
        conn.close()
        
        logger.info("Extended database schema with user_reading_list table")
        return True
    except Exception as e:
        logger.error(f"Error extending database schema: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def save_to_reading_list(session_id: str, title: str) -> bool:
    """
    Save a book to the user's reading list.
    
    Args:
        session_id: User's session ID
        title: Title of the book to save
        
    Returns:
        Boolean indicating success
    """
    if not session_id or not title:
        logger.warning("Missing session_id or title for save_to_reading_list")
        return False
    
    try:
        # Connect to the database
        conn = sqlite3.connect("user_history.db")
        cursor = conn.cursor()
        
        # Insert or replace the entry
        cursor.execute(
            "INSERT OR REPLACE INTO user_reading_list (session_id, title, added_at) VALUES (?, ?, ?)",
            (session_id, title, datetime.now())
        )
        
        # Commit changes and close connection
        conn.commit()
        conn.close()
        
        logger.info(f"Saved '{title}' to reading list for session {session_id}")
        
        # Invalidate cache for this book
        cache_key = f"{session_id}_{title}"
        if cache_key in book_cache:
            del book_cache[cache_key]
            logger.info(f"Invalidated cache for book: {title}")
        
        return True
    except Exception as e:
        logger.error(f"Error saving to reading list: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def is_in_reading_list(session_id: str, title: str) -> bool:
    """
    Check if a book is in the user's reading list.
    
    Args:
        session_id: User's session ID
        title: Title of the book to check
        
    Returns:
        Boolean indicating if the book is saved
    """
    if not session_id or not title:
        return False
    
    try:
        # Connect to the database
        conn = sqlite3.connect("user_history.db")
        cursor = conn.cursor()
        
        # Check if the entry exists
        cursor.execute(
            "SELECT 1 FROM user_reading_list WHERE session_id = ? AND title = ?",
            (session_id, title)
        )
        
        result = cursor.fetchone() is not None
        
        # Close connection
        conn.close()
        
        return result
    except Exception as e:
        logger.error(f"Error checking reading list: {str(e)}")
        return False

def get_book_details(title: str, session_id: str = None) -> Optional[Dict]:
    """
    Get detailed information about a book using Perplexity API.
    
    Args:
        title: Title of the book
        session_id: Optional user session ID
        
    Returns:
        Dictionary with book details or None if error
    """
    if not title or not PERPLEXITY_API_KEY:
        logger.error("Missing title or Perplexity API key for get_book_details")
        return None
    
    # Generate cache key
    cache_key = f"{session_id}_{title}" if session_id else f"anon_{title}"
    
    # Check cache first
    if cache_key in book_cache:
        logger.info(f"Using cached book details for: {title}")
        return book_cache[cache_key]
    
    try:
        # Query 1: Basic book details and quotes
        details_prompt = f"""Provide details for the book/literary work titled '{title}':
1. Full title (confirm exact title)
2. Author's full name
3. Type (novel, short story, poem, etc.)
4. Publication year
5. A 2-3 sentence summary highlighting key themes
6. 1-2 notable quotes from the work with attribution

Format your response with clear labels for each section."""
        
        # Query 2: Recent news from X and web
        news_prompt = f"""Find recent mentions (last 30 days if possible) of the book/literary work '{title}' from:
1. 2-3 posts or discussions from X (Twitter)
2. 2-3 recent web articles or news items

For each item, provide:
- Source (X username or website name)
- Brief 1-sentence summary of what was said
- Date if available

Format with clear sections for X and Web."""
        
        # Query 3: Related books
        related_prompt = f"""Recommend 3-5 narrative books similar to '{title}' based on themes, style, or author.

For each recommendation, provide:
- Title
- Author
- Type (novel, short story, etc.)
- Brief reason for recommendation (1 sentence)

Format with numbered entries and clear labels."""
        
        # Make API calls to Perplexity
        details_response = query_perplexity(details_prompt)
        news_response = query_perplexity(news_prompt)
        related_response = query_perplexity(related_prompt)
        
        if not details_response:
            logger.warning(f"Failed to get book details for: {title}")
            return None
        
        # Parse responses
        book_info = parse_book_details(details_response)
        news_info = parse_news_info(news_response)
        related_books = parse_related_books(related_response)
        
        # Get match score from recommendations cache if available
        match_score = 50  # Default score
        if session_id and session_id in recs_cache:
            for rec_group in ["core", "trending"]:
                for item, score, _ in recs_cache[session_id].get(rec_group, []):
                    if item.title.lower() == title.lower():
                        match_score = item.match_score
                        break
        
        # Check if book is saved to reading list
        saved = is_in_reading_list(session_id, title) if session_id else False
        
        # Combine all information
        book_details = {
            "title": book_info.get("title", title),
            "author": book_info.get("author", "Unknown"),
            "type": book_info.get("type", "book"),
            "year": book_info.get("year", "Unknown"),
            "summary": book_info.get("summary", "No summary available."),
            "quotes": book_info.get("quotes", []),
            "news_x": news_info.get("x", []),
            "news_web": news_info.get("web", []),
            "related": related_books,
            "match_score": match_score,
            "saved": saved
        }
        
        # Cache the result
        book_cache[cache_key] = book_details
        logger.info(f"Cached book details for: {title}")
        
        return book_details
    except Exception as e:
        logger.error(f"Error getting book details: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def query_perplexity(prompt: str) -> Optional[str]:
    """
    Query the Perplexity API with a prompt.
    
    Args:
        prompt: The prompt to send to Perplexity
        
    Returns:
        The response text or None if error
    """
    try:
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
                        "content": "You are a literary expert with extensive knowledge of books, authors, and literary trends."
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
            
            # Extract the content from the response
            if "choices" in response_data and len(response_data["choices"]) > 0:
                content = response_data["choices"][0]["message"]["content"].strip()
                return content
            else:
                logger.warning(f"Unexpected response structure from Perplexity: {response_data}")
        else:
            logger.warning(f"Failed to query Perplexity: {response.status_code} - {response.text}")
        
        return None
    except Exception as e:
        logger.error(f"Error querying Perplexity: {str(e)}")
        return None

def parse_book_details(text: str) -> Dict:
    """
    Parse book details from Perplexity response.
    
    Args:
        text: The text response from Perplexity
        
    Returns:
        Dictionary with parsed book details
    """
    result = {
        "title": "",
        "author": "",
        "type": "",
        "year": "",
        "summary": "",
        "quotes": []
    }
    
    if not text:
        return result
    
    # Extract title
    title_match = re.search(r"(?:Title|Full title):\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if title_match:
        result["title"] = title_match.group(1).strip()
    
    # Extract author
    author_match = re.search(r"Author(?:'s full name)?:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if author_match:
        result["author"] = author_match.group(1).strip()
    
    # Extract type
    type_match = re.search(r"Type:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if type_match:
        result["type"] = type_match.group(1).strip()
    
    # Extract year
    year_match = re.search(r"(?:Publication year|Year|Published):\s*(\d{4})", text, re.IGNORECASE)
    if year_match:
        result["year"] = year_match.group(1).strip()
    
    # Extract summary
    summary_match = re.search(r"Summary:\s*(.+?)(?:\n\n|\n[A-Z]|$)", text, re.IGNORECASE | re.DOTALL)
    if summary_match:
        result["summary"] = summary_match.group(1).strip()
    
    # Extract quotes
    quotes_section = re.search(r"(?:Notable )?Quotes?:(.+?)(?:\n\n|\n[A-Z]|$)", text, re.IGNORECASE | re.DOTALL)
    if quotes_section:
        quotes_text = quotes_section.group(1).strip()
        # Split by numbered items or bullet points
        quotes = re.split(r"\n\d+\.|\n-|\n\*", quotes_text)
        # Clean up quotes
        result["quotes"] = [q.strip() for q in quotes if q.strip()]
    
    return result

def parse_news_info(text: str) -> Dict:
    """
    Parse news information from Perplexity response.
    
    Args:
        text: The text response from Perplexity
        
    Returns:
        Dictionary with parsed news information
    """
    result = {
        "x": [],
        "web": []
    }
    
    if not text:
        return result
    
    # Extract X (Twitter) posts
    x_section = re.search(r"(?:X|Twitter)(?:\s+posts|\s+mentions)?:(.+?)(?:Web|$)", text, re.IGNORECASE | re.DOTALL)
    if x_section:
        x_text = x_section.group(1).strip()
        # Split by numbered items or bullet points
        x_items = re.split(r"\n\d+\.|\n-|\n\*", x_text)
        # Clean up items
        result["x"] = [item.strip() for item in x_items if item.strip()]
    
    # Extract Web mentions
    web_section = re.search(r"Web(?:\s+mentions|\s+articles)?:(.+?)(?:\n\n\n|$)", text, re.IGNORECASE | re.DOTALL)
    if web_section:
        web_text = web_section.group(1).strip()
        # Split by numbered items or bullet points
        web_items = re.split(r"\n\d+\.|\n-|\n\*", web_text)
        # Clean up items
        result["web"] = [item.strip() for item in web_items if item.strip()]
    
    return result

def parse_related_books(text: str) -> List[Dict]:
    """
    Parse related books from Perplexity response.
    
    Args:
        text: The text response from Perplexity
        
    Returns:
        List of dictionaries with related book information
    """
    related_books = []
    
    if not text:
        return related_books
    
    # Split by numbered items
    book_sections = re.split(r"\n\d+\.\s+", text)
    
    for section in book_sections[1:]:  # Skip the first empty section
        book = {}
        
        # Extract title
        title_match = re.search(r"(?:Title|Book):\s*(.+?)(?:\n|$)", section, re.IGNORECASE)
        if title_match:
            book["title"] = title_match.group(1).strip()
        else:
            # Try to extract title from the first line if no explicit label
            first_line = section.split("\n")[0].strip()
            if first_line and ":" not in first_line:
                book["title"] = first_line
        
        # Extract author
        author_match = re.search(r"Author:\s*(.+?)(?:\n|$)", section, re.IGNORECASE)
        if author_match:
            book["author"] = author_match.group(1).strip()
        
        # Extract type
        type_match = re.search(r"Type:\s*(.+?)(?:\n|$)", section, re.IGNORECASE)
        if type_match:
            book["type"] = type_match.group(1).strip()
        else:
            book["type"] = "book"  # Default
        
        # Extract reason
        reason_match = re.search(r"(?:Reason|Why):\s*(.+?)(?:\n|$)", section, re.IGNORECASE)
        if reason_match:
            book["reason"] = reason_match.group(1).strip()
        
        # Only add if we have at least a title
        if "title" in book:
            related_books.append(book)
    
    return related_books

# Initialize the extended database schema
extend_db_schema()

if __name__ == "__main__":
    # Test the book details functionality
    print("Testing book details functionality...")
    details = get_book_details("Love in the Time of Cholera")
    if details:
        print(f"Successfully retrieved details for: {details['title']} by {details['author']}")
        print(f"Summary: {details['summary']}")
        print(f"Quotes: {details['quotes']}")
        print(f"Related books: {len(details['related'])}")
    else:
        print("Failed to retrieve book details")
