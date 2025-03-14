#!/usr/bin/env python3
"""
Test script for book cover retrieval functionality.
This script tests the enhanced book cover retrieval system with different scenarios.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the necessary functions
from LiteratureDiscovery.literature_logic import get_book_cover, fetch_book_cover_api

def test_book_cover_retrieval():
    """Test different scenarios for book cover retrieval."""
    
    print("\n===== TESTING BOOK COVER RETRIEVAL =====\n")
    
    # Test case 1: Using a known Goodreads ID
    print("Test Case 1: Book with known Goodreads ID")
    goodreads_id = "2767052"  # The Hunger Games
    title = "The Hunger Games"
    image_url, returned_id = get_book_cover(title, goodreads_id=goodreads_id)
    print(f"Title: {title}")
    print(f"Goodreads ID: {goodreads_id}")
    print(f"Image URL: {image_url}")
    print(f"Returned ID: {returned_id}")
    print("\n" + "-"*50 + "\n")
    
    # Test case 2: Using only a title
    print("Test Case 2: Book with only title")
    title = "To Kill a Mockingbird"
    image_url, returned_id = get_book_cover(title)
    print(f"Title: {title}")
    print(f"Image URL: {image_url}")
    print(f"Returned ID: {returned_id}")
    print("\n" + "-"*50 + "\n")
    
    # Test case 3: Using title and author
    print("Test Case 3: Book with title and author")
    title = "1984"
    author = "George Orwell"
    image_url, returned_id = get_book_cover(title, author)
    print(f"Title: {title}")
    print(f"Author: {author}")
    print(f"Image URL: {image_url}")
    print(f"Returned ID: {returned_id}")
    print("\n" + "-"*50 + "\n")
    
    # Test case 4: Non-existent book (should return placeholder)
    print("Test Case 4: Non-existent book (should return placeholder)")
    title = "This Book Definitely Does Not Exist 12345"
    image_url, returned_id = get_book_cover(title)
    print(f"Title: {title}")
    print(f"Image URL: {image_url}")
    print(f"Returned ID: {returned_id}")
    print("\n" + "-"*50 + "\n")
    
    # Test case 5: Direct API test
    print("Test Case 5: Direct Open Library API test")
    goodreads_id = "2767052"  # The Hunger Games
    api_url, success = fetch_book_cover_api(goodreads_id)
    print(f"Goodreads ID: {goodreads_id}")
    print(f"API URL: {api_url}")
    print(f"Success: {success}")
    
    print("\n===== TESTING COMPLETE =====\n")

if __name__ == "__main__":
    test_book_cover_retrieval()
