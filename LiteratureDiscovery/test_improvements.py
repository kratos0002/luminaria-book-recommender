"""
Test script for the improved recommendation logic in LiteratureDiscovery.

This script allows testing the improvements without modifying the main app.py file.
It focuses on the specific issue with "The Brothers Karamazov" recommendations.
"""

import os
import sys
import logging
from typing import Dict, List, Tuple, Optional, Set
from dotenv import load_dotenv

# Import from the main application
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app import app, openai_client, PERPLEXITY_API_KEY, cache_key, STOPWORDS
from models import LiteratureItem
from database import get_user_history

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def test_karamazov_recommendations():
    """Test improvements for The Brothers Karamazov recommendations."""
    logger.info("Testing improved recommendations for 'The Brothers Karamazov'")
    
    # Simulate user input
    user_input = "the brothers karamazov"
    
    # Special case for The Brothers Karamazov
    logger.info("Detected 'The Brothers Karamazov' query, using special case handling")
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
    
    # Create some sample literature items for testing
    sample_items = [
        LiteratureItem(
            title="Crime and Punishment",
            author="Fyodor Dostoevsky",
            description="A novel about a poor ex-student who murders a pawnbroker for her money, dealing with themes of morality, guilt, and redemption.",
            item_type="novel"
        ),
        LiteratureItem(
            title="The Idiot",
            author="Fyodor Dostoevsky",
            description="A novel about a saintly man who returns to Russia after years in a Swiss sanatorium and finds himself a stranger in a society obsessed with wealth and power.",
            item_type="novel"
        ),
        LiteratureItem(
            title="Notes from Underground",
            author="Fyodor Dostoevsky",
            description="A short novel about an isolated, unnamed narrator who is a retired civil servant living in St. Petersburg. Considered one of the first existentialist novels.",
            item_type="novel"
        ),
        LiteratureItem(
            title="The Scarlet Letter",
            author="Nathaniel Hawthorne",
            description="A work of historical fiction set in Puritan Massachusetts Bay Colony during the years 1642 to 1649. It tells the story of Hester Prynne, who conceives a daughter through an affair.",
            item_type="novel"
        ),
        LiteratureItem(
            title="War and Peace",
            author="Leo Tolstoy",
            description="A novel that chronicles the French invasion of Russia and the impact of the Napoleonic era on Tsarist society through the stories of five Russian aristocratic families.",
            item_type="novel"
        )
    ]
    
    # Test the improved recommendation scoring
    recommendations = improved_recommend_literature(sample_items, terms, user_input)
    
    # Print the results
    logger.info(f"Recommendations for '{user_input}':")
    for i, (item, score, matched_terms) in enumerate(recommendations, 1):
        logger.info(f"{i}. {item.title} by {item.author} (Score: {score})")
        logger.info(f"   Matched terms: {', '.join(matched_terms)}")
    
    return recommendations

def improved_recommend_literature(trending_items: List[LiteratureItem], user_terms: List[str], literature_input: str = None) -> List[Tuple[LiteratureItem, float, List[str]]]:
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
    
    for item in trending_items:
        # Skip self-recommendations (if the item title matches the user input)
        if literature_input and item.title.lower() == literature_input.lower():
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
            score += 2.0
            logger.info(f"Applied author bonus to: {item.title} (author: {item.author})")
        
        # Add to scored items if there's at least one match
        if matched_terms:
            scored_items.append((item, score, list(matched_terms)))
    
    logger.info(f"Scored {len(scored_items)} items with at least one match")
    
    # Sort by score in descending order
    scored_items.sort(key=lambda x: x[1], reverse=True)
    
    # Return top 5 items
    top_items = scored_items[:5]
    
    # Log top scores for debugging
    if top_items:
        top_scores = [f"{item[0].title[:20]}... ({item[1]})" for item in top_items[:3]]
        logger.info(f"Top scores: {', '.join(top_scores)}")
    
    return top_items

if __name__ == "__main__":
    test_karamazov_recommendations()
