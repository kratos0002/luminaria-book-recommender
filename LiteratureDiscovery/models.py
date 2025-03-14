from datetime import datetime
from typing import Optional, List, Set
import re

class LiteratureItem:
    """
    Class representing a literature item (book, poem, essay, etc.)
    with its metadata.
    """
    def __init__(self, title: str, author: str, description: str = None, item_type: str = "book", is_trending: bool = False, image_url: str = None, goodreads_id: str = None, publication_date: str = None, genre: str = None, summary: str = None):
        self.title = title
        self.author = author
        self.description = description
        self.item_type = item_type.lower() if item_type else "book"
        self.is_trending = is_trending
        self.image_url = image_url
        self.goodreads_id = goodreads_id
        self.publication_date = publication_date
        self.genre = genre
        self.summary = summary
        self.score = 0.0  # Recommendation score
        self.matched_terms = set()  # Terms that matched this item
        self.match_score = 0  # Match score on a 0-100 scale

    @property
    def category(self) -> str:
        """
        Determine the category of the literature item based on its type.
        
        Returns:
            str: One of 'novels', 'papers', 'poems', or 'other'
        """
        if not self.item_type:
            return "other"
            
        # Force lowercase for consistent comparison
        item_type_lower = self.item_type.lower()
        
        # Direct string contains check
        if "novel" in item_type_lower:
            return "novels"
        if "book" in item_type_lower:
            return "novels"
        if "fiction" in item_type_lower:
            return "novels"
        if "paper" in item_type_lower:
            return "papers"
        if "article" in item_type_lower:
            return "papers"
        if "research" in item_type_lower:
            return "papers"
        if "poem" in item_type_lower:
            return "poems"
        if "poetry" in item_type_lower:
            return "poems"
        if "verse" in item_type_lower:
            return "poems"
            
        # Default category
        return "other"

    @property
    def has_valid_goodreads_id(self) -> bool:
        """Check if the item has a valid Goodreads ID."""
        return self.goodreads_id is not None and self.goodreads_id.strip() != ""

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            "title": self.title,
            "author": self.author,
            "publication_date": self.publication_date,
            "genre": self.genre,
            "description": self.description,
            "item_type": self.item_type,
            "is_trending": self.is_trending,
            "image_url": self.image_url,
            "goodreads_id": self.goodreads_id,
            "score": self.score,
            "matched_terms": list(self.matched_terms),
            "match_score": self.match_score,
            "category": self.category
        }
