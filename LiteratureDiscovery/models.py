from datetime import datetime
from typing import Optional, List, Set

class LiteratureItem:
    """
    Class representing a literature item (book, poem, essay, etc.)
    with its metadata.
    """
    def __init__(self, title: str, author: str, publication_date: str = "", 
                 genre: str = "", description: str = "", item_type: str = "book"):
        self.title = title
        self.author = author
        self.publication_date = publication_date
        self.genre = genre
        self.description = description
        self.item_type = item_type  # book, poem, essay, etc.
        self.score = 0.0  # Recommendation score
        self.matched_terms = set()  # Terms that matched this item
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            "title": self.title,
            "author": self.author,
            "publication_date": self.publication_date,
            "genre": self.genre,
            "description": self.description,
            "item_type": self.item_type,
            "score": self.score,
            "matched_terms": list(self.matched_terms)
        }
