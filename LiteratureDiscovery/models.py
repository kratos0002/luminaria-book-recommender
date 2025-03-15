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
        self.personalized = False  # Flag to indicate if this item has been personalized

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
            "category": self.category,
            "personalized": self.personalized
        }

class UserProfile:
    """
    Class representing a user's reading preferences and history.
    Used for personalized recommendations.
    """
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.genre_preferences = {}  # Weighted genre preferences
        self.author_preferences = {}  # Weighted author preferences
        self.reading_history = []  # Books read/saved
        self.last_updated = datetime.now()
        
    def update_preferences(self, item: LiteratureItem, interaction_type: str):
        """
        Update user preferences based on interaction with a literature item.
        
        Args:
            item: The literature item the user interacted with
            interaction_type: Type of interaction (view, save, rate)
        """
        # Set weights based on interaction type
        weight = self._get_interaction_weight(interaction_type)
        
        # Update genre preferences
        if item.genre:
            self.genre_preferences[item.genre] = self.genre_preferences.get(item.genre, 0) + weight
        
        # Update author preferences
        if item.author:
            self.author_preferences[item.author] = self.author_preferences.get(item.author, 0) + weight
        
        # Add to reading history if not already present
        if interaction_type in ['save', 'rate'] and item.goodreads_id:
            if item.goodreads_id not in [book['goodreads_id'] for book in self.reading_history]:
                self.reading_history.append({
                    'goodreads_id': item.goodreads_id,
                    'title': item.title,
                    'author': item.author,
                    'timestamp': datetime.now(),
                    'interaction_type': interaction_type
                })
        
        self.last_updated = datetime.now()
    
    def _get_interaction_weight(self, interaction_type: str) -> float:
        """
        Get the weight for a specific interaction type.
        
        Args:
            interaction_type: Type of interaction (view, save, rate)
            
        Returns:
            float: Weight value for the interaction
        """
        weights = {
            'view': 0.5,    # Viewing a book details
            'save': 1.0,    # Saving a book
            'rate': 1.5,    # Rating a book
            'finish': 2.0   # Finishing a book
        }
        return weights.get(interaction_type, 0.1)  # Default weight for unknown interactions
    
    def get_top_genres(self, limit: int = 5) -> List[tuple]:
        """
        Get the user's top preferred genres.
        
        Args:
            limit: Maximum number of genres to return
            
        Returns:
            List of (genre, weight) tuples sorted by weight
        """
        sorted_genres = sorted(self.genre_preferences.items(), key=lambda x: x[1], reverse=True)
        return sorted_genres[:limit]
    
    def get_top_authors(self, limit: int = 5) -> List[tuple]:
        """
        Get the user's top preferred authors.
        
        Args:
            limit: Maximum number of authors to return
            
        Returns:
            List of (author, weight) tuples sorted by weight
        """
        sorted_authors = sorted(self.author_preferences.items(), key=lambda x: x[1], reverse=True)
        return sorted_authors[:limit]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'user_id': self.user_id,
            'genre_preferences': self.genre_preferences,
            'author_preferences': self.author_preferences,
            'reading_history': self.reading_history,
            'last_updated': self.last_updated.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'UserProfile':
        """Create a UserProfile from a dictionary"""
        profile = cls(data['user_id'])
        profile.genre_preferences = data.get('genre_preferences', {})
        profile.author_preferences = data.get('author_preferences', {})
        profile.reading_history = data.get('reading_history', [])
        profile.last_updated = datetime.fromisoformat(data.get('last_updated', datetime.now().isoformat()))
        return profile
