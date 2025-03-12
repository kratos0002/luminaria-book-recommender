"""
Update script for LiteratureDiscovery application.
This script contains focused improvements to the recommendation logic.
"""

# Special case handling for specific literary works
SPECIAL_CASES = {
    "brothers karamazov": {
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

# Additional stopwords to filter out from literary terms
ADDITIONAL_STOPWORDS = {
    "book", "novel", "story", "literature", "literary", "fiction", "nonfiction", 
    "read", "reading", "author", "writer", "books", "novels", "stories", "poem", 
    "poetry", "essay", "articles", "text", "publication", "publish", "published",
    "also", "prominent", "pursue", "character", "theme", "plot", "narrative",
    "chapter", "page", "write", "written", "work", "reader"
}

# Enhanced OpenAI prompt for literary term extraction
ENHANCED_PROMPT = """Analyze: {combined_input}

Return 5-7 unique literary themes, genres, or styles (e.g., 'moral dilemma', 'existentialism') as a comma-separated list. 

Focus on:
- Specific literary genres (e.g., 'magical realism', 'dystopian fiction')
- Thematic elements (e.g., 'moral ambiguity', 'coming of age')
- Writing styles (e.g., 'stream of consciousness', 'unreliable narrator')
- Time periods or movements (e.g., 'victorian era', 'beat generation')

Avoid duplicates (e.g., 'psychological' if 'psychological complexity' exists) and generic terms ('book', 'novel', 'also').

Return ONLY a comma-separated list with no additional text."""

# Enhanced Perplexity prompt for context extraction
PERPLEXITY_CONTEXT_PROMPT = """Summarize themes of {literature_input} in 2-3 sentences, focusing on literary elements.
        
If you recognize this as a specific work, please include the author's name and any relevant literary movement or time period.

Focus on themes, style, and genre rather than plot summary."""

# Enhanced Perplexity prompt for trending literature
PERPLEXITY_TRENDING_PROMPT = """List 10 narrative books or short stories (no nonfiction, monographs) matching these themes: {terms_text}.{exclusion_text}

For each item, provide the following information in this exact format:

Title: [Full title]
Author: [Author's full name]
Type: [book, short story, novella, etc.]
Description: [Brief description highlighting themes related to: {terms_text}]

Please ensure each entry follows this exact format with clear labels for each field."""

def deduplicate_terms(terms):
    """
    Remove duplicate terms and subsets of other terms.
    For example, if we have both "psychological" and "psychological complexity",
    we'll keep only "psychological complexity".
    """
    deduplicated_terms = []
    for term in terms:
        # Check if this term is a subset of any other term
        if not any(term != other_term and term in other_term for other_term in terms):
            deduplicated_terms.append(term)
    return deduplicated_terms

def enhance_recommendation_scoring(item, user_terms, author_name=None):
    """
    Enhanced scoring function for literature recommendations.
    
    Args:
        item: LiteratureItem object
        user_terms: List of user preference terms
        author_name: Optional author name to prioritize
        
    Returns:
        Tuple of (score, matched_terms)
    """
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
    
    # Author bonus: if the author matches the input author, add bonus points
    if author_name and author_name in author_lower:
        score += 2.0
    
    return score, matched_terms

def extract_author_from_input(literature_input):
    """
    Extract potential author name from user input.
    
    Args:
        literature_input: User input string
        
    Returns:
        Optional author name or None
    """
    author_name = None
    if literature_input:
        # Check for common authors in the input
        if "dostoevsky" in literature_input.lower() or any(name in literature_input.lower() for name in ["karamazov", "crime and punishment", "idiot"]):
            author_name = "dostoevsky"
        elif "tolkien" in literature_input.lower() or any(name in literature_input.lower() for name in ["lord of the rings", "hobbit", "middle earth"]):
            author_name = "tolkien"
        elif "austen" in literature_input.lower() or any(name in literature_input.lower() for name in ["pride and prejudice", "emma", "sense and sensibility"]):
            author_name = "austen"
    
    return author_name
