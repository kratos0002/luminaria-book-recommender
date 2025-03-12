"""
Script to update the LiteratureDiscovery app with feedback functionality and richer context.
This script adds the following features:
1. User feedback tracking (thumbs up/down)
2. Enhanced SQLite database with feedback table
3. Richer context with summaries via Perplexity API
4. Match scores for recommendation quality
"""

import os
import re
import shutil
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def backup_file(file_path):
    """Create a backup of the file before modifying it."""
    backup_path = f"{file_path}.bak"
    shutil.copy2(file_path, backup_path)
    logger.info(f"Created backup at {backup_path}")

def update_literature_logic():
    """Update the literature_logic.py file with new functionality."""
    file_path = "literature_logic.py"
    
    # Create a backup
    backup_file(file_path)
    
    # Read the current content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Update the LiteratureItem class to include summary and match_score
    content = re.sub(
        r'class LiteratureItem:.*?def __init__\(self, title: str, author: str, publication_date: str = "", .*?genre: str = "", description: str = "", item_type: str = "book"\):.*?self\.matched_terms = set\(\).*?\n',
        '''class LiteratureItem:
    """Class representing a literature item (book, poem, essay, etc.)
    with its metadata."""
    
    def __init__(self, title: str, author: str, publication_date: str = "", 
                 genre: str = "", description: str = "", item_type: str = "book", 
                 summary: str = ""):
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
    
''',
        content,
        flags=re.DOTALL
    )
    
    # Update the to_dict method to include summary and match_score
    content = re.sub(
        r'def to_dict\(self\):.*?return \{.*?"matched_terms": list\(self\.matched_terms\).*?\}',
        '''def to_dict(self):
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
            "match_score": self.match_score
        }''',
        content,
        flags=re.DOTALL
    )
    
    # Update the init_db function to add user_feedback table
    content = re.sub(
        r'def init_db\(\):.*?CREATE TABLE IF NOT EXISTS user_inputs \(.*?UNIQUE\(session_id, input_text\).*?\).*?conn\.commit\(\)',
        '''def init_db():
    """Initialize the SQLite database for user history tracking."""
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_history.db")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create user_inputs table if it doesn't exist
    cursor.execute(\'\'\'
    CREATE TABLE IF NOT EXISTS user_inputs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        input_text TEXT,
        timestamp DATETIME,
        UNIQUE(session_id, input_text)
    )
    \'\'\')
    
    # Create user_feedback table if it doesn't exist
    cursor.execute(\'\'\'
    CREATE TABLE IF NOT EXISTS user_feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        title TEXT,
        feedback INTEGER,
        timestamp DATETIME,
        UNIQUE(session_id, title)
    )
    \'\'\')
    
    conn.commit()''',
        content,
        flags=re.DOTALL
    )
    
    # Add feedback functions after get_user_history
    feedback_functions = '''
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
        cursor.execute(\'\'\'
        INSERT OR REPLACE INTO user_feedback (session_id, title, feedback, timestamp)
        VALUES (?, ?, ?, ?)
        \'\'\', (session_id, title, feedback, datetime.now()))
        
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
        cursor.execute(\'\'\'
        SELECT title, feedback FROM user_feedback
        WHERE session_id = ?
        \'\'\', (session_id,))
        
        feedback_dict = {row[0]: row[1] for row in cursor.fetchall()}
        
        conn.close()
        
        logger.info(f"Retrieved {len(feedback_dict)} feedback items for session {session_id}")
        return feedback_dict
    except Exception as e:
        logger.error(f"Error retrieving user feedback: {e}")
        traceback.print_exc()
        return {}
'''
    
    # Insert feedback functions after get_user_history
    content = re.sub(
        r'(def get_user_history.*?return \[\].*?\n)',
        r'\1' + feedback_functions,
        content,
        flags=re.DOTALL
    )
    
    # Update get_trending_literature to include summaries
    content = re.sub(
        r'def get_trending_literature.*?prompt = \(.*?f"List.*?matching themes.*?\)',
        '''def get_trending_literature(user_terms: List[str] = None, literature_input: str = None):
    """
    Use Perplexity API to search for classic literature across categories.
    
    Args:
        user_terms: Optional list of terms to focus the search
        literature_input: Original user input to exclude from recommendations
        
    Returns:
        List of LiteratureItem objects representing classic literature
    """
    if not PERPLEXITY_API_KEY:
        logger.warning("Perplexity API key not set, cannot get trending literature")
        return []
    
    # Generate a cache key based on the input
    key = cache_key("trends", (user_terms, literature_input))
    if key in trends_cache:
        logger.info("Using cached trending literature results")
        return trends_cache[key]
    
    # Prepare terms for the query
    terms_str = ", ".join(user_terms) if user_terms else "various themes"
    
    # Prepare the exclusion part
    exclude_str = f"Exclude {literature_input}." if literature_input else ""
    
    # Construct the prompt for Perplexity
    prompt = (
        f"List 10 narrative books or short stories (no plays, nonfiction, essays, poetry) "
        f"from 19th-century or classic literature matching themes [{terms_str}]. "
        f"{exclude_str} Include title, type, source, description with author name, "
        f"a 2-3 sentence summary."
    )''',
        content,
        flags=re.DOTALL
    )
    
    # Update parse_literature_items to extract summaries
    content = re.sub(
        r'def parse_literature_items\(text: str\):.*?items = \[\].*?for part in parts:.*?item = LiteratureItem\(.*?title=title,.*?author=author,.*?item_type=item_type,.*?description=description.*?\)',
        '''def parse_literature_items(text: str):
    """
    Parse the response from Perplexity API into LiteratureItem objects.
    
    Args:
        text: The text response from Perplexity
        
    Returns:
        List of LiteratureItem objects
    """
    items = []
    
    # Split the text by numbered items (1., 2., etc.)
    parts = re.split(r'\\n\\s*\\d+\\.\\s+', text)
    
    # Skip the first part if it's just an introduction
    if not re.search(r'(?i)title|author|by', parts[0]):
        parts = parts[1:]
    
    for part in parts:
        if not part.strip():
            continue
        
        try:
            # Extract title
            title_match = re.search(r'(?i)"([^"]+)"|(?i)title:\\s*([^,\\n]+)', part)
            if not title_match:
                title_match = re.search(r'(?i)^([^,\\n:]+)', part)
            
            title = (title_match.group(1) or title_match.group(2)).strip() if title_match else "Unknown Title"
            
            # Extract author
            author_match = re.search(r'(?i)by\\s+([^,\\n]+)|(?i)author:\\s*([^,\\n]+)', part)
            author = (author_match.group(1) or author_match.group(2)).strip() if author_match else "Unknown Author"
            
            # Extract type
            type_match = re.search(r'(?i)type:\\s*([^,\\n]+)|(?i)\\(([^)]*novel[^)]*)\\)', part)
            item_type = type_match.group(1) or type_match.group(2) if type_match else "book"
            
            # Extract description
            description = part.strip()
            
            # Extract summary - look for a section that seems like a summary
            summary_match = re.search(r'(?i)summary:?\\s*([^.]+\\.[^.]+\\.[^.]+\\.)', part)
            if not summary_match:
                # Try to find 2-3 sentences that might be a summary
                summary_match = re.search(r'(?i)(?:it|the book|the novel|the story)\\s+[^.]+\\.[^.]+\\.[^.]+\\.', part)
            
            summary = summary_match.group(1) if summary_match else ""
            if not summary:
                # Just take the last 2-3 sentences as a fallback
                sentences = re.findall(r'[^.!?]+[.!?]', part)
                if len(sentences) >= 3:
                    summary = ''.join(sentences[-3:])
                elif sentences:
                    summary = ''.join(sentences)
            
            # Create and add the item
            item = LiteratureItem(
                title=title,
                author=author,
                item_type=item_type,
                description=description,
                summary=summary
            )''',
        content,
        flags=re.DOTALL
    )
    
    # Update get_literary_trends function
    content = re.sub(
        r'def get_literary_trends\(user_terms: List\[str\] = None\):.*?prompt = \(.*?f"List.*?matching themes.*?\)',
        '''def get_literary_trends(user_terms: List[str] = None):
    """
    Use Perplexity API to search for trending recent literature across categories.
    
    Args:
        user_terms: Optional list of terms to focus the search
        
    Returns:
        List of LiteratureItem objects representing trending recent literature
    """
    if not PERPLEXITY_API_KEY:
        logger.warning("Perplexity API key not set, cannot get literary trends")
        return []
    
    # Generate a cache key based on the input
    key = cache_key("literary_trends", user_terms)
    if key in trends_cache:
        logger.info("Using cached literary trends results")
        return trends_cache[key]
    
    # Prepare terms for the query
    terms_str = ", ".join(user_terms) if user_terms else "various themes"
    
    # Construct the prompt for Perplexity
    prompt = (
        f"List 5 trending narrative books or short stories from recent years matching themes [{terms_str}]. "
        f"Include title, type, source, description with author name, a 2-3 sentence summary."
    )''',
        content,
        flags=re.DOTALL
    )
    
    # Update recommend_literature to include feedback and match scores
    content = re.sub(
        r'def recommend_literature\(trending_items: List\[LiteratureItem\], user_terms: List\[str\], literature_input: str = None\):.*?# Score each item.*?scored_items = \[\]',
        '''def recommend_literature(trending_items: List[LiteratureItem], user_terms: List[str], literature_input: str = None, session_id: str = None):
    """
    Score and recommend literature items based on user terms.
    
    Args:
        trending_items: List of LiteratureItem objects to score
        user_terms: List of user preference terms
        literature_input: Original user input to avoid self-recommendation
        session_id: User's session ID for retrieving feedback
        
    Returns:
        List of tuples (LiteratureItem, score, matched_terms)
    """
    if not trending_items or not user_terms:
        return []
    
    # Get user feedback if session_id is provided
    user_feedback = get_user_feedback(session_id) if session_id else {}
    
    # Get the author of the input literature if available
    input_author = None
    if literature_input:
        input_author = get_author(literature_input)
    
    # Score each item
    scored_items = []''',
        content,
        flags=re.DOTALL
    )
    
    # Update the scoring logic in recommend_literature
    content = re.sub(
        r'for item in trending_items:.*?# Calculate final score.*?final_score = .*?# Add to scored items.*?scored_items\.append\(\(item, final_score, list\(matched_terms\)\)\)',
        '''for item in trending_items:
        # Skip if this is the same as the input
        if literature_input and literature_input.lower() in item.title.lower():
            continue
        
        # Initialize score components
        term_matches = 0
        author_match = 0
        feedback_score = 0
        
        # Check for term matches in title and description
        matched_terms = set()
        for term in user_terms:
            term_lower = term.lower()
            if (term_lower in item.title.lower() or 
                term_lower in item.description.lower() or 
                term_lower in item.author.lower()):
                term_matches += 1
                matched_terms.add(term)
        
        # Store matched terms
        item.matched_terms = matched_terms
        
        # Calculate base score: 30 points per matched term (max 5 terms = 150, capped at 100)
        base_score = min(30 * term_matches, 100)
        
        # Bonus for matching multiple terms: +10 if ≥3 terms match
        term_bonus = 10 if term_matches >= 3 else 0
        
        # Author match bonus: +20 if the author matches
        if input_author and input_author.lower() in item.author.lower():
            author_match = 20
        
        # Feedback adjustment: +20 for thumbs up, -20 for thumbs down
        if item.title in user_feedback:
            feedback_score = 20 * user_feedback[item.title]
        
        # Calculate final score (0-100 scale)
        final_score = max(0, min(100, base_score + term_bonus + author_match + feedback_score))
        
        # Store match score in the item
        item.match_score = final_score
        
        # Add to scored items
        scored_items.append((item, final_score, list(matched_terms)))''',
        content,
        flags=re.DOTALL
    )
    
    # Update get_recommendations to pass session_id to recommend_literature
    content = re.sub(
        r'def get_recommendations\(literature_input: str, session_id: str = None\):.*?# Score and recommend classic literature.*?core_recommendations = recommend_literature\(classic_items, user_terms, literature_input\)',
        '''def get_recommendations(literature_input: str, session_id: str = None):
    """
    Get both core and trending recommendations for a literature input.
    
    Args:
        literature_input: The literature input from the user
        session_id: Optional user session ID for history tracking
        
    Returns:
        Dictionary with core and trending recommendations
    """
    # Store the user input if session_id is provided
    if session_id:
        store_user_input(session_id, literature_input)
    
    # Get user preferences
    user_terms, context, history_used = get_user_preferences(literature_input, session_id)
    
    if not user_terms:
        logger.warning("No user terms extracted, cannot generate recommendations")
        return {
            "core": [],
            "trending": [],
            "terms": [],
            "history": []
        }
    
    # Get classic literature recommendations
    classic_items = get_trending_literature(user_terms, literature_input)
    
    # Get trending literature recommendations
    trending_items = get_literary_trends(user_terms)
    
    # Score and recommend classic literature
    core_recommendations = recommend_literature(classic_items, user_terms, literature_input, session_id)''',
        content,
        flags=re.DOTALL
    )
    
    # Update trending recommendations to pass session_id
    content = re.sub(
        r'# Score and recommend trending literature.*?trending_recommendations = recommend_literature\(trending_items, user_terms, literature_input\)',
        '''# Score and recommend trending literature
    trending_recommendations = recommend_literature(trending_items, user_terms, literature_input, session_id)''',
        content,
        flags=re.DOTALL
    )
    
    # Add submit_feedback function at the end
    submit_feedback_func = '''
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
'''
    
    # Add the submit_feedback function before the if __name__ == "__main__" block
    content = re.sub(
        r'(if __name__ == "__main__":)',
        submit_feedback_func + r'\n\1',
        content
    )
    
    # Write the updated content back to the file
    with open(file_path, 'w') as f:
        f.write(content)
    
    logger.info(f"Updated {file_path} with feedback functionality and richer context")

def update_recommendations_template():
    """Update the recommendations.html template to include feedback buttons and match scores."""
    file_path = "templates/recommendations.html"
    
    # Create a backup
    backup_file(file_path)
    
    # Read the current content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Add CSS for feedback buttons
    content = re.sub(
        r'<style>',
        '''<style>
        .feedback-buttons {
            margin-top: 10px;
            display: flex;
            gap: 10px;
        }
        .feedback-btn {
            cursor: pointer;
            background: none;
            border: 1px solid #ddd;
            border-radius: 50%;
            width: 36px;
            height: 36px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            transition: all 0.2s;
        }
        .feedback-btn:hover {
            background-color: #f0f0f0;
        }
        .feedback-btn.thumbs-up:hover {
            border-color: #4CAF50;
            color: #4CAF50;
        }
        .feedback-btn.thumbs-down:hover {
            border-color: #F44336;
            color: #F44336;
        }
        .feedback-btn.active.thumbs-up {
            background-color: #E8F5E9;
            border-color: #4CAF50;
            color: #4CAF50;
        }
        .feedback-btn.active.thumbs-down {
            background-color: #FFEBEE;
            border-color: #F44336;
            color: #F44336;
        }
        .match-score {
            font-weight: bold;
            margin-bottom: 5px;
        }
        .match-score-note {
            font-size: 0.8em;
            color: #666;
            margin-top: 0;
        }
        .summary {
            font-style: italic;
            margin: 10px 0;
            color: #555;
            line-height: 1.4;
        }''',
        content
    )
    
    # Update the core recommendations section to include summary and feedback buttons
    content = re.sub(
        r'<div class="recommendation-card">.*?<h2 class="title">{{ item\.title }}</h2>.*?<p class="description">{{ item\.description }}</p>.*?<div class="match-info">.*?<p class="match-score">Relevance score: {{ "%.2f"\|format\(score\) }}</p>.*?</div>.*?</div>',
        '''<div class="recommendation-card" data-title="{{ item.title }}">
                        <h2 class="title">{{ item.title }}</h2>
                        <div class="meta">
                            <span class="author">{{ item.author }}</span>
                            {% if item.item_type %}
                            <span class="type">{{ item.item_type }}</span>
                            {% endif %}
                            {% if item.genre %}
                            <span class="genre">{{ item.genre }}</span>
                            {% endif %}
                            {% if item.publication_date %}
                            <span class="date">{{ item.publication_date }}</span>
                            {% endif %}
                        </div>
                        {% if item.summary %}
                        <p class="summary">{{ item.summary }}</p>
                        {% endif %}
                        <p class="description">{{ item.description }}</p>
                        <div class="match-info">
                            <p class="match-score">Match Score: {{ item.match_score }}/100</p>
                            <p class="match-score-note">How well this fits your input</p>
                            {% if matched_terms %}
                            <p class="match-terms">Why this? Matches: {{ matched_terms|join(", ") }}</p>
                            {% endif %}
                        </div>
                        <div class="feedback-buttons">
                            <button class="feedback-btn thumbs-up" data-title="{{ item.title }}" data-feedback="1" title="This recommendation is helpful">👍</button>
                            <button class="feedback-btn thumbs-down" data-title="{{ item.title }}" data-feedback="-1" title="This recommendation is not helpful">👎</button>
                        </div>
                    </div>''',
        content,
        flags=re.DOTALL
    )
    
    # Update the trending recommendations section similarly
    content = re.sub(
        r'<div class="recommendation-card trending">.*?<h2 class="title">{{ item\.title }}</h2>.*?<p class="description">{{ item\.description }}</p>.*?<div class="match-info">.*?<p class="match-score">Relevance score: {{ "%.2f"\|format\(score\) }}</p>.*?</div>.*?</div>',
        '''<div class="recommendation-card trending" data-title="{{ item.title }}">
                        <h2 class="title">{{ item.title }}</h2>
                        <div class="meta">
                            <span class="author">{{ item.author }}</span>
                            {% if item.item_type %}
                            <span class="type">{{ item.item_type }}</span>
                            {% endif %}
                            {% if item.genre %}
                            <span class="genre">{{ item.genre }}</span>
                            {% endif %}
                            {% if item.publication_date %}
                            <span class="date">{{ item.publication_date }}</span>
                            {% endif %}
                        </div>
                        {% if item.summary %}
                        <p class="summary">{{ item.summary }}</p>
                        {% endif %}
                        <p class="description">{{ item.description }}</p>
                        <div class="match-info">
                            <p class="match-score">Match Score: {{ item.match_score }}/100</p>
                            <p class="match-score-note">How well this fits your input</p>
                            {% if matched_terms %}
                            <p class="match-terms">Why this? Matches: {{ matched_terms|join(", ") }}</p>
                            {% endif %}
                        </div>
                        <div class="feedback-buttons">
                            <button class="feedback-btn thumbs-up" data-title="{{ item.title }}" data-feedback="1" title="This recommendation is helpful">👍</button>
                            <button class="feedback-btn thumbs-down" data-title="{{ item.title }}" data-feedback="-1" title="This recommendation is not helpful">👎</button>
                        </div>
                    </div>''',
        content,
        flags=re.DOTALL
    )
    
    # Add JavaScript for handling feedback
    content = re.sub(
        r'</body>',
        '''    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Set up feedback buttons
            document.querySelectorAll('.feedback-btn').forEach(button => {
                button.addEventListener('click', function() {
                    const title = this.getAttribute('data-title');
                    const feedback = parseInt(this.getAttribute('data-feedback'));
                    const card = this.closest('.recommendation-card');
                    
                    // Remove active class from both buttons in this card
                    card.querySelectorAll('.feedback-btn').forEach(btn => {
                        btn.classList.remove('active');
                    });
                    
                    // Add active class to the clicked button
                    this.classList.add('active');
                    
                    // Send feedback to the server
                    fetch('/feedback', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            title: title,
                            feedback: feedback
                        })
                    })
                    .then(response => response.json())
                    .then(data => {
                        console.log('Feedback submitted:', data);
                    })
                    .catch(error => {
                        console.error('Error submitting feedback:', error);
                    });
                });
            });
        });
    </script>
</body>''',
        content
    )
    
    # Write the updated content back to the file
    with open(file_path, 'w') as f:
        f.write(content)
    
    logger.info(f"Updated {file_path} with feedback buttons and match scores")

if __name__ == "__main__":
    # Update the literature_logic.py file
    update_literature_logic()
    
    # Update the recommendations.html template
    update_recommendations_template()
    
    print("Updates completed successfully!")
    print("The LiteratureDiscovery app now has:")
    print("1. User feedback tracking (thumbs up/down)")
    print("2. Enhanced SQLite database with feedback table")
    print("3. Richer context with summaries via Perplexity API")
    print("4. Match scores for recommendation quality (0-100 scale)")
    print("\nTo test the changes, run the Flask app and submit a literature input.")
