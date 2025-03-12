"""
Script to update the recommendation functions to include summaries and match scores
"""
import re
import os
import logging
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def backup_file(file_path):
    """Create a backup of the file before modifying it."""
    backup_path = f"{file_path}.bak"
    if not os.path.exists(backup_path):
        shutil.copy2(file_path, backup_path)
        logger.info(f"Created backup at {backup_path}")

def update_trending_literature():
    """Update the get_trending_literature function to include summaries."""
    file_path = "literature_logic.py"
    
    # Create a backup
    backup_file(file_path)
    
    # Read the current content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the prompt in get_trending_literature
    prompt_pattern = r'prompt = \(\s*f"List.*?matching themes.*?\)'
    prompt_match = re.search(prompt_pattern, content, re.DOTALL)
    
    if not prompt_match:
        logger.error("Could not find prompt in get_trending_literature")
        return False
    
    # Replace with updated prompt
    updated_prompt = '''prompt = (
        f"List 10 narrative books or short stories (no plays, nonfiction, essays, poetry) "
        f"from 19th-century or classic literature matching themes [{terms_str}]. "
        f"{exclude_str} Include title, type, source, description with author name, "
        f"a 2-3 sentence summary."
    )'''
    
    content = content.replace(prompt_match.group(0), updated_prompt)
    
    # Write the updated content back to the file
    with open(file_path, 'w') as f:
        f.write(content)
    
    logger.info("Updated get_trending_literature function to include summaries")
    return True

def update_literary_trends():
    """Update the get_literary_trends function to include summaries."""
    file_path = "literature_logic.py"
    
    # Read the current content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the prompt in get_literary_trends
    prompt_pattern = r'def get_literary_trends.*?prompt = \(\s*f"List.*?matching themes.*?\)'
    prompt_match = re.search(prompt_pattern, content, re.DOTALL)
    
    if not prompt_match:
        logger.error("Could not find prompt in get_literary_trends")
        return False
    
    # Extract the existing function definition
    function_def = re.search(r'def get_literary_trends.*?""".*?"""', prompt_match.group(0), re.DOTALL).group(0)
    
    # Replace with updated prompt
    updated_prompt = function_def + '''
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
    )'''
    
    content = content.replace(prompt_match.group(0), updated_prompt)
    
    # Write the updated content back to the file
    with open(file_path, 'w') as f:
        f.write(content)
    
    logger.info("Updated get_literary_trends function to include summaries")
    return True

def update_parse_literature_items():
    """Update the parse_literature_items function to extract summaries."""
    file_path = "literature_logic.py"
    
    # Read the current content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the item creation in parse_literature_items
    item_pattern = r'# Create and add the item.*?item = LiteratureItem\(.*?title=title,.*?author=author,.*?item_type=item_type,.*?description=description.*?\)'
    item_match = re.search(item_pattern, content, re.DOTALL)
    
    if not item_match:
        logger.error("Could not find item creation in parse_literature_items")
        return False
    
    # Insert summary extraction before item creation
    summary_extraction = '''            # Extract summary - look for a section that seems like a summary
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
            
            # Create and add the item'''
    
    content = content.replace("            # Create and add the item", summary_extraction)
    
    # Update the LiteratureItem creation to include summary
    updated_item = '''            item = LiteratureItem(
                title=title,
                author=author,
                item_type=item_type,
                description=description,
                summary=summary
            )'''
    
    content = re.sub(r'item = LiteratureItem\(.*?title=title,.*?author=author,.*?item_type=item_type,.*?description=description.*?\)', 
                    updated_item, content, flags=re.DOTALL)
    
    # Write the updated content back to the file
    with open(file_path, 'w') as f:
        f.write(content)
    
    logger.info("Updated parse_literature_items function to extract summaries")
    return True

def update_recommend_literature():
    """Update the recommend_literature function to include feedback and match scores."""
    file_path = "literature_logic.py"
    
    # Read the current content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Update the function signature to include session_id
    old_signature = "def recommend_literature(trending_items: List[LiteratureItem], user_terms: List[str], literature_input: str = None):"
    new_signature = "def recommend_literature(trending_items: List[LiteratureItem], user_terms: List[str], literature_input: str = None, session_id: str = None):"
    
    content = content.replace(old_signature, new_signature)
    
    # Add user feedback retrieval at the beginning of the function
    feedback_retrieval = '''    # Get user feedback if session_id is provided
    user_feedback = get_user_feedback(session_id) if session_id else {}
    
    # Get the author of the input literature if available
    input_author = None
    if literature_input:
        input_author = get_author(literature_input)
    '''
    
    # Find the beginning of the scoring logic
    scoring_start = "    # Score each item"
    
    content = content.replace(scoring_start, feedback_retrieval + "\n" + scoring_start)
    
    # Update the scoring logic
    old_scoring = r'for item in trending_items:.*?# Calculate final score.*?final_score = .*?# Add to scored items.*?scored_items\.append\(\(item, final_score, list\(matched_terms\)\)\)'
    old_scoring_match = re.search(old_scoring, content, re.DOTALL)
    
    if not old_scoring_match:
        logger.error("Could not find scoring logic in recommend_literature")
        return False
    
    new_scoring = '''    for item in trending_items:
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
        scored_items.append((item, final_score, list(matched_terms)))'''
    
    content = content.replace(old_scoring_match.group(0), new_scoring)
    
    # Write the updated content back to the file
    with open(file_path, 'w') as f:
        f.write(content)
    
    logger.info("Updated recommend_literature function to include feedback and match scores")
    return True

def update_get_recommendations():
    """Update the get_recommendations function to pass session_id to recommend_literature."""
    file_path = "literature_logic.py"
    
    # Read the current content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Update the core recommendations call
    old_core_rec = "core_recommendations = recommend_literature(classic_items, user_terms, literature_input)"
    new_core_rec = "core_recommendations = recommend_literature(classic_items, user_terms, literature_input, session_id)"
    
    content = content.replace(old_core_rec, new_core_rec)
    
    # Update the trending recommendations call
    old_trending_rec = "trending_recommendations = recommend_literature(trending_items, user_terms, literature_input)"
    new_trending_rec = "trending_recommendations = recommend_literature(trending_items, user_terms, literature_input, session_id)"
    
    content = content.replace(old_trending_rec, new_trending_rec)
    
    # Write the updated content back to the file
    with open(file_path, 'w') as f:
        f.write(content)
    
    logger.info("Updated get_recommendations function to pass session_id to recommend_literature")
    return True

def add_submit_feedback():
    """Add the submit_feedback function."""
    file_path = "literature_logic.py"
    
    # Read the current content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Add submit_feedback function before the if __name__ == "__main__" block
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
    
    # Find the if __name__ == "__main__" block
    main_block = "if __name__ == \"__main__\":"
    
    content = content.replace(main_block, submit_feedback_func + main_block)
    
    # Write the updated content back to the file
    with open(file_path, 'w') as f:
        f.write(content)
    
    logger.info("Added submit_feedback function")
    return True

if __name__ == "__main__":
    update_trending_literature()
    update_literary_trends()
    update_parse_literature_items()
    update_recommend_literature()
    update_get_recommendations()
    add_submit_feedback()
    print("Recommendation functions updated successfully!")
