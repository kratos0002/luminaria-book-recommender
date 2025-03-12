"""
Script to update the recommend_literature function in literature_logic.py
"""

import re
import os

def update_recommend_literature():
    """Update the recommend_literature function"""
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "literature_logic.py")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the function definition
    pattern = r'def recommend_literature\(trending_items:.*?\):.*?return top_items'
    
    # New function code
    new_code = '''def recommend_literature(trending_items: List[LiteratureItem], user_terms: List[str], literature_input: str = None) -> List[Tuple[LiteratureItem, float, List[str]]]:
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
        # Add other author detections as needed
    
    # Normalize input for comparison
    literature_input_normalized = literature_input.lower().strip() if literature_input else None
    
    for item in trending_items:
        # Skip self-recommendations (if the item title matches the user input)
        if literature_input_normalized and item.title.lower().strip() == literature_input_normalized:
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
            score += 3.0  # Increased from 2.0 to 3.0
            logger.info(f"Applied author bonus to: {item.title} (author: {item.author})")
        
        # Add to scored items if there's at least one match
        if matched_terms:
            # Store matched terms and score in the item
            item.matched_terms = matched_terms
            item.score = score
            
            scored_items.append((item, score, list(matched_terms)))
    
    logger.info(f"Scored {len(scored_items)} items with at least one match")
    
    # Sort by score in descending order
    scored_items.sort(key=lambda x: x[1], reverse=True)
    
    # Get top 5 items
    top_items = scored_items[:5]
    
    # If we have fewer than 5 items with matches, pad with highest non-zero scores
    if len(top_items) < 5 and len(scored_items) > len(top_items):
        remaining_items = scored_items[len(top_items):]
        # Sort remaining items by score
        remaining_items.sort(key=lambda x: x[1], reverse=True)
        # Add highest scoring remaining items until we have 5 or run out
        for item in remaining_items:
            if item[1] > 0 and len(top_items) < 5:
                top_items.append(item)
    
    # Log top scores for debugging
    if top_items:
        top_scores = [f"{item[0].title[:20]}... ({item[1]})" for item in top_items[:3]]
        logger.info(f"Top scores: {', '.join(top_scores)}")
    
    return top_items'''
    
    # Replace the function
    updated_content = re.sub(pattern, new_code, content, flags=re.DOTALL)
    
    # Write back to the file
    with open(file_path, 'w') as f:
        f.write(updated_content)
    
    print("Updated recommend_literature function")

if __name__ == "__main__":
    update_recommend_literature()
