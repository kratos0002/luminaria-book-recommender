"""
Script to update the recommend_literature function in literature_logic.py
"""
import os
import re

def read_file(file_path):
    with open(file_path, 'r') as f:
        return f.read()

def write_file(file_path, content):
    with open(file_path, 'w') as f:
        f.write(content)

def update_function():
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "literature_logic.py")
    content = read_file(file_path)
    
    # Find the function
    function_start = content.find("def recommend_literature")
    if function_start == -1:
        print("recommend_literature function not found")
        return
    
    # Find the end of the function
    next_def = content.find("def ", function_start + 10)
    if next_def == -1:
        next_def = len(content)
    
    # Extract the function
    old_function = content[function_start:next_def]
    
    # Create the updated function
    new_function = """\def recommend_literature(trending_items: List[LiteratureItem], user_terms: List[str], literature_input: str = None) -> List[Tuple[LiteratureItem, float, List[str]]]:
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
    
    # Get the author of the input literature if available
    input_author = None
    if literature_input:
        input_author = get_author(literature_input)
        if input_author:
            logger.info(f"Found author for input: {input_author}")
    
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
        if input_author and input_author.lower() in author_lower:
            score += 4.0
            logger.info(f"Applied author bonus to: {item.title} (author: {item.author})")
        
        # Add to scored items if there's at least one match
        if matched_terms:
            # Store matched terms and score in the item
            item.matched_terms = list(matched_terms)
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
    
    return top_items
"""
    
    # Replace the old function with the new one
    updated_content = content.replace(old_function, new_function)
    
    # Write the updated content
    write_file(file_path, updated_content)
    print("Updated recommend_literature function in literature_logic.py")

if __name__ == "__main__":
    update_function()
