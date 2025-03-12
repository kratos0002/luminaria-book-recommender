"""
Test script to verify the improvements to the LiteratureDiscovery application
"""

import os
import sys
from literature_logic import (
    init_db, store_user_input, get_user_history, 
    get_user_preferences, get_trending_literature, recommend_literature,
    test_recommendations
)

def main():
    # Initialize the database
    print("Initializing database...")
    init_db()
    
    # Test session ID
    session_id = "test_session_123"
    
    # Test inputs
    test_inputs = [
        "The Brothers Karamazov",
        "Crime and Punishment",
        "The Idiot"
    ]
    
    # Store test inputs
    for input_text in test_inputs:
        store_user_input(session_id, input_text)
        print(f"Stored input: {input_text}")
    
    # Get user history
    history = get_user_history(session_id)
    print(f"\nUser history: {history}")
    
    # Test recommendations for each input
    for input_text in test_inputs:
        print("\n" + "="*80)
        print(f"TESTING: {input_text}")
        print("="*80)
        
        # Use the test_recommendations function
        test_recommendations(input_text, session_id)
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()
