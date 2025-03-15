"""
Script to run the Luminaria Book Recommender with settings to allow external connections.
"""
import os
import sys
import argparse

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the Flask app
from LiteratureDiscovery.app import app

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the Luminaria Book Recommender')
    parser.add_argument('--port', type=int, default=8080, help='Port to run the server on')
    args = parser.parse_args()
    
    print("Starting Luminaria Book Recommender...")
    print(f"Application will be accessible at http://localhost:{args.port}")
    
    # Run with host='0.0.0.0' to allow external connections
    app.run(debug=True, host='0.0.0.0', port=args.port)
