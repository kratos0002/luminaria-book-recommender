"""
Book routes for the Luminaria application.
This module contains Flask routes for book details and reading list functionality.
"""

from flask import Blueprint, request, jsonify, render_template, redirect, url_for, session, flash
from LiteratureDiscovery.book_details import get_book_details, save_to_reading_list, is_in_reading_list, recs_cache, get_reading_list

# Create a Blueprint for book routes
book_bp = Blueprint('book', __name__)

@book_bp.route('/book', methods=['GET'])
def book_details():
    """
    Route for displaying book details page.
    """
    # Get the book title from the query parameter
    title = request.args.get('title')
    goodreads_id = request.args.get('goodreads_id', '')
    
    if not title:
        flash("No book title provided")
        return redirect(url_for('get_recommendations_route'))
    
    # Get the session ID from the cookie
    session_id = request.cookies.get('session_id')
    
    # Get book details
    book_info = get_book_details(title, session_id)
    if not book_info:
        flash(f"Could not retrieve details for '{title}'")
        return redirect(url_for('get_recommendations_route'))
    
    # Add Goodreads ID to book info if provided
    if goodreads_id and 'goodreads_id' not in book_info:
        book_info['goodreads_id'] = goodreads_id
    
    # Check if the book is in the reading list
    book_info['is_saved'] = is_in_reading_list(session_id, title, goodreads_id)
    
    # Render the book details template
    return render_template('book.html', book=book_info)

@book_bp.route('/save', methods=['POST'])
def save_book():
    """
    Route for saving a book to the user's reading list.
    """
    # Get the book title and Goodreads ID from the request
    if request.is_json:
        data = request.get_json()
        title = data.get('title')
        goodreads_id = data.get('goodreads_id', '')
    else:
        title = request.form.get('title')
        goodreads_id = request.form.get('goodreads_id', '')
    
    if not title:
        return jsonify({"success": False, "error": "No book title provided"}), 400
    
    # Get the session ID from the cookie
    session_id = request.cookies.get('session_id')
    if not session_id:
        return jsonify({"success": False, "error": "No session ID found"}), 400
    
    # Save the book to the reading list
    success = save_to_reading_list(session_id, title, goodreads_id)
    
    return jsonify({"success": success})

@book_bp.route('/feedback', methods=['POST'])
def book_feedback():
    """
    Route for submitting feedback on a book recommendation.
    """
    # Get the feedback data from the request
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form
    
    title = data.get('title')
    feedback = data.get('feedback')
    
    if not title or feedback is None:
        return jsonify({"success": False, "error": "Missing title or feedback"}), 400
    
    # Get the session ID from the cookie
    session_id = request.cookies.get('session_id')
    if not session_id:
        return jsonify({"success": False, "error": "No session ID found"}), 400
    
    # Store the feedback (in a real application, you would save this to a database)
    # For now, we'll just log it
    print(f"Received feedback for '{title}': {feedback} from session {session_id}")
    
    return jsonify({"success": True})

@book_bp.route('/my_books', methods=['GET'])
def my_books():
    """
    Route for displaying the user's reading list.
    """
    # Get the session ID from the cookie
    session_id = request.cookies.get('session_id')
    
    if not session_id:
        flash("Please browse some books first to create a session")
        return redirect(url_for('index'))
    
    # Get the user's reading list
    books = get_reading_list(session_id)
    
    # Render the my_books template
    return render_template('my_books.html', books=books)

# Function to register the blueprint with the Flask app
def register_book_routes(app):
    app.register_blueprint(book_bp)
    
    # Update the recommendations route to store recommendations in the cache
    original_get_recommendations = app.view_functions['get_recommendations_route']
    
    def get_recommendations_with_cache():
        response = original_get_recommendations()
        
        # If the response is a template response, extract the recommendations
        if hasattr(response, 'context') and 'recommendations' in response.context:
            session_id = request.cookies.get('session_id')
            if session_id:
                recs = response.context['recommendations']
                recs_cache[session_id] = {
                    'core': [(item, score, terms) for item, score, terms in recs.get('core', [])],
                    'trending': [(item, score, terms) for item, score, terms in recs.get('trending', [])]
                }
        
        return response
    
    # Replace the original function with our wrapped version
    app.view_functions['get_recommendations_route'] = get_recommendations_with_cache
    
    return app
