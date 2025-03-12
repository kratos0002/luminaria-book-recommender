# Luminaria Upgrade Guide

This guide provides instructions for upgrading the LiteratureDiscovery application to Luminaria, including the implementation of book details pages and reading list functionality.

## Files Created

1. **book_details.py**: Contains functionality for fetching book details, managing reading lists, and caching.
2. **book_routes.py**: Contains Flask routes for the book details page and reading list operations.
3. **templates/book.html**: HTML template for displaying detailed information about a book.

## Required Changes to Existing Files

### 1. Update app.py

Add the following imports at the top of the file:

```python
# Import book details functionality
from book_details import extend_db_schema, book_cache, recs_cache

# Import book routes
from book_routes import register_book_routes
```

Add the following code just before the `if __name__ == "__main__":` section:

```python
# Initialize the extended database schema for book functionality
extend_db_schema()

# Register book routes
app = register_book_routes(app)
```

Update the startup messages in the `__main__` section:

```python
if __name__ == "__main__":
    # Print startup message
    print("Caching enabled for Luminaria") if CACHING_ENABLED else print("Caching disabled for Luminaria")
    print("Frontend enabled for Luminaria") if FRONTEND_ENABLED else print("Frontend disabled for Luminaria")
    print("Single-input recommendation system active")
    print("Enhanced recommendation quality active")
    print("User history tracking active")
    print("Book details functionality active")
    print("Reading list functionality active")
    print("Luminaria recommendation engine ready!")
    
    # Initialize the database
    init_db()
    
    # Run the Flask application
    # Parse command line arguments for port
    import argparse
    parser = argparse.ArgumentParser(description='Run the Luminaria Flask application')
    parser.add_argument('--port', type=int, default=5003, help='Port to run the server on')
    args = parser.parse_args()

    # Run the Flask application
    app.run(debug=True, port=args.port)
```

### 2. Update templates/recommendations.html

Change the title in the head section:

```html
<title>Your Personalized Recommendations - Luminaria</title>
```

Update the card template to include links to the book details page. Find the section where recommendation cards are rendered and update the title to include a link:

For core recommendations (around line 400):
```html
<h5 class="card-title"><a href="{{ url_for('book.book_details', title=item.title) }}">{{ item.title }}</a></h5>
```

For trending recommendations (around line 460):
```html
<h5 class="card-title"><a href="{{ url_for('book.book_details', title=item.title) }}">{{ item.title }}</a></h5>
```

Update the footer:
```html
<footer>
    <p>&copy; 2025 Luminaria - Powered by Perplexity API</p>
</footer>
```

## Testing the Implementation

1. Start the Flask application:
   ```
   python app.py --port 5003
   ```

2. Open a web browser and navigate to `http://localhost:5003`

3. Enter a query like "love in the time of cholera" and submit

4. Click on one of the book recommendations to view its details page

5. Test the "Save to Reading List" functionality by clicking the save button

## Troubleshooting

If you encounter any issues:

1. **Missing module errors**: Ensure all required modules are installed:
   ```
   pip install flask requests python-dotenv openai==0.28 cachetools
   ```

2. **API key errors**: Verify that your `.env` file contains valid API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   PERPLEXITY_API_KEY=your_perplexity_api_key
   ```

3. **Database errors**: Check that the database is properly initialized:
   ```python
   from literature_logic import init_db
   from book_details import extend_db_schema
   init_db()
   extend_db_schema()
   ```

4. **Template errors**: Ensure that the templates directory contains all required HTML files.

## Important Notes

1. The OpenAI API configuration uses version 0.28 with the module-level approach. Do not attempt to upgrade to newer OpenAI client patterns without explicit permission.

2. The Perplexity API is configured to use the "sonar" model. Do not change this model name.

3. The application handles both JSON and form data submissions to ensure compatibility with both API calls and web interface interactions.

4. Book details are cached for 24 hours to improve performance and reduce API calls.

5. The reading list functionality is implemented using SQLite for persistence.

## Next Steps

Consider the following enhancements for future development:

1. Implement user authentication for personalized reading lists
2. Add book cover images using a book cover API
3. Enhance the recommendation algorithm with collaborative filtering
4. Add social sharing features for book recommendations
5. Implement a search function for finding books in the database
