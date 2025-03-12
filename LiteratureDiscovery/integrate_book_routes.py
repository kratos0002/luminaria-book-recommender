"""
Integration script for Luminaria (formerly LiteratureDiscovery)

This script helps integrate the book details functionality into the main app.py file.
Follow the instructions below to update your application.
"""

print("Luminaria Integration Guide")
print("==========================")
print("\nFollow these steps to update your application:\n")

print("1. Add the following imports to app.py at the top of the file:")
print("""
# Import book details functionality
from book_details import extend_db_schema, book_cache, recs_cache

# Import book routes
from book_routes import register_book_routes
""")

print("\n2. Add the following code at the end of app.py, just before 'if __name__ == \"__main__\":':")
print("""
# Initialize the extended database schema for book functionality
extend_db_schema()

# Register book routes
app = register_book_routes(app)
""")

print("\n3. Update the startup messages in the '__main__' section:")
print("""
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
""")

print("\n4. Update the recommendations.html template to link to the book details page:")
print("""
In the card template section, change the title to include a link:
<h5 class="card-title"><a href="{{ url_for('book.book_details', title=item.title) }}">{{ item.title }}</a></h5>
""")

print("\nAfter making these changes, run your application with:")
print("python app.py --port 5003")
print("\nYou should now have a fully functional Luminaria application with book details!")
