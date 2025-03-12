# Luminaria - Book Recommendation System

Luminaria is a sophisticated book recommendation system that helps users discover new literature based on their interests and preferences.

## Features

- **Personalized Recommendations**: Get book recommendations based on your literary interests
- **Trending Books**: Discover trending books that match your preferences
- **Book Details**: View detailed information about each recommended book
- **Reading List**: Save books to your reading list for future reference
- **Match Scoring**: See how well each recommendation matches your search criteria

## Technology Stack

- **Backend**: Flask (Python)
- **APIs**: 
  - OpenAI API for understanding user preferences
  - Perplexity API for book recommendations and details
- **Database**: SQLite for user history and reading list storage
- **Frontend**: HTML, CSS, JavaScript

## Installation

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Create a `.env` file with your API keys (see `.env.example`)
6. Run the application: `python LiteratureDiscovery/app.py`

## Usage

1. Enter a book title, author, or literary theme in the search box
2. View personalized recommendations based on your input
3. Click on book titles to view detailed information
4. Add books to your reading list for future reference

## License

MIT License

## Acknowledgements

- OpenAI for providing the GPT-3.5 API
- Perplexity for their powerful search API
