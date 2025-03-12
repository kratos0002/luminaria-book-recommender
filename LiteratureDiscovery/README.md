# LiteratureDiscovery

A personalized literature recommendation engine that uses Perplexity API and Grok API to suggest diverse literary works including books, research papers, poems, philosophy papers, and short stories.

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
- Copy `.env.example` to `.env`
- Add your API keys in the `.env` file:
  - `PERPLEXITY_API_KEY`: Your Perplexity API key
  - `GROK_API_KEY`: Your Grok API key

## Running the Application

Start the Flask server:
```bash
python app.py
```

The server will run on `http://localhost:5000`

## Project Structure

- `app.py`: Main Flask application with API routes and core logic
- `models.py`: Data models for literature items
- `requirements.txt`: Project dependencies
- `.env`: Environment variables (not tracked in git)
- `.env.example`: Template for environment variables

## Features

- Fetch trending literature across various categories
- Process user preferences and interests
- Generate personalized literature recommendations
- Support for multiple literature types:
  - Books
  - Research papers
  - Poems
  - Philosophy papers
  - Short stories
