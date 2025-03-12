"""
Script to update the recommendations.html template to include feedback buttons and match scores
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

def update_recommendations_template():
    """Update the recommendations.html template to include feedback buttons and match scores."""
    file_path = "templates/recommendations.html"
    
    # Create a backup
    backup_file(file_path)
    
    # Read the current content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Add CSS for feedback buttons
    style_tag = "<style>"
    feedback_css = """<style>
        .recommendation-card.trending {
            border-left: 4px solid #ff6b6b;
            background-color: #fff5f5;
        }
        
        .recommendations-container h2 {
            margin-top: 30px;
            margin-bottom: 15px;
            color: #333;
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
        }
        
        .core-recommendations, .trending-recommendations {
            margin-bottom: 30px;
        }
        
        .feedback-buttons {
            margin-top: 10px;
            display: flex;
            gap: 10px;
        }
        .feedback-btn {
            cursor: pointer;
            background: none;
            border: 1px solid #ddd;
            border-radius: 50%;
            width: 36px;
            height: 36px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            transition: all 0.2s;
        }
        .feedback-btn:hover {
            background-color: #f0f0f0;
        }
        .feedback-btn.thumbs-up:hover {
            border-color: #4CAF50;
            color: #4CAF50;
        }
        .feedback-btn.thumbs-down:hover {
            border-color: #F44336;
            color: #F44336;
        }
        .feedback-btn.active.thumbs-up {
            background-color: #E8F5E9;
            border-color: #4CAF50;
            color: #4CAF50;
        }
        .feedback-btn.active.thumbs-down {
            background-color: #FFEBEE;
            border-color: #F44336;
            color: #F44336;
        }
        .match-score {
            font-weight: bold;
            margin-bottom: 5px;
        }
        .match-score-note {
            font-size: 0.8em;
            color: #666;
            margin-top: 0;
        }
        .summary {
            font-style: italic;
            margin: 10px 0;
            color: #555;
            line-height: 1.4;
        }"""
    
    content = content.replace(style_tag, feedback_css)
    
    # Update the core recommendations section to include summary and feedback buttons
    core_card_pattern = r'<div class="recommendation-card">.*?<h2 class="title">{{ item\.title }}</h2>.*?<p class="description">{{ item\.description }}</p>.*?<div class="match-info">.*?<p class="match-score">Relevance score: {{ "%.2f"\|format\(score\) }}</p>.*?</div>.*?</div>'
    core_card_match = re.search(core_card_pattern, content, re.DOTALL)
    
    if not core_card_match:
        logger.error("Could not find core recommendation card")
        return False
    
    updated_core_card = '''<div class="recommendation-card" data-title="{{ item.title }}">
                        <h2 class="title">{{ item.title }}</h2>
                        <div class="meta">
                            <span class="author">{{ item.author }}</span>
                            {% if item.item_type %}
                            <span class="type">{{ item.item_type }}</span>
                            {% endif %}
                            {% if item.genre %}
                            <span class="genre">{{ item.genre }}</span>
                            {% endif %}
                            {% if item.publication_date %}
                            <span class="date">{{ item.publication_date }}</span>
                            {% endif %}
                        </div>
                        {% if item.summary %}
                        <p class="summary">{{ item.summary }}</p>
                        {% endif %}
                        <p class="description">{{ item.description }}</p>
                        <div class="match-info">
                            <p class="match-score">Match Score: {{ item.match_score }}/100</p>
                            <p class="match-score-note">How well this fits your input</p>
                            {% if matched_terms %}
                            <p class="match-terms">Why this? Matches: {{ matched_terms|join(", ") }}</p>
                            {% endif %}
                        </div>
                        <div class="feedback-buttons">
                            <button class="feedback-btn thumbs-up" data-title="{{ item.title }}" data-feedback="1" title="This recommendation is helpful">👍</button>
                            <button class="feedback-btn thumbs-down" data-title="{{ item.title }}" data-feedback="-1" title="This recommendation is not helpful">👎</button>
                        </div>
                    </div>'''
    
    content = content.replace(core_card_match.group(0), updated_core_card)
    
    # Update the trending recommendations section similarly
    trending_card_pattern = r'<div class="recommendation-card trending">.*?<h2 class="title">{{ item\.title }}</h2>.*?<p class="description">{{ item\.description }}</p>.*?<div class="match-info">.*?<p class="match-score">Relevance score: {{ "%.2f"\|format\(score\) }}</p>.*?</div>.*?</div>'
    trending_card_match = re.search(trending_card_pattern, content, re.DOTALL)
    
    if not trending_card_match:
        logger.error("Could not find trending recommendation card")
        return False
    
    updated_trending_card = '''<div class="recommendation-card trending" data-title="{{ item.title }}">
                        <h2 class="title">{{ item.title }}</h2>
                        <div class="meta">
                            <span class="author">{{ item.author }}</span>
                            {% if item.item_type %}
                            <span class="type">{{ item.item_type }}</span>
                            {% endif %}
                            {% if item.genre %}
                            <span class="genre">{{ item.genre }}</span>
                            {% endif %}
                            {% if item.publication_date %}
                            <span class="date">{{ item.publication_date }}</span>
                            {% endif %}
                        </div>
                        {% if item.summary %}
                        <p class="summary">{{ item.summary }}</p>
                        {% endif %}
                        <p class="description">{{ item.description }}</p>
                        <div class="match-info">
                            <p class="match-score">Match Score: {{ item.match_score }}/100</p>
                            <p class="match-score-note">How well this fits your input</p>
                            {% if matched_terms %}
                            <p class="match-terms">Why this? Matches: {{ matched_terms|join(", ") }}</p>
                            {% endif %}
                        </div>
                        <div class="feedback-buttons">
                            <button class="feedback-btn thumbs-up" data-title="{{ item.title }}" data-feedback="1" title="This recommendation is helpful">👍</button>
                            <button class="feedback-btn thumbs-down" data-title="{{ item.title }}" data-feedback="-1" title="This recommendation is not helpful">👎</button>
                        </div>
                    </div>'''
    
    content = content.replace(trending_card_match.group(0), updated_trending_card)
    
    # Add JavaScript for handling feedback
    body_end_tag = "</body>"
    feedback_js = '''    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Set up feedback buttons
            document.querySelectorAll('.feedback-btn').forEach(button => {
                button.addEventListener('click', function() {
                    const title = this.getAttribute('data-title');
                    const feedback = parseInt(this.getAttribute('data-feedback'));
                    const card = this.closest('.recommendation-card');
                    
                    // Remove active class from both buttons in this card
                    card.querySelectorAll('.feedback-btn').forEach(btn => {
                        btn.classList.remove('active');
                    });
                    
                    // Add active class to the clicked button
                    this.classList.add('active');
                    
                    // Send feedback to the server
                    fetch('/feedback', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            title: title,
                            feedback: feedback
                        })
                    })
                    .then(response => response.json())
                    .then(data => {
                        console.log('Feedback submitted:', data);
                    })
                    .catch(error => {
                        console.error('Error submitting feedback:', error);
                    });
                });
            });
        });
    </script>
</body>'''
    
    content = content.replace(body_end_tag, feedback_js)
    
    # Write the updated content back to the file
    with open(file_path, 'w') as f:
        f.write(content)
    
    logger.info("Updated recommendations.html template with feedback buttons and match scores")
    return True

if __name__ == "__main__":
    update_recommendations_template()
    print("Recommendations template updated successfully!")
