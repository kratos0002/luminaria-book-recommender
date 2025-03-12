"""
Script to update recommendations.html template to show core and trending sections
"""
import os

def read_file(file_path):
    with open(file_path, 'r') as f:
        return f.read()

def write_file(file_path, content):
    with open(file_path, 'w') as f:
        f.write(content)

def update_template():
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                "templates", "recommendations.html")
    
    # Check if the file exists
    if not os.path.exists(template_path):
        print(f"Template file not found: {template_path}")
        return
    
    # Read the current template
    content = read_file(template_path)
    
    # Find the recommendations section
    recs_start = content.find('<div class="recommendations">')
    if recs_start == -1:
        print("Recommendations section not found in template")
        return
    
    # Find the end of the recommendations section
    recs_end = content.find('</div>', recs_start)
    if recs_end == -1:
        print("End of recommendations section not found in template")
        return
    
    # Find the next div after the recommendations section
    next_div = content.find('<div class="actions">', recs_end)
    if next_div == -1:
        print("Actions section not found in template")
        return
    
    # Extract the recommendations section
    old_recs_section = content[recs_start:next_div]
    
    # Create the new recommendations section with core and trending
    new_recs_section = """<div class="recommendations-container">
                <h2>Core Recommendations</h2>
                <div class="recommendations core-recommendations">
                {% if recommendations.core %}
                    {% for item, score, matched_terms in recommendations.core %}
                    <div class="recommendation-card">
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
                        <p class="description">{{ item.description }}</p>
                        <div class="match-info">
                            <p class="match-score">Relevance score: {{ "%.2f"|format(score) }}</p>
                            {% if matched_terms %}
                            <p class="match-terms">Why this? Matches: {{ matched_terms|join(", ") }}</p>
                            {% endif %}
                        </div>
                    </div>
                    {% endfor %}
                {% else %}
                    <div class="no-results">
                        <p>No core recommendations found. Please try different preferences.</p>
                    </div>
                {% endif %}
                </div>
                
                <h2>Trending Now</h2>
                <div class="recommendations trending-recommendations">
                {% if recommendations.trending %}
                    {% for item, score, matched_terms in recommendations.trending %}
                    <div class="recommendation-card trending">
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
                        <p class="description">{{ item.description }}</p>
                        <div class="match-info">
                            <p class="match-score">Relevance score: {{ "%.2f"|format(score) }}</p>
                            {% if matched_terms %}
                            <p class="match-terms">Why this? Matches: {{ matched_terms|join(", ") }}</p>
                            {% endif %}
                        </div>
                    </div>
                    {% endfor %}
                {% else %}
                    <div class="no-results">
                        <p>No trending recommendations found. Please try different preferences.</p>
                    </div>
                {% endif %}
                </div>
            </div>"""
    
    # Replace the old recommendations section with the new one
    updated_content = content.replace(old_recs_section, new_recs_section)
    
    # Add CSS for the trending section
    style_tag = '<link rel="stylesheet" href="{{ url_for(\'static\', filename=\'style.css\') }}">'
    style_pos = updated_content.find(style_tag)
    if style_pos != -1:
        style_end = style_pos + len(style_tag)
        trending_css = """
    <style>
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
    </style>"""
        updated_content = updated_content[:style_end] + trending_css + updated_content[style_end:]
    
    # Write the updated template
    write_file(template_path, updated_content)
    print("Updated recommendations.html template with core and trending sections")

if __name__ == "__main__":
    update_template()
