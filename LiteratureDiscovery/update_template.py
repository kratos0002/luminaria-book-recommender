"""
Script to update recommendations.html template
"""
import os

def update_recommendations_template():
    """Update the recommendations.html template to show core and trending sections"""
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                "templates", "recommendations.html")
    
    # Check if the file exists
    if not os.path.exists(template_path):
        print(f"Template file not found: {template_path}")
        return
    
    # Read the current template
    with open(template_path, 'r') as f:
        content = f.read()
    
    # Check if the template already has core and trending sections
    if "Core Recommendations" in content and "Trending Now" in content:
        print("Template already has core and trending sections")
        return
    
    # Find the recommendations section
    recs_start = content.find("{% if recommendations %}")
    if recs_start == -1:
        print("Recommendations section not found in template")
        return
    
    # Find the end of the recommendations section
    recs_end = content.find("{% endif %}", recs_start)
    if recs_end == -1:
        print("End of recommendations section not found in template")
        return
    
    # Extract the recommendations section
    recs_section = content[recs_start:recs_end + 10]  # Include the {% endif %}
    
    # Create the new recommendations section with core and trending
    new_recs_section = """{% if recommendations %}
    <div class="recommendations-container">
        <h2>Core Recommendations</h2>
        <div class="recommendations">
            {% for item, score, matched_terms in recommendations.core %}
            <div class="recommendation-card">
                <h3>{{ item.title }}</h3>
                <p class="author">by {{ item.author }}</p>
                <p class="type">{{ item.item_type }}</p>
                <p class="description">{{ item.description }}</p>
                <div class="matched-terms">
                    <span class="score">Score: {{ score|round(1) }}</span>
                    <span class="terms">Matched: {{ matched_terms|join(', ') }}</span>
                </div>
            </div>
            {% endfor %}
        </div>
        
        <h2>Trending Now</h2>
        <div class="recommendations">
            {% for item, score, matched_terms in recommendations.trending %}
            <div class="recommendation-card trending">
                <h3>{{ item.title }}</h3>
                <p class="author">by {{ item.author }}</p>
                <p class="type">{{ item.item_type }}</p>
                <p class="description">{{ item.description }}</p>
                <div class="matched-terms">
                    <span class="score">Score: {{ score|round(1) }}</span>
                    <span class="terms">Matched: {{ matched_terms|join(', ') }}</span>
                </div>
            </div>
            {% endfor %}
        </div>
        
        <div class="terms-container">
            <h3>Based on these themes:</h3>
            <div class="terms">
                {% for term in recommendations.terms %}
                <span class="term">{{ term }}</span>
                {% endfor %}
            </div>
        </div>
        
        {% if recommendations.history %}
        <div class="history-container">
            <h3>Your recent searches:</h3>
            <div class="history">
                {% for item in recommendations.history %}
                <span class="history-item">{{ item }}</span>
                {% endfor %}
            </div>
        </div>
        {% endif %}
    </div>
{% endif %}"""
    
    # Replace the old recommendations section with the new one
    updated_content = content.replace(recs_section, new_recs_section)
    
    # Add some CSS for the trending section
    if "<style>" in updated_content:
        style_end = updated_content.find("</style>")
        if style_end != -1:
            trending_css = """
    .recommendation-card.trending {
        border-left: 4px solid #ff6b6b;
        background-color: #fff5f5;
    }
    
    h2 {
        margin-top: 30px;
        margin-bottom: 15px;
        color: #333;
        border-bottom: 1px solid #ddd;
        padding-bottom: 5px;
    }"""
            updated_content = updated_content[:style_end] + trending_css + updated_content[style_end:]
    
    # Write the updated template
    with open(template_path, 'w') as f:
        f.write(updated_content)
    
    print("Updated recommendations.html template with core and trending sections")

if __name__ == "__main__":
    update_recommendations_template()
