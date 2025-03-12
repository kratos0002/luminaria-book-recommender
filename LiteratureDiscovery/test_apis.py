import os
import sys
import json
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

def test_openai_api():
    """Test the OpenAI API with the module-level approach"""
    print("\n--- Testing OpenAI API ---")
    
    if not OPENAI_API_KEY:
        print("Error: OpenAI API key not found in environment variables")
        return False
    
    try:
        # Use module-level approach for OpenAI (compatible with OpenAI 1.12.0)
        import openai
        openai.api_key = OPENAI_API_KEY
        
        print("Querying OpenAI for themes...")
        
        # Using the older API pattern that works with OpenAI 1.12.0
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a literary expert helping extract themes from user input."},
                {"role": "user", "content": """
                Extract 3-5 key literary themes or interests from this text: "Crime and Punishment"
                
                Return ONLY a comma-separated list of single words or short phrases (2-3 words max).
                Example: mystery, psychological thriller, redemption
                
                DO NOT include any explanations, headers, or additional text.
                """}
            ],
            temperature=0.3,
            max_tokens=100
        )
        
        # Extract the content from the response
        if hasattr(response, 'choices') and len(response.choices) > 0:
            content = response.choices[0].message.content.strip()
            print(f"OpenAI returned themes: {content}")
            
            # Split by commas and clean up
            themes = [theme.strip() for theme in content.split(',')]
            print(f"Parsed themes: {themes}")
            return True
        else:
            print("Error: Unexpected response structure from OpenAI")
            return False
    except Exception as e:
        print(f"Error querying OpenAI API: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

def test_perplexity_api():
    """Test the Perplexity API"""
    print("\n--- Testing Perplexity API ---")
    
    if not PERPLEXITY_API_KEY:
        print("Error: Perplexity API key not found in environment variables")
        return False
    
    try:
        # Prepare the prompt for Perplexity
        prompt = """List 3 books related to existentialism.
        For each item, provide:
        1. Title
        2. Author
        3. Type (book, poem, essay, etc.)
        4. Brief description highlighting key themes
        
        Format your response as a list with clear sections for each item.
        """
        
        print("Querying Perplexity API...")
        
        # Prepare the API call with the correct model name
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "sonar",  # Using just "sonar" as specified
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are a literary expert specializing in book recommendations."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 1000
            }
        )
        
        if response.status_code == 200:
            response_data = response.json()
            print(f"Received response from Perplexity API with status code: {response.status_code}")
            
            # Extract the content from the response
            if "choices" in response_data and len(response_data["choices"]) > 0:
                content = response_data["choices"][0]["message"]["content"].strip()
                print(f"Perplexity content preview: {content[:200]}...")
                return True
            else:
                print(f"Error: Unexpected response structure from Perplexity: {response_data}")
                return False
        else:
            print(f"Error: Failed to query Perplexity: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"Error querying Perplexity API: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    print("Testing API connections...")
    
    openai_success = test_openai_api()
    perplexity_success = test_perplexity_api()
    
    print("\n--- Test Results ---")
    print(f"OpenAI API: {'SUCCESS' if openai_success else 'FAILED'}")
    print(f"Perplexity API: {'SUCCESS' if perplexity_success else 'FAILED'}")
    
    if openai_success and perplexity_success:
        print("\nAll API tests passed successfully!")
        sys.exit(0)
    else:
        print("\nOne or more API tests failed.")
        sys.exit(1)
