import openai
import os 
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()
client = openai.OpenAI()

def generate_content(summary):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant who creates student-friendly presentation scripts."},
            {"role": "user", "content": f"Create a student-friendly presentation script from this summary: {summary}. Return data in valid JSON format with an array of slides, each having 'title' and 'body' fields."},
        ],
        max_tokens=1000
    )
    
    content = response.choices[0].message.content
    
    # Try to parse the JSON response
    try:
        # Extract JSON if it's wrapped in markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        # Parse the JSON string into Python object
        slides = json.loads(content)
        
        return slides
        
    except json.JSONDecodeError:
        # If JSON parsing fails, fall back to the text parsing method
        print("Failed to parse JSON. Using text parsing fallback.")
        return content