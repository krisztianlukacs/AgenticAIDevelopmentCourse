import os
from dotenv import load_dotenv  # Ensure this is installed

#from google import genai
import google.generativeai as genai  # Ensure google-generativeai is installed

load_dotenv()

# Configure Gemini API key globally
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

genai.configure(api_key=api_key)

# Create a generative model instance
model = genai.GenerativeModel(model_name="gemini-2.5-flash")

# Example: Generate content
response = model.generate_content("Hello, Gemini!")
print(response.text)
