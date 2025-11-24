import os
from dotenv import load_dotenv  # Ensure this is installed
import ollama

"""
pip install ollama
ollama pull gpt-oss:120b-cloud
ollama signin
ollama signout
"""

"""
ollama pull llama3-8b-8192
ollama pull mistral
ollama pull codellama

"""
load_dotenv()

response = ollama.chat(model='gpt-oss:120b-cloud', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])
print(response['message']['content'])
