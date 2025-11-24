import os
from dotenv import load_dotenv  # Ensure this is installed

from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")
response = llm.invoke("What is agentic AI?")
print(response)
