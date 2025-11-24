import os
from dotenv import load_dotenv  # Ensure this is installed

from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
response = llm.invoke("Explain prompt engineering in simple terms.")
print(response)
