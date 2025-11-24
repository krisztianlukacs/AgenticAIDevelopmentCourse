import os
from dotenv import load_dotenv  # Ensure this is installed

from langchain_community.chat_models import ChatOllama

load_dotenv()

llm = ChatOllama(model="mistral")
response = llm.invoke("Summarize how local models work.")
print(response)