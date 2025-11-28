import os
from dotenv import load_dotenv  # Ensure this is installed

import chromadb

load_dotenv()
api_key = os.getenv("CHROMADB_API_KEY")

client = chromadb.CloudClient(
  api_key=api_key,
  tenant='a82d9ca3-e4a0-4e83-9045-70f53b5ca272',
  database='DigitalInvestment'
)