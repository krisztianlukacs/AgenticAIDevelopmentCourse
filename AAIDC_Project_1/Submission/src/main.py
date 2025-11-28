from .app import RAGAssistant
from dotenv import load_dotenv
from .logmanager import LogManager
import logging
import datetime
from .documentloader import DocumentLoader
import os
import warnings

"""
The main applications entry point for the RAG-based AI assistant. 

Instructions to run:
1. Activate virtual environment (sourve venv/bin/activate
2. Install dependencies (pip install -r requirements.txt)
3. Run the application (python -m main)

Features:

1. Load documents from the ./data/ folder (MarkDown files, text files)
2. Chunk documents into smaller pieces
3. Store document chunks in a vector database (ChromaDB) with embeddings from HuggingFace
4. Use a Retrieval-Augmented Generation (RAG) approach to answer user questions by
5. Answer questions using the information it found
6. Combine multiple sources to give comprehensive answers 

"""

# Disable ChromaDB telemetry before any imports
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

# Suppress ChromaDB telemetry warnings by setting logging level
chromadb_logger = logging.getLogger('chromadb')
chromadb_logger.setLevel(logging.ERROR)

# Load environment variables
load_dotenv()
log = LogManager()
log.add_logfile("app")

def main():
    """
    Main function to run the RAG assistant.
    """

    try:
        # Initialize the RAG assistant
        print("Initializing RAG Assistant...")
        assistant = RAGAssistant()

        # Load sample documents
        print("\nLoading documents...")
        document_loader = DocumentLoader()
        sample_docs = document_loader.load_documents()
        print(f"Loaded {len(sample_docs)} sample documents")

        assistant.add_documents(sample_docs)

        done = False

        # Example interaction loop
        print("Welcome to the RAG-based AI Assistant! Type 'exit' to quit.")
        while True:
            question = input("You: ")
            if question.lower() == "quit" or question.lower() == "exit":
                print("Goodbye!")
                break

            # Get assistant response
            response = assistant.invoke(question)
            print(f"Assistant: {response}")
    except Exception as e:
        print(f"Error running RAG assistant: {e}")
        print("Make sure you have set up your .env file with at least one API key:")
        print("- OPENAI_API_KEY (OpenAI GPT models)")
        print("- GROQ_API_KEY (Groq Llama models)")
        print("- GOOGLE_API_KEY (Google Gemini models)")

if __name__ == "__main__":
    log.write_log("app", logging.INFO, "Initializing RAGAssistant Application")
    main()
    log.write_log("app", logging.INFO, "RAGAssistant Application terminated")
