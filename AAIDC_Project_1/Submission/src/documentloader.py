from typing import List
import os
from langchain_community.document_loaders import TextLoader
from .logmanager import LogManager
import logging

"""
Document Loader Module
----------------------
This module provides functionality to load documents from the ./data/ folder. It supports loading text files (.txt) and markdown files (.md).
"""
class DocumentLoader:
    """
    A simple document loader to load documents from the ./data/ folder.
    Supports .txt and .md files.
    """

    def __init__(self, document_path: str = "./data/"):
        """Initialize the DocumentLoader."""
        self.log = LogManager()
        self.log.add_logfile("documentloader")
        self.log.write_log("documentloader", logging.INFO, "DocumentLoader initialized")
        self.document_path = document_path

    def load_documents(self) -> List[str]:
        """
        Load documents from the ./data/ folder. (.txt and .md files)    
        Returns:
            List of sample documents
        """
        results = []

        # List to store all documents
        documents = []

        # Load each .txt file in the documents folder
        for file in os.listdir(self.document_path):
            if file.endswith(".md") or file.endswith(".txt"):
                file_path = os.path.join(self.document_path, file)
                try:
                    loader = TextLoader(file_path)
                    loaded_docs = loader.load()
                    documents.extend(loaded_docs)
                    self.log.write_log("documentloader", logging.INFO, f"Successfully loaded: {file}")
                except Exception as e:
                    self.log.write_log("documentloader", logging.ERROR, f"Error loading {file}: {str(e)}")
        
        self.log.write_log("documentloader", logging.INFO, f"Total documents loaded: {len(documents)}")
        
        # Extract content as strings and return
        results = []
        for doc in documents:
            results.append(doc.page_content)
        return results