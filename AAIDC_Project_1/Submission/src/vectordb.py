import logging
import os
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from .logmanager import LogManager
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter


class VectorDB:
    """
    A simple vector database wrapper using ChromaDB with HuggingFace embeddings.
    """

    def __init__(self, collection_name: str = "", embedding_model: str = ""):
        """
        Initialize the vector database.

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: HuggingFace model name for embeddings
        """
        self.log = LogManager()
        self.log.add_logfile("vectordb")
        self.log.write_log("vectordb", logging.INFO, "VectorDB initialized")
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "rag_documents"
        )
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize ChromaDB client with telemetry disabled
        settings = Settings(
            anonymized_telemetry=False
        )
        self.client = chromadb.PersistentClient(path="./chroma_db", settings=settings)

        # Load embedding model
        self.log.write_log("vectordb", logging.INFO, f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document collection"},
        )

        self.log.write_log("vectordb", logging.INFO, f"Vector database initialized with collection: {self.collection_name}")

    def chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """
        Break text into searchable chunks using LangChain's RecursiveCharacterTextSplitter.
        This method intelligently splits on paragraph breaks, sentences, and words
        while maintaining context through overlapping chunks.

        Args:
            text: Input text to chunk
            chunk_size: Approximate number of characters per chunk (default: 1000)

        Returns:
            List of text chunks
        """
        # Use LangChain's RecursiveCharacterTextSplitter for intelligent chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,          # ~200-250 words per chunk
            chunk_overlap=100,              # Overlap to preserve context between chunks
            separators=["\n\n", "\n", ". ", " ", ""],  # Try to split on these in order
            length_function=len,
        )
        
        chunks = text_splitter.split_text(text)
        
        return chunks

    def add_documents(self, documents: List) -> None:
        """
        Add documents to the vector database.

        Args:
            documents: List of documents
        """
        # TODO: Implement document ingestion logic
        # HINT: Loop through each document in the documents list
        # HINT: Extract 'content' and 'metadata' from each document dict
        # HINT: Use self.chunk_text() to split each document into chunks
        # HINT: Create unique IDs for each chunk (e.g., "doc_0_chunk_0")
        # HINT: Use self.embedding_model.encode() to create embeddings for all chunks
        # HINT: Store the embeddings, documents, metadata, and IDs in your vector database
        # HINT: Print progress messages to inform the user

        self.log.write_log("vectordb", logging.INFO, f"Processing {len(documents)} documents...")
        
        all_chunks = []
        all_ids = []
        all_metadatas = []
        
        for doc_idx, doc in enumerate(documents):
            # Handle both string documents and dict documents
            if isinstance(doc, str):
                content = doc
                metadata = {"source": f"document_{doc_idx}"}
            else:
                content = doc.get('content', str(doc))
                metadata = doc.get('metadata', {"source": f"document_{doc_idx}"})
            
            # Split document into chunks with better chunk size
            chunks = self.chunk_text(content, chunk_size=250)
            
            # Create IDs and metadata for each chunk
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_ids.append(f"doc_{doc_idx}_chunk_{chunk_idx}")
                chunk_metadata = metadata.copy()
                chunk_metadata['chunk_index'] = str(chunk_idx)  # Convert to string for ChromaDB
                all_metadatas.append(chunk_metadata)
        
        if all_chunks:
            # Create embeddings for all chunks
            embeddings = self.embedding_model.encode(all_chunks)
            
            # Add to ChromaDB collection
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=all_chunks,
                metadatas=all_metadatas,
                ids=all_ids
            )
            self.log.write_log("vectordb", logging.INFO, f"Added {len(all_chunks)} chunks from {len(documents)} documents to vector database")
        
        self.log.write_log("vectordb", logging.INFO, "Documents added to vector database")

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search for similar documents in the vector database.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            Dictionary containing search results with keys: 'documents', 'metadatas', 'distances', 'ids'
        """
        # TODO: Implement similarity search logic
        # HINT: Use self.embedding_model.encode([query]) to create query embedding
        # HINT: Convert the embedding to appropriate format for your vector database
        # HINT: Use your vector database's search/query method with the query embedding and n_results
        # HINT: Return a dictionary with keys: 'documents', 'metadatas', 'distances', 'ids'
        # HINT: Handle the case where results might be empty

        # Create query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results
        )
        
        # Return results in the expected format
        return {
            "documents": results.get("documents", []),
            "metadatas": results.get("metadatas", []),
            "distances": results.get("distances", []),
            "ids": results.get("ids", []),
        }