"""
    Pinecone Helper
    Utilities for vector database operations using Pinecone
"""
import time
from pinecone import Pinecone, ServerlessSpec
from config import Config


def initialize_pinecone_index(name):
    """
    Initialize Pinecone index
    
    Args:
        name (str): Name of the index to create or connect to
        
    Returns:
        pinecone.Index: Connected index object if successful, None otherwise
    """
    pc = Pinecone(api_key=Config.PINECONE_API_KEY)
    try:
        # Check if index exists
        if name not in pc.list_indexes().names():
            pc.create_index(
                name=name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"  # Virginia - única región disponible en Starter Plan
                )
            )
            print(f"✅ Index '{name}' created successfully")
            # Allow time for index to initialize
            time.sleep(10)
        else:
            print(f"ℹ️ Index '{name}' already exists")
        
        # Connect to the index
        index = pc.Index(name)
        return index
    except Exception as e:
        print(f"❌ Error with Pinecone index: {e}")
        return None


class PineconeVectorDB:
    """
    Class for managing Pinecone vector database operations
    
    Handles creation, connection, and operations with Pinecone vector indexes
    """
    def __init__(self, name):
        """
        Initialize PineconeVectorDB instance
        
        Args:
            name (str): Name of the Pinecone index to use
        """
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        self.name = name
        self.index = None
        self._create_and_connect_index()
        
    def _create_and_connect_index(self):
        """Private method to create and connect to index"""
        try:
            # Check if index exists
            if self.name not in self.pc.list_indexes().names():
                self.pc.create_index(
                    name=self.name,
                    dimension=1536,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"  # Virginia - única región disponible en Starter Plan
                    )
                )
                print("✅ Index created successfully")
                # Wait to ensure index is ready
                time.sleep(20)
            else:
                print("ℹ️ Index already exists, continuing...")

            # Connect to index
            self.index = self.pc.Index(self.name)
            print("✅ Connection to index established")
        except Exception as e:
            print(f"❌ Error with index: {e}")
            
    def upsert_vectors(self, vectors):
        """
        Insert or update vectors in the index
        
        Args:
            vectors (list): List of vector records with id, values, and metadata
            
        Returns:
            bool: Success status
        """
        if not self.index:
            print("❌ No active index connection")
            return False
            
        try:
            self.index.upsert(vectors=vectors)
            return True
        except Exception as e:
            print(f"❌ Error upserting vectors: {e}")
            return False
            
    def query(self, vector, top_k=5, include_metadata=True):
        """
        Query the index for similar vectors
        
        Args:
            vector (list): Query vector
            top_k (int): Number of results to return
            include_metadata (bool): Whether to include metadata in results
            
        Returns:
            dict: Query results or None if error
        """
        if not self.index:
            print("❌ No active index connection")
            return None
            
        try:
            results = self.index.query(
                vector=vector,
                top_k=top_k,
                include_metadata=include_metadata
            )
            return results
        except Exception as e:
            print(f"❌ Error querying index: {e}")
            return None
            
    def delete_vectors(self, ids):
        """
        Delete vectors from the index
        
        Args:
            ids (list): List of vector IDs to delete
            
        Returns:
            bool: Success status
        """
        if not self.index:
            print("❌ No active index connection")
            return False
            
        try:
            self.index.delete(ids=ids)
            return True
        except Exception as e:
            print(f"❌ Error deleting vectors: {e}")
            return False
