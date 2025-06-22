"""
Supabase client for vector database operations.
"""

import os
import json
import uuid
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from supabase import create_client, Client
from app.config import SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_SERVICE_KEY


class SupabaseVectorClient:
    """Client for Supabase vector database operations."""
    
    def __init__(self, use_service_key: bool = False):
        """Initialize Supabase client.
        
        Args:
            use_service_key: Whether to use service key for admin operations
        """
        if not SUPABASE_URL:
            raise ValueError("SUPABASE_URL environment variable is required")
        
        key = SUPABASE_SERVICE_KEY if use_service_key and SUPABASE_SERVICE_KEY else SUPABASE_ANON_KEY
        if not key:
            raise ValueError("SUPABASE_ANON_KEY or SUPABASE_SERVICE_KEY environment variable is required")
        
        self.client: Client = create_client(SUPABASE_URL, key)
        self.use_service_key = use_service_key
    
    def create_collection(self, name: str, **kwargs) -> Dict[str, Any]:
        """Create a new collection (dataset).
        
        Args:
            name: Collection name
            **kwargs: Additional metadata
            
        Returns:
            Collection metadata
        """
        collection_data = {
            "name": name,
            "metadata": kwargs
        }
        
        result = self.client.table("collections").insert(collection_data).execute()
        
        if result.data:
            return {"name": name, "id": result.data[0]["id"]}
        else:
            raise Exception(f"Failed to create collection: {result}")
    
    def _get_collection_data(self, name: str) -> Dict[str, Any]:
        """Get raw collection data by name.
        
        Args:
            name: Collection name
            
        Returns:
            Dictionary with collection data
            
        Raises:
            Exception: If collection not found
        """
        result = self.client.table("collections").select("*").eq("name", name).execute()
        
        if not result.data:
            raise Exception(f"Collection '{name}' not found")
        
        return result.data[0]

    def get_collection(self, name: str):
        """Get collection by name.
        
        Args:
            name: Collection name
            
        Returns:
            SupabaseCollection object
            
        Raises:
            Exception: If collection not found
        """
        collection_data = self._get_collection_data(name)
        return SupabaseCollection(self, name, collection_data)
    
    def get_or_create_collection(self, name: str, **kwargs):
        """Get existing collection or create new one.
        
        Args:
            name: Collection name
            **kwargs: Additional metadata for creation
            
        Returns:
            SupabaseCollection object
        """
        try:
            # get_collection already returns a SupabaseCollection object
            return self.get_collection(name)
        except Exception:
            # create_collection returns collection data dict
            collection_data = self.create_collection(name, **kwargs)
            return SupabaseCollection(self, name, collection_data)
    
    def list_collections(self) -> List[Dict[str, Any]]:
        """List all collections.
        
        Returns:
            List of collection data
        """
        result = self.client.table("collections").select("*").execute()
        return result.data or []
    
    def update_collection_metadata(self, name: str, **kwargs) -> bool:
        """Update collection metadata.
        
        Args:
            name: Collection name
            **kwargs: Metadata fields to update
            
        Returns:
            True if successful
        """
        try:
            # Get current collection data
            collection = self._get_collection_data(name)
            current_metadata = collection.get("metadata", {})
            
            # Update metadata with new values
            updated_metadata = {**current_metadata, **kwargs}
            
            # Update in database
            result = self.client.table("collections").update({
                "metadata": updated_metadata
            }).eq("name", name).execute()
            
            return bool(result.data)
        except Exception as e:
            print(f"Error updating collection metadata for {name}: {e}")
            return False
    
    def delete_collection(self, name: str) -> bool:
        """Delete a collection and all its documents.
        
        Args:
            name: Collection name
            
        Returns:
            True if successful
        """
        try:
            # Get collection
            collection = self._get_collection_data(name)
            collection_id = collection["id"]
            
            # Delete all documents in the collection
            self.client.table("documents").delete().eq("collection_id", collection_id).execute()
            
            # Delete the collection
            result = self.client.table("collections").delete().eq("name", name).execute()
            
            return True
        except Exception as e:
            print(f"Error deleting collection {name}: {e}")
            return False
    
    def add_documents(
        self, 
        collection_name: str, 
        documents: List[str], 
        embeddings: List[List[float]], 
        metadatas: List[Dict[str, Any]] = None,
        ids: List[str] = None
    ) -> bool:
        """Add documents to a collection.
        
        Args:
            collection_name: Name of the collection
            documents: List of document texts
            embeddings: List of embedding vectors
            metadatas: Optional metadata for each document
            ids: Optional custom IDs for documents
            
        Returns:
            True if successful
        """
        try:
            collection = self._get_collection_data(collection_name)
            collection_id = collection["id"]
            
            if metadatas is None:
                metadatas = [{}] * len(documents)
            
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in range(len(documents))]
            
            # Prepare documents for insertion
            doc_data = []
            for doc, embedding, metadata, doc_id in zip(documents, embeddings, metadatas, ids):
                doc_data.append({
                    "id": doc_id,
                    "collection_id": collection_id,
                    "content": doc,
                    "embedding": embedding,  # Supabase will handle the vector type
                    "metadata": metadata
                })
            
            # Insert documents in smaller batches to avoid SSL errors
            batch_size = 50  # Reduced batch size further
            successful_batches = 0
            total_batches = (len(doc_data) + batch_size - 1) // batch_size
            
            for i in range(0, len(doc_data), batch_size):
                batch = doc_data[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                # Enhanced retry logic for SSL errors
                max_retries = 5  # Increased retry count
                batch_success = False
                
                for attempt in range(max_retries):
                    try:
                        # Use upsert to handle duplicate IDs gracefully
                        result = self.client.table("documents").upsert(batch).execute()
                        
                        if result.data:
                            batch_success = True
                            successful_batches += 1
                            print(f"Successfully inserted batch {batch_num}/{total_batches}")
                            break  # Success, exit retry loop
                        else:
                            print(f"Warning: Batch {batch_num} returned empty result")
                            
                    except Exception as batch_error:
                        error_msg = str(batch_error).lower()
                        is_ssl_error = any(ssl_keyword in error_msg for ssl_keyword in [
                            'eof occurred in violation of protocol',
                            'ssl',
                            'connection reset',
                            'connection aborted',
                            'broken pipe'
                        ])
                        
                        if is_ssl_error:
                            if attempt < max_retries - 1:
                                wait_time = min(2 ** attempt, 10)  # Cap at 10 seconds
                                print(f"SSL/Connection error on batch {batch_num}, attempt {attempt + 1}/{max_retries}. Retrying in {wait_time}s...")
                                import time
                                time.sleep(wait_time)
                                
                                # Try even smaller batch size if multiple failures
                                if attempt >= 2 and len(batch) > 10:
                                    print(f"Reducing batch size for batch {batch_num}")
                                    mini_batch_size = max(len(batch) // 2, 10)
                                    mini_batches = [batch[j:j + mini_batch_size] for j in range(0, len(batch), mini_batch_size)]
                                    
                                    mini_success = True
                                    for mini_batch in mini_batches:
                                        try:
                                            mini_result = self.client.table("documents").upsert(mini_batch).execute()
                                            if not mini_result.data:
                                                mini_success = False
                                                break
                                        except Exception:
                                            mini_success = False
                                            break
                                    
                                    if mini_success:
                                        batch_success = True
                                        successful_batches += 1
                                        print(f"Successfully inserted batch {batch_num}/{total_batches} using smaller chunks")
                                        break
                            else:
                                print(f"Failed to insert batch {batch_num} after {max_retries} attempts. Continuing with next batch...")
                                # Don't raise error, just continue with next batch
                                break
                        else:
                            print(f"Non-SSL error on batch {batch_num}: {batch_error}")
                            # For non-SSL errors, still try once more
                            if attempt < max_retries - 1:
                                import time
                                time.sleep(1)
                            else:
                                print(f"Failed to insert batch {batch_num} due to non-SSL error")
                                break
            
            # Report final status
            print(f"Document insertion completed. {successful_batches}/{total_batches} batches successful.")
            
            # Consider it successful if at least some batches worked
            if successful_batches == 0:
                print("Warning: No batches were successfully inserted")
                return False
            elif successful_batches < total_batches:
                print(f"Warning: Only {successful_batches} out of {total_batches} batches were inserted")
                # Still return True as partial success
                return True
            else:
                print("All batches inserted successfully!")
                return True
            
        except Exception as e:
            print(f"Error adding documents to collection {collection_name}: {e}")
            return False
    
    def query_collection(
        self, 
        collection_name: str, 
        query_embedding: List[float], 
        n_results: int = 5,
        where: Dict[str, Any] = None
    ) -> Dict[str, List[Any]]:
        """Query a collection for similar documents.
        
        Args:
            collection_name: Name of the collection
            query_embedding: Query embedding vector
            n_results: Number of results to return
            where: Optional metadata filters
            
        Returns:
            Dictionary with documents, metadatas, distances, and ids
        """
        try:
            collection = self._get_collection_data(collection_name)
            collection_id = collection["id"]
            
            # Build the query
            query = self.client.table("documents").select("*")
            query = query.eq("collection_id", collection_id)
            
            # Add metadata filters if provided
            if where:
                for key, value in where.items():
                    # Handle metadata filtering (this is a simplified approach)
                    # For more complex filtering, you might need to use PostgreSQL JSON operators
                    pass
            
            # Execute similarity search using pgvector
            # Note: This uses the RPC function we'll create in the database
            rpc_result = self.client.rpc(
                "match_documents",
                {
                    "collection_id": collection_id,
                    "query_embedding": query_embedding,
                    "match_threshold": 0.0,
                    "match_count": n_results
                }
            ).execute()
            
            if not rpc_result.data:
                return {
                    "documents": [[]],
                    "metadatas": [[]],
                    "distances": [[]],
                    "ids": [[]]
                }
            
            # Format results to match ChromaDB format
            documents = []
            metadatas = []
            distances = []
            ids = []
            
            for doc in rpc_result.data:
                documents.append(doc["content"])
                metadatas.append(doc["metadata"] or {})
                distances.append(doc["similarity"])
                ids.append(doc["id"])
            
            return {
                "documents": [documents],
                "metadatas": [metadatas],
                "distances": [distances],
                "ids": [ids]
            }
            
        except Exception as e:
            print(f"Error querying collection {collection_name}: {e}")
            return {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
                "ids": [[]]
            }
    
    def get_documents(
        self, 
        collection_name: str, 
        ids: List[str] = None,
        where: Dict[str, Any] = None,
        include: List[str] = None
    ) -> Dict[str, List[Any]]:
        """Get documents from a collection.
        
        Args:
            collection_name: Name of the collection
            ids: Optional list of document IDs to retrieve
            where: Optional metadata filters
            include: List of fields to include (documents, metadatas, embeddings)
            
        Returns:
            Dictionary with requested document data
        """
        try:
            collection = self._get_collection_data(collection_name)
            collection_id = collection["id"]
            
            # Build the query
            query = self.client.table("documents").select("*")
            query = query.eq("collection_id", collection_id)
            
            # Filter by IDs if provided
            if ids:
                query = query.in_("id", ids)
            
            # Add metadata filters if provided
            if where:
                for key, value in where.items():
                    # Handle metadata filtering
                    pass
            
            result = query.execute()
            
            documents = []
            metadatas = []
            embeddings = []
            doc_ids = []
            
            for doc in result.data or []:
                if not include or "documents" in include:
                    documents.append(doc["content"])
                if not include or "metadatas" in include:
                    metadatas.append(doc["metadata"] or {})
                if not include or "embeddings" in include:
                    embeddings.append(doc["embedding"])
                doc_ids.append(doc["id"])
            
            result_dict = {"ids": doc_ids}
            if not include or "documents" in include:
                result_dict["documents"] = documents
            if not include or "metadatas" in include:
                result_dict["metadatas"] = metadatas
            if not include or "embeddings" in include:
                result_dict["embeddings"] = embeddings
            
            return result_dict
            
        except Exception as e:
            print(f"Error getting documents from {collection_name}: {e}")
            return {"ids": [], "documents": [], "metadatas": [], "embeddings": []}
    
    def search_documents_by_content(
        self, 
        collection_name: str, 
        search_text: str, 
        n_results: int = 10
    ) -> Dict[str, List[Any]]:
        """Search documents by content using full-text search.
        
        Args:
            collection_name: Name of the collection
            search_text: Text to search for
            n_results: Number of results to return
            
        Returns:
            Dictionary with search results in ChromaDB format
        """
        try:
            collection = self._get_collection_data(collection_name)
            collection_id = collection["id"]
            
            # Use Supabase full-text search on content field
            result = self.client.table("documents")\
                .select("*")\
                .eq("collection_id", collection_id)\
                .text_search("content", search_text)\
                .limit(n_results)\
                .execute()
            
            if not result.data:
                # Fallback to ILIKE search if full-text search returns nothing
                result = self.client.table("documents")\
                    .select("*")\
                    .eq("collection_id", collection_id)\
                    .ilike("content", f"%{search_text}%")\
                    .limit(n_results)\
                    .execute()
            
            # Format results to match ChromaDB format
            documents = []
            metadatas = []
            distances = []
            ids = []
            
            for doc in result.data or []:
                documents.append(doc["content"])
                metadatas.append(doc["metadata"] or {})
                distances.append(1.0)  # Default distance for keyword matches
                ids.append(doc["id"])
            
            return {
                "documents": [documents],
                "metadatas": [metadatas],
                "distances": [distances],
                "ids": [ids]
            }
            
        except Exception as e:
            print(f"Error searching documents in collection {collection_name}: {e}")
            return {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
                "ids": [[]]
            }
    
    def delete_documents(self, collection_name: str, ids: List[str]) -> bool:
        """Delete documents by IDs.
        
        Args:
            collection_name: Name of the collection
            ids: List of document IDs to delete
            
        Returns:
            True if successful
        """
        try:
            collection = self._get_collection_data(collection_name)
            collection_id = collection["id"]
            
            result = self.client.table("documents").delete().eq("collection_id", collection_id).in_("id", ids).execute()
            
            return True
            
        except Exception as e:
            print(f"Error deleting documents from collection {collection_name}: {e}")
            return False
    
    def count_documents(self, collection_name: str) -> int:
        """Count documents in a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Number of documents
        """
        try:
            collection = self._get_collection_data(collection_name)
            collection_id = collection["id"]
            
            result = self.client.table("documents").select("id", count="exact").eq("collection_id", collection_id).execute()
            
            return result.count or 0
            
        except Exception as e:
            print(f"Error counting documents in collection {collection_name}: {e}")
            return 0


class SupabaseCollection:
    """Wrapper class to mimic ChromaDB Collection interface."""
    
    def __init__(self, client: SupabaseVectorClient, name: str, collection_data: Dict[str, Any]):
        self.client = client
        self.name = name
        self.collection_data = collection_data
    
    def add(self, documents: List[str], embeddings: List[List[float]] = None, 
            metadatas: List[Dict[str, Any]] = None, ids: List[str] = None):
        """Add documents to the collection."""
        if embeddings is None:
            raise ValueError("Embeddings are required for Supabase implementation")
        
        return self.client.add_documents(
            collection_name=self.name,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
    
    def query(self, query_embeddings: List[List[float]], n_results: int = 5, 
              where: Dict[str, Any] = None, **kwargs):
        """Query the collection."""
        if not query_embeddings or not query_embeddings[0]:
            raise ValueError("Query embeddings are required")
        
        return self.client.query_collection(
            collection_name=self.name,
            query_embedding=query_embeddings[0],
            n_results=n_results,
            where=where
        )
    
    def get(self, ids: List[str] = None, where: Dict[str, Any] = None, 
            include: List[str] = None, **kwargs):
        """Get documents from the collection."""
        return self.client.get_documents(
            collection_name=self.name,
            ids=ids,
            where=where,
            include=include
        )
    
    def delete(self, ids: List[str], **kwargs):
        """Delete documents from the collection."""
        return self.client.delete_documents(
            collection_name=self.name,
            ids=ids
        )
    
    def count(self) -> int:
        """Count documents in the collection."""
        return self.client.count_documents(self.name)