
import os
import numpy as np
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import logging
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


class MedicalRetriever:
    """Optimized retrieval system for medical documents using ChromaDB"""

    def __init__(self, persist_directory: str = "./vectordb_store", collection_name: str = "medical_documents"):
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.collection = self._initialize_chromadb()

    def _initialize_chromadb(self):
        """Initialize ChromaDB client and collection"""
        try:
            self.persist_directory.mkdir(parents=True, exist_ok=True)

            client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(anonymized_telemetry=False, allow_reset=True)
            )

            try:
                collection = client.get_collection(name=self.collection_name)
                logger.info(f"Loaded existing collection: {self.collection_name}")
            except ValueError:
                collection = client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"description": "Medical documents"}
                )
                logger.info(f"Created new collection: {self.collection_name}")

            return collection

        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {str(e)}")
            raise

    def add_documents(self, chunks: List[Dict[str, Any]]) -> bool:
        """Add document chunks to vector store"""
        try:
            documents, embeddings, ids, metadatas = [], [], [], []

            for chunk in chunks:
                # Skip chunks without embeddings
                if 'embedding' not in chunk:
                    continue

                chunk_id = chunk.get('chunk_id', str(uuid.uuid4()))
                embedding = chunk['embedding']
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()

                # Simplified metadata - only essential fields
                metadata = {
                    'source': chunk.get('document_source', 'Unknown'),
                    'section': chunk.get('section', 'Unknown'),
                    'size': len(chunk['content'])
                }

                # Add medical context flags if available
                medical_context = chunk.get('medical_context', {})
                if medical_context:
                    metadata.update({
                        'has_diagnosis': medical_context.get('contains_diagnosis', False),
                        'has_treatment': medical_context.get('contains_treatment', False),
                        'has_symptoms': medical_context.get('contains_symptoms', False)
                    })

                documents.append(chunk['content'])
                embeddings.append(embedding)
                ids.append(chunk_id)
                metadatas.append(metadata)

            if documents:
                self.collection.add(
                    documents=documents,
                    embeddings=embeddings,
                    ids=ids,
                    metadatas=metadatas
                )
                logger.info(f"Added {len(documents)} documents to vector store")
                return True

            logger.warning("No valid documents to add")
            return False

        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            return False

    def search(self, query_embedding: np.ndarray,
               n_results: int = 5,
               filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()

            search_params = {
                'query_embeddings': [query_embedding],
                'n_results': n_results,
                'include': ['documents', 'metadatas', 'distances']
            }

            if filters:
                search_params['where'] = filters

            results = self.collection.query(**search_params)

            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'similarity_score': 1 - results['distances'][0][i]
                    })

            return formatted_results

        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            return []

    def search_by_context(self, query_embedding: np.ndarray,
                          context: str,
                          n_results: int = 5) -> List[Dict[str, Any]]:
        """Search with medical context filter"""
        context_filters = {
            'diagnosis': {'has_diagnosis': True},
            'treatment': {'has_treatment': True},
            'symptoms': {'has_symptoms': True}
        }

        filters = context_filters.get(context.lower())
        return self.search(query_embedding, n_results, filters)

    def search_by_source(self, query_embedding: np.ndarray,
                         source: str,
                         n_results: int = 5) -> List[Dict[str, Any]]:
        """Search within specific document source"""
        filters = {'source': source}
        return self.search(query_embedding, n_results, filters)

    def keyword_search(self, query_embedding: np.ndarray,
                       keywords: List[str],
                       n_results: int = 5) -> List[Dict[str, Any]]:
        """Combine vector search with keyword filtering"""
        # Get broader results first
        results = self.search(query_embedding, n_results * 2)

        # Filter by keywords
        filtered = []
        for result in results:
            content_lower = result['content'].lower()
            if any(keyword.lower() in content_lower for keyword in keywords):
                filtered.append(result)

        return filtered[:n_results]

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            count = self.collection.count()

            if count == 0:
                return {'total_documents': 0}

            # Sample documents for stats
            sample_size = min(50, count)
            sample = self.collection.get(limit=sample_size, include=['metadatas'])

            stats = {'total_documents': count}

            if sample['metadatas']:
                sources = {}
                sections = {}

                for metadata in sample['metadatas']:
                    source = metadata.get('source', 'Unknown')
                    sources[source] = sources.get(source, 0) + 1

                    section = metadata.get('section', 'Unknown')
                    sections[section] = sections.get(section, 0) + 1

                stats.update({
                    'sources': sources,
                    'sections': sections
                })

            return stats

        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {'error': str(e)}

    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents by IDs"""
        try:
            self.collection.delete(ids=document_ids)
            logger.info(f"Deleted {len(document_ids)} documents")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            return False

    def clear_collection(self) -> bool:
        """Clear all documents from collection"""
        try:
            # Get client reference to recreate collection
            client = self.collection._client
            client.delete_collection(name=self.collection_name)

            self.collection = client.create_collection(
                name=self.collection_name,
                metadata={"description": "Medical documents"}
            )

            logger.info(f"Cleared collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            return False

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific document by ID"""
        try:
            result = self.collection.get(
                ids=[document_id],
                include=['documents', 'metadatas']
            )

            if result['documents']:
                return {
                    'content': result['documents'][0],
                    'metadata': result['metadatas'][0]
                }
            return None

        except Exception as e:
            logger.error(f"Error getting document {document_id}: {str(e)}")
            return None

    def update_document(self, document_id: str, content: str,
                        embedding: np.ndarray, metadata: Dict[str, Any]) -> bool:
        """Update an existing document"""
        try:
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()

            self.collection.update(
                ids=[document_id],
                documents=[content],
                embeddings=[embedding],
                metadatas=[metadata]
            )

            logger.info(f"Updated document {document_id}")
            return True

        except Exception as e:
            logger.error(f"Error updating document {document_id}: {str(e)}")
            return False

    def bulk_add(self, chunks: List[Dict[str, Any]], batch_size: int = 100) -> bool:
        """Add documents in batches for better performance"""
        try:
            total_added = 0

            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                if self.add_documents(batch):
                    total_added += len(batch)

            logger.info(f"Bulk added {total_added} documents")
            return total_added > 0

        except Exception as e:
            logger.error(f"Error in bulk add: {str(e)}")
            return False