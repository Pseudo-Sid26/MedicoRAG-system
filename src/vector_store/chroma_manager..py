
import chromadb
from chromadb.config import Settings as ChromaSettings
import uuid
from typing import List, Dict, Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


class ChromaManager:
    """Optimized ChromaDB manager for medical documents"""

    def __init__(self, persist_directory: str = "./vectordb_store", collection_name: str = "medical_docs"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Initialize client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True)
        )

        # Single collection approach - simpler and more efficient
        self.collection = self._get_or_create_collection()
        logger.info(f"ChromaManager initialized: {persist_directory}")

    def _get_or_create_collection(self):
        """Get or create the main collection"""
        try:
            return self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Medical documents and literature"}
            )
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """Add chunks to collection"""
        if not chunks:
            return False

        try:
            ids, embeddings, documents, metadatas = [], [], [], []

            for chunk in chunks:
                # Skip chunks without embeddings
                if 'embedding' not in chunk or 'content' not in chunk:
                    continue

                # Generate ID
                chunk_id = chunk.get('chunk_id', str(uuid.uuid4()))
                ids.append(chunk_id)

                # Get embedding (convert numpy to list if needed)
                embedding = chunk['embedding']
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()
                embeddings.append(embedding)

                # Get content
                documents.append(chunk['content'])

                # Simplified metadata - only essential fields
                metadata = {
                    'source': chunk.get('source', 'unknown'),
                    'doc_type': chunk.get('document_type', 'general'),
                    'section': chunk.get('section', 'unknown')
                }

                # Add medical context if available
                medical_context = chunk.get('medical_context', {})
                if medical_context:
                    metadata.update({
                        'has_diagnosis': medical_context.get('contains_diagnosis', False),
                        'has_treatment': medical_context.get('contains_treatment', False),
                        'has_symptoms': medical_context.get('contains_symptoms', False)
                    })

                metadatas.append(metadata)

            # Add to collection
            if ids:
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas
                )
                logger.info(f"Added {len(ids)} chunks to collection")
                return True

            return False

        except Exception as e:
            logger.error(f"Error adding chunks: {e}")
            return False

    def search(self, query_embedding: List[float],
               n_results: int = 5,
               filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Search for relevant chunks"""
        try:
            # Convert numpy array if needed
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()

            # Build search parameters
            search_params = {
                'query_embeddings': [query_embedding],
                'n_results': n_results,
                'include': ['documents', 'metadatas', 'distances']
            }

            if filters:
                search_params['where'] = filters

            # Perform search
            results = self.collection.query(**search_params)

            # Format results
            formatted_results = {
                'documents': results.get('documents', [[]]),
                'metadatas': results.get('metadatas', [[]]),
                'distances': results.get('distances', [[]]),
                'count': len(results.get('documents', [[]])[0]) if results.get('documents') else 0
            }

            logger.info(f"Retrieved {formatted_results['count']} results")
            return formatted_results

        except Exception as e:
            logger.error(f"Error searching: {e}")
            return {'documents': [[]], 'metadatas': [[]], 'distances': [[]], 'count': 0}

    def search_by_type(self, query_embedding: List[float],
                       doc_type: str,
                       n_results: int = 5) -> Dict[str, Any]:
        """Search within specific document type"""
        filters = {'doc_type': doc_type}
        return self.search(query_embedding, n_results, filters)

    def search_by_context(self, query_embedding: List[float],
                          context_type: str,
                          n_results: int = 5) -> Dict[str, Any]:
        """Search by medical context"""
        context_filters = {
            'diagnosis': {'has_diagnosis': True},
            'treatment': {'has_treatment': True},
            'symptoms': {'has_symptoms': True}
        }

        filters = context_filters.get(context_type.lower())
        return self.search(query_embedding, n_results, filters)

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            count = self.collection.count()

            # Get sample for analysis
            if count > 0:
                sample_size = min(50, count)
                sample = self.collection.get(limit=sample_size, include=['metadatas'])

                # Analyze document types
                doc_types = {}
                sources = {}

                for metadata in sample.get('metadatas', []):
                    doc_type = metadata.get('doc_type', 'unknown')
                    doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

                    source = metadata.get('source', 'unknown')
                    sources[source] = sources.get(source, 0) + 1

                return {
                    'total_chunks': count,
                    'collection_name': self.collection_name,
                    'doc_types': doc_types,
                    'sources': dict(list(sources.items())[:10])  # Top 10 sources
                }
            else:
                return {
                    'total_chunks': 0,
                    'collection_name': self.collection_name,
                    'doc_types': {},
                    'sources': {}
                }

        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {'error': str(e)}

    def delete_chunks(self, chunk_ids: List[str]) -> bool:
        """Delete specific chunks"""
        try:
            self.collection.delete(ids=chunk_ids)
            logger.info(f"Deleted {len(chunk_ids)} chunks")
            return True
        except Exception as e:
            logger.error(f"Error deleting chunks: {e}")
            return False

    def delete_by_source(self, source: str) -> bool:
        """Delete all chunks from a specific source"""
        try:
            self.collection.delete(where={'source': source})
            logger.info(f"Deleted chunks from source: {source}")
            return True
        except Exception as e:
            logger.error(f"Error deleting by source: {e}")
            return False

    def clear_collection(self) -> bool:
        """Clear all data from collection"""
        try:
            # Delete and recreate collection
            self.client.delete_collection(self.collection_name)
            self.collection = self._get_or_create_collection()
            logger.info("Collection cleared")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get specific chunk by ID"""
        try:
            result = self.collection.get(
                ids=[chunk_id],
                include=['documents', 'metadatas']
            )

            if result['documents']:
                return {
                    'content': result['documents'][0],
                    'metadata': result['metadatas'][0]
                }
            return None

        except Exception as e:
            logger.error(f"Error getting chunk {chunk_id}: {e}")
            return None

    def update_chunk(self, chunk_id: str, content: str,
                     embedding: List[float], metadata: Dict[str, Any]) -> bool:
        """Update existing chunk"""
        try:
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()

            self.collection.update(
                ids=[chunk_id],
                documents=[content],
                embeddings=[embedding],
                metadatas=[metadata]
            )

            logger.info(f"Updated chunk {chunk_id}")
            return True

        except Exception as e:
            logger.error(f"Error updating chunk: {e}")
            return False

    def batch_add(self, chunks: List[Dict[str, Any]], batch_size: int = 100) -> int:
        """Add chunks in batches for better performance"""
        total_added = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            if self.add_chunks(batch):
                total_added += len([c for c in batch if 'embedding' in c and 'content' in c])

        logger.info(f"Batch added {total_added} chunks total")
        return total_added

    def health_check(self) -> Dict[str, Any]:
        """Check system health"""
        try:
            count = self.collection.count()
            # Try a simple query
            test_embedding = [0.0] * 384  # Common embedding dimension
            test_result = self.search(test_embedding, n_results=1)

            return {
                'status': 'healthy',
                'total_chunks': count,
                'search_working': test_result['count'] >= 0,
                'collection_name': self.collection_name
            }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'collection_name': self.collection_name
            }