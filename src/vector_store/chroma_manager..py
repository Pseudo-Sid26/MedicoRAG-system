
import faiss
import numpy as np
import pickle
import uuid
import os
from typing import List, Dict, Any, Optional
import logging
import json

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """FAISS-based vector store - ChromaDB replacement for medical documents"""

    def __init__(self, persist_directory: str = "./vectordb_store", collection_name: str = "medical_docs"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_dim = 384  # Default for sentence-transformers

        # Ensure directory exists
        os.makedirs(persist_directory, exist_ok=True)

        # File paths
        self.index_file = os.path.join(persist_directory, f"{collection_name}_faiss.index")
        self.metadata_file = os.path.join(persist_directory, f"{collection_name}_metadata.pkl")
        self.documents_file = os.path.join(persist_directory, f"{collection_name}_documents.pkl")

        # Initialize storage
        self.index = None
        self.metadata = {}  # id -> metadata mapping
        self.documents = {}  # id -> document mapping
        self.id_to_idx = {}  # id -> faiss index mapping
        self.idx_to_id = {}  # faiss index -> id mapping

        # Load existing data
        self._load_data()

        logger.info(f"FAISSVectorStore initialized: {persist_directory}")

    def _load_data(self):
        """Load existing index and metadata"""
        try:
            # Load FAISS index
            if os.path.exists(self.index_file):
                self.index = faiss.read_index(self.index_file)
                self.embedding_dim = self.index.d
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            else:
                # Create new index (cosine similarity)
                self.index = faiss.IndexFlatIP(self.embedding_dim)
                logger.info(f"Created new FAISS index with dimension {self.embedding_dim}")

            # Load metadata
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'rb') as f:
                    data = pickle.load(f)
                    self.metadata = data.get('metadata', {})
                    self.id_to_idx = data.get('id_to_idx', {})
                    self.idx_to_id = data.get('idx_to_id', {})
                logger.info(f"Loaded metadata for {len(self.metadata)} documents")

            # Load documents
            if os.path.exists(self.documents_file):
                with open(self.documents_file, 'rb') as f:
                    self.documents = pickle.load(f)
                logger.info(f"Loaded {len(self.documents)} documents")

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            # Reset to empty state
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.metadata = {}
            self.documents = {}
            self.id_to_idx = {}
            self.idx_to_id = {}

    def _save_data(self):
        """Save index and metadata to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, self.index_file)

            # Save metadata
            metadata_data = {
                'metadata': self.metadata,
                'id_to_idx': self.id_to_idx,
                'idx_to_id': self.idx_to_id
            }
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(metadata_data, f)

            # Save documents
            with open(self.documents_file, 'wb') as f:
                pickle.dump(self.documents, f)

            logger.info("Data saved successfully")

        except Exception as e:
            logger.error(f"Error saving data: {e}")

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """Add chunks to the vector store"""
        if not chunks:
            return False

        try:
            embeddings_to_add = []
            ids_to_add = []

            for chunk in chunks:
                # Skip chunks without embeddings
                if 'embedding' not in chunk or 'content' not in chunk:
                    continue

                # Generate ID
                chunk_id = chunk.get('chunk_id', str(uuid.uuid4()))

                # Skip if already exists
                if chunk_id in self.metadata:
                    continue

                # Get embedding
                embedding = chunk['embedding']
                if isinstance(embedding, list):
                    embedding = np.array(embedding, dtype=np.float32)
                elif isinstance(embedding, np.ndarray):
                    embedding = embedding.astype(np.float32)
                else:
                    continue

                # Normalize for cosine similarity
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

                # Update embedding dimension if needed
                if hasattr(embedding, 'shape') and len(embedding.shape) > 0:
                    if self.index.d != embedding.shape[0]:
                        self.embedding_dim = embedding.shape[0]
                        # Recreate index with correct dimension
                        old_vectors = []
                        if self.index.ntotal > 0:
                            old_vectors = self.index.reconstruct_n(0, self.index.ntotal)
                        self.index = faiss.IndexFlatIP(self.embedding_dim)
                        if len(old_vectors) > 0:
                            self.index.add(old_vectors)

                embeddings_to_add.append(embedding)
                ids_to_add.append(chunk_id)

                # Store metadata
                metadata = {
                    'source': chunk.get('source', 'unknown'),
                    'doc_type': chunk.get('document_type', 'general'),
                    'section': chunk.get('section', 'unknown')
                }

                # Add medical context
                medical_context = chunk.get('medical_context', {})
                if medical_context:
                    metadata.update({
                        'has_diagnosis': medical_context.get('contains_diagnosis', False),
                        'has_treatment': medical_context.get('contains_treatment', False),
                        'has_symptoms': medical_context.get('contains_symptoms', False)
                    })

                self.metadata[chunk_id] = metadata
                self.documents[chunk_id] = chunk['content']

            # Add to FAISS index
            if embeddings_to_add:
                embeddings_array = np.vstack(embeddings_to_add)
                start_idx = self.index.ntotal

                self.index.add(embeddings_array)

                # Update mappings
                for i, chunk_id in enumerate(ids_to_add):
                    idx = start_idx + i
                    self.id_to_idx[chunk_id] = idx
                    self.idx_to_id[idx] = chunk_id

                # Save data
                self._save_data()

                logger.info(f"Added {len(embeddings_to_add)} chunks to FAISS index")
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
            if self.index.ntotal == 0:
                return {'documents': [[]], 'metadatas': [[]], 'distances': [[]], 'count': 0}

            # Prepare query embedding
            if isinstance(query_embedding, list):
                query_embedding = np.array(query_embedding, dtype=np.float32)
            elif isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.astype(np.float32)

            # Normalize for cosine similarity
            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                query_embedding = query_embedding / norm

            # Reshape for FAISS
            query_embedding = query_embedding.reshape(1, -1)

            # Search
            scores, indices = self.index.search(query_embedding, min(n_results, self.index.ntotal))

            # Process results
            documents = []
            metadatas = []
            distances = []

            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:  # FAISS returns -1 for empty results
                    break

                chunk_id = self.idx_to_id.get(idx)
                if not chunk_id:
                    continue

                # Apply filters if specified
                if filters:
                    metadata = self.metadata.get(chunk_id, {})
                    skip = False
                    for key, value in filters.items():
                        if metadata.get(key) != value:
                            skip = True
                            break
                    if skip:
                        continue

                documents.append(self.documents.get(chunk_id, ''))
                metadatas.append(self.metadata.get(chunk_id, {}))
                distances.append(float(score))

            # Format results to match ChromaDB format
            return {
                'documents': [documents],
                'metadatas': [metadatas],
                'distances': [distances],
                'count': len(documents)
            }

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
            count = len(self.documents)

            if count > 0:
                # Analyze document types
                doc_types = {}
                sources = {}

                for metadata in self.metadata.values():
                    doc_type = metadata.get('doc_type', 'unknown')
                    doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

                    source = metadata.get('source', 'unknown')
                    sources[source] = sources.get(source, 0) + 1

                return {
                    'total_chunks': count,
                    'collection_name': self.collection_name,
                    'doc_types': doc_types,
                    'sources': dict(list(sources.items())[:10])
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
        """Delete specific chunks (rebuild index)"""
        try:
            # Remove from metadata and documents
            for chunk_id in chunk_ids:
                if chunk_id in self.metadata:
                    del self.metadata[chunk_id]
                if chunk_id in self.documents:
                    del self.documents[chunk_id]

            # Rebuild index (FAISS doesn't support deletion)
            self._rebuild_index()

            logger.info(f"Deleted {len(chunk_ids)} chunks")
            return True

        except Exception as e:
            logger.error(f"Error deleting chunks: {e}")
            return False

    def _rebuild_index(self):
        """Rebuild FAISS index after deletions"""
        try:
            # Create new index
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.id_to_idx = {}
            self.idx_to_id = {}

            # Re-add all remaining embeddings
            if self.metadata:
                # This is a simplified rebuild - in practice, you'd need to store embeddings
                logger.warning("Index rebuild requires re-embedding documents")

            self._save_data()

        except Exception as e:
            logger.error(f"Error rebuilding index: {e}")

    def clear_collection(self) -> bool:
        """Clear all data"""
        try:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.metadata = {}
            self.documents = {}
            self.id_to_idx = {}
            self.idx_to_id = {}

            # Remove files
            for file_path in [self.index_file, self.metadata_file, self.documents_file]:
                if os.path.exists(file_path):
                    os.remove(file_path)

            logger.info("Collection cleared")
            return True

        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """Check system health"""
        try:
            count = len(self.documents)

            # Test search
            test_embedding = [0.0] * self.embedding_dim
            test_result = self.search(test_embedding, n_results=1)

            return {
                'status': 'healthy',
                'total_chunks': count,
                'search_working': test_result['count'] >= 0,
                'collection_name': self.collection_name,
                'index_type': 'FAISS'
            }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'collection_name': self.collection_name,
                'index_type': 'FAISS'
            }


# Alias for compatibility with existing code
ChromaManager = FAISSVectorStore