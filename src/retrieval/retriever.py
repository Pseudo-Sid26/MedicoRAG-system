#
# import os
# import numpy as np
# from typing import List, Dict, Any, Optional
# import chromadb
# from chromadb.config import Settings
# import logging
# import uuid
# from pathlib import Path
#
# logger = logging.getLogger(__name__)
#
#
# class MedicalRetriever:
#     """Optimized retrieval system for medical documents using ChromaDB"""
#
#     def __init__(self, persist_directory: str = "./vectordb_store", collection_name: str = "medical_documents"):
#         self.persist_directory = Path(persist_directory)
#         self.collection_name = collection_name
#         self.collection = self._initialize_chromadb()
#
#     def _initialize_chromadb(self):
#         """Initialize ChromaDB client and collection"""
#         try:
#             self.persist_directory.mkdir(parents=True, exist_ok=True)
#
#             client = chromadb.PersistentClient(
#                 path=str(self.persist_directory),
#                 settings=Settings(anonymized_telemetry=False, allow_reset=True)
#             )
#
#             try:
#                 collection = client.get_collection(name=self.collection_name)
#                 logger.info(f"Loaded existing collection: {self.collection_name}")
#             except ValueError:
#                 collection = client.get_or_create_collection(
#                     name=self.collection_name,
#                     metadata={"description": "Medical documents"}
#                 )
#                 logger.info(f"Created new collection: {self.collection_name}")
#
#             return collection
#
#         except Exception as e:
#             logger.error(f"Error initializing ChromaDB: {str(e)}")
#             raise
#
#     def add_documents(self, chunks: List[Dict[str, Any]]) -> bool:
#         """Add document chunks to vector store"""
#         try:
#             documents, embeddings, ids, metadatas = [], [], [], []
#
#             for chunk in chunks:
#                 # Skip chunks without embeddings
#                 if 'embedding' not in chunk:
#                     continue
#
#                 chunk_id = chunk.get('chunk_id', str(uuid.uuid4()))
#                 embedding = chunk['embedding']
#                 if isinstance(embedding, np.ndarray):
#                     embedding = embedding.tolist()
#
#                 # Simplified metadata - only essential fields
#                 metadata = {
#                     'source': chunk.get('document_source', 'Unknown'),
#                     'section': chunk.get('section', 'Unknown'),
#                     'size': len(chunk['content'])
#                 }
#
#                 # Add medical context flags if available
#                 medical_context = chunk.get('medical_context', {})
#                 if medical_context:
#                     metadata.update({
#                         'has_diagnosis': medical_context.get('contains_diagnosis', False),
#                         'has_treatment': medical_context.get('contains_treatment', False),
#                         'has_symptoms': medical_context.get('contains_symptoms', False)
#                     })
#
#                 documents.append(chunk['content'])
#                 embeddings.append(embedding)
#                 ids.append(chunk_id)
#                 metadatas.append(metadata)
#
#             if documents:
#                 self.collection.add(
#                     documents=documents,
#                     embeddings=embeddings,
#                     ids=ids,
#                     metadatas=metadatas
#                 )
#                 logger.info(f"Added {len(documents)} documents to vector store")
#                 return True
#
#             logger.warning("No valid documents to add")
#             return False
#
#         except Exception as e:
#             logger.error(f"Error adding documents: {str(e)}")
#             return False
#
#     def search(self, query_embedding: np.ndarray,
#                n_results: int = 5,
#                filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
#         """Search for similar documents"""
#         try:
#             if isinstance(query_embedding, np.ndarray):
#                 query_embedding = query_embedding.tolist()
#
#             search_params = {
#                 'query_embeddings': [query_embedding],
#                 'n_results': n_results,
#                 'include': ['documents', 'metadatas', 'distances']
#             }
#
#             if filters:
#                 search_params['where'] = filters
#
#             results = self.collection.query(**search_params)
#
#             # Format results
#             formatted_results = []
#             if results['documents'] and results['documents'][0]:
#                 for i in range(len(results['documents'][0])):
#                     formatted_results.append({
#                         'content': results['documents'][0][i],
#                         'metadata': results['metadatas'][0][i],
#                         'similarity_score': 1 - results['distances'][0][i]
#                     })
#
#             return formatted_results
#
#         except Exception as e:
#             logger.error(f"Error during search: {str(e)}")
#             return []
#
#     def search_by_context(self, query_embedding: np.ndarray,
#                           context: str,
#                           n_results: int = 5) -> List[Dict[str, Any]]:
#         """Search with medical context filter"""
#         context_filters = {
#             'diagnosis': {'has_diagnosis': True},
#             'treatment': {'has_treatment': True},
#             'symptoms': {'has_symptoms': True}
#         }
#
#         filters = context_filters.get(context.lower())
#         return self.search(query_embedding, n_results, filters)
#
#     def search_by_source(self, query_embedding: np.ndarray,
#                          source: str,
#                          n_results: int = 5) -> List[Dict[str, Any]]:
#         """Search within specific document source"""
#         filters = {'source': source}
#         return self.search(query_embedding, n_results, filters)
#
#     def keyword_search(self, query_embedding: np.ndarray,
#                        keywords: List[str],
#                        n_results: int = 5) -> List[Dict[str, Any]]:
#         """Combine vector search with keyword filtering"""
#         # Get broader results first
#         results = self.search(query_embedding, n_results * 2)
#
#         # Filter by keywords
#         filtered = []
#         for result in results:
#             content_lower = result['content'].lower()
#             if any(keyword.lower() in content_lower for keyword in keywords):
#                 filtered.append(result)
#
#         return filtered[:n_results]
#
#     def get_stats(self) -> Dict[str, Any]:
#         """Get collection statistics"""
#         try:
#             count = self.collection.count()
#
#             if count == 0:
#                 return {'total_documents': 0}
#
#             # Sample documents for stats
#             sample_size = min(50, count)
#             sample = self.collection.get(limit=sample_size, include=['metadatas'])
#
#             stats = {'total_documents': count}
#
#             if sample['metadatas']:
#                 sources = {}
#                 sections = {}
#
#                 for metadata in sample['metadatas']:
#                     source = metadata.get('source', 'Unknown')
#                     sources[source] = sources.get(source, 0) + 1
#
#                     section = metadata.get('section', 'Unknown')
#                     sections[section] = sections.get(section, 0) + 1
#
#                 stats.update({
#                     'sources': sources,
#                     'sections': sections
#                 })
#
#             return stats
#
#         except Exception as e:
#             logger.error(f"Error getting stats: {str(e)}")
#             return {'error': str(e)}
#
#     def delete_documents(self, document_ids: List[str]) -> bool:
#         """Delete documents by IDs"""
#         try:
#             self.collection.delete(ids=document_ids)
#             logger.info(f"Deleted {len(document_ids)} documents")
#             return True
#         except Exception as e:
#             logger.error(f"Error deleting documents: {str(e)}")
#             return False
#
#     def clear_collection(self) -> bool:
#         """Clear all documents from collection"""
#         try:
#             # Get client reference to recreate collection
#             client = self.collection._client
#             client.delete_collection(name=self.collection_name)
#
#             self.collection = client.create_collection(
#                 name=self.collection_name,
#                 metadata={"description": "Medical documents"}
#             )
#
#             logger.info(f"Cleared collection: {self.collection_name}")
#             return True
#         except Exception as e:
#             logger.error(f"Error clearing collection: {str(e)}")
#             return False
#
#     def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
#         """Get a specific document by ID"""
#         try:
#             result = self.collection.get(
#                 ids=[document_id],
#                 include=['documents', 'metadatas']
#             )
#
#             if result['documents']:
#                 return {
#                     'content': result['documents'][0],
#                     'metadata': result['metadatas'][0]
#                 }
#             return None
#
#         except Exception as e:
#             logger.error(f"Error getting document {document_id}: {str(e)}")
#             return None
#
#     def update_document(self, document_id: str, content: str,
#                         embedding: np.ndarray, metadata: Dict[str, Any]) -> bool:
#         """Update an existing document"""
#         try:
#             if isinstance(embedding, np.ndarray):
#                 embedding = embedding.tolist()
#
#             self.collection.update(
#                 ids=[document_id],
#                 documents=[content],
#                 embeddings=[embedding],
#                 metadatas=[metadata]
#             )
#
#             logger.info(f"Updated document {document_id}")
#             return True
#
#         except Exception as e:
#             logger.error(f"Error updating document {document_id}: {str(e)}")
#             return False
#
#     def bulk_add(self, chunks: List[Dict[str, Any]], batch_size: int = 100) -> bool:
#         """Add documents in batches for better performance"""
#         try:
#             total_added = 0
#
#             for i in range(0, len(chunks), batch_size):
#                 batch = chunks[i:i + batch_size]
#                 if self.add_documents(batch):
#                     total_added += len(batch)
#
#             logger.info(f"Bulk added {total_added} documents")
#             return total_added > 0
#
#         except Exception as e:
#             logger.error(f"Error in bulk add: {str(e)}")
#             return False

import os
import numpy as np
from typing import List, Dict, Any, Optional
import logging
import uuid
import json
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """FAISS-based vector store for medical documents"""

    def __init__(self, persist_directory: str = "./vectordb_store", collection_name: str = "medical_docs"):
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            logger.error("FAISS not installed. Please install with: pip install faiss-cpu")
            raise

        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embedding_dim = 384  # Standard dimension

        # Ensure directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # File paths
        self.index_file = self.persist_directory / f"{collection_name}_faiss.index"
        self.metadata_file = self.persist_directory / f"{collection_name}_metadata.json"
        self.documents_file = self.persist_directory / f"{collection_name}_documents.json"

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
            if self.index_file.exists():
                self.index = self.faiss.read_index(str(self.index_file))
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            else:
                # Create new index (cosine similarity)
                self.index = self.faiss.IndexFlatIP(self.embedding_dim)
                logger.info(f"Created new FAISS index with dimension {self.embedding_dim}")

            # Load metadata
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    self.metadata = data.get('metadata', {})
                    self.id_to_idx = data.get('id_to_idx', {})
                    # Convert string keys back to integers for idx_to_id
                    idx_to_id_str = data.get('idx_to_id', {})
                    self.idx_to_id = {int(k): v for k, v in idx_to_id_str.items() if k.isdigit()}
                logger.info(f"Loaded metadata for {len(self.metadata)} documents")

            # Load documents
            if self.documents_file.exists():
                with open(self.documents_file, 'r') as f:
                    self.documents = json.load(f)
                logger.info(f"Loaded {len(self.documents)} documents")

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            # Reset to empty state
            self.index = self.faiss.IndexFlatIP(self.embedding_dim)
            self.metadata = {}
            self.documents = {}
            self.id_to_idx = {}
            self.idx_to_id = {}

    def _save_data(self):
        """Save index and metadata to disk"""
        try:
            # Save FAISS index
            self.faiss.write_index(self.index, str(self.index_file))

            # Save metadata
            metadata_data = {
                'metadata': self.metadata,
                'id_to_idx': self.id_to_idx,
                'idx_to_id': {str(k): v for k, v in self.idx_to_id.items()}  # Convert int keys to strings for JSON
            }
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata_data, f, indent=2)

            # Save documents
            with open(self.documents_file, 'w') as f:
                json.dump(self.documents, f, indent=2)

            logger.info("Data saved successfully")

        except Exception as e:
            logger.error(f"Error saving data: {e}")

    def add(self, ids: List[str], embeddings: List[np.ndarray],
            documents: List[str], metadatas: List[Dict[str, Any]]) -> bool:
        """Add documents to the vector store"""
        try:
            if not ids or len(embeddings) == 0 or not documents:
                return False

            embeddings_to_add = []
            ids_to_add = []

            for i, (doc_id, embedding, document, metadata) in enumerate(zip(ids, embeddings, documents, metadatas)):
                # Skip if already exists
                if doc_id in self.metadata:
                    logger.warning(f"Document {doc_id} already exists, updating")
                    # Update existing document
                    self.documents[doc_id] = document
                    self.metadata[doc_id] = metadata
                    continue

                # Ensure embedding is correct format
                if isinstance(embedding, list):
                    embedding = np.array(embedding, dtype=np.float32)
                elif isinstance(embedding, np.ndarray):
                    embedding = embedding.astype(np.float32)
                else:
                    logger.warning(f"Invalid embedding format for {doc_id}")
                    continue

                # Normalize for cosine similarity
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

                embeddings_to_add.append(embedding)
                ids_to_add.append(doc_id)

                # Store metadata and document
                self.metadata[doc_id] = metadata
                self.documents[doc_id] = document

            # Add to FAISS index
            if len(embeddings_to_add) > 0:
                embeddings_array = np.vstack(embeddings_to_add)
                start_idx = self.index.ntotal

                self.index.add(embeddings_array)

                # Update mappings
                for i, doc_id in enumerate(ids_to_add):
                    idx = start_idx + i
                    self.id_to_idx[doc_id] = idx
                    self.idx_to_id[idx] = doc_id

                # Save data
                self._save_data()

                logger.info(f"Added {len(embeddings_to_add)} documents to FAISS index")
                return True

            # Save even if no new embeddings (metadata updates)
            self._save_data()
            return True

        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False

    def query(self, query_embeddings: List[np.ndarray], n_results: int = 5,
              where: Optional[Dict[str, Any]] = None,
              include: List[str] = None) -> Dict[str, Any]:
        """Query the vector store"""
        try:
            if self.index.ntotal == 0:
                return {'documents': [[]], 'metadatas': [[]], 'distances': [[]], 'ids': [[]]}

            # Use first query embedding
            query_embedding = query_embeddings[0] if query_embeddings else None
            if query_embedding is None:
                return {'documents': [[]], 'metadatas': [[]], 'distances': [[]], 'ids': [[]]}

            # Ensure correct format
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

            # Search with more results to allow for filtering
            search_results = min(n_results * 3, self.index.ntotal)
            scores, indices = self.index.search(query_embedding, search_results)

            # Process results
            documents = []
            metadatas = []
            distances = []
            ids = []

            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for empty results
                    break

                doc_id = self.idx_to_id.get(idx)
                if not doc_id:
                    continue

                # Apply filters if specified
                if where:
                    metadata = self.metadata.get(doc_id, {})
                    skip = False
                    for key, value in where.items():
                        if metadata.get(key) != value:
                            skip = True
                            break
                    if skip:
                        continue

                documents.append(self.documents.get(doc_id, ''))
                metadatas.append(self.metadata.get(doc_id, {}))
                distances.append(float(score))
                ids.append(doc_id)

                # Stop when we have enough results
                if len(documents) >= n_results:
                    break

            # Format results to match ChromaDB format
            return {
                'documents': [documents],
                'metadatas': [metadatas],
                'distances': [distances],
                'ids': [ids]
            }

        except Exception as e:
            logger.error(f"Error querying: {e}")
            return {'documents': [[]], 'metadatas': [[]], 'distances': [[]], 'ids': [[]]}

    def count(self) -> int:
        """Get number of documents"""
        return len(self.documents)

    def get(self, ids: Optional[List[str]] = None, limit: Optional[int] = None,
            include: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get documents by IDs or get all documents"""
        try:
            if ids:
                # Get specific documents
                documents = []
                metadatas = []
                result_ids = []

                for doc_id in ids:
                    if doc_id in self.documents:
                        documents.append(self.documents[doc_id])
                        metadatas.append(self.metadata.get(doc_id, {}))
                        result_ids.append(doc_id)

                return {
                    'documents': documents,
                    'metadatas': metadatas,
                    'ids': result_ids
                }
            else:
                # Get all or limited documents
                all_ids = list(self.documents.keys())
                if limit:
                    all_ids = all_ids[:limit]

                documents = [self.documents[doc_id] for doc_id in all_ids]
                metadatas = [self.metadata.get(doc_id, {}) for doc_id in all_ids]

                return {
                    'documents': documents,
                    'metadatas': metadatas,
                    'ids': all_ids
                }

        except Exception as e:
            logger.error(f"Error getting documents: {e}")
            return {'documents': [], 'metadatas': [], 'ids': []}

    def delete(self, ids: List[str]):
        """Delete documents by IDs (removes from metadata and documents)"""
        try:
            for doc_id in ids:
                if doc_id in self.metadata:
                    del self.metadata[doc_id]
                if doc_id in self.documents:
                    del self.documents[doc_id]
                if doc_id in self.id_to_idx:
                    idx = self.id_to_idx[doc_id]
                    if idx in self.idx_to_id:
                        del self.idx_to_id[idx]
                    del self.id_to_idx[doc_id]

            # Note: FAISS doesn't support deletion, so we keep the vectors
            # but they won't be returned since we removed their metadata
            self._save_data()
            logger.info(f"Deleted {len(ids)} documents from metadata")

        except Exception as e:
            logger.error(f"Error deleting documents: {e}")

    def delete_collection(self, name: str):
        """Delete collection (clear all data)"""
        try:
            self.index = self.faiss.IndexFlatIP(self.embedding_dim)
            self.metadata = {}
            self.documents = {}
            self.id_to_idx = {}
            self.idx_to_id = {}

            # Remove files
            for file_path in [self.index_file, self.metadata_file, self.documents_file]:
                if file_path.exists():
                    file_path.unlink()

            logger.info("Collection cleared")

        except Exception as e:
            logger.error(f"Error clearing collection: {e}")


class MedicalRetriever:
    """Medical document retriever using FAISS vector store"""

    def __init__(self, collection_name: str = "medical_documents",
                 persist_directory: str = "./vectordb_store"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # Initialize embedding manager
        from src.embeddings.simple_embedding_manager import EmbeddingManager
        self.embedding_manager = EmbeddingManager()

        # Initialize vector store (acts like ChromaDB collection)
        self.collection = FAISSVectorStore(persist_directory, collection_name)

        logger.info(f"MedicalRetriever initialized with collection: {collection_name}")

    def add_documents(self, chunks: List[Dict[str, Any]]) -> bool:
        """Add document chunks to vector store (maintains original API)"""
        try:
            if len(chunks) == 0:
                return False

            documents, embeddings, ids, metadatas = [], [], [], []

            for chunk in chunks:
                # Skip chunks without content
                if 'content' not in chunk:
                    continue

                # Get or generate ID
                chunk_id = chunk.get('chunk_id', str(uuid.uuid4()))
                ids.append(chunk_id)

                # Get content
                content = chunk['content']
                documents.append(content)

                # Get or generate embedding
                if 'embedding' in chunk:
                    embedding = chunk['embedding']
                    if isinstance(embedding, np.ndarray):
                        embedding = embedding.tolist()
                else:
                    # Generate embedding if not provided
                    embedding = self.embedding_manager.embed_text(content)
                    if isinstance(embedding, np.ndarray):
                        embedding = embedding.tolist()

                embeddings.append(embedding)

                # Prepare metadata (simplified for compatibility)
                metadata = {
                    'source': chunk.get('document_source', chunk.get('source', 'Unknown')),
                    'section': chunk.get('section', 'Unknown'),
                    'size': len(content),
                    'timestamp': datetime.now().isoformat()
                }

                # Add medical context flags if available
                medical_context = chunk.get('medical_context', {})
                if medical_context:
                    metadata.update({
                        'has_diagnosis': medical_context.get('contains_diagnosis', False),
                        'has_treatment': medical_context.get('contains_treatment', False),
                        'has_symptoms': medical_context.get('contains_symptoms', False)
                    })

                metadatas.append(metadata)

            # Add to vector store using ChromaDB-like interface
            if len(documents) > 0:
                success = self.collection.add(ids, embeddings, documents, metadatas)
                if success:
                    logger.info(f"Added {len(documents)} documents to vector store")
                    return True
                else:
                    logger.error("Failed to add documents to vector store")
                    return False

            logger.warning("No valid documents to add")
            return False

        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False

    def search(self, query_embedding: np.ndarray,
               n_results: int = 5,
               filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar documents (maintains original API)"""
        try:
            # Convert numpy array to list if needed
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()

            # Query the vector store
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filters,
                include=['documents', 'metadatas', 'distances']
            )

            # Format results to match original API
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]

                for i in range(len(documents)):
                    formatted_results.append({
                        'content': documents[i],
                        'metadata': metadatas[i],
                        'similarity_score': float(distances[i])  # FAISS returns cosine similarity directly
                    })

            logger.info(f"Search returned {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            logger.error(f"Error during search: {e}")
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
                return {
                    'total_documents': 0,
                    'total_chunks': 0,  # For compatibility
                    'collection_name': self.collection_name
                }

            # Sample documents for stats
            sample_size = min(50, count)
            sample = self.collection.get(limit=sample_size, include=['metadatas'])

            stats = {
                'total_documents': count,
                'total_chunks': count,  # For compatibility
                'collection_name': self.collection_name,
                'embedding_dimension': self.embedding_manager.get_embedding_dimension(),
                'vector_store_type': 'FAISS'
            }

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
            logger.error(f"Error getting stats: {e}")
            return {'error': str(e)}

    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents by IDs"""
        try:
            self.collection.delete(document_ids)
            logger.info(f"Deleted {len(document_ids)} documents")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False

    def clear_collection(self) -> bool:
        """Clear all documents from collection"""
        try:
            self.collection.delete_collection(self.collection_name)
            logger.info(f"Cleared collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
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
            logger.error(f"Error getting document {document_id}: {e}")
            return None

    def update_document(self, document_id: str, content: str,
                        embedding: np.ndarray, metadata: Dict[str, Any]) -> bool:
        """Update an existing document"""
        try:
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()

            # FAISS doesn't support updates, so we delete and re-add
            self.collection.delete([document_id])
            success = self.collection.add([document_id], [embedding], [content], [metadata])

            if success:
                logger.info(f"Updated document {document_id}")
                return True
            else:
                logger.error(f"Failed to update document {document_id}")
                return False

        except Exception as e:
            logger.error(f"Error updating document {document_id}: {e}")
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
            logger.error(f"Error in bulk add: {e}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """Check system health"""
        try:
            stats = self.get_stats()

            return {
                'status': 'healthy',
                'total_documents': stats.get('total_documents', 0),
                'embedding_manager': 'loaded',
                'vector_store': 'FAISS',
                'collection_name': self.collection_name
            }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'collection_name': self.collection_name
            }

    # Additional methods for enhanced medical search capabilities
    def search_medical_query(self, query: str, top_k: int = 5,
                             filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search using text query (generates embedding automatically)"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_manager.create_medical_query_embedding(query)

            # Perform search
            results = self.search(query_embedding, top_k, filters)

            logger.info(f"Medical query search returned {len(results)} results for: {query}")
            return results

        except Exception as e:
            logger.error(f"Error in medical query search: {e}")
            return []

    def semantic_search(self, query: str, specialty: str = None,
                        doc_type: str = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """Semantic search with medical context"""
        try:
            # Build filters
            filters = {}
            if specialty:
                filters['specialty'] = specialty
            if doc_type:
                filters['doc_type'] = doc_type

            # Enhance query with context
            enhanced_query = query
            if specialty:
                enhanced_query = f"[{specialty}] {query}"

            # Generate embedding and search
            query_embedding = self.embedding_manager.create_medical_query_embedding(enhanced_query)
            results = self.search(query_embedding, top_k, filters)

            return results

        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []