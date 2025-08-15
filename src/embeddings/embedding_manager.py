#
# import numpy as np
# from typing import List, Dict, Any, Optional
# import logging
# import pickle
# import os
# import hashlib
# import re
# from sklearn.feature_extraction.text import TfidfVectorizer
#
# logger = logging.getLogger(__name__)
#
#
# class MinimalEmbeddingManager:
#     """Minimal embedding manager using TF-IDF (no external model dependencies)"""
#
#     def __init__(self, model_name: str = "tfidf", cache_dir: str = "./models"):
#         """Initialize with TF-IDF vectorizer"""
#         self.model_name = model_name
#         self.cache_dir = cache_dir
#         self.embedding_dim = 384  # Match sentence-transformers dimension
#
#         os.makedirs(cache_dir, exist_ok=True)
#
#         # Initialize TF-IDF vectorizer with safer settings
#         self.vectorizer = TfidfVectorizer(
#             max_features=self.embedding_dim,
#             stop_words='english',
#             ngram_range=(1, 2),
#             min_df=1,  # Minimum document frequency
#             max_df=1.0,  # Maximum document frequency (allow all)
#             lowercase=True,
#             strip_accents='unicode'
#         )
#
#         # Track if vectorizer is fitted
#         self.is_fitted = False
#
#         logger.info(f"Initialized minimal embedding manager with TF-IDF")
#
#     def _preprocess_text(self, text: str) -> str:
#         """Basic text preprocessing"""
#         if not text:
#             return ""
#
#         # Convert to lowercase
#         text = text.lower()
#
#         # Remove special characters but keep medical terms
#         text = re.sub(r'[^\w\s\-\.]', ' ', text)
#
#         # Remove extra whitespace
#         text = ' '.join(text.split())
#
#         return text
#
#     def _ensure_fitted(self, texts: List[str]):
#         """Ensure vectorizer is fitted on some corpus"""
#         if not self.is_fitted:
#             # Fit on provided texts
#             processed_texts = [self._preprocess_text(text) for text in texts]
#             # Filter empty texts
#             valid_texts = [t for t in processed_texts if t.strip()]
#
#             # Need at least 2 documents for TF-IDF
#             if len(valid_texts) >= 2:
#                 try:
#                     self.vectorizer.fit(valid_texts)
#                     self.is_fitted = True
#                     logger.info(f"TF-IDF vectorizer fitted on {len(valid_texts)} documents")
#                 except Exception as e:
#                     logger.error(f"Error fitting TF-IDF: {e}")
#                     self._fit_fallback()
#             else:
#                 # Not enough documents, use fallback
#                 self._fit_fallback()
#
#     def _fit_fallback(self):
#         """Fallback fitting with medical vocabulary"""
#         try:
#             # Extended medical vocabulary for better embeddings
#             medical_texts = [
#                 "patient diagnosis treatment medical condition",
#                 "symptoms disease medicine health care",
#                 "doctor hospital clinic surgery operation",
#                 "medication dosage prescription therapy treatment",
#                 "blood pressure heart rate temperature",
#                 "infection virus bacteria antibiotic",
#                 "pain fever headache nausea vomiting",
#                 "examination test results laboratory",
#                 "chronic acute condition illness",
#                 "respiratory cardiovascular neurological"
#             ]
#
#             # Use simpler TF-IDF settings for fallback
#             fallback_vectorizer = TfidfVectorizer(
#                 max_features=self.embedding_dim,
#                 stop_words='english',
#                 min_df=1,
#                 max_df=1.0,
#                 ngram_range=(1, 1)  # Only unigrams for stability
#             )
#
#             fallback_vectorizer.fit(medical_texts)
#             self.vectorizer = fallback_vectorizer
#             self.is_fitted = True
#             logger.info("TF-IDF fitted with medical vocabulary fallback")
#
#         except Exception as e:
#             logger.error(f"Fallback fitting failed: {e}")
#             # Ultimate fallback - simple hash-based embeddings
#             self.is_fitted = True
#
#     def _hash_embed(self, text: str) -> np.ndarray:
#         """Hash-based embedding fallback"""
#         try:
#             # Create multiple hash values for better distribution
#             hash_funcs = [
#                 lambda x: hashlib.md5(x.encode()).hexdigest(),
#                 lambda x: hashlib.sha1(x.encode()).hexdigest(),
#             ]
#
#             # Combine hash values
#             combined_hash = ""
#             for hash_func in hash_funcs:
#                 combined_hash += hash_func(text)
#
#             # Convert to numeric array
#             hash_bytes = bytes.fromhex(combined_hash[:192])  # Use hex for bytes
#             hash_array = np.frombuffer(hash_bytes, dtype=np.uint8).astype(np.float32)
#
#             # Pad to full dimension
#             if len(hash_array) < self.embedding_dim:
#                 padding = np.random.normal(0, 0.1, self.embedding_dim - len(hash_array)).astype(np.float32)
#                 hash_array = np.concatenate([hash_array, padding])
#             else:
#                 hash_array = hash_array[:self.embedding_dim]
#
#             # Normalize
#             norm = np.linalg.norm(hash_array)
#             if norm > 0:
#                 hash_array = hash_array / norm
#
#             return hash_array.astype(np.float32)
#
#         except Exception as e:
#             logger.error(f"Hash embedding failed: {e}")
#             return np.random.normal(0, 0.1, self.embedding_dim).astype(np.float32)
#
#     def embed_text(self, text: str) -> np.ndarray:
#         """Generate TF-IDF embedding for text"""
#         try:
#             if not text or not text.strip():
#                 return np.zeros(self.embedding_dim, dtype=np.float32)
#
#             # Ensure vectorizer is fitted
#             if not self.is_fitted:
#                 self._ensure_fitted([text])
#
#             processed_text = self._preprocess_text(text)
#
#             # Handle case where TF-IDF fitting failed
#             if not hasattr(self.vectorizer, 'vocabulary_') or not self.vectorizer.vocabulary_:
#                 # Use hash-based embedding as ultimate fallback
#                 return self._hash_embed(processed_text)
#
#             try:
#                 # Get TF-IDF vector
#                 tfidf_vector = self.vectorizer.transform([processed_text]).toarray()[0]
#
#                 # Ensure exact dimension
#                 if len(tfidf_vector) < self.embedding_dim:
#                     padding = np.zeros(self.embedding_dim - len(tfidf_vector))
#                     tfidf_vector = np.concatenate([tfidf_vector, padding])
#                 else:
#                     tfidf_vector = tfidf_vector[:self.embedding_dim]
#
#                 return tfidf_vector.astype(np.float32)
#
#             except Exception as tfidf_error:
#                 logger.warning(f"TF-IDF transform failed: {tfidf_error}, using hash fallback")
#                 return self._hash_embed(processed_text)
#
#         except Exception as e:
#             logger.error(f"Error embedding text: {e}")
#             return np.zeros(self.embedding_dim, dtype=np.float32)
#
#     def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
#         """Generate embeddings for multiple texts"""
#         try:
#             if not texts:
#                 return []
#
#             # Ensure vectorizer is fitted
#             if not self.is_fitted:
#                 self._ensure_fitted(texts)
#
#             # Process all texts
#             processed_texts = [self._preprocess_text(text) if text else "" for text in texts]
#
#             # Handle case where TF-IDF fitting failed
#             if not hasattr(self.vectorizer, 'vocabulary_') or not self.vectorizer.vocabulary_:
#                 # Use hash-based embeddings
#                 return [self._hash_embed(text) for text in processed_texts]
#
#             try:
#                 # Get TF-IDF vectors
#                 tfidf_matrix = self.vectorizer.transform(processed_texts).toarray()
#
#                 embeddings = []
#                 for tfidf_vector in tfidf_matrix:
#                     # Ensure correct dimension
#                     if len(tfidf_vector) < self.embedding_dim:
#                         padding = np.zeros(self.embedding_dim - len(tfidf_vector))
#                         tfidf_vector = np.concatenate([tfidf_vector, padding])
#                     else:
#                         tfidf_vector = tfidf_vector[:self.embedding_dim]
#
#                     embeddings.append(tfidf_vector.astype(np.float32))
#
#                 return embeddings
#
#             except Exception as tfidf_error:
#                 logger.warning(f"TF-IDF batch transform failed: {tfidf_error}, using hash fallback")
#                 return [self._hash_embed(text) for text in processed_texts]
#
#         except Exception as e:
#             logger.error(f"Error embedding texts: {e}")
#             return [np.zeros(self.embedding_dim, dtype=np.float32) for _ in texts]
#
#     def embed_query(self, query: str) -> np.ndarray:
#         """Generate embedding for search query"""
#         return self.embed_text(query)
#
#     def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
#         """Calculate cosine similarity"""
#         try:
#             norm1 = np.linalg.norm(embedding1)
#             norm2 = np.linalg.norm(embedding2)
#
#             if norm1 == 0 or norm2 == 0:
#                 return 0.0
#
#             similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
#             return float(similarity)
#
#         except Exception as e:
#             logger.error(f"Error calculating similarity: {e}")
#             return 0.0
#
#     def find_most_similar(self, query_embedding: np.ndarray,
#                           embeddings: List[np.ndarray],
#                           top_k: int = 5) -> List[tuple]:
#         """Find most similar embeddings"""
#         try:
#             similarities = []
#             for i, emb in enumerate(embeddings):
#                 sim = self.similarity(query_embedding, emb)
#                 similarities.append((i, sim))
#
#             similarities.sort(key=lambda x: x[1], reverse=True)
#             return similarities[:top_k]
#
#         except Exception as e:
#             logger.error(f"Error finding similar embeddings: {e}")
#             return []
#
#     def embed_medical_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#         """Embed medical document chunks"""
#         try:
#             if not chunks:
#                 return []
#
#             # Extract content from chunks
#             texts = []
#             for chunk in chunks:
#                 content = chunk.get('content', '')
#                 if isinstance(content, str):
#                     texts.append(content)
#                 else:
#                     texts.append(str(content))
#
#             # Generate embeddings
#             embeddings = self.embed_texts(texts)
#
#             # Add embeddings to chunks
#             enriched_chunks = []
#             for i, chunk in enumerate(chunks):
#                 new_chunk = chunk.copy()
#                 if i < len(embeddings):
#                     new_chunk['embedding'] = embeddings[i]
#                 else:
#                     new_chunk['embedding'] = np.zeros(self.embedding_dim, dtype=np.float32)
#                 enriched_chunks.append(new_chunk)
#
#             logger.info(f"Successfully embedded {len(enriched_chunks)} medical chunks")
#             return enriched_chunks
#
#         except Exception as e:
#             logger.error(f"Error embedding medical chunks: {e}")
#             return chunks
#
#     def create_medical_query_embedding(self, query: str, context: str = None) -> np.ndarray:
#         """Create embedding for medical query with optional context"""
#         try:
#             # Combine query with context if provided
#             if context and context.strip():
#                 enhanced_query = f"{context} {query}"
#             else:
#                 enhanced_query = query
#
#             return self.embed_text(enhanced_query)
#
#         except Exception as e:
#             logger.error(f"Error creating medical query embedding: {e}")
#             return np.zeros(self.embedding_dim, dtype=np.float32)
#
#     def get_embedding_dimension(self) -> int:
#         """Get the embedding dimension"""
#         return self.embedding_dim
#
#     def get_model_info(self) -> Dict[str, Any]:
#         """Get model information"""
#         return {
#             'model_name': self.model_name,
#             'embedding_dimension': self.embedding_dim,
#             'model_type': 'TF-IDF',
#             'is_fitted': self.is_fitted,
#             'cache_dir': self.cache_dir
#         }
#
#
# # Use minimal manager as default
# EmbeddingManager = MinimalEmbeddingManager

import numpy as np
from typing import List, Dict, Any, Optional
import logging
import pickle
import os
import hashlib
import re

logger = logging.getLogger(__name__)

# Delayed import to avoid threading issues
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except (ImportError, RuntimeError):
    SKLEARN_AVAILABLE = False
    TfidfVectorizer = None


class MinimalEmbeddingManager:
    """Minimal embedding manager using TF-IDF (no external model dependencies)"""

    def __init__(self, model_name: str = "tfidf", cache_dir: str = "./models"):
        """Initialize with TF-IDF vectorizer"""
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.embedding_dim = 384  # Match sentence-transformers dimension

        os.makedirs(cache_dir, exist_ok=True)

        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn not available, using simple vectorizer")
            self.vectorizer = None
            self.is_fitted = False
            return

        # Initialize TF-IDF vectorizer with safer settings
        self.vectorizer = TfidfVectorizer(
            max_features=self.embedding_dim,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,  # Minimum document frequency
            max_df=1.0,  # Maximum document frequency (allow all)
            lowercase=True,
            strip_accents='unicode'
        )

        # Track if vectorizer is fitted
        self.is_fitted = False

        logger.info(f"Initialized minimal embedding manager with TF-IDF")

    def _preprocess_text(self, text: str) -> str:
        """Basic text preprocessing"""
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove special characters but keep medical terms
        text = re.sub(r'[^\w\s\-\.]', ' ', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def _ensure_fitted(self, texts: List[str]):
        """Ensure vectorizer is fitted on some corpus"""
        if not self.is_fitted:
            # Fit on provided texts
            processed_texts = [self._preprocess_text(text) for text in texts]
            # Filter empty texts
            valid_texts = [t for t in processed_texts if t.strip()]

            # Need at least 2 documents for TF-IDF
            if len(valid_texts) >= 2:
                try:
                    self.vectorizer.fit(valid_texts)
                    self.is_fitted = True
                    logger.info(f"TF-IDF vectorizer fitted on {len(valid_texts)} documents")
                except Exception as e:
                    logger.error(f"Error fitting TF-IDF: {e}")
                    self._fit_fallback()
            else:
                # Not enough documents, use fallback
                self._fit_fallback()

    def _fit_fallback(self):
        """Fallback fitting with medical vocabulary"""
        try:
            # Extended medical vocabulary for better embeddings
            medical_texts = [
                "patient diagnosis treatment medical condition",
                "symptoms disease medicine health care",
                "doctor hospital clinic surgery operation",
                "medication dosage prescription therapy treatment",
                "blood pressure heart rate temperature",
                "infection virus bacteria antibiotic",
                "pain fever headache nausea vomiting",
                "examination test results laboratory",
                "chronic acute condition illness",
                "respiratory cardiovascular neurological"
            ]

            # Use simpler TF-IDF settings for fallback
            fallback_vectorizer = TfidfVectorizer(
                max_features=self.embedding_dim,
                stop_words='english',
                min_df=1,
                max_df=1.0,
                ngram_range=(1, 1)  # Only unigrams for stability
            )

            fallback_vectorizer.fit(medical_texts)
            self.vectorizer = fallback_vectorizer
            self.is_fitted = True
            logger.info("TF-IDF fitted with medical vocabulary fallback")

        except Exception as e:
            logger.error(f"Fallback fitting failed: {e}")
            # Ultimate fallback - simple hash-based embeddings
            self.is_fitted = True

    def _hash_embed(self, text: str) -> np.ndarray:
        """Hash-based embedding fallback"""
        try:
            # Create multiple hash values for better distribution
            hash_funcs = [
                lambda x: hashlib.md5(x.encode()).hexdigest(),
                lambda x: hashlib.sha1(x.encode()).hexdigest(),
            ]

            # Combine hash values
            combined_hash = ""
            for hash_func in hash_funcs:
                combined_hash += hash_func(text)

            # Convert to numeric array (ensure we have enough hex characters)
            hex_needed = min(192, len(combined_hash))
            hash_bytes = bytes.fromhex(combined_hash[:hex_needed])
            hash_array = np.frombuffer(hash_bytes, dtype=np.uint8).astype(np.float32)

            # Pad to full dimension
            if len(hash_array) < self.embedding_dim:
                padding = np.random.normal(0, 0.1, self.embedding_dim - len(hash_array)).astype(np.float32)
                hash_array = np.concatenate([hash_array, padding])
            else:
                hash_array = hash_array[:self.embedding_dim]

            # Normalize
            norm = np.linalg.norm(hash_array)
            if norm > 0:
                hash_array = hash_array / norm

            return hash_array.astype(np.float32)

        except Exception as e:
            logger.error(f"Hash embedding failed: {e}")
            return np.random.normal(0, 0.1, self.embedding_dim).astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        """Generate TF-IDF embedding for text"""
        try:
            if not text or not text.strip():
                return np.zeros(self.embedding_dim, dtype=np.float32)

            # Ensure vectorizer is fitted
            if not self.is_fitted:
                self._ensure_fitted([text])

            processed_text = self._preprocess_text(text)

            # Handle case where TF-IDF fitting failed
            if not hasattr(self.vectorizer, 'vocabulary_') or not self.vectorizer.vocabulary_:
                # Use hash-based embedding as ultimate fallback
                return self._hash_embed(processed_text)

            try:
                # Get TF-IDF vector
                tfidf_vector = self.vectorizer.transform([processed_text]).toarray()[0]

                # Ensure exact dimension
                if len(tfidf_vector) < self.embedding_dim:
                    padding = np.zeros(self.embedding_dim - len(tfidf_vector))
                    tfidf_vector = np.concatenate([tfidf_vector, padding])
                else:
                    tfidf_vector = tfidf_vector[:self.embedding_dim]

                return tfidf_vector.astype(np.float32)

            except Exception as tfidf_error:
                logger.warning(f"TF-IDF transform failed: {tfidf_error}, using hash fallback")
                return self._hash_embed(processed_text)

        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Generate embeddings for multiple texts"""
        try:
            if not texts:
                return []

            # Ensure vectorizer is fitted
            if not self.is_fitted:
                self._ensure_fitted(texts)

            # Process all texts
            processed_texts = [self._preprocess_text(text) if text else "" for text in texts]

            # Handle case where TF-IDF fitting failed
            if not hasattr(self.vectorizer, 'vocabulary_') or not self.vectorizer.vocabulary_:
                # Use hash-based embeddings
                return [self._hash_embed(text) for text in processed_texts]

            try:
                # Get TF-IDF vectors
                tfidf_matrix = self.vectorizer.transform(processed_texts).toarray()

                embeddings = []
                for tfidf_vector in tfidf_matrix:
                    # Ensure correct dimension
                    if len(tfidf_vector) < self.embedding_dim:
                        padding = np.zeros(self.embedding_dim - len(tfidf_vector))
                        tfidf_vector = np.concatenate([tfidf_vector, padding])
                    else:
                        tfidf_vector = tfidf_vector[:self.embedding_dim]

                    embeddings.append(tfidf_vector.astype(np.float32))

                return embeddings

            except Exception as tfidf_error:
                logger.warning(f"TF-IDF batch transform failed: {tfidf_error}, using hash fallback")
                return [self._hash_embed(text) for text in processed_texts]

        except Exception as e:
            logger.error(f"Error embedding texts: {e}")
            return [np.zeros(self.embedding_dim, dtype=np.float32) for _ in texts]

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for search query"""
        return self.embed_text(query)

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity"""
        try:
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)

        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

    def find_most_similar(self, query_embedding: np.ndarray,
                          embeddings: List[np.ndarray],
                          top_k: int = 5) -> List[tuple]:
        """Find most similar embeddings"""
        try:
            similarities = []
            for i, emb in enumerate(embeddings):
                sim = self.similarity(query_embedding, emb)
                similarities.append((i, sim))

            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]

        except Exception as e:
            logger.error(f"Error finding similar embeddings: {e}")
            return []

    def embed_medical_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Embed medical document chunks"""
        try:
            if not chunks:
                return []

            # Extract content from chunks
            texts = []
            for chunk in chunks:
                content = chunk.get('content', '')
                if isinstance(content, str):
                    texts.append(content)
                else:
                    texts.append(str(content))

            # Generate embeddings
            embeddings = self.embed_texts(texts)

            # Add embeddings to chunks
            enriched_chunks = []
            for i, chunk in enumerate(chunks):
                new_chunk = chunk.copy()
                if i < len(embeddings):
                    new_chunk['embedding'] = embeddings[i]
                else:
                    new_chunk['embedding'] = np.zeros(self.embedding_dim, dtype=np.float32)
                enriched_chunks.append(new_chunk)

            logger.info(f"Successfully embedded {len(enriched_chunks)} medical chunks")
            return enriched_chunks

        except Exception as e:
            logger.error(f"Error embedding medical chunks: {e}")
            return chunks

    def create_medical_query_embedding(self, query: str, context: str = None) -> np.ndarray:
        """Create embedding for medical query with optional context"""
        try:
            # Combine query with context if provided
            if context and context.strip():
                enhanced_query = f"{context} {query}"
            else:
                enhanced_query = query

            return self.embed_text(enhanced_query)

        except Exception as e:
            logger.error(f"Error creating medical query embedding: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)

    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension"""
        return self.embedding_dim

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dim,
            'model_type': 'TF-IDF',
            'is_fitted': self.is_fitted,
            'cache_dir': self.cache_dir
        }

    # Additional methods for compatibility
    def create_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Alternative method name for embed_texts"""
        return self.embed_texts(texts)

    def get_embedding(self, text: str) -> np.ndarray:
        """Alternative method name for embed_text"""
        return self.embed_text(text)

    def encode_medical_text(self, text: str, text_type: str = "general") -> np.ndarray:
        """Encode medical text with type specification"""
        if text_type and text_type != "general":
            contextualized_text = f"[{text_type}] {text}"
        else:
            contextualized_text = text
        return self.embed_text(contextualized_text)

    def batch_encode(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Batch encode texts"""
        return self.embed_texts(texts, batch_size=batch_size)

    def encode_query(self, query: str) -> np.ndarray:
        """Alternative method name for embed_query"""
        return self.embed_query(query)

    def encode_document(self, document: str) -> np.ndarray:
        """Encode a single document"""
        return self.embed_text(document)

    def create_query_embedding(self, query: str) -> np.ndarray:
        """Create embedding for query"""
        return self.embed_query(query)

    def create_document_embedding(self, document: str) -> np.ndarray:
        """Create embedding for document"""
        return self.embed_text(document)

    def is_model_loaded(self) -> bool:
        """Check if the model is loaded and ready"""
        return True  # TF-IDF is always ready once initialized

    def get_vocabulary_size(self) -> int:
        """Get vocabulary size"""
        if self.is_fitted and hasattr(self.vectorizer, 'vocabulary_'):
            return len(self.vectorizer.vocabulary_)
        return 0


# Use minimal manager as default
EmbeddingManager = MinimalEmbeddingManager