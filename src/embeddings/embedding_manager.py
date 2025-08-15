

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import logging
import pickle
import os
from pathlib import Path
import hashlib
import threading

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages embeddings for medical documents using HuggingFace models"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.embedding_cache = {}
        self.cache_file = f"embeddings_cache_{model_name.split('/')[-1]}.pkl"
        self._cache_lock = threading.Lock()

        self._load_model()
        self._load_cache()

    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            try:
                self.model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded fallback model: {self.model_name}")
            except Exception as e2:
                logger.error(f"Failed to load fallback model: {str(e2)}")
                raise

    def _load_cache(self):
        """Load embedding cache from disk"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Could not load embedding cache: {str(e)}")
                self.embedding_cache = {}

    def _save_cache(self):
        """Save embedding cache to disk"""
        with self._cache_lock:
            try:
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(self.embedding_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(f"Saved {len(self.embedding_cache)} embeddings to cache")
            except Exception as e:
                logger.warning(f"Could not save embedding cache: {str(e)}")

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        content = f"{self.model_name}:{text.strip()}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]

    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        if not text.strip():
            return np.zeros(self.model.get_sentence_embedding_dimension())

        cache_key = self._get_cache_key(text)

        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        try:
            embedding = self.model.encode(text.strip(), convert_to_numpy=True, normalize_embeddings=True)
            self.embedding_cache[cache_key] = embedding
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    def embed_texts(self, texts: List[str], batch_size: int = 64) -> List[np.ndarray]:
        """Generate embeddings for multiple texts"""
        embeddings = []
        cached_embeddings = {}
        new_texts = []
        new_indices = []

        for i, text in enumerate(texts):
            if not text.strip():
                cached_embeddings[i] = np.zeros(self.model.get_sentence_embedding_dimension())
                continue

            cache_key = self._get_cache_key(text)
            if cache_key in self.embedding_cache:
                cached_embeddings[i] = self.embedding_cache[cache_key]
            else:
                new_texts.append(text.strip())
                new_indices.append(i)

        if new_texts:
            try:
                logger.info(f"Generating embeddings for {len(new_texts)} new texts")
                new_embeddings = self.model.encode(
                    new_texts,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )

                for i, embedding in enumerate(new_embeddings):
                    original_index = new_indices[i]
                    cache_key = self._get_cache_key(new_texts[i])
                    self.embedding_cache[cache_key] = embedding
                    cached_embeddings[original_index] = embedding

            except Exception as e:
                logger.error(f"Error generating batch embeddings: {str(e)}")
                raise

        embeddings = [cached_embeddings[i] for i in range(len(texts))]

        if new_texts:
            self._save_cache()

        return embeddings

    def embed_medical_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for medical document chunks"""
        logger.info(f"Generating embeddings for {len(chunks)} chunks")

        texts_to_embed = []
        for chunk in chunks:
            content = chunk['content']
            parts = []

            if chunk.get('section') and chunk['section'] != 'Unknown':
                parts.append(f"Section: {chunk['section']}")

            if chunk.get('medical_context'):
                context = chunk['medical_context']
                context_text = []
                for key, label in [
                    ('contains_diagnosis', 'diagnosis'),
                    ('contains_treatment', 'treatment'),
                    ('contains_symptoms', 'symptoms'),
                    ('contains_procedures', 'procedures')
                ]:
                    if context.get(key):
                        context_text.append(label)

                if context_text:
                    parts.append(f"Medical context: {', '.join(context_text)}")

            parts.append(content)
            texts_to_embed.append('. '.join(parts))

        embeddings = self.embed_texts(texts_to_embed)

        enhanced_chunks = []
        for i, chunk in enumerate(chunks):
            enhanced_chunk = {
                **chunk,
                'embedding': embeddings[i],
                'embedding_model': self.model_name,
                'embedding_dimension': len(embeddings[i])
            }
            enhanced_chunks.append(enhanced_chunk)

        logger.info("Embeddings generated successfully")
        return enhanced_chunks

    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            return float(np.dot(embedding1, embedding2))
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0

    def find_similar_chunks(self, query_embedding: np.ndarray,
                            chunk_embeddings: List[np.ndarray],
                            top_k: int = 5) -> List[Dict[str, Any]]:
        """Find most similar chunks to a query embedding"""
        if not chunk_embeddings:
            return []

        similarities = np.dot(np.vstack(chunk_embeddings), query_embedding)
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [{'index': int(idx), 'similarity': float(similarities[idx])}
                for idx in top_indices]

    def create_medical_query_embedding(self, query: str, query_type: str = "general") -> np.ndarray:
        """Create optimized embedding for medical queries"""
        query_prefixes = {
            "diagnosis": "Medical diagnosis: ",
            "treatment": "Medical treatment: ",
            "symptoms": "Medical symptoms: ",
            "drug_interaction": "Drug interaction: ",
            "guidelines": "Clinical guidelines: "
        }

        enhanced_query = query_prefixes.get(query_type, "") + query
        return self.embed_text(enhanced_query)

    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about embeddings"""
        return {
            'model_name': self.model_name,
            'cached_embeddings': len(self.embedding_cache),
            'embedding_dimension': self.model.get_sentence_embedding_dimension() if self.model else None,
            'cache_file_exists': os.path.exists(self.cache_file)
        }

    def clear_cache(self):
        """Clear embedding cache"""
        self.embedding_cache = {}
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        logger.info("Embedding cache cleared")

    def export_embeddings(self, chunks: List[Dict[str, Any]], output_path: str):
        """Export embeddings to file"""
        embeddings_data = []
        for chunk in chunks:
            if 'embedding' in chunk:
                embeddings_data.append({
                    'chunk_id': chunk.get('chunk_id', ''),
                    'content': chunk['content'][:200],
                    'embedding': chunk['embedding'].tolist(),
                    'section': chunk.get('section', ''),
                    'document_source': chunk.get('document_source', '')
                })

        with open(output_path, 'wb') as f:
            pickle.dump(embeddings_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"Exported {len(embeddings_data)} embeddings to {output_path}")

    def load_embeddings(self, input_path: str) -> List[Dict[str, Any]]:
        """Load embeddings from file"""
        with open(input_path, 'rb') as f:
            embeddings_data = pickle.load(f)

        for item in embeddings_data:
            item['embedding'] = np.array(item['embedding'])

        logger.info(f"Loaded {len(embeddings_data)} embeddings from {input_path}")
        return embeddings_data