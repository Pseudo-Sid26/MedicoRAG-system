"""
Simple Embedding Manager for Medical RAG System
Handles text embeddings without complex dependencies
"""

import numpy as np
from typing import List, Dict, Any, Optional
import logging
import pickle
import os
import hashlib
import re
from collections import Counter
import json

logger = logging.getLogger(__name__)


class SimpleEmbeddingManager:
    """Simple embedding manager using basic TF-IDF implementation"""

    def __init__(self, model_name: str = "simple_tfidf", cache_dir: str = "./models"):
        """Initialize with simple TF-IDF"""
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.embedding_dim = 384  # Standard dimension
        
        os.makedirs(cache_dir, exist_ok=True)
        
        # Simple vocabulary for medical terms
        self.vocabulary = {}
        self.idf_scores = {}
        self.is_fitted = False
        
        # Medical vocabulary enhancement
        self.medical_terms = {
            'hypertension', 'diabetes', 'medication', 'treatment', 'diagnosis',
            'patient', 'symptoms', 'therapy', 'clinical', 'medical', 'drug',
            'disease', 'condition', 'healthcare', 'prescription', 'dosage',
            'adverse', 'effect', 'contraindication', 'indication', 'chronic',
            'acute', 'syndrome', 'disorder', 'infection', 'inflammation'
        }
        
        logger.info("Initialized simple embedding manager")

    def _simple_tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        if not text:
            return []
        
        # Convert to lowercase and split
        text = text.lower()
        # Keep alphanumeric and basic punctuation
        text = re.sub(r'[^\w\s\-]', ' ', text)
        tokens = text.split()
        
        # Filter tokens
        tokens = [t for t in tokens if len(t) > 2 and not t.isdigit()]
        
        return tokens

    def _build_vocabulary(self, texts: List[str]) -> None:
        """Build vocabulary from texts"""
        all_tokens = []
        doc_frequencies = Counter()
        
        for text in texts:
            tokens = self._simple_tokenize(text)
            all_tokens.extend(tokens)
            doc_frequencies.update(set(tokens))
        
        # Create vocabulary with most common terms
        token_counts = Counter(all_tokens)
        
        # Prioritize medical terms
        vocab_items = []
        for term in self.medical_terms:
            if term in token_counts:
                vocab_items.append((term, token_counts[term]))
        
        # Add other frequent terms
        for token, count in token_counts.most_common(self.embedding_dim):
            if (token, count) not in vocab_items:
                vocab_items.append((token, count))
        
        # Build vocabulary mapping
        self.vocabulary = {token: idx for idx, (token, _) in enumerate(vocab_items[:self.embedding_dim])}
        
        # Calculate IDF scores
        num_docs = len(texts)
        for token in self.vocabulary:
            df = doc_frequencies[token]
            self.idf_scores[token] = np.log(num_docs / (df + 1))
        
        self.is_fitted = True
        logger.info(f"Built vocabulary with {len(self.vocabulary)} terms")

    def _text_to_vector(self, text: str) -> np.ndarray:
        """Convert text to TF-IDF vector"""
        if not self.is_fitted:
            return np.zeros(self.embedding_dim)
        
        tokens = self._simple_tokenize(text)
        if not tokens:
            return np.zeros(self.embedding_dim)
        
        # Calculate TF scores
        tf_scores = Counter(tokens)
        total_tokens = len(tokens)
        
        # Create vector
        vector = np.zeros(self.embedding_dim)
        
        for token, tf in tf_scores.items():
            if token in self.vocabulary:
                idx = self.vocabulary[token]
                tf_normalized = tf / total_tokens
                idf = self.idf_scores.get(token, 1.0)
                vector[idx] = tf_normalized * idf
        
        # Normalize vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector

    def fit_vocabulary(self, texts: List[str]) -> None:
        """Fit vocabulary on texts"""
        if not texts:
            logger.warning("No texts provided for vocabulary fitting")
            return
        
        self._build_vocabulary(texts)
        logger.info("TF-IDF fitted with medical vocabulary fallback")

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts"""
        if not texts:
            return np.array([]).reshape(0, self.embedding_dim)
        
        # If not fitted, fit on the provided texts
        if not self.is_fitted:
            self.fit_vocabulary(texts)
        
        embeddings = []
        for text in texts:
            vector = self._text_to_vector(text)
            embeddings.append(vector)
        
        result = np.array(embeddings)
        logger.info(f"Successfully embedded {len(texts)} medical chunks")
        
        return result

    def embed_medical_chunks(self, chunks: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Embed medical chunks (compatibility method) - returns chunks with embeddings added"""
        if not chunks:
            return []
        
        # Extract text content from chunks
        texts = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                # Try different possible keys for text content
                text = chunk.get('content', chunk.get('text', chunk.get('chunk', str(chunk))))
            else:
                text = str(chunk)
            texts.append(text)
        
        # Generate embeddings
        embeddings = self.get_embeddings(texts)
        
        # Add embeddings to chunks
        result_chunks = []
        for i, chunk in enumerate(chunks):
            # Create a copy of the chunk
            new_chunk = chunk.copy() if isinstance(chunk, dict) else {'content': str(chunk)}
            # Add the embedding
            new_chunk['embedding'] = embeddings[i]
            result_chunks.append(new_chunk)
        
        logger.info(f"Successfully embedded {len(chunks)} medical chunks")
        
        return result_chunks

    def save_model(self, filepath: str) -> None:
        """Save the model"""
        model_data = {
            'vocabulary': self.vocabulary,
            'idf_scores': self.idf_scores,
            'is_fitted': self.is_fitted,
            'embedding_dim': self.embedding_dim
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load the model"""
        if not os.path.exists(filepath):
            logger.warning(f"Model file not found: {filepath}")
            return
        
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        self.vocabulary = model_data['vocabulary']
        self.idf_scores = model_data['idf_scores']
        self.is_fitted = model_data['is_fitted']
        self.embedding_dim = model_data['embedding_dim']
        
        logger.info(f"Model loaded from {filepath}")


# Additional compatibility methods
    def embed_text(self, text: str) -> np.ndarray:
        """Embed single text"""
        if not text:
            return np.zeros(self.embedding_dim)
        return self.get_embeddings([text])[0]

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Embed multiple texts"""
        embeddings = self.get_embeddings(texts)
        return [embeddings[i] for i in range(len(embeddings))]

    def embed_query(self, query: str) -> np.ndarray:
        """Embed query text"""
        return self.embed_text(query)

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        return self.embedding_dim

    def create_medical_query_embedding(self, query: str, context: str = None) -> np.ndarray:
        """Create medical query embedding with context"""
        # Enhance query with medical context
        if context:
            enhanced_query = f"medical {context} query: {query}"
        else:
            enhanced_query = f"medical query: {query}"
        return self.embed_text(enhanced_query)

    def get_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Get embeddings in batches"""
        return self.get_embeddings(texts)

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Alternative method name for get_embeddings"""
        return self.get_embeddings(texts)


# Alias for compatibility
EmbeddingManager = SimpleEmbeddingManager
