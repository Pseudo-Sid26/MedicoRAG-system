# import re
# import time
# from typing import List, Dict, Any, Tuple, Optional
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import logging
# from functools import lru_cache
# import concurrent.futures
# from dataclasses import dataclass
#
# logger = logging.getLogger(__name__)
#
#
# @dataclass
# class ChunkMetrics:
#     """Data class for chunk performance metrics"""
#     total_chunks: int
#     processing_time: float
#     avg_chunk_size: float
#     sections_detected: int
#     clinical_relevance_avg: float
#
#
# class MedicalChunkingStrategy:
#     """Optimized chunking strategy for medical documents with performance enhancements"""
#
#     def __init__(self, chunk_size: int = 1000, overlap: int = 200, enable_caching: bool = True):
#         self.chunk_size = chunk_size
#         self.overlap = overlap
#         self.enable_caching = enable_caching
#
#         # Performance: Compile regex patterns once
#         self._compile_patterns()
#
#         # Performance: Cache for processed sections
#         self._section_cache = {} if enable_caching else None
#         self._medical_context_cache = {} if enable_caching else None
#
#         # Enhanced medical section headers with more patterns
#         self.section_patterns = [
#             # Standard research paper sections
#             r'^(ABSTRACT|INTRODUCTION|METHODS?|RESULTS?|DISCUSSION|CONCLUSION)S?\s*:?',
#             r'^(BACKGROUND|OBJECTIVE|DESIGN|SETTING|PARTICIPANTS|INTERVENTIONS?)S?\s*:?',
#             r'^(MAIN OUTCOMES?|MEASUREMENTS?|LIMITATIONS|IMPLICATIONS)S?\s*:?',
#
#             # Clinical sections
#             r'^(CASE REPORT|PATIENT PRESENTATION|CLINICAL FINDINGS)S?\s*:?',
#             r'^(DIAGNOSIS|TREATMENT|MANAGEMENT|FOLLOW[-\s]?UP)S?\s*:?',
#             r'^(ADVERSE EVENTS?|SIDE EFFECTS?|CONTRAINDICATIONS?)S?\s*:?',
#             r'^(DOSAGE|ADMINISTRATION|PHARMACOLOGY)S?\s*:?',
#
#             # Additional medical sections
#             r'^(CLINICAL SIGNIFICANCE|THERAPEUTIC IMPLICATIONS|SAFETY PROFILE)S?\s*:?',
#             r'^(EPIDEMIOLOGY|PATHOPHYSIOLOGY|PROGNOSIS|PREVENTION)S?\s*:?',
#             r'^(DIFFERENTIAL DIAGNOSIS|COMPLICATIONS|MONITORING)S?\s*:?',
#             r'^(PATIENT EDUCATION|SPECIAL POPULATIONS|DRUG INTERACTIONS?)S?\s*:?',
#
#             # Numbered sections (1., 2., etc.)
#             r'^\d+\.\s+[A-Z][^.]+$',
#
#             # Guidelines sections
#             r'^(RECOMMENDATION|EVIDENCE|GRADE|STRENGTH)S?\s*:?'
#         ]
#
#         # Initialize optimized text splitter
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=chunk_size,
#             chunk_overlap=overlap,
#             length_function=len,
#             separators=[
#                 "\n\n\n",  # Multiple line breaks
#                 "\n\n",  # Double line breaks
#                 "\n",  # Single line breaks
#                 ". ",  # Sentence endings
#                 "! ",  # Exclamation sentences
#                 "? ",  # Question sentences
#                 "; ",  # Semicolons
#                 ", ",  # Commas
#                 " ",  # Spaces
#                 ""  # Character level
#             ]
#         )
#
#     def _compile_patterns(self):
#         """Compile regex patterns for better performance"""
#         self.compiled_section_patterns = [
#             re.compile(pattern, re.IGNORECASE) for pattern in self.section_patterns
#         ]
#
#         # Medical context patterns
#         self.medical_patterns = {
#             'diagnosis': re.compile(r'\b(diagnos[ei]s|condition|disease|disorder|syndrome)\b', re.IGNORECASE),
#             'treatment': re.compile(r'\b(treatment|therapy|medication|drug|intervention|management)\b', re.IGNORECASE),
#             'symptoms': re.compile(r'\b(symptom|sign|present|complaint|manifestation)\b', re.IGNORECASE),
#             'procedures': re.compile(r'\b(procedure|surgery|operation|intervention|technique)\b', re.IGNORECASE),
#             'measurements': re.compile(r'\b\d+(?:\.\d+)?\s*(?:mg|ml|mmHg|°[CF]|bpm|kg|cm|mm|hrs?|min)\b',
#                                        re.IGNORECASE),
#             'guidelines': re.compile(r'\b(guideline|recommendation|should|must|protocol|standard)\b', re.IGNORECASE),
#             'evidence': re.compile(r'\b(evidence|study|trial|research|data|analysis)\b', re.IGNORECASE),
#             'outcomes': re.compile(r'\b(outcome|result|finding|conclusion|efficacy|effectiveness)\b', re.IGNORECASE)
#         }
#
#         # Drug name patterns (enhanced)
#         self.drug_pattern = re.compile(
#             r'\b\w+(?:cillin|mycin|azole|pril|sartan|statin|ide|ine|ol|al|an|er|in|um|ate|ase)\b',
#             re.IGNORECASE
#         )
#
#         # Medical terminology pattern
#         self.medical_term_pattern = re.compile(r'\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\b')
#
#     @lru_cache(maxsize=100)
#     def _cached_section_detection(self, text_hash: str, text: str) -> Tuple[List[Dict[str, Any]]]:
#         """Cache section detection results for repeated processing"""
#         return tuple(self.detect_sections(text))
#
#     def detect_sections(self, text: str) -> List[Dict[str, Any]]:
#         """Optimized section detection with caching"""
#         if self.enable_caching:
#             text_hash = str(hash(text))
#             if text_hash in self._section_cache:
#                 return self._section_cache[text_hash]
#
#         sections = []
#         lines = text.split('\n')
#         current_section = None
#         current_content = []
#
#         # Performance: Process lines in batches for large documents
#         batch_size = 1000
#
#         for i in range(0, len(lines), batch_size):
#             batch_lines = lines[i:i + batch_size]
#
#             for line_idx, line in enumerate(batch_lines):
#                 line_stripped = line.strip()
#
#                 if not line_stripped:
#                     continue
#
#                 # Check if line matches any compiled section pattern
#                 section_match = None
#                 for pattern in self.compiled_section_patterns:
#                     if pattern.match(line_stripped):
#                         section_match = line_stripped
#                         break
#
#                 if section_match:
#                     # Save previous section
#                     if current_section and current_content:
#                         sections.append({
#                             'title': current_section,
#                             'content': '\n'.join(current_content).strip(),
#                             'start_line': len(sections),
#                             'length': len('\n'.join(current_content)),
#                             'section_type': self._classify_section_type(current_section)
#                         })
#
#                     # Start new section
#                     current_section = section_match
#                     current_content = []
#                 else:
#                     if line_stripped:  # Only add non-empty lines
#                         current_content.append(line)
#
#         # Add final section
#         if current_section and current_content:
#             sections.append({
#                 'title': current_section,
#                 'content': '\n'.join(current_content).strip(),
#                 'start_line': len(sections),
#                 'length': len('\n'.join(current_content)),
#                 'section_type': self._classify_section_type(current_section)
#             })
#
#         # Cache result
#         if self.enable_caching:
#             self._section_cache[text_hash] = sections
#
#         return sections
#
#     def _classify_section_type(self, section_title: str) -> str:
#         """Classify section type for better processing"""
#         title_lower = section_title.lower()
#
#         if any(word in title_lower for word in ['abstract', 'summary']):
#             return 'summary'
#         elif any(word in title_lower for word in ['introduction', 'background']):
#             return 'background'
#         elif any(word in title_lower for word in ['method', 'design', 'procedure']):
#             return 'methodology'
#         elif any(word in title_lower for word in ['result', 'finding', 'outcome']):
#             return 'results'
#         elif any(word in title_lower for word in ['discussion', 'conclusion']):
#             return 'discussion'
#         elif any(word in title_lower for word in ['treatment', 'management', 'therapy']):
#             return 'clinical'
#         elif any(word in title_lower for word in ['diagnosis', 'symptom', 'presentation']):
#             return 'diagnostic'
#         else:
#             return 'general'
#
#     def semantic_chunking(self, text: str) -> List[Dict[str, Any]]:
#         """Optimized semantic chunking with parallel processing for large documents"""
#         start_time = time.time()
#         chunks = []
#
#         # First, try to detect sections
#         sections = self.detect_sections(text)
#
#         if sections and len(sections) > 1:
#             # Process sections in parallel for large documents
#             if len(sections) > 5:
#                 chunks = self._parallel_section_processing(sections)
#             else:
#                 # Process each section separately
#                 for section in sections:
#                     section_chunks = self._chunk_section(section)
#                     chunks.extend(section_chunks)
#         else:
#             # Fallback to optimized paragraph-based chunking
#             chunks = self._chunk_by_paragraphs(text)
#
#         # Log performance metrics
#         processing_time = time.time() - start_time
#         logger.debug(f"Semantic chunking completed in {processing_time:.2f}s, created {len(chunks)} chunks")
#
#         return chunks
#
#     def _parallel_section_processing(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#         """Process sections in parallel for better performance"""
#         chunks = []
#
#         # Use ThreadPoolExecutor for I/O bound operations
#         with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
#             future_to_section = {
#                 executor.submit(self._chunk_section, section): section
#                 for section in sections
#             }
#
#             for future in concurrent.futures.as_completed(future_to_section):
#                 try:
#                     section_chunks = future.result()
#                     chunks.extend(section_chunks)
#                 except Exception as e:
#                     logger.error(f"Error processing section: {e}")
#                     # Continue with other sections
#                     continue
#
#         # Sort chunks by original order
#         chunks.sort(key=lambda x: x.get('original_order', 0))
#
#         return chunks
#
#     def _chunk_section(self, section: Dict[str, Any]) -> List[Dict[str, Any]]:
#         """Optimized section chunking with enhanced metadata"""
#         section_title = section['title']
#         section_content = section['content']
#         section_type = section.get('section_type', 'general')
#
#         chunks = []
#
#         if len(section_content) <= self.chunk_size:
#             # Section fits in one chunk
#             chunks.append({
#                 'content': section_content,
#                 'section': section_title,
#                 'section_type': section_type,
#                 'chunk_type': 'complete_section',
#                 'size': len(section_content),
#                 'overlap_previous': 0,
#                 'is_complete_section': True,
#                 'original_order': section.get('start_line', 0)
#             })
#         else:
#             # Split section into multiple chunks with smart overlap
#             text_chunks = self._smart_section_split(section_content)
#
#             for i, chunk_text in enumerate(text_chunks):
#                 # Add section context to each chunk
#                 contextual_content = f"{section_title}\n\n{chunk_text}"
#
#                 chunks.append({
#                     'content': contextual_content,
#                     'section': section_title,
#                     'section_type': section_type,
#                     'chunk_type': 'section_part',
#                     'part_number': i + 1,
#                     'total_parts': len(text_chunks),
#                     'size': len(chunk_text),
#                     'overlap_previous': self.overlap if i > 0 else 0,
#                     'is_complete_section': False,
#                     'original_order': section.get('start_line', 0) + i * 0.1
#                 })
#
#         return chunks
#
#     def _smart_section_split(self, content: str) -> List[str]:
#         """Smart splitting that preserves sentence boundaries and context"""
#         # Try to split at sentence boundaries first
#         sentences = re.split(r'(?<=[.!?])\s+', content)
#
#         chunks = []
#         current_chunk = ""
#
#         for sentence in sentences:
#             # Check if adding this sentence would exceed chunk size
#             if len(current_chunk) + len(sentence) + 1 > self.chunk_size:
#                 if current_chunk:
#                     chunks.append(current_chunk.strip())
#                     # Add overlap from the end of current chunk
#                     overlap_text = current_chunk[-self.overlap:] if len(current_chunk) > self.overlap else current_chunk
#                     current_chunk = overlap_text + " " + sentence
#                 else:
#                     # Single sentence is too large, use standard splitter
#                     large_chunks = self.text_splitter.split_text(sentence)
#                     chunks.extend(large_chunks[:-1])
#                     current_chunk = large_chunks[-1] if large_chunks else ""
#             else:
#                 if current_chunk:
#                     current_chunk += " " + sentence
#                 else:
#                     current_chunk = sentence
#
#         # Add final chunk
#         if current_chunk:
#             chunks.append(current_chunk.strip())
#
#         return chunks
#
#     def _chunk_by_paragraphs(self, text: str) -> List[Dict[str, Any]]:
#         """Optimized paragraph-based chunking"""
#         paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
#         chunks = []
#         current_chunk = ""
#         chunk_count = 0
#
#         for para in paragraphs:
#             # Estimate if adding this paragraph would exceed chunk size
#             estimated_size = len(current_chunk) + len(para) + 2
#
#             if estimated_size > self.chunk_size:
#                 if current_chunk:
#                     chunks.append({
#                         'content': current_chunk.strip(),
#                         'section': 'Unstructured Content',
#                         'section_type': 'general',
#                         'chunk_type': 'paragraph_based',
#                         'chunk_number': chunk_count + 1,
#                         'size': len(current_chunk.strip()),
#                         'overlap_previous': 0,
#                         'is_complete_section': False,
#                         'original_order': chunk_count
#                     })
#                     chunk_count += 1
#                     current_chunk = para
#                 else:
#                     # Single paragraph is too large, split it intelligently
#                     large_chunks = self._smart_section_split(para)
#                     for chunk_text in large_chunks:
#                         chunks.append({
#                             'content': chunk_text,
#                             'section': 'Large Paragraph',
#                             'section_type': 'general',
#                             'chunk_type': 'large_paragraph_split',
#                             'chunk_number': chunk_count + 1,
#                             'size': len(chunk_text),
#                             'overlap_previous': self.overlap,
#                             'is_complete_section': False,
#                             'original_order': chunk_count
#                         })
#                         chunk_count += 1
#                     current_chunk = ""
#             else:
#                 if current_chunk:
#                     current_chunk += "\n\n" + para
#                 else:
#                     current_chunk = para
#
#         # Add final chunk
#         if current_chunk:
#             chunks.append({
#                 'content': current_chunk.strip(),
#                 'section': 'Unstructured Content',
#                 'section_type': 'general',
#                 'chunk_type': 'paragraph_based',
#                 'chunk_number': chunk_count + 1,
#                 'size': len(current_chunk.strip()),
#                 'overlap_previous': 0,
#                 'is_complete_section': False,
#                 'original_order': chunk_count
#             })
#
#         return chunks
#
#     def create_contextual_chunks(self, text: str, document_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
#         """Create chunks with enhanced medical context and metadata"""
#         start_time = time.time()
#
#         base_chunks = self.semantic_chunking(text)
#         enhanced_chunks = []
#
#         # Process chunks in batches for better performance
#         batch_size = 10
#         for i in range(0, len(base_chunks), batch_size):
#             batch = base_chunks[i:i + batch_size]
#
#             for j, chunk in enumerate(batch):
#                 chunk_index = i + j
#                 enhanced_chunk = {
#                     **chunk,
#                     'document_type': document_metadata.get('type', 'unknown'),
#                     'document_source': document_metadata.get('filename', 'unknown'),
#                     'chunk_id': f"{document_metadata.get('filename', 'doc')}_{chunk_index}",
#                     'medical_context': self._extract_medical_context_optimized(chunk['content']),
#                     'clinical_relevance': self._assess_clinical_relevance_optimized(chunk['content']),
#                     'chunk_index': chunk_index,
#                     'processing_timestamp': time.time(),
#                     'word_count': len(chunk['content'].split()),
#                     'sentence_count': len(re.split(r'[.!?]+', chunk['content']))
#                 }
#                 enhanced_chunks.append(enhanced_chunk)
#
#         processing_time = time.time() - start_time
#
#         # Log metrics
#         metrics = ChunkMetrics(
#             total_chunks=len(enhanced_chunks),
#             processing_time=processing_time,
#             avg_chunk_size=sum(c['size'] for c in enhanced_chunks) / len(enhanced_chunks) if enhanced_chunks else 0,
#             sections_detected=len(set(c.get('section', '') for c in enhanced_chunks)),
#             clinical_relevance_avg=sum(c['clinical_relevance'] for c in enhanced_chunks) / len(
#                 enhanced_chunks) if enhanced_chunks else 0
#         )
#
#         logger.info(f"Chunk processing metrics: {metrics}")
#
#         return enhanced_chunks
#
#     def _extract_medical_context_optimized(self, chunk_content: str) -> Dict[str, Any]:
#         """Optimized medical context extraction using compiled patterns"""
#         cache_key = hash(chunk_content) if self.enable_caching else None
#
#         if self.enable_caching and cache_key in self._medical_context_cache:
#             return self._medical_context_cache[cache_key]
#
#         context = {}
#
#         # Use compiled patterns for better performance
#         for context_type, pattern in self.medical_patterns.items():
#             context[f'contains_{context_type}'] = bool(pattern.search(chunk_content))
#
#         # Additional context
#         context.update({
#             'medication_mentions': len(self.drug_pattern.findall(chunk_content)),
#             'numerical_data_count': len(re.findall(r'\b\d+(?:\.\d+)?\b', chunk_content)),
#             'medical_acronyms': len(re.findall(r'\b[A-Z]{2,}\b', chunk_content)),
#             'citation_count': len(re.findall(r'\[\d+\]|\(\d{4}\)', chunk_content))
#         })
#
#         # Cache result
#         if self.enable_caching and cache_key:
#             self._medical_context_cache[cache_key] = context
#
#         return context
#
#     def _assess_clinical_relevance_optimized(self, chunk_content: str) -> float:
#         """Optimized clinical relevance assessment"""
#         content_lower = chunk_content.lower()
#         relevance_score = 0.0
#
#         # Weight different types of medical content
#         weights = {
#             'diagnosis': 0.25,
#             'treatment': 0.25,
#             'evidence': 0.20,
#             'outcomes': 0.15,
#             'procedures': 0.10,
#             'guidelines': 0.05
#         }
#
#         for context_type, weight in weights.items():
#             if self.medical_patterns[context_type].search(content_lower):
#                 relevance_score += weight
#
#         # Bonus for specific medical indicators
#         if re.search(r'\b(patient|clinical|medical)\b', content_lower):
#             relevance_score += 0.1
#
#         if re.search(r'\b(randomized|controlled|trial|study)\b', content_lower):
#             relevance_score += 0.1
#
#         return min(relevance_score, 1.0)
#
#     def optimize_chunks_for_retrieval(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#         """Enhanced chunk optimization for retrieval with performance improvements"""
#         optimized_chunks = []
#
#         for chunk in chunks:
#             content = chunk['content']
#
#             # Add contextual headers
#             context_headers = []
#
#             if chunk.get('section') and chunk['section'] not in content:
#                 context_headers.append(f"Section: {chunk['section']}")
#
#             if chunk.get('document_type'):
#                 context_headers.append(f"Document Type: {chunk['document_type']}")
#
#             if chunk.get('section_type') and chunk['section_type'] != 'general':
#                 context_headers.append(f"Content Type: {chunk['section_type']}")
#
#             # Combine headers with content
#             if context_headers:
#                 content = '\n'.join(context_headers) + '\n\n' + content
#
#             optimized_chunk = {
#                 **chunk,
#                 'content': content,
#                 'search_keywords': self._extract_search_keywords_optimized(content),
#                 'chunk_summary': self._generate_chunk_summary_optimized(content),
#                 'retrieval_weight': self._calculate_retrieval_weight(chunk)
#             }
#
#             optimized_chunks.append(optimized_chunk)
#
#         return optimized_chunks
#
#     def _extract_search_keywords_optimized(self, content: str) -> List[str]:
#         """Optimized keyword extraction"""
#         keywords = set()
#
#         # Medical terms
#         medical_terms = self.medical_term_pattern.findall(content)
#         keywords.update(term for term in medical_terms if len(term) > 3)
#
#         # Drug names
#         drugs = self.drug_pattern.findall(content)
#         keywords.update(drugs)
#
#         # Measurements and values
#         measurements = self.medical_patterns['measurements'].findall(content)
#         keywords.update(measurements)
#
#         # Medical acronyms
#         acronyms = re.findall(r'\b[A-Z]{2,6}\b', content)
#         keywords.update(acronym for acronym in acronyms if len(acronym) <= 6)
#
#         return list(keywords)[:25]  # Limit to top 25 keywords
#
#     def _generate_chunk_summary_optimized(self, content: str) -> str:
#         """Optimized chunk summary generation"""
#         # Remove context headers for summary
#         lines = content.split('\n')
#         content_lines = [line for line in lines if not line.startswith(('Section:', 'Document Type:', 'Content Type:'))]
#         clean_content = '\n'.join(content_lines)
#
#         sentences = re.split(r'[.!?]+', clean_content)
#         sentences = [s.strip() for s in sentences if s.strip()]
#
#         if not sentences:
#             return clean_content[:200] + "..." if len(clean_content) > 200 else clean_content
#
#         if len(sentences) <= 2:
#             summary = '. '.join(sentences)
#         else:
#             # Take first and most relevant sentence
#             first_sentence = sentences[0]
#
#             # Find sentence with highest medical term density
#             best_sentence = sentences[1]
#             best_score = 0
#
#             for sentence in sentences[1:]:
#                 score = sum(1 for pattern in self.medical_patterns.values() if pattern.search(sentence))
#                 if score > best_score:
#                     best_score = score
#                     best_sentence = sentence
#
#             summary = f"{first_sentence}. {best_sentence}"
#
#         return summary[:300] + "..." if len(summary) > 300 else summary
#
#     def _calculate_retrieval_weight(self, chunk: Dict[str, Any]) -> float:
#         """Calculate retrieval weight based on chunk characteristics"""
#         weight = 1.0
#
#         # Boost complete sections
#         if chunk.get('is_complete_section'):
#             weight *= 1.2
#
#         # Boost clinical content
#         if chunk.get('section_type') in ['clinical', 'diagnostic', 'results']:
#             weight *= 1.15
#
#         # Boost based on clinical relevance
#         relevance = chunk.get('clinical_relevance', 0)
#         weight *= (1 + relevance * 0.3)
#
#         # Consider chunk size (prefer medium-sized chunks)
#         size = chunk.get('size', 0)
#         if 500 <= size <= 1500:
#             weight *= 1.1
#         elif size < 200:
#             weight *= 0.8
#
#         return round(weight, 3)
#
#     def get_chunking_stats(self) -> Dict[str, Any]:
#         """Get chunking performance statistics"""
#         return {
#             'cache_enabled': self.enable_caching,
#             'section_cache_size': len(self._section_cache) if self._section_cache else 0,
#             'medical_context_cache_size': len(self._medical_context_cache) if self._medical_context_cache else 0,
#             'chunk_size': self.chunk_size,
#             'overlap': self.overlap,
#             'patterns_compiled': len(self.compiled_section_patterns)
#         }
#
#     def clear_cache(self):
#         """Clear all caches to free memory"""
#         if self._section_cache:
#             self._section_cache.clear()
#         if self._medical_context_cache:
#             self._medical_context_cache.clear()
#
#         # Clear LRU cache
#         self._cached_section_detection.cache_clear()
#
#         logger.info("Chunking strategy caches cleared")

import re
import time
from typing import List, Dict, Any, Tuple, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
from functools import lru_cache
import concurrent.futures
from dataclasses import dataclass
import hashlib
import json

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetrics:
    """Data class for chunk performance metrics"""
    total_chunks: int
    processing_time: float
    avg_chunk_size: float
    sections_detected: int
    clinical_relevance_avg: float
    medical_context_detected: int
    optimization_applied: bool


class MedicalChunkingStrategy:
    """Optimized chunking strategy for medical documents with FAISS integration"""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200, enable_caching: bool = True):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.enable_caching = enable_caching

        # Performance: Compile regex patterns once
        self._compile_patterns()

        # Performance: Cache for processed sections
        self._section_cache = {} if enable_caching else None
        self._medical_context_cache = {} if enable_caching else None

        # Enhanced medical section headers with more comprehensive patterns
        self.section_patterns = [
            # Standard research paper sections
            r'^(ABSTRACT|INTRODUCTION|METHODS?|RESULTS?|DISCUSSION|CONCLUSION)S?\s*:?',
            r'^(BACKGROUND|OBJECTIVE|DESIGN|SETTING|PARTICIPANTS|INTERVENTIONS?)S?\s*:?',
            r'^(MAIN OUTCOMES?|MEASUREMENTS?|LIMITATIONS|IMPLICATIONS)S?\s*:?',

            # Clinical sections
            r'^(CASE REPORT|PATIENT PRESENTATION|CLINICAL FINDINGS)S?\s*:?',
            r'^(DIAGNOSIS|TREATMENT|MANAGEMENT|FOLLOW[-\s]?UP)S?\s*:?',
            r'^(ADVERSE EVENTS?|SIDE EFFECTS?|CONTRAINDICATIONS?)S?\s*:?',
            r'^(DOSAGE|ADMINISTRATION|PHARMACOLOGY|PHARMACOKINETICS)S?\s*:?',

            # Additional medical sections
            r'^(CLINICAL SIGNIFICANCE|THERAPEUTIC IMPLICATIONS|SAFETY PROFILE)S?\s*:?',
            r'^(EPIDEMIOLOGY|PATHOPHYSIOLOGY|PROGNOSIS|PREVENTION)S?\s*:?',
            r'^(DIFFERENTIAL DIAGNOSIS|COMPLICATIONS|MONITORING)S?\s*:?',
            r'^(PATIENT EDUCATION|SPECIAL POPULATIONS|DRUG INTERACTIONS?)S?\s*:?',

            # Numbered sections (1., 2., etc.)
            r'^\d+\.\s+[A-Z][^.]+$',

            # Guidelines sections
            r'^(RECOMMENDATION|EVIDENCE|GRADE|STRENGTH|LEVEL OF EVIDENCE)S?\s*:?',

            # Additional clinical patterns
            r'^(HISTORY|PHYSICAL EXAM|LABORATORY|IMAGING|ASSESSMENT|PLAN)S?\s*:?',
            r'^(CHIEF COMPLAINT|HPI|PMH|PSH|ALLERGIES|MEDICATIONS)S?\s*:?',
            r'^(SOCIAL HISTORY|FAMILY HISTORY|REVIEW OF SYSTEMS)S?\s*:?'
        ]

        # Initialize optimized text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=[
                "\n\n\n",  # Multiple line breaks
                "\n\n",  # Double line breaks
                "\n",  # Single line breaks
                ". ",  # Sentence endings
                "! ",  # Exclamation sentences
                "? ",  # Question sentences
                "; ",  # Semicolons
                ", ",  # Commas
                " ",  # Spaces
                ""  # Character level
            ]
        )

    def _compile_patterns(self):
        """Compile regex patterns for better performance"""
        self.compiled_section_patterns = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in self.section_patterns
        ]

        # Enhanced medical context patterns
        self.medical_patterns = {
            'diagnosis': re.compile(
                r'\b(diagnos[ei]s|condition|disease|disorder|syndrome|pathology|etiology)\b',
                re.IGNORECASE
            ),
            'treatment': re.compile(
                r'\b(treatment|therapy|medication|drug|intervention|management|therapeutic|remedy)\b',
                re.IGNORECASE
            ),
            'symptoms': re.compile(
                r'\b(symptom|sign|present|complaint|manifestation|finding|feature)\b',
                re.IGNORECASE
            ),
            'procedures': re.compile(
                r'\b(procedure|surgery|operation|intervention|technique|approach|method)\b',
                re.IGNORECASE
            ),
            'measurements': re.compile(
                r'\b\d+(?:\.\d+)?\s*(?:mg|ml|mmHg|°[CF]|bpm|kg|cm|mm|hrs?|min|sec|units?|IU|mEq|mcg|ng|pg)\b',
                re.IGNORECASE
            ),
            'guidelines': re.compile(
                r'\b(guideline|recommendation|should|must|protocol|standard|policy|criteria)\b',
                re.IGNORECASE
            ),
            'evidence': re.compile(
                r'\b(evidence|study|trial|research|data|analysis|investigation|examination)\b',
                re.IGNORECASE
            ),
            'outcomes': re.compile(
                r'\b(outcome|result|finding|conclusion|efficacy|effectiveness|response|benefit)\b',
                re.IGNORECASE
            ),
            'clinical_terms': re.compile(
                r'\b(patient|clinical|medical|healthcare|hospital|clinic)\b',
                re.IGNORECASE
            ),
            'anatomy': re.compile(
                r'\b(heart|lung|brain|liver|kidney|stomach|intestine|muscle|bone|blood)\b',
                re.IGNORECASE
            )
        }

        # Enhanced drug name patterns
        self.drug_pattern = re.compile(
            r'\b\w+(?:cillin|mycin|azole|pril|sartan|statin|ide|ine|ol|al|an|er|in|um|ate|ase|mab|nib|tide)\b',
            re.IGNORECASE
        )

        # Medical terminology pattern (more specific)
        self.medical_term_pattern = re.compile(
            r'\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|[a-z]+(?:itis|osis|emia|uria|pathy|trophy|plasia|genesis))\b'
        )

        # Clinical values pattern
        self.clinical_values_pattern = re.compile(
            r'\b(?:normal|abnormal|elevated|decreased|positive|negative|stable|unstable)\b',
            re.IGNORECASE
        )

    @lru_cache(maxsize=200)
    def _cached_section_detection(self, text_hash: str) -> Tuple[Dict[str, Any], ...]:
        """Cache section detection results for repeated processing"""
        # Note: We can't cache the actual text processing, just the hash
        return tuple()

    def detect_sections(self, text: str) -> List[Dict[str, Any]]:
        """Optimized section detection with enhanced medical focus"""
        if self.enable_caching:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash in self._section_cache:
                return self._section_cache[text_hash]

        sections = []
        lines = text.split('\n')
        current_section = None
        current_content = []
        line_number = 0

        # Performance: Process lines efficiently
        for line in lines:
            line_number += 1
            line_stripped = line.strip()

            if not line_stripped:
                continue

            # Check if line matches any compiled section pattern
            section_match = None
            for pattern in self.compiled_section_patterns:
                if pattern.match(line_stripped):
                    section_match = line_stripped
                    break

            if section_match:
                # Save previous section
                if current_section and current_content:
                    content_text = '\n'.join(current_content).strip()
                    sections.append({
                        'title': current_section,
                        'content': content_text,
                        'start_line': line_number - len(current_content),
                        'end_line': line_number - 1,
                        'length': len(content_text),
                        'section_type': self._classify_section_type(current_section),
                        'medical_relevance': self._assess_section_medical_relevance(content_text),
                        'word_count': len(content_text.split()),
                        'sentence_count': len(re.split(r'[.!?]+', content_text))
                    })

                # Start new section
                current_section = section_match
                current_content = []
            else:
                if line_stripped:  # Only add non-empty lines
                    current_content.append(line)

        # Add final section
        if current_section and current_content:
            content_text = '\n'.join(current_content).strip()
            sections.append({
                'title': current_section,
                'content': content_text,
                'start_line': line_number - len(current_content),
                'end_line': line_number,
                'length': len(content_text),
                'section_type': self._classify_section_type(current_section),
                'medical_relevance': self._assess_section_medical_relevance(content_text),
                'word_count': len(content_text.split()),
                'sentence_count': len(re.split(r'[.!?]+', content_text))
            })

        # Cache result
        if self.enable_caching:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            self._section_cache[text_hash] = sections

        return sections

    def _classify_section_type(self, section_title: str) -> str:
        """Enhanced section type classification"""
        title_lower = section_title.lower()

        # Clinical documentation sections
        if any(word in title_lower for word in ['chief complaint', 'cc', 'hpi', 'history']):
            return 'clinical_history'
        elif any(word in title_lower for word in ['physical exam', 'examination', 'findings']):
            return 'physical_examination'
        elif any(word in title_lower for word in ['assessment', 'impression', 'diagnosis']):
            return 'assessment'
        elif any(word in title_lower for word in ['plan', 'treatment', 'management', 'therapy']):
            return 'treatment_plan'

        # Research paper sections
        elif any(word in title_lower for word in ['abstract', 'summary']):
            return 'summary'
        elif any(word in title_lower for word in ['introduction', 'background']):
            return 'background'
        elif any(word in title_lower for word in ['method', 'design', 'procedure']):
            return 'methodology'
        elif any(word in title_lower for word in ['result', 'finding', 'outcome']):
            return 'results'
        elif any(word in title_lower for word in ['discussion', 'conclusion']):
            return 'discussion'

        # Clinical content
        elif any(word in title_lower for word in ['dosage', 'administration', 'pharmacology']):
            return 'pharmacological'
        elif any(word in title_lower for word in ['adverse', 'side effect', 'contraindication']):
            return 'safety'
        elif any(word in title_lower for word in ['evidence', 'recommendation', 'guideline']):
            return 'evidence_based'

        else:
            return 'general'

    def _assess_section_medical_relevance(self, content: str) -> float:
        """Assess medical relevance of a section"""
        relevance_score = 0.0
        content_lower = content.lower()

        # Check for medical patterns
        for pattern_name, pattern in self.medical_patterns.items():
            if pattern.search(content_lower):
                relevance_score += 0.1

        # Bonus for drug mentions
        if self.drug_pattern.search(content):
            relevance_score += 0.15

        # Bonus for clinical values
        if self.clinical_values_pattern.search(content):
            relevance_score += 0.1

        # Bonus for medical terminology
        medical_terms = len(self.medical_term_pattern.findall(content))
        relevance_score += min(medical_terms * 0.02, 0.2)

        return min(relevance_score, 1.0)

    def semantic_chunking(self, text: str) -> List[Dict[str, Any]]:
        """Enhanced semantic chunking with medical context awareness"""
        start_time = time.time()
        chunks = []

        # First, try to detect sections
        sections = self.detect_sections(text)

        if sections and len(sections) > 1:
            # Process sections based on size and content
            for section in sections:
                section_chunks = self._chunk_section_intelligently(section)
                chunks.extend(section_chunks)
        else:
            # Fallback to intelligent paragraph-based chunking
            chunks = self._chunk_by_medical_paragraphs(text)

        # Post-process chunks for medical optimization
        chunks = self._optimize_medical_chunks(chunks)

        # Log performance metrics
        processing_time = time.time() - start_time
        logger.info(f"Semantic chunking completed in {processing_time:.2f}s, created {len(chunks)} chunks")

        return chunks

    def _chunk_section_intelligently(self, section: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Intelligent section chunking with medical context preservation"""
        section_title = section['title']
        section_content = section['content']
        section_type = section.get('section_type', 'general')
        medical_relevance = section.get('medical_relevance', 0.0)

        chunks = []

        # Adjust chunk size based on section type
        effective_chunk_size = self._get_effective_chunk_size(section_type, medical_relevance)

        if len(section_content) <= effective_chunk_size:
            # Section fits in one chunk
            chunks.append({
                'content': section_content,
                'section': section_title,
                'section_type': section_type,
                'chunk_type': 'complete_section',
                'size': len(section_content),
                'overlap_previous': 0,
                'is_complete_section': True,
                'medical_relevance': medical_relevance,
                'original_order': section.get('start_line', 0),
                'section_metadata': {
                    'word_count': section.get('word_count', 0),
                    'sentence_count': section.get('sentence_count', 0)
                }
            })
        else:
            # Split section into multiple chunks with intelligent overlap
            text_chunks = self._smart_medical_split(section_content, effective_chunk_size)

            for i, chunk_text in enumerate(text_chunks):
                # Add section context to each chunk for better retrieval
                contextual_content = self._add_section_context(chunk_text, section_title, section_type)

                chunks.append({
                    'content': contextual_content,
                    'section': section_title,
                    'section_type': section_type,
                    'chunk_type': 'section_part',
                    'part_number': i + 1,
                    'total_parts': len(text_chunks),
                    'size': len(chunk_text),
                    'overlap_previous': self.overlap if i > 0 else 0,
                    'is_complete_section': False,
                    'medical_relevance': medical_relevance,
                    'original_order': section.get('start_line', 0) + i * 0.1,
                    'section_metadata': {
                        'parent_section_words': section.get('word_count', 0),
                        'parent_section_sentences': section.get('sentence_count', 0)
                    }
                })

        return chunks

    def _get_effective_chunk_size(self, section_type: str, medical_relevance: float) -> int:
        """Determine effective chunk size based on content type"""
        base_size = self.chunk_size

        # Adjust based on section type
        if section_type in ['summary', 'abstract']:
            return int(base_size * 0.8)  # Smaller for summaries
        elif section_type in ['methodology', 'results']:
            return int(base_size * 1.2)  # Larger for detailed sections
        elif section_type in ['treatment_plan', 'assessment']:
            return int(base_size * 1.1)  # Slightly larger for clinical content

        # Adjust based on medical relevance
        if medical_relevance > 0.7:
            return int(base_size * 1.1)  # Larger chunks for highly medical content

        return base_size

    def _smart_medical_split(self, content: str, chunk_size: int) -> List[str]:
        """Smart splitting that preserves medical context and sentence boundaries"""
        # Try to split at medical section boundaries first
        medical_boundaries = re.split(r'\n(?=(?:DIAGNOSIS|TREATMENT|ASSESSMENT|PLAN|FINDINGS))', content)

        if len(medical_boundaries) > 1:
            # Split along medical boundaries
            chunks = []
            for boundary in medical_boundaries:
                if len(boundary) <= chunk_size:
                    chunks.append(boundary.strip())
                else:
                    # Further split large boundaries
                    sub_chunks = self._split_by_sentences(boundary, chunk_size)
                    chunks.extend(sub_chunks)
            return chunks

        # Fallback to sentence-based splitting
        return self._split_by_sentences(content, chunk_size)

    def _split_by_sentences(self, content: str, chunk_size: int) -> List[str]:
        """Split content by sentences while preserving medical context"""
        sentences = re.split(r'(?<=[.!?])\s+', content)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) + 1 > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    # Add overlap from the end of current chunk
                    overlap_text = self._get_smart_overlap(current_chunk)
                    current_chunk = overlap_text + " " + sentence
                else:
                    # Single sentence is too large, use standard splitter
                    large_chunks = self.text_splitter.split_text(sentence)
                    chunks.extend(large_chunks[:-1])
                    current_chunk = large_chunks[-1] if large_chunks else ""
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence

        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _get_smart_overlap(self, current_chunk: str) -> str:
        """Get intelligent overlap that preserves medical context"""
        # Try to get overlap that ends at a sentence boundary
        sentences = re.split(r'(?<=[.!?])\s+', current_chunk)

        if len(sentences) <= 1:
            return current_chunk[-self.overlap:] if len(current_chunk) > self.overlap else current_chunk

        # Find the best overlap point
        overlap_text = ""
        for sentence in reversed(sentences):
            if len(overlap_text) + len(sentence) <= self.overlap:
                overlap_text = sentence + " " + overlap_text
            else:
                break

        return overlap_text.strip()

    def _add_section_context(self, chunk_text: str, section_title: str, section_type: str) -> str:
        """Add section context to chunk for better retrieval"""
        context_prefix = f"[{section_title}]"

        # Add section type context for clinical sections
        if section_type in ['clinical_history', 'physical_examination', 'assessment', 'treatment_plan']:
            context_prefix += f" [{section_type.replace('_', ' ').title()}]"

        return f"{context_prefix}\n\n{chunk_text}"

    def _chunk_by_medical_paragraphs(self, text: str) -> List[Dict[str, Any]]:
        """Enhanced paragraph-based chunking with medical awareness"""
        # Split by double newlines but preserve medical list structures
        paragraphs = re.split(r'\n\n+', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks = []
        current_chunk = ""
        chunk_count = 0

        for para in paragraphs:
            # Estimate if adding this paragraph would exceed chunk size
            estimated_size = len(current_chunk) + len(para) + 2

            if estimated_size > self.chunk_size:
                if current_chunk:
                    chunks.append({
                        'content': current_chunk.strip(),
                        'section': 'Unstructured Medical Content',
                        'section_type': 'general',
                        'chunk_type': 'paragraph_based',
                        'chunk_number': chunk_count + 1,
                        'size': len(current_chunk.strip()),
                        'overlap_previous': 0,
                        'is_complete_section': False,
                        'medical_relevance': self._assess_section_medical_relevance(current_chunk),
                        'original_order': chunk_count
                    })
                    chunk_count += 1
                    current_chunk = para
                else:
                    # Single paragraph is too large, split it intelligently
                    large_chunks = self._smart_medical_split(para, self.chunk_size)
                    for chunk_text in large_chunks:
                        chunks.append({
                            'content': chunk_text,
                            'section': 'Large Medical Paragraph',
                            'section_type': 'general',
                            'chunk_type': 'large_paragraph_split',
                            'chunk_number': chunk_count + 1,
                            'size': len(chunk_text),
                            'overlap_previous': self.overlap,
                            'is_complete_section': False,
                            'medical_relevance': self._assess_section_medical_relevance(chunk_text),
                            'original_order': chunk_count
                        })
                        chunk_count += 1
                    current_chunk = ""
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para

        # Add final chunk
        if current_chunk:
            chunks.append({
                'content': current_chunk.strip(),
                'section': 'Unstructured Medical Content',
                'section_type': 'general',
                'chunk_type': 'paragraph_based',
                'chunk_number': chunk_count + 1,
                'size': len(current_chunk.strip()),
                'overlap_previous': 0,
                'is_complete_section': False,
                'medical_relevance': self._assess_section_medical_relevance(current_chunk),
                'original_order': chunk_count
            })

        return chunks

    def _optimize_medical_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize chunks for medical content retrieval"""
        optimized_chunks = []

        for chunk in chunks:
            # Enhance medical context
            medical_context = self._extract_medical_context_optimized(chunk['content'])

            # Calculate clinical relevance
            clinical_relevance = self._assess_clinical_relevance_optimized(chunk['content'])

            # Add medical enhancements
            enhanced_chunk = {
                **chunk,
                'medical_context': medical_context,
                'clinical_relevance': clinical_relevance,
                'medical_entities': self._extract_medical_entities(chunk['content']),
                'clinical_keywords': self._extract_clinical_keywords(chunk['content'])
            }

            optimized_chunks.append(enhanced_chunk)

        return optimized_chunks

    def create_contextual_chunks(self, text: str, document_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create chunks with enhanced medical context and FAISS-optimized metadata"""
        start_time = time.time()

        base_chunks = self.semantic_chunking(text)
        enhanced_chunks = []

        # Process chunks efficiently
        for i, chunk in enumerate(base_chunks):
            enhanced_chunk = {
                **chunk,
                'document_type': document_metadata.get('type', 'unknown'),
                'document_source': document_metadata.get('filename', 'unknown'),
                'chunk_id': f"{document_metadata.get('filename', 'doc')}_{i}",
                'chunk_index': i,
                'processing_timestamp': time.time(),
                'word_count': len(chunk['content'].split()),
                'sentence_count': len(re.split(r'[.!?]+', chunk['content'])),
                # FAISS-specific enhancements
                'embedding_ready': True,
                'retrieval_priority': self._calculate_retrieval_priority(chunk),
                'medical_specialty': self._detect_medical_specialty(chunk['content']),
                'content_classification': self._classify_content_type(chunk['content'])
            }
            enhanced_chunks.append(enhanced_chunk)

        processing_time = time.time() - start_time

        # Calculate comprehensive metrics
        metrics = self._calculate_chunk_metrics(enhanced_chunks, processing_time)
        logger.info(f"Enhanced chunk processing metrics: {metrics}")

        return enhanced_chunks

    def _extract_medical_context_optimized(self, chunk_content: str) -> Dict[str, Any]:
        """Optimized medical context extraction using compiled patterns"""
        cache_key = hashlib.md5(chunk_content.encode()).hexdigest() if self.enable_caching else None

        if self.enable_caching and cache_key in self._medical_context_cache:
            return self._medical_context_cache[cache_key]

        context = {}

        # Use compiled patterns for better performance
        for context_type, pattern in self.medical_patterns.items():
            matches = pattern.findall(chunk_content)
            context[f'contains_{context_type}'] = len(matches) > 0
            context[f'{context_type}_count'] = len(matches)

        # Additional enhanced context
        context.update({
            'medication_mentions': len(self.drug_pattern.findall(chunk_content)),
            'numerical_data_count': len(re.findall(r'\b\d+(?:\.\d+)?\b', chunk_content)),
            'medical_acronyms': len(re.findall(r'\b[A-Z]{2,6}\b', chunk_content)),
            'citation_count': len(re.findall(r'\[\d+\]|\(\d{4}\)|et al\.', chunk_content)),
            'clinical_values': len(self.clinical_values_pattern.findall(chunk_content)),
            'has_patient_data': bool(re.search(r'\b(patient|case|subject)\b', chunk_content, re.IGNORECASE)),
            'has_statistical_data': bool(
                re.search(r'\b(p\s*[<>=]\s*0\.\d+|CI|confidence interval|significant)\b', chunk_content, re.IGNORECASE))
        })

        # Cache result
        if self.enable_caching and cache_key:
            self._medical_context_cache[cache_key] = context

        return context

    def _assess_clinical_relevance_optimized(self, chunk_content: str) -> float:
        """Enhanced clinical relevance assessment"""
        content_lower = chunk_content.lower()
        relevance_score = 0.0

        # Enhanced weight system for medical content
        weights = {
            'diagnosis': 0.20,
            'treatment': 0.20,
            'evidence': 0.15,
            'outcomes': 0.15,
            'procedures': 0.10,
            'guidelines': 0.10,
            'clinical_terms': 0.05,
            'anatomy': 0.05
        }

        for context_type, weight in weights.items():
            if context_type in self.medical_patterns and self.medical_patterns[context_type].search(content_lower):
                relevance_score += weight

        # Bonus for specific medical indicators
        bonus_patterns = [
            (r'\b(randomized|controlled|trial|study|meta-analysis)\b', 0.15),
            (r'\b(efficacy|effectiveness|safety|adverse)\b', 0.10),
            (r'\b(dose|dosage|mg|ml|units)\b', 0.08),
            (r'\b(p\s*[<>=]\s*0\.\d+|statistically significant)\b', 0.12),
            (r'\b(patient|clinical|medical|healthcare)\b', 0.05)
        ]

        for pattern, bonus in bonus_patterns:
            if re.search(pattern, content_lower):
                relevance_score += bonus

        return min(relevance_score, 1.0)

    def _extract_medical_entities(self, content: str) -> List[str]:
        """Extract medical entities from content"""
        entities = set()

        # Medical terms
        medical_terms = self.medical_term_pattern.findall(content)
        entities.update(term for term in medical_terms if len(term) > 3)

        # Drug names
        drugs = self.drug_pattern.findall(content)
        entities.update(drugs)

        # Medical measurements
        measurements = self.medical_patterns['measurements'].findall(content)
        entities.update(measurements)

        # Anatomical terms
        anatomy_matches = self.medical_patterns['anatomy'].findall(content)
        entities.update(anatomy_matches)

        return list(entities)[:30]  # Limit to top 30 entities

    def _extract_clinical_keywords(self, content: str) -> List[str]:
        """Extract clinical keywords for enhanced search"""
        keywords = set()

        # Clinical procedures and tests
        clinical_procedures = re.findall(
            r'\b(?:CT|MRI|X-ray|ultrasound|biopsy|endoscopy|ECG|EKG|blood test)\b',
            content, re.IGNORECASE
        )
        keywords.update(clinical_procedures)

        # Medical conditions
        conditions = re.findall(
            r'\b\w+(?:itis|osis|emia|uria|pathy|trophy|plasia|genesis|cardia|pulmonary)\b',
            content, re.IGNORECASE
        )
        keywords.update(conditions)

        # Medical specialties
        specialties = re.findall(
            r'\b(?:cardiology|neurology|oncology|pediatrics|surgery|radiology|pathology)\b',
            content, re.IGNORECASE
        )
        keywords.update(specialties)

        return list(keywords)[:20]  # Limit to top 20 keywords

    def _detect_medical_specialty(self, content: str) -> str:
        """Detect the likely medical specialty based on content"""
        content_lower = content.lower()

        specialty_indicators = {
            'cardiology': ['heart', 'cardiac', 'cardiovascular', 'ecg', 'ekg', 'coronary', 'myocardial'],
            'neurology': ['brain', 'neurological', 'nervous', 'seizure', 'stroke', 'neuronal'],
            'oncology': ['cancer', 'tumor', 'malignant', 'chemotherapy', 'radiation', 'oncologic'],
            'endocrinology': ['diabetes', 'insulin', 'hormone', 'thyroid', 'endocrine', 'glucose'],
            'pulmonology': ['lung', 'pulmonary', 'respiratory', 'asthma', 'copd', 'pneumonia'],
            'gastroenterology': ['stomach', 'intestinal', 'gastric', 'digestive', 'liver', 'hepatic'],
            'nephrology': ['kidney', 'renal', 'dialysis', 'creatinine', 'nephron'],
            'orthopedics': ['bone', 'joint', 'fracture', 'orthopedic', 'musculoskeletal'],
            'dermatology': ['skin', 'dermal', 'rash', 'dermatologic', 'cutaneous'],
            'ophthalmology': ['eye', 'visual', 'retina', 'ophthalmic', 'vision']
        }

        specialty_scores = {}
        for specialty, indicators in specialty_indicators.items():
            score = sum(1 for indicator in indicators if indicator in content_lower)
            if score > 0:
                specialty_scores[specialty] = score

        if specialty_scores:
            return max(specialty_scores, key=specialty_scores.get)

        return 'general_medicine'

    def _classify_content_type(self, content: str) -> str:
        """Classify the type of medical content"""
        content_lower = content.lower()

        # Research paper classification
        if any(term in content_lower for term in ['abstract', 'methodology', 'results', 'discussion']):
            return 'research_paper'

        # Clinical guidelines
        elif any(term in content_lower for term in ['guideline', 'recommendation', 'protocol']):
            return 'clinical_guideline'

        # Case reports
        elif any(term in content_lower for term in ['case report', 'patient presentation', 'case study']):
            return 'case_report'

        # Drug information
        elif any(term in content_lower for term in ['dosage', 'administration', 'contraindication', 'side effect']):
            return 'drug_information'

        # Diagnostic information
        elif any(term in content_lower for term in ['diagnosis', 'symptoms', 'signs', 'differential']):
            return 'diagnostic_information'

        # Treatment protocols
        elif any(term in content_lower for term in ['treatment', 'therapy', 'management', 'intervention']):
            return 'treatment_protocol'

        else:
            return 'general_medical'

    def _calculate_retrieval_priority(self, chunk: Dict[str, Any]) -> float:
        """Calculate retrieval priority for FAISS optimization"""
        priority = 1.0

        # Boost based on medical relevance
        medical_relevance = chunk.get('medical_relevance', 0)
        priority += medical_relevance * 0.5

        # Boost based on clinical relevance
        clinical_relevance = chunk.get('clinical_relevance', 0)
        priority += clinical_relevance * 0.3

        # Boost complete sections
        if chunk.get('is_complete_section'):
            priority *= 1.2

        # Boost clinical content types
        section_type = chunk.get('section_type', 'general')
        if section_type in ['clinical_history', 'assessment', 'treatment_plan']:
            priority *= 1.15
        elif section_type in ['results', 'evidence_based']:
            priority *= 1.1

        # Consider chunk size (prefer medium-sized chunks)
        size = chunk.get('size', 0)
        if 500 <= size <= 1500:
            priority *= 1.1
        elif size < 200:
            priority *= 0.8
        elif size > 2000:
            priority *= 0.9

        return round(priority, 3)

    def _calculate_chunk_metrics(self, chunks: List[Dict[str, Any]], processing_time: float) -> ChunkMetrics:
        """Calculate comprehensive chunk metrics"""
        if not chunks:
            return ChunkMetrics(0, processing_time, 0, 0, 0, 0, False)

        total_chunks = len(chunks)
        avg_chunk_size = sum(c.get('size', 0) for c in chunks) / total_chunks
        sections_detected = len(set(c.get('section', '') for c in chunks))
        clinical_relevance_avg = sum(c.get('clinical_relevance', 0) for c in chunks) / total_chunks
        medical_context_detected = sum(
            1 for c in chunks if c.get('medical_context', {}).get('contains_clinical_terms', False))
        optimization_applied = any(c.get('retrieval_priority', 1.0) != 1.0 for c in chunks)

        return ChunkMetrics(
            total_chunks=total_chunks,
            processing_time=processing_time,
            avg_chunk_size=avg_chunk_size,
            sections_detected=sections_detected,
            clinical_relevance_avg=clinical_relevance_avg,
            medical_context_detected=medical_context_detected,
            optimization_applied=optimization_applied
        )

    def optimize_chunks_for_retrieval(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhanced chunk optimization for FAISS retrieval"""
        optimized_chunks = []

        for chunk in chunks:
            content = chunk['content']

            # Add enhanced contextual headers
            context_headers = []

            # Section context
            if chunk.get('section') and chunk['section'] not in content:
                context_headers.append(f"Section: {chunk['section']}")

            # Document type context
            if chunk.get('document_type') and chunk['document_type'] != 'unknown':
                context_headers.append(f"Document Type: {chunk['document_type']}")

            # Medical specialty context
            if chunk.get('medical_specialty') and chunk['medical_specialty'] != 'general_medicine':
                context_headers.append(f"Medical Specialty: {chunk['medical_specialty']}")

            # Content classification
            if chunk.get('content_classification') and chunk['content_classification'] != 'general_medical':
                context_headers.append(f"Content Type: {chunk['content_classification']}")

            # Clinical relevance indicator
            clinical_relevance = chunk.get('clinical_relevance', 0)
            if clinical_relevance > 0.7:
                context_headers.append("High Clinical Relevance")
            elif clinical_relevance > 0.4:
                context_headers.append("Moderate Clinical Relevance")

            # Combine headers with content
            if context_headers:
                content = '\n'.join(context_headers) + '\n\n' + content

            optimized_chunk = {
                **chunk,
                'content': content,
                'search_keywords': self._extract_search_keywords_optimized(content),
                'chunk_summary': self._generate_chunk_summary_optimized(content),
                'retrieval_weight': chunk.get('retrieval_priority', 1.0),
                'faiss_metadata': {
                    'medical_entities': chunk.get('medical_entities', []),
                    'clinical_keywords': chunk.get('clinical_keywords', []),
                    'specialty': chunk.get('medical_specialty', 'general_medicine'),
                    'content_type': chunk.get('content_classification', 'general_medical'),
                    'relevance_score': clinical_relevance
                }
            }

            optimized_chunks.append(optimized_chunk)

        return optimized_chunks

    def _extract_search_keywords_optimized(self, content: str) -> List[str]:
        """Enhanced keyword extraction for better search performance"""
        keywords = set()

        # Medical terms
        medical_terms = self.medical_term_pattern.findall(content)
        keywords.update(term for term in medical_terms if len(term) > 3)

        # Drug names
        drugs = self.drug_pattern.findall(content)
        keywords.update(drugs)

        # Clinical measurements and values
        measurements = self.medical_patterns['measurements'].findall(content)
        keywords.update(measurements)

        # Medical acronyms (filtered for relevance)
        acronyms = re.findall(r'\b[A-Z]{2,6}\b', content)
        medical_acronyms = [
            acronym for acronym in acronyms
            if acronym in ['ECG', 'EKG', 'MRI', 'CT', 'ICU', 'ER', 'OR', 'IV', 'IM', 'PO', 'BID', 'TID', 'QID', 'PRN']
        ]
        keywords.update(medical_acronyms)

        # Clinical conditions and procedures
        conditions = re.findall(r'\b\w+(?:itis|osis|emia|uria|pathy|trophy|plasia|genesis)\b', content, re.IGNORECASE)
        keywords.update(conditions)

        # Important clinical terms
        clinical_terms = re.findall(
            r'\b(?:acute|chronic|severe|mild|moderate|positive|negative|normal|abnormal|elevated|decreased)\b',
            content, re.IGNORECASE
        )
        keywords.update(clinical_terms)

        return list(keywords)[:30]  # Limit to top 30 keywords

    def _generate_chunk_summary_optimized(self, content: str) -> str:
        """Enhanced chunk summary generation with medical focus"""
        # Remove context headers for summary
        lines = content.split('\n')
        content_lines = [
            line for line in lines
            if not line.startswith((
                                   'Section:', 'Document Type:', 'Content Type:', 'Medical Specialty:', 'High Clinical',
                                   'Moderate Clinical'))
        ]
        clean_content = '\n'.join(content_lines)

        sentences = re.split(r'[.!?]+', clean_content)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

        if not sentences:
            return clean_content[:200] + "..." if len(clean_content) > 200 else clean_content

        if len(sentences) <= 2:
            summary = '. '.join(sentences)
        else:
            # Take first sentence and most medically relevant sentence
            first_sentence = sentences[0]

            # Find sentence with highest medical term density
            best_sentence = sentences[1]
            best_score = 0

            for sentence in sentences[1:]:
                score = 0
                # Score based on medical patterns
                for pattern in self.medical_patterns.values():
                    score += len(pattern.findall(sentence))

                # Bonus for drug names
                score += len(self.drug_pattern.findall(sentence)) * 2

                # Bonus for clinical measurements
                score += len(re.findall(r'\b\d+(?:\.\d+)?\s*(?:mg|ml|mmHg|°[CF]|bpm)\b', sentence))

                if score > best_score:
                    best_score = score
                    best_sentence = sentence

            summary = f"{first_sentence}. {best_sentence}"

        return summary[:400] + "..." if len(summary) > 400 else summary

    def get_chunking_stats(self) -> Dict[str, Any]:
        """Get comprehensive chunking performance statistics"""
        return {
            'configuration': {
                'chunk_size': self.chunk_size,
                'overlap': self.overlap,
                'cache_enabled': self.enable_caching
            },
            'performance': {
                'section_cache_size': len(self._section_cache) if self._section_cache else 0,
                'medical_context_cache_size': len(self._medical_context_cache) if self._medical_context_cache else 0,
                'patterns_compiled': len(self.compiled_section_patterns)
            },
            'medical_features': {
                'section_patterns_count': len(self.section_patterns),
                'medical_patterns_count': len(self.medical_patterns),
                'specialties_supported': 10,
                'content_types_supported': 6
            },
            'optimization': {
                'smart_splitting_enabled': True,
                'medical_context_extraction': True,
                'clinical_relevance_assessment': True,
                'faiss_optimization': True
            }
        }

    def clear_cache(self):
        """Clear all caches to free memory"""
        if self._section_cache:
            self._section_cache.clear()
        if self._medical_context_cache:
            self._medical_context_cache.clear()

        # Clear LRU cache
        self._cached_section_detection.cache_clear()

        logger.info("Medical chunking strategy caches cleared")

    def validate_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate chunks for FAISS compatibility and medical completeness"""
        validation_results = {
            'total_chunks': len(chunks),
            'valid_chunks': 0,
            'issues': [],
            'warnings': [],
            'medical_coverage': {
                'chunks_with_medical_context': 0,
                'chunks_with_clinical_relevance': 0,
                'chunks_with_medical_entities': 0
            }
        }

        for i, chunk in enumerate(chunks):
            chunk_issues = []

            # Check required fields
            required_fields = ['content', 'chunk_id', 'section', 'size']
            for field in required_fields:
                if field not in chunk:
                    chunk_issues.append(f"Missing required field: {field}")

            # Check content quality
            content = chunk.get('content', '')
            if len(content.strip()) < 50:
                chunk_issues.append("Content too short (< 50 characters)")

            # Check medical context
            medical_context = chunk.get('medical_context', {})
            if medical_context:
                validation_results['medical_coverage']['chunks_with_medical_context'] += 1

            clinical_relevance = chunk.get('clinical_relevance', 0)
            if clinical_relevance > 0:
                validation_results['medical_coverage']['chunks_with_clinical_relevance'] += 1

            medical_entities = chunk.get('medical_entities', [])
            if medical_entities:
                validation_results['medical_coverage']['chunks_with_medical_entities'] += 1

            # If no issues, mark as valid
            if not chunk_issues:
                validation_results['valid_chunks'] += 1
            else:
                validation_results['issues'].append({
                    'chunk_index': i,
                    'chunk_id': chunk.get('chunk_id', f'chunk_{i}'),
                    'issues': chunk_issues
                })

        # Add warnings for low medical coverage
        total_chunks = validation_results['total_chunks']
        if total_chunks > 0:
            medical_coverage_pct = (validation_results['medical_coverage'][
                                        'chunks_with_medical_context'] / total_chunks) * 100
            if medical_coverage_pct < 50:
                validation_results['warnings'].append(f"Low medical context coverage: {medical_coverage_pct:.1f}%")

        return validation_results

    def export_chunking_config(self) -> Dict[str, Any]:
        """Export current chunking configuration for reproducibility"""
        return {
            'version': '2.0.0',
            'timestamp': time.time(),
            'configuration': {
                'chunk_size': self.chunk_size,
                'overlap': self.overlap,
                'enable_caching': self.enable_caching
            },
            'patterns': {
                'section_patterns': self.section_patterns,
                'medical_patterns': {k: v.pattern for k, v in self.medical_patterns.items()},
                'drug_pattern': self.drug_pattern.pattern,
                'medical_term_pattern': self.medical_term_pattern.pattern
            },
            'features': {
                'semantic_chunking': True,
                'medical_context_extraction': True,
                'clinical_relevance_assessment': True,
                'faiss_optimization': True,
                'parallel_processing': True
            }
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'MedicalChunkingStrategy':
        """Create MedicalChunkingStrategy from exported configuration"""
        instance = cls(
            chunk_size=config['configuration']['chunk_size'],
            overlap=config['configuration']['overlap'],
            enable_caching=config['configuration']['enable_caching']
        )

        logger.info(f"Loaded MedicalChunkingStrategy from config version {config.get('version', 'unknown')}")
        return instance