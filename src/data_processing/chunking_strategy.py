# import re
# from typing import List, Dict, Any, Tuple
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import logging
#
# logger = logging.getLogger(__name__)
#
#
# class MedicalChunkingStrategy:
#     """Specialized chunking strategy for medical documents"""
#
#     def __init__(self, chunk_size: int = 1000, overlap: int = 200):
#         self.chunk_size = chunk_size
#         self.overlap = overlap
#
#         # Medical section headers
#         self.section_patterns = [
#             r'^(ABSTRACT|INTRODUCTION|METHODS?|RESULTS?|DISCUSSION|CONCLUSION)S?\s*:?',
#             r'^(BACKGROUND|OBJECTIVE|DESIGN|SETTING|PARTICIPANTS|INTERVENTIONS?)S?\s*:?',
#             r'^(MAIN OUTCOMES?|MEASUREMENTS?|LIMITATIONS|IMPLICATIONS)S?\s*:?',
#             r'^(CASE REPORT|PATIENT PRESENTATION|CLINICAL FINDINGS)S?\s*:?',
#             r'^(DIAGNOSIS|TREATMENT|MANAGEMENT|FOLLOW[-\s]?UP)S?\s*:?',
#             r'^(ADVERSE EVENTS?|SIDE EFFECTS?|CONTRAINDICATIONS?)S?\s*:?',
#             r'^(DOSAGE|ADMINISTRATION|PHARMACOLOGY)S?\s*:?'
#         ]
#
#         # Initialize text splitter
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=chunk_size,
#             chunk_overlap=overlap,
#             length_function=len,
#             separators=["\n\n", "\n", ". ", " ", ""]
#         )
#
#     def detect_sections(self, text: str) -> List[Dict[str, Any]]:
#         """Detect medical document sections"""
#         sections = []
#         lines = text.split('\n')
#         current_section = None
#         current_content = []
#
#         for i, line in enumerate(lines):
#             line_stripped = line.strip()
#
#             # Check if line matches any section pattern
#             section_match = None
#             for pattern in self.section_patterns:
#                 if re.match(pattern, line_stripped, re.IGNORECASE):
#                     section_match = line_stripped
#                     break
#
#             if section_match:
#                 # Save previous section
#                 if current_section and current_content:
#                     sections.append({
#                         'title': current_section,
#                         'content': '\n'.join(current_content).strip(),
#                         'start_line': len(sections),
#                         'length': len('\n'.join(current_content))
#                     })
#
#                 # Start new section
#                 current_section = section_match
#                 current_content = []
#             else:
#                 if line_stripped:  # Only add non-empty lines
#                     current_content.append(line)
#
#         # Add final section
#         if current_section and current_content:
#             sections.append({
#                 'title': current_section,
#                 'content': '\n'.join(current_content).strip(),
#                 'start_line': len(sections),
#                 'length': len('\n'.join(current_content))
#             })
#
#         return sections
#
#     def semantic_chunking(self, text: str) -> List[Dict[str, Any]]:
#         """Create semantically meaningful chunks for medical content"""
#         chunks = []
#
#         # First, try to detect sections
#         sections = self.detect_sections(text)
#
#         if sections:
#             # Process each section separately
#             for section in sections:
#                 section_chunks = self._chunk_section(section)
#                 chunks.extend(section_chunks)
#         else:
#             # Fallback to paragraph-based chunking
#             chunks = self._chunk_by_paragraphs(text)
#
#         return chunks
#
#     def _chunk_section(self, section: Dict[str, Any]) -> List[Dict[str, Any]]:
#         """Chunk a specific section of medical document"""
#         section_title = section['title']
#         section_content = section['content']
#
#         chunks = []
#
#         if len(section_content) <= self.chunk_size:
#             # Section fits in one chunk
#             chunks.append({
#                 'content': section_content,
#                 'section': section_title,
#                 'chunk_type': 'complete_section',
#                 'size': len(section_content),
#                 'overlap_previous': 0
#             })
#         else:
#             # Split section into multiple chunks
#             text_chunks = self.text_splitter.split_text(section_content)
#
#             for i, chunk_text in enumerate(text_chunks):
#                 chunks.append({
#                     'content': f"{section_title}\n\n{chunk_text}",
#                     'section': section_title,
#                     'chunk_type': 'section_part',
#                     'part_number': i + 1,
#                     'total_parts': len(text_chunks),
#                     'size': len(chunk_text),
#                     'overlap_previous': self.overlap if i > 0 else 0
#                 })
#
#         return chunks
#
#     def _chunk_by_paragraphs(self, text: str) -> List[Dict[str, Any]]:
#         """Chunk text by paragraphs when no clear sections are found"""
#         paragraphs = text.split('\n\n')
#         chunks = []
#         current_chunk = ""
#         chunk_count = 0
#
#         for para in paragraphs:
#             para = para.strip()
#             if not para:
#                 continue
#
#             # Check if adding this paragraph would exceed chunk size
#             if len(current_chunk) + len(para) + 2 > self.chunk_size:
#                 if current_chunk:
#                     chunks.append({
#                         'content': current_chunk.strip(),
#                         'section': 'Unknown',
#                         'chunk_type': 'paragraph_based',
#                         'chunk_number': chunk_count + 1,
#                         'size': len(current_chunk.strip()),
#                         'overlap_previous': 0
#                     })
#                     chunk_count += 1
#                     current_chunk = para
#                 else:
#                     # Single paragraph is too large, split it
#                     large_chunks = self.text_splitter.split_text(para)
#                     for chunk_text in large_chunks:
#                         chunks.append({
#                             'content': chunk_text,
#                             'section': 'Unknown',
#                             'chunk_type': 'large_paragraph_split',
#                             'chunk_number': chunk_count + 1,
#                             'size': len(chunk_text),
#                             'overlap_previous': self.overlap
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
#                 'section': 'Unknown',
#                 'chunk_type': 'paragraph_based',
#                 'chunk_number': chunk_count + 1,
#                 'size': len(current_chunk.strip()),
#                 'overlap_previous': 0
#             })
#
#         return chunks
#
#     def create_contextual_chunks(self, text: str, document_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
#         """Create chunks with additional medical context"""
#         base_chunks = self.semantic_chunking(text)
#
#         enhanced_chunks = []
#         for i, chunk in enumerate(base_chunks):
#             enhanced_chunk = {
#                 **chunk,
#                 'document_type': document_metadata.get('type', 'unknown'),
#                 'document_source': document_metadata.get('filename', 'unknown'),
#                 'chunk_id': f"{document_metadata.get('filename', 'doc')}_{i}",
#                 'medical_context': self._extract_medical_context(chunk['content']),
#                 'clinical_relevance': self._assess_clinical_relevance(chunk['content'])
#             }
#             enhanced_chunks.append(enhanced_chunk)
#
#         return enhanced_chunks
#
#     def _extract_medical_context(self, chunk_content: str) -> Dict[str, Any]:
#         """Extract medical context from chunk"""
#         context = {
#             'contains_diagnosis': bool(
#                 re.search(r'\b(diagnos[ei]s|condition|disease)\b', chunk_content, re.IGNORECASE)),
#             'contains_treatment': bool(
#                 re.search(r'\b(treatment|therapy|medication|drug)\b', chunk_content, re.IGNORECASE)),
#             'contains_symptoms': bool(re.search(r'\b(symptom|sign|present|complaint)\b', chunk_content, re.IGNORECASE)),
#             'contains_procedures': bool(
#                 re.search(r'\b(procedure|surgery|operation|intervention)\b', chunk_content, re.IGNORECASE)),
#             'contains_measurements': bool(re.search(r'\b\d+\s*(mg|ml|mmHg|°[CF]|bpm)\b', chunk_content, re.IGNORECASE)),
#             'contains_guidelines': bool(
#                 re.search(r'\b(guideline|recommendation|should|must)\b', chunk_content, re.IGNORECASE))
#         }
#
#         return context
#
#     def _assess_clinical_relevance(self, chunk_content: str) -> float:
#         """Assess clinical relevance score (0-1)"""
#         relevance_indicators = [
#             r'\b(patient|clinical|medical|treatment|diagnosis)\b',
#             r'\b(evidence|study|trial|research)\b',
#             r'\b(efficacy|safety|adverse|benefit)\b',
#             r'\b(guideline|recommendation|protocol)\b',
#             r'\b(outcome|result|finding|conclusion)\b'
#         ]
#
#         content_lower = chunk_content.lower()
#         matches = sum(1 for pattern in relevance_indicators if re.search(pattern, content_lower))
#
#         return min(matches / len(relevance_indicators), 1.0)
#
#     def optimize_chunks_for_retrieval(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#         """Optimize chunks for better retrieval performance"""
#         optimized_chunks = []
#
#         for chunk in chunks:
#             # Add section context to chunk content if missing
#             content = chunk['content']
#             if chunk.get('section') and chunk['section'] not in content:
#                 content = f"Section: {chunk['section']}\n\n{content}"
#
#             # Add document type context
#             if chunk.get('document_type'):
#                 content = f"Document Type: {chunk['document_type']}\n{content}"
#
#             optimized_chunk = {
#                 **chunk,
#                 'content': content,
#                 'search_keywords': self._extract_search_keywords(content),
#                 'chunk_summary': self._generate_chunk_summary(content)
#             }
#
#             optimized_chunks.append(optimized_chunk)
#
#         return optimized_chunks
#
#     def _extract_search_keywords(self, content: str) -> List[str]:
#         """Extract relevant keywords for search optimization"""
#         # Medical terminology patterns
#         medical_terms = re.findall(r'\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\b', content)
#
#         # Drug names (often end with common suffixes)
#         drug_patterns = r'\b\w+(?:cillin|mycin|azole|pril|sartan|statin|ide|ine)\b'
#         drugs = re.findall(drug_patterns, content, re.IGNORECASE)
#
#         # Measurements and values
#         measurements = re.findall(r'\d+\s*(?:mg|ml|mmHg|°[CF]|bpm)', content)
#
#         keywords = list(set(medical_terms + drugs + measurements))
#         return keywords[:20]  # Limit to top 20 keywords
#
#     def _generate_chunk_summary(self, content: str) -> str:
#         """Generate a brief summary of the chunk"""
#         sentences = content.split('. ')
#         if len(sentences) <= 2:
#             return content[:200] + "..." if len(content) > 200 else content
#
#         # Take first and last sentence for summary
#         summary = sentences[0] + ". " + sentences[-1]
#         return summary[:200] + "..." if len(summary) > 200 else summary


import re
import time
from typing import List, Dict, Any, Tuple, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
from functools import lru_cache
import concurrent.futures
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetrics:
    """Data class for chunk performance metrics"""
    total_chunks: int
    processing_time: float
    avg_chunk_size: float
    sections_detected: int
    clinical_relevance_avg: float


class MedicalChunkingStrategy:
    """Optimized chunking strategy for medical documents with performance enhancements"""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200, enable_caching: bool = True):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.enable_caching = enable_caching

        # Performance: Compile regex patterns once
        self._compile_patterns()

        # Performance: Cache for processed sections
        self._section_cache = {} if enable_caching else None
        self._medical_context_cache = {} if enable_caching else None

        # Enhanced medical section headers with more patterns
        self.section_patterns = [
            # Standard research paper sections
            r'^(ABSTRACT|INTRODUCTION|METHODS?|RESULTS?|DISCUSSION|CONCLUSION)S?\s*:?',
            r'^(BACKGROUND|OBJECTIVE|DESIGN|SETTING|PARTICIPANTS|INTERVENTIONS?)S?\s*:?',
            r'^(MAIN OUTCOMES?|MEASUREMENTS?|LIMITATIONS|IMPLICATIONS)S?\s*:?',

            # Clinical sections
            r'^(CASE REPORT|PATIENT PRESENTATION|CLINICAL FINDINGS)S?\s*:?',
            r'^(DIAGNOSIS|TREATMENT|MANAGEMENT|FOLLOW[-\s]?UP)S?\s*:?',
            r'^(ADVERSE EVENTS?|SIDE EFFECTS?|CONTRAINDICATIONS?)S?\s*:?',
            r'^(DOSAGE|ADMINISTRATION|PHARMACOLOGY)S?\s*:?',

            # Additional medical sections
            r'^(CLINICAL SIGNIFICANCE|THERAPEUTIC IMPLICATIONS|SAFETY PROFILE)S?\s*:?',
            r'^(EPIDEMIOLOGY|PATHOPHYSIOLOGY|PROGNOSIS|PREVENTION)S?\s*:?',
            r'^(DIFFERENTIAL DIAGNOSIS|COMPLICATIONS|MONITORING)S?\s*:?',
            r'^(PATIENT EDUCATION|SPECIAL POPULATIONS|DRUG INTERACTIONS?)S?\s*:?',

            # Numbered sections (1., 2., etc.)
            r'^\d+\.\s+[A-Z][^.]+$',

            # Guidelines sections
            r'^(RECOMMENDATION|EVIDENCE|GRADE|STRENGTH)S?\s*:?'
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
            re.compile(pattern, re.IGNORECASE) for pattern in self.section_patterns
        ]

        # Medical context patterns
        self.medical_patterns = {
            'diagnosis': re.compile(r'\b(diagnos[ei]s|condition|disease|disorder|syndrome)\b', re.IGNORECASE),
            'treatment': re.compile(r'\b(treatment|therapy|medication|drug|intervention|management)\b', re.IGNORECASE),
            'symptoms': re.compile(r'\b(symptom|sign|present|complaint|manifestation)\b', re.IGNORECASE),
            'procedures': re.compile(r'\b(procedure|surgery|operation|intervention|technique)\b', re.IGNORECASE),
            'measurements': re.compile(r'\b\d+(?:\.\d+)?\s*(?:mg|ml|mmHg|°[CF]|bpm|kg|cm|mm|hrs?|min)\b',
                                       re.IGNORECASE),
            'guidelines': re.compile(r'\b(guideline|recommendation|should|must|protocol|standard)\b', re.IGNORECASE),
            'evidence': re.compile(r'\b(evidence|study|trial|research|data|analysis)\b', re.IGNORECASE),
            'outcomes': re.compile(r'\b(outcome|result|finding|conclusion|efficacy|effectiveness)\b', re.IGNORECASE)
        }

        # Drug name patterns (enhanced)
        self.drug_pattern = re.compile(
            r'\b\w+(?:cillin|mycin|azole|pril|sartan|statin|ide|ine|ol|al|an|er|in|um|ate|ase)\b',
            re.IGNORECASE
        )

        # Medical terminology pattern
        self.medical_term_pattern = re.compile(r'\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\b')

    @lru_cache(maxsize=100)
    def _cached_section_detection(self, text_hash: str, text: str) -> Tuple[List[Dict[str, Any]]]:
        """Cache section detection results for repeated processing"""
        return tuple(self.detect_sections(text))

    def detect_sections(self, text: str) -> List[Dict[str, Any]]:
        """Optimized section detection with caching"""
        if self.enable_caching:
            text_hash = str(hash(text))
            if text_hash in self._section_cache:
                return self._section_cache[text_hash]

        sections = []
        lines = text.split('\n')
        current_section = None
        current_content = []

        # Performance: Process lines in batches for large documents
        batch_size = 1000

        for i in range(0, len(lines), batch_size):
            batch_lines = lines[i:i + batch_size]

            for line_idx, line in enumerate(batch_lines):
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
                        sections.append({
                            'title': current_section,
                            'content': '\n'.join(current_content).strip(),
                            'start_line': len(sections),
                            'length': len('\n'.join(current_content)),
                            'section_type': self._classify_section_type(current_section)
                        })

                    # Start new section
                    current_section = section_match
                    current_content = []
                else:
                    if line_stripped:  # Only add non-empty lines
                        current_content.append(line)

        # Add final section
        if current_section and current_content:
            sections.append({
                'title': current_section,
                'content': '\n'.join(current_content).strip(),
                'start_line': len(sections),
                'length': len('\n'.join(current_content)),
                'section_type': self._classify_section_type(current_section)
            })

        # Cache result
        if self.enable_caching:
            self._section_cache[text_hash] = sections

        return sections

    def _classify_section_type(self, section_title: str) -> str:
        """Classify section type for better processing"""
        title_lower = section_title.lower()

        if any(word in title_lower for word in ['abstract', 'summary']):
            return 'summary'
        elif any(word in title_lower for word in ['introduction', 'background']):
            return 'background'
        elif any(word in title_lower for word in ['method', 'design', 'procedure']):
            return 'methodology'
        elif any(word in title_lower for word in ['result', 'finding', 'outcome']):
            return 'results'
        elif any(word in title_lower for word in ['discussion', 'conclusion']):
            return 'discussion'
        elif any(word in title_lower for word in ['treatment', 'management', 'therapy']):
            return 'clinical'
        elif any(word in title_lower for word in ['diagnosis', 'symptom', 'presentation']):
            return 'diagnostic'
        else:
            return 'general'

    def semantic_chunking(self, text: str) -> List[Dict[str, Any]]:
        """Optimized semantic chunking with parallel processing for large documents"""
        start_time = time.time()
        chunks = []

        # First, try to detect sections
        sections = self.detect_sections(text)

        if sections and len(sections) > 1:
            # Process sections in parallel for large documents
            if len(sections) > 5:
                chunks = self._parallel_section_processing(sections)
            else:
                # Process each section separately
                for section in sections:
                    section_chunks = self._chunk_section(section)
                    chunks.extend(section_chunks)
        else:
            # Fallback to optimized paragraph-based chunking
            chunks = self._chunk_by_paragraphs(text)

        # Log performance metrics
        processing_time = time.time() - start_time
        logger.debug(f"Semantic chunking completed in {processing_time:.2f}s, created {len(chunks)} chunks")

        return chunks

    def _parallel_section_processing(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process sections in parallel for better performance"""
        chunks = []

        # Use ThreadPoolExecutor for I/O bound operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_section = {
                executor.submit(self._chunk_section, section): section
                for section in sections
            }

            for future in concurrent.futures.as_completed(future_to_section):
                try:
                    section_chunks = future.result()
                    chunks.extend(section_chunks)
                except Exception as e:
                    logger.error(f"Error processing section: {e}")
                    # Continue with other sections
                    continue

        # Sort chunks by original order
        chunks.sort(key=lambda x: x.get('original_order', 0))

        return chunks

    def _chunk_section(self, section: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Optimized section chunking with enhanced metadata"""
        section_title = section['title']
        section_content = section['content']
        section_type = section.get('section_type', 'general')

        chunks = []

        if len(section_content) <= self.chunk_size:
            # Section fits in one chunk
            chunks.append({
                'content': section_content,
                'section': section_title,
                'section_type': section_type,
                'chunk_type': 'complete_section',
                'size': len(section_content),
                'overlap_previous': 0,
                'is_complete_section': True,
                'original_order': section.get('start_line', 0)
            })
        else:
            # Split section into multiple chunks with smart overlap
            text_chunks = self._smart_section_split(section_content)

            for i, chunk_text in enumerate(text_chunks):
                # Add section context to each chunk
                contextual_content = f"{section_title}\n\n{chunk_text}"

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
                    'original_order': section.get('start_line', 0) + i * 0.1
                })

        return chunks

    def _smart_section_split(self, content: str) -> List[str]:
        """Smart splitting that preserves sentence boundaries and context"""
        # Try to split at sentence boundaries first
        sentences = re.split(r'(?<=[.!?])\s+', content)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) + 1 > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    # Add overlap from the end of current chunk
                    overlap_text = current_chunk[-self.overlap:] if len(current_chunk) > self.overlap else current_chunk
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

    def _chunk_by_paragraphs(self, text: str) -> List[Dict[str, Any]]:
        """Optimized paragraph-based chunking"""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
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
                        'section': 'Unstructured Content',
                        'section_type': 'general',
                        'chunk_type': 'paragraph_based',
                        'chunk_number': chunk_count + 1,
                        'size': len(current_chunk.strip()),
                        'overlap_previous': 0,
                        'is_complete_section': False,
                        'original_order': chunk_count
                    })
                    chunk_count += 1
                    current_chunk = para
                else:
                    # Single paragraph is too large, split it intelligently
                    large_chunks = self._smart_section_split(para)
                    for chunk_text in large_chunks:
                        chunks.append({
                            'content': chunk_text,
                            'section': 'Large Paragraph',
                            'section_type': 'general',
                            'chunk_type': 'large_paragraph_split',
                            'chunk_number': chunk_count + 1,
                            'size': len(chunk_text),
                            'overlap_previous': self.overlap,
                            'is_complete_section': False,
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
                'section': 'Unstructured Content',
                'section_type': 'general',
                'chunk_type': 'paragraph_based',
                'chunk_number': chunk_count + 1,
                'size': len(current_chunk.strip()),
                'overlap_previous': 0,
                'is_complete_section': False,
                'original_order': chunk_count
            })

        return chunks

    def create_contextual_chunks(self, text: str, document_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create chunks with enhanced medical context and metadata"""
        start_time = time.time()

        base_chunks = self.semantic_chunking(text)
        enhanced_chunks = []

        # Process chunks in batches for better performance
        batch_size = 10
        for i in range(0, len(base_chunks), batch_size):
            batch = base_chunks[i:i + batch_size]

            for j, chunk in enumerate(batch):
                chunk_index = i + j
                enhanced_chunk = {
                    **chunk,
                    'document_type': document_metadata.get('type', 'unknown'),
                    'document_source': document_metadata.get('filename', 'unknown'),
                    'chunk_id': f"{document_metadata.get('filename', 'doc')}_{chunk_index}",
                    'medical_context': self._extract_medical_context_optimized(chunk['content']),
                    'clinical_relevance': self._assess_clinical_relevance_optimized(chunk['content']),
                    'chunk_index': chunk_index,
                    'processing_timestamp': time.time(),
                    'word_count': len(chunk['content'].split()),
                    'sentence_count': len(re.split(r'[.!?]+', chunk['content']))
                }
                enhanced_chunks.append(enhanced_chunk)

        processing_time = time.time() - start_time

        # Log metrics
        metrics = ChunkMetrics(
            total_chunks=len(enhanced_chunks),
            processing_time=processing_time,
            avg_chunk_size=sum(c['size'] for c in enhanced_chunks) / len(enhanced_chunks) if enhanced_chunks else 0,
            sections_detected=len(set(c.get('section', '') for c in enhanced_chunks)),
            clinical_relevance_avg=sum(c['clinical_relevance'] for c in enhanced_chunks) / len(
                enhanced_chunks) if enhanced_chunks else 0
        )

        logger.info(f"Chunk processing metrics: {metrics}")

        return enhanced_chunks

    def _extract_medical_context_optimized(self, chunk_content: str) -> Dict[str, Any]:
        """Optimized medical context extraction using compiled patterns"""
        cache_key = hash(chunk_content) if self.enable_caching else None

        if self.enable_caching and cache_key in self._medical_context_cache:
            return self._medical_context_cache[cache_key]

        context = {}

        # Use compiled patterns for better performance
        for context_type, pattern in self.medical_patterns.items():
            context[f'contains_{context_type}'] = bool(pattern.search(chunk_content))

        # Additional context
        context.update({
            'medication_mentions': len(self.drug_pattern.findall(chunk_content)),
            'numerical_data_count': len(re.findall(r'\b\d+(?:\.\d+)?\b', chunk_content)),
            'medical_acronyms': len(re.findall(r'\b[A-Z]{2,}\b', chunk_content)),
            'citation_count': len(re.findall(r'\[\d+\]|\(\d{4}\)', chunk_content))
        })

        # Cache result
        if self.enable_caching and cache_key:
            self._medical_context_cache[cache_key] = context

        return context

    def _assess_clinical_relevance_optimized(self, chunk_content: str) -> float:
        """Optimized clinical relevance assessment"""
        content_lower = chunk_content.lower()
        relevance_score = 0.0

        # Weight different types of medical content
        weights = {
            'diagnosis': 0.25,
            'treatment': 0.25,
            'evidence': 0.20,
            'outcomes': 0.15,
            'procedures': 0.10,
            'guidelines': 0.05
        }

        for context_type, weight in weights.items():
            if self.medical_patterns[context_type].search(content_lower):
                relevance_score += weight

        # Bonus for specific medical indicators
        if re.search(r'\b(patient|clinical|medical)\b', content_lower):
            relevance_score += 0.1

        if re.search(r'\b(randomized|controlled|trial|study)\b', content_lower):
            relevance_score += 0.1

        return min(relevance_score, 1.0)

    def optimize_chunks_for_retrieval(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhanced chunk optimization for retrieval with performance improvements"""
        optimized_chunks = []

        for chunk in chunks:
            content = chunk['content']

            # Add contextual headers
            context_headers = []

            if chunk.get('section') and chunk['section'] not in content:
                context_headers.append(f"Section: {chunk['section']}")

            if chunk.get('document_type'):
                context_headers.append(f"Document Type: {chunk['document_type']}")

            if chunk.get('section_type') and chunk['section_type'] != 'general':
                context_headers.append(f"Content Type: {chunk['section_type']}")

            # Combine headers with content
            if context_headers:
                content = '\n'.join(context_headers) + '\n\n' + content

            optimized_chunk = {
                **chunk,
                'content': content,
                'search_keywords': self._extract_search_keywords_optimized(content),
                'chunk_summary': self._generate_chunk_summary_optimized(content),
                'retrieval_weight': self._calculate_retrieval_weight(chunk)
            }

            optimized_chunks.append(optimized_chunk)

        return optimized_chunks

    def _extract_search_keywords_optimized(self, content: str) -> List[str]:
        """Optimized keyword extraction"""
        keywords = set()

        # Medical terms
        medical_terms = self.medical_term_pattern.findall(content)
        keywords.update(term for term in medical_terms if len(term) > 3)

        # Drug names
        drugs = self.drug_pattern.findall(content)
        keywords.update(drugs)

        # Measurements and values
        measurements = self.medical_patterns['measurements'].findall(content)
        keywords.update(measurements)

        # Medical acronyms
        acronyms = re.findall(r'\b[A-Z]{2,6}\b', content)
        keywords.update(acronym for acronym in acronyms if len(acronym) <= 6)

        return list(keywords)[:25]  # Limit to top 25 keywords

    def _generate_chunk_summary_optimized(self, content: str) -> str:
        """Optimized chunk summary generation"""
        # Remove context headers for summary
        lines = content.split('\n')
        content_lines = [line for line in lines if not line.startswith(('Section:', 'Document Type:', 'Content Type:'))]
        clean_content = '\n'.join(content_lines)

        sentences = re.split(r'[.!?]+', clean_content)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return clean_content[:200] + "..." if len(clean_content) > 200 else clean_content

        if len(sentences) <= 2:
            summary = '. '.join(sentences)
        else:
            # Take first and most relevant sentence
            first_sentence = sentences[0]

            # Find sentence with highest medical term density
            best_sentence = sentences[1]
            best_score = 0

            for sentence in sentences[1:]:
                score = sum(1 for pattern in self.medical_patterns.values() if pattern.search(sentence))
                if score > best_score:
                    best_score = score
                    best_sentence = sentence

            summary = f"{first_sentence}. {best_sentence}"

        return summary[:300] + "..." if len(summary) > 300 else summary

    def _calculate_retrieval_weight(self, chunk: Dict[str, Any]) -> float:
        """Calculate retrieval weight based on chunk characteristics"""
        weight = 1.0

        # Boost complete sections
        if chunk.get('is_complete_section'):
            weight *= 1.2

        # Boost clinical content
        if chunk.get('section_type') in ['clinical', 'diagnostic', 'results']:
            weight *= 1.15

        # Boost based on clinical relevance
        relevance = chunk.get('clinical_relevance', 0)
        weight *= (1 + relevance * 0.3)

        # Consider chunk size (prefer medium-sized chunks)
        size = chunk.get('size', 0)
        if 500 <= size <= 1500:
            weight *= 1.1
        elif size < 200:
            weight *= 0.8

        return round(weight, 3)

    def get_chunking_stats(self) -> Dict[str, Any]:
        """Get chunking performance statistics"""
        return {
            'cache_enabled': self.enable_caching,
            'section_cache_size': len(self._section_cache) if self._section_cache else 0,
            'medical_context_cache_size': len(self._medical_context_cache) if self._medical_context_cache else 0,
            'chunk_size': self.chunk_size,
            'overlap': self.overlap,
            'patterns_compiled': len(self.compiled_section_patterns)
        }

    def clear_cache(self):
        """Clear all caches to free memory"""
        if self._section_cache:
            self._section_cache.clear()
        if self._medical_context_cache:
            self._medical_context_cache.clear()

        # Clear LRU cache
        self._cached_section_detection.cache_clear()

        logger.info("Chunking strategy caches cleared")