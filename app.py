

import streamlit as st
import os
import time
import hashlib
import json
from datetime import datetime
import re
from typing import Dict, List, Any

# Import custom modules
from config.settings import Settings
from src.data_processing.document_loader import DocumentLoader
from src.data_processing.text_preprocessor import TextPreprocessor
from src.data_processing.chunking_strategy import MedicalChunkingStrategy
from src.embeddings.simple_embedding_manager import EmbeddingManager
from src.retrieval.retriever import MedicalRetriever
from src.generation.groq_response_generator import GroqResponseGenerator
from src.medical_nlp.drug_interaction_checker import DrugInteractionChecker
from src.medical_nlp.terminology_mapper import MedicalTerminologyMapper
from src.utils.groq_utils import GroqUtils

# Validate configuration after Streamlit is initialized
try:
    # For Streamlit Cloud, try to access secrets directly as a fallback
    if not os.getenv("GROQ_API_KEY"):
        try:
            if "GROQ_API_KEY" in st.secrets:
                os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
                st.success("✅ API key loaded from Streamlit secrets")
        except Exception as e:
            st.warning(f"Could not access Streamlit secrets: {e}")
    
    Settings.validate_config()
    st.success("✅ Configuration validated successfully")
    
except Exception as e:
    st.error("## 🔧 Configuration Issue")
    st.error(str(e))
    
    # Show debugging information
    with st.expander("🔍 Debug Information"):
        st.write("**Environment Variables:**")
        env_vars = {k: "***" if "KEY" in k or "TOKEN" in k else v 
                   for k, v in os.environ.items() if k.startswith(("GROQ", "STREAMLIT"))}
        st.json(env_vars)
        
        st.write("**Streamlit Secrets:**")
        try:
            available_secrets = list(st.secrets.keys())
            st.write(f"Available: {available_secrets}")
        except Exception as secrets_error:
            st.write(f"Error accessing secrets: {secrets_error}")
    
    st.info("💡 **To fix this:** Add your GROQ_API_KEY to Streamlit Cloud secrets")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Medical RAG System",
    page_icon="🏥",
    layout="wide"
)

# Minimal CSS for clean look
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }

    .status-ok { color: #22c55e; }
    .status-error { color: #ef4444; }

    .stButton > button {
        width: 100%;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)


# Simple cache for queries
class QueryCache:
    def __init__(self):
        if 'query_cache' not in st.session_state:
            st.session_state.query_cache = {}

    def get_key(self, query: str) -> str:
        return hashlib.md5(query.lower().encode()).hexdigest()[:10]

    def get(self, query: str):
        key = self.get_key(query)
        return st.session_state.query_cache.get(key)

    def set(self, query: str, response):
        key = self.get_key(query)
        # Keep only 10 cached responses
        if len(st.session_state.query_cache) >= 10:
            oldest = list(st.session_state.query_cache.keys())[0]
            del st.session_state.query_cache[oldest]
        st.session_state.query_cache[key] = response


query_cache = QueryCache()


def init_session_state():
    """Initialize session state"""
    defaults = {
        'api_key_valid': False,
        'docs_loaded': False,
        'chat_history': [],
        'components_ready': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Auto-validate API key from environment on first load
    if not st.session_state.api_key_valid and not st.session_state.get('api_key_checked', False):
        auto_validate_api_key()
        st.session_state.api_key_checked = True

def auto_validate_api_key():
    """Automatically validate API key from environment"""
    api_key = Settings.GROQ_API_KEY

    if api_key:
        try:
            validation = GroqUtils.validate_api_key(api_key)

            if validation['valid']:
                st.session_state.api_key_valid = True
                st.session_state.groq_api_key = api_key
                st.session_state.response_generator = GroqResponseGenerator(
                    api_key=api_key,
                    model="llama3-8b-8192"
                )
                st.session_state.selected_model = "llama3-8b-8192"
                # Show success message only once
                if not st.session_state.get('api_success_shown', False):
                    st.success("✅ API key automatically loaded from environment")
                    st.session_state.api_success_shown = True
            else:
                st.error(f"❌ Invalid API key in environment: {validation['message']}")
        except Exception as e:
            st.error(f"❌ Error validating environment API key: {e}")
    else:
        st.warning("⚠️ No API key found in environment. Please set GROQ_API_KEY in your .env file.")


@st.cache_resource
def init_components():
    """Initialize system components with caching"""
    try:
        # Create a compatibility wrapper for the optimized chunking strategy
        class CompatibleMedicalChunker:
            """Wrapper to make optimized chunking strategy compatible"""

            def __init__(self, chunk_size=1000, chunk_overlap=200):
                try:
                    # Try optimized version first (uses 'overlap')
                    self.chunker = MedicalChunkingStrategy(
                        chunk_size=chunk_size,
                        overlap=chunk_overlap  # optimized version parameter
                    )
                    self.strategy_type = "optimized"
                except Exception:
                    try:
                        # Try original version (uses 'chunk_overlap')
                        self.chunker = MedicalChunkingStrategy(
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap
                        )
                        self.strategy_type = "original"
                    except Exception:
                        # Create minimal chunker
                        self.chunker = None
                        self.strategy_type = "manual"

                # Set compatibility attributes
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap

                # Add section_patterns attribute for compatibility
                if hasattr(self.chunker, 'section_patterns'):
                    self.section_patterns = self.chunker.section_patterns
                else:
                    # Provide basic section patterns
                    self.section_patterns = [
                        r'^(ABSTRACT|INTRODUCTION|METHODS?|RESULTS?|DISCUSSION|CONCLUSION)S?\s*:?',
                        r'^(BACKGROUND|OBJECTIVE|TREATMENT|DIAGNOSIS)S?\s*:?'
                    ]

            def create_contextual_chunks(self, text, metadata=None):
                """Main chunking method with multiple fallbacks"""
                if self.chunker is None:
                    return self._manual_chunk(text, metadata)

                try:
                    # Try optimized method
                    if hasattr(self.chunker, 'create_contextual_chunks'):
                        return self.chunker.create_contextual_chunks(text, metadata or {})
                    elif hasattr(self.chunker, 'semantic_chunking'):
                        # Convert optimized output to expected format
                        chunks = self.chunker.semantic_chunking(text)
                        return self._convert_chunks(chunks, metadata)
                    else:
                        return self._manual_chunk(text, metadata)
                except Exception as e:
                    st.warning(f"Advanced chunking failed: {e}")
                    return self._manual_chunk(text, metadata)

            def create_chunks(self, text, metadata=None):
                """Alias for create_contextual_chunks"""
                return self.create_contextual_chunks(text, metadata)

            def _convert_chunks(self, chunks, metadata):
                """Convert optimized chunk format to expected format"""
                converted = []
                for i, chunk in enumerate(chunks):
                    converted_chunk = {
                        'content': chunk.get('content', ''),
                        'chunk_id': f"chunk_{i}",
                        'source': metadata.get('filename', 'unknown') if metadata else 'unknown',
                        'document_type': metadata.get('type', 'general') if metadata else 'general',
                        'metadata': metadata or {},
                        'section': chunk.get('section', 'unknown'),
                        'chunk_type': chunk.get('chunk_type', 'unknown'),
                        'size': chunk.get('size', len(chunk.get('content', '')))
                    }
                    converted.append(converted_chunk)
                return converted

            def _manual_chunk(self, text, metadata):
                """Manual fallback chunking"""
                words = text.split()
                chunks = []
                chunk_size = 800  # Slightly smaller for safety

                for i in range(0, len(words), chunk_size):
                    chunk_words = words[i:i + chunk_size]
                    chunk_text = ' '.join(chunk_words)

                    chunks.append({
                        'content': chunk_text,
                        'chunk_id': f"manual_chunk_{i // chunk_size}",
                        'source': metadata.get('filename', 'unknown') if metadata else 'unknown',
                        'document_type': metadata.get('type', 'general') if metadata else 'general',
                        'metadata': metadata or {},
                        'section': 'unknown',
                        'chunk_type': 'manual',
                        'size': len(chunk_text)
                    })

                return chunks

        # Use the compatibility wrapper
        chunking_strategy = CompatibleMedicalChunker(
            Settings.CHUNK_SIZE,
            Settings.CHUNK_OVERLAP
        )

        return {
            'document_loader': DocumentLoader(),
            'preprocessor': TextPreprocessor(),
            'chunking_strategy': chunking_strategy,
            'embedding_manager': EmbeddingManager(model_name=Settings.EMBEDDING_MODEL),
            'retriever': MedicalRetriever(
                persist_directory=Settings.VECTOR_STORE_DIRECTORY,
                collection_name=Settings.COLLECTION_NAME
            ),
            'drug_checker': DrugInteractionChecker(),
            'terminology_mapper': MedicalTerminologyMapper()
        }

    except Exception as e:
        st.error(f"All initialization methods failed: {e}")
        return None


def setup_components():
    """Setup components in session state"""
    if not st.session_state.components_ready:
        with st.spinner("🚀 Loading system components..."):
            components = init_components()
            if components:
                for name, component in components.items():
                    st.session_state[name] = component
                st.session_state.components_ready = True
                st.success("✅ Components loaded successfully")
            else:
                st.error("❌ Failed to initialize components")
                st.info("Please check that all modules are properly installed and imported")


def render_header():
    """Simple header"""
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Medical Literature RAG</h1>
        <p>AI-Powered Clinical Decision Support</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Compact sidebar"""
    with st.sidebar:
        st.markdown("## ⚙️ Setup")

        # API Key status and manual override option
        if st.session_state.api_key_valid:
            st.success("✅ API Key Active")

            # Show current API key (masked)
            current_key = st.session_state.get('groq_api_key', '')
            if current_key:
                masked_key = current_key[:8] + "..." + current_key[-4:] if len(current_key) > 12 else "***"
                st.caption(f"Key: {masked_key}")

            # Option to change API key
            with st.expander("🔄 Change API Key", expanded=False):
                with st.form("change_api_form"):
                    new_api_key = st.text_input("New Groq API Key", type="password")
                    if st.form_submit_button("Update"):
                        if new_api_key:
                            validate_api_key(new_api_key)
                        else:
                            st.error("Please enter a new API key")
        else:
            # Show manual input only if environment key failed or doesn't exist
            st.warning("🔑 API Key Required")
            with st.form("api_form"):
                api_key = st.text_input("Groq API Key", type="password",
                                        help="Enter manually if environment key failed")
                if st.form_submit_button("Validate"):
                    validate_api_key(api_key)

        # Model selection (only show if API is valid)
        if st.session_state.api_key_valid:
            st.markdown("### 🤖 Model")
            models = {
                "llama3-8b-8192": "Llama 3 8B (Fast)",
                "llama3-70b-8192": "Llama 3 70B (Accurate)"
            }

            current_model = st.session_state.get('selected_model', "llama3-8b-8192")
            selected = st.selectbox("Model", list(models.keys()),
                                    format_func=lambda x: models[x],
                                    index=list(models.keys()).index(current_model))

            if selected != current_model:
                st.session_state.selected_model = selected
                if hasattr(st.session_state, 'response_generator'):
                    st.session_state.response_generator.model = selected
                    st.success(f"✅ Switched to {models[selected]}")

        # Status
        st.markdown("### 📊 Status")
        api_status = "✅" if st.session_state.api_key_valid else "❌"
        docs_status = "✅" if st.session_state.docs_loaded else "❌"
        st.markdown(f"API: {api_status} | Docs: {docs_status}")

        if st.session_state.docs_loaded:
            try:
                stats = st.session_state.retriever.get_stats()
                st.caption(f"📄 {stats.get('total_documents', 0)} documents")
            except:
                pass

        # Quick actions - UPDATED SECTION
        st.markdown("### ⚡ Actions")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧪 Test", key="test_system"):
                test_system()
        with col2:
            if st.button("🔄 Reset", key="reset_system"):
                reset_system()

        # Add Clear Database button - NEW ADDITION
        if st.button("🗑️ Clear Database", key="clear_db", type="secondary"):
            try:
                if hasattr(st.session_state, 'retriever'):
                    st.session_state.retriever.clear_collection()
                    st.success("Database cleared successfully!")
                    # Reset docs loaded status
                    st.session_state.docs_loaded = False
                    st.rerun()  # Updated method name for newer Streamlit versions
                else:
                    st.error("Retriever not initialized")
            except Exception as e:
                st.error(f"Error clearing database: {e}")


def validate_api_key(api_key: str):
    """Validate API key"""
    if not api_key:
        st.error("Please enter API key")
        return

    with st.spinner("Validating..."):
        validation = GroqUtils.validate_api_key(api_key)

        if validation['valid']:
            st.session_state.api_key_valid = True
            st.session_state.groq_api_key = api_key
            st.session_state.response_generator = GroqResponseGenerator(
                api_key=api_key,
                model="llama3-8b-8192"
            )
            st.success("✅ API key validated")
            st.rerun()
        else:
            st.error(f"❌ {validation['message']}")


def test_system():
    """Quick system test"""
    if st.session_state.api_key_valid and st.session_state.docs_loaded:
        st.success("✅ System ready!")
    else:
        st.error("❌ Setup incomplete")


def reset_system():
    """Reset system"""
    for key in list(st.session_state.keys()):
        if key not in ['query_cache']:
            del st.session_state[key]
    st.rerun()


def document_section():
    """Document upload and processing"""
    st.markdown("## 📄 Documents")

    uploaded_files = st.file_uploader(
        "Upload medical documents",
        type=['pdf', 'txt', 'docx'],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} files selected")

        if st.button("🚀 Process Documents", type="primary"):
            process_documents(uploaded_files)


def process_documents(files):
    """Process uploaded documents with proper file handling"""
    if not st.session_state.components_ready:
        st.error("Components not ready")
        return

    progress = st.progress(0)
    status = st.empty()

    try:
        all_chunks = []

        # Process files
        for i, file in enumerate(files):
            status.text(f"Processing: {file.name}")
            progress.progress((i + 1) / len(files) * 0.6)

            try:
                # Read file content directly from uploaded file object
                file_content = None

                if file.type == "application/pdf":
                    # For PDF files, try to read content directly
                    try:
                        # Reset file pointer
                        file.seek(0)

                        # Create a simple document data structure
                        doc_data = {
                            'content': f"PDF content from {file.name}. Size: {len(file.getvalue())} bytes.",
                            'source': file.name,
                            'document_type': 'pdf',
                            'metadata': {'filename': file.name, 'size': len(file.getvalue())}
                        }

                        # Try to extract text if PDF reader is available
                        try:
                            import PyPDF2
                            import io

                            file.seek(0)
                            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.getvalue()))
                            text_content = ""
                            for page in pdf_reader.pages:
                                text_content += page.extract_text() + "\n"

                            if text_content.strip():
                                doc_data['content'] = text_content

                        except ImportError:
                            st.warning(f"PyPDF2 not available. Using filename for {file.name}")
                        except Exception as pdf_error:
                            st.warning(f"PDF reading failed for {file.name}: {pdf_error}")

                    except Exception as e:
                        st.warning(f"Error reading PDF {file.name}: {e}")
                        continue

                elif file.type == "text/plain":
                    # For text files
                    try:
                        file.seek(0)
                        content = file.getvalue().decode('utf-8')
                        doc_data = {
                            'content': content,
                            'source': file.name,
                            'document_type': 'text',
                            'metadata': {'filename': file.name, 'size': len(content)}
                        }
                    except Exception as e:
                        st.warning(f"Error reading text file {file.name}: {e}")
                        continue

                else:
                    # For other file types, create basic structure
                    file.seek(0)
                    doc_data = {
                        'content': f"Document: {file.name}. Content type: {file.type}. Size: {len(file.getvalue())} bytes.",
                        'source': file.name,
                        'document_type': 'unknown',
                        'metadata': {'filename': file.name, 'type': file.type}
                    }

                # Preprocess content
                try:
                    processed = st.session_state.preprocessor.preprocess_document(
                        doc_data['content'], anonymize=True
                    )
                except Exception as preprocess_error:
                    st.warning(f"Preprocessing failed for {file.name}: {preprocess_error}")
                    # Use original content if preprocessing fails
                    processed = {'processed_content': doc_data['content']}

                # Create chunks - try multiple method names to ensure compatibility
                try:
                    # First try the optimized method
                    if hasattr(st.session_state.chunking_strategy, 'create_contextual_chunks'):
                        chunks = st.session_state.chunking_strategy.create_contextual_chunks(
                            processed['processed_content'], doc_data
                        )
                    elif hasattr(st.session_state.chunking_strategy, 'create_chunks'):
                        chunks = st.session_state.chunking_strategy.create_chunks(
                            processed['processed_content'], doc_data
                        )
                    elif hasattr(st.session_state.chunking_strategy, 'semantic_chunking'):
                        # For optimized chunking strategy
                        base_chunks = st.session_state.chunking_strategy.semantic_chunking(
                            processed['processed_content']
                        )
                        # Convert to expected format
                        chunks = []
                        for i, chunk in enumerate(base_chunks):
                            chunk_data = {
                                'content': chunk.get('content', ''),
                                'chunk_id': f"{file.name}_chunk_{i}",
                                'source': file.name,
                                'document_type': doc_data.get('document_type', 'general'),
                                'metadata': doc_data.get('metadata', {}),
                                'section': chunk.get('section', 'unknown'),
                                'chunk_type': chunk.get('chunk_type', 'unknown')
                            }
                            chunks.append(chunk_data)
                    else:
                        # Manual chunking as absolute fallback
                        words = processed['processed_content'].split()
                        chunk_size = 1000
                        chunks = []
                        for j in range(0, len(words), chunk_size):
                            chunk_text = ' '.join(words[j:j + chunk_size])
                            chunks.append({
                                'content': chunk_text,
                                'chunk_id': f"{file.name}_chunk_{j // chunk_size}",
                                'source': file.name,
                                'document_type': doc_data.get('document_type', 'general'),
                                'metadata': doc_data.get('metadata', {})
                            })

                    all_chunks.extend(chunks)

                except Exception as chunk_error:
                    st.warning(f"Chunking failed for {file.name}: {chunk_error}")
                    # Create single chunk as fallback
                    chunks = [{
                        'content': processed['processed_content'][:2000],  # Limit size
                        'chunk_id': f"{file.name}_single_chunk",
                        'source': file.name,
                        'document_type': doc_data.get('document_type', 'general'),
                        'metadata': doc_data.get('metadata', {})
                    }]
                    all_chunks.extend(chunks)

            except Exception as file_error:
                st.warning(f"Skipping {file.name} due to error: {file_error}")
                continue

        if not all_chunks:
            st.error("❌ No content could be extracted from uploaded files")
            return

        # Generate embeddings
        status.text("Generating embeddings...")
        progress.progress(0.8)

        try:
            embedded_chunks = st.session_state.embedding_manager.embed_medical_chunks(all_chunks)
        except Exception as embed_error:
            st.error(f"❌ Embedding generation failed: {embed_error}")
            return

        # Store in database
        status.text("Storing in database...")
        progress.progress(0.9)

        try:
            success = st.session_state.retriever.add_documents(embedded_chunks)

            if success:
                progress.progress(1.0)
                st.session_state.docs_loaded = True
                st.success(f"✅ Processed {len(files)} files into {len(embedded_chunks)} chunks")
            else:
                st.error("❌ Failed to store documents in database")
        except Exception as store_error:
            st.error(f"❌ Document storage failed: {store_error}")

    except Exception as e:
        st.error(f"❌ Processing error: {e}")
    finally:
        progress.empty()
        status.empty()


def chat_section():
    """Main chat interface"""
    st.markdown("## 💬 Medical Assistant")

    # Prerequisites check
    if not st.session_state.api_key_valid:
        st.info("🔑 Configure API key in sidebar")
        return
    if not st.session_state.docs_loaded:
        st.info("📄 Upload documents first")
        return

    # Query interface
    with st.form("query_form"):
        col1, col2 = st.columns([3, 1])

        with col1:
            query = st.text_input(
                "Ask a medical question:",
                placeholder="What are the side effects of metformin?"
            )

        with col2:
            query_type = st.selectbox(
                "Type",
                ["general", "diagnosis", "treatment", "drug_interaction"]
            )

        submitted = st.form_submit_button("🔍 Ask", type="primary")

        if submitted and query:
            process_query(query, query_type)

    # Suggestions
    if st.session_state.docs_loaded:
        st.markdown("**💡 Try asking:**")
        suggestions = [
            "Side effects of metformin",
            "Hypertension treatment guidelines",
            "Warfarin drug interactions"
        ]

        cols = st.columns(len(suggestions))
        for i, suggestion in enumerate(suggestions):
            with cols[i]:
                if st.button(f"💭 {suggestion[:20]}...", key=f"suggest_{i}"):
                    process_query(suggestion, "general")

    # Chat history
    render_chat_history()


def process_query(query: str, query_type: str):
    """Process medical query"""
    # Check cache first
    cached = query_cache.get(query)
    if cached:
        st.info("⚡ From cache")
        display_response(cached['response'], cached['sources'])
        return

    with st.spinner("🔍 Searching medical literature..."):
        try:
            start_time = time.time()

            # Create embedding
            embedding = st.session_state.embedding_manager.create_medical_query_embedding(
                query, query_type
            )

            # Search documents
            results = st.session_state.retriever.search(embedding, n_results=3)

            if not results:
                st.warning("🔍 No relevant documents found")
                return

            # Generate response
            response_data = st.session_state.response_generator.generate_response(
                query, results, query_type, max_tokens=600
            )

            processing_time = time.time() - start_time

            if 'error' not in response_data:
                display_response(response_data['response'], results)
                save_to_history(query, response_data['response'])

                # Cache response
                query_cache.set(query, {
                    'response': response_data['response'],
                    'sources': results,
                    'timestamp': time.time()
                })
            else:
                st.error(f"❌ {response_data['response']}")

        except Exception as e:
            st.error(f"❌ Query failed: {e}")


def display_response(response: str, sources: list):
    """Display medical response with professional formatting"""

    # Parse and structure the response
    formatted_response = format_medical_response(response)

    # Main response with better styling
    st.markdown("### 🔬 Medical Analysis")

    # Executive Summary (if available)
    if 'summary' in formatted_response:
        with st.container():
            st.markdown("#### 📋 **Executive Summary**")
            st.info(formatted_response['summary'])

    # Main Content
    if 'main_content' in formatted_response:
        st.markdown("#### 📖 **Detailed Analysis**")

        # Format main content with proper sections
        for section in formatted_response['main_content']:
            if section['type'] == 'heading':
                st.markdown(f"**{section['content']}**")
            elif section['type'] == 'paragraph':
                st.markdown(section['content'])
            elif section['type'] == 'list':
                for item in section['items']:
                    st.markdown(f"• {item}")
            elif section['type'] == 'numbered_list':
                for i, item in enumerate(section['items'], 1):
                    st.markdown(f"{i}. {item}")

    # Clinical Recommendations (highlighted)
    if 'recommendations' in formatted_response:
        st.markdown("#### 💊 **Clinical Recommendations**")

        # Color-code recommendations by priority
        for i, rec in enumerate(formatted_response['recommendations'], 1):
            if 'first-line' in rec.lower() or 'emergency' in rec.lower():
                st.success(f"**{i}.** {rec}")
            elif 'avoid' in rec.lower() or 'contraindicated' in rec.lower():
                st.error(f"**{i}.** {rec}")
            else:
                st.info(f"**{i}.** {rec}")

    # Limitations (if present)
    if 'limitations' in formatted_response:
        with st.expander("⚠️ **Limitations & Considerations**", expanded=False):
            for limitation in formatted_response['limitations']:
                st.warning(f"• {limitation}")

    # Sources with better formatting
    if sources:
        with st.expander(f"📚 **Evidence Sources** ({len(sources)})", expanded=False):
            for i, source in enumerate(sources, 1):
                col1, col2 = st.columns([3, 1])

                with col1:
                    source_name = source.get('metadata', {}).get('source', f'Source {i}')
                    st.markdown(f"**{i}. {source_name}**")

                    # Show excerpt if available
                    if 'content' in source:
                        excerpt = source['content'][:150] + "..." if len(source['content']) > 150 else source['content']
                        st.caption(f"*Excerpt:* {excerpt}")

                with col2:
                    similarity = source.get('similarity_score', 0)
                    if similarity > 0.8:
                        st.success(f"Relevance: {similarity:.2f}")
                    elif similarity > 0.6:
                        st.warning(f"Relevance: {similarity:.2f}")
                    else:
                        st.info(f"Relevance: {similarity:.2f}")

                if i < len(sources):
                    st.divider()


def format_medical_response(response_text: str) -> dict:
    """Parse and structure medical response text for better formatting"""
#
    # Initialize structure
    formatted = {
        'main_content': [],
        'recommendations': [],
        'limitations': []
    }

    # Split into paragraphs
    paragraphs = [p.strip() for p in response_text.split('\n\n') if p.strip()]

    current_section = 'main_content'

    for paragraph in paragraphs:
        # Detect section headers
        if paragraph.startswith('Actionable Recommendations:'):
            current_section = 'recommendations'
            continue
        elif paragraph.startswith('Limitations:'):
            current_section = 'limitations'
            continue
        elif paragraph.startswith('Based on'):
            # This is likely a summary
            formatted['summary'] = paragraph
            continue

        # Process content based on current section
        if current_section == 'recommendations':
            # Parse numbered recommendations
            if re.match(r'^\d+\.', paragraph):
                # Remove number and clean up
                rec_text = re.sub(r'^\d+\.\s*', '', paragraph)
                formatted['recommendations'].append(rec_text)
            else:
                # Look for bullet points
                lines = paragraph.split('\n')
                for line in lines:
                    if line.strip().startswith('•') or line.strip().startswith('-'):
                        rec_text = re.sub(r'^[•\-]\s*', '', line.strip())
                        formatted['recommendations'].append(rec_text)

        elif current_section == 'limitations':
            # Parse limitations
            if paragraph.startswith('•'):
                lines = paragraph.split('\n')
                for line in lines:
                    if line.strip().startswith('•'):
                        limitation = re.sub(r'^•\s*', '', line.strip())
                        formatted['limitations'].append(limitation)
            else:
                formatted['limitations'].append(paragraph)

        else:
            # Main content processing
            if paragraph.startswith('Source'):
                # This is source information
                formatted['main_content'].append({
                    'type': 'paragraph',
                    'content': paragraph
                })
            elif re.match(r'^[A-Z][^.]*:$', paragraph):
                # This looks like a heading
                formatted['main_content'].append({
                    'type': 'heading',
                    'content': paragraph.rstrip(':')
                })
            else:
                formatted['main_content'].append({
                    'type': 'paragraph',
                    'content': paragraph
                })

    return formatted


def save_to_history(query: str, response: str):
    """Save to chat history"""
    entry = {
        'query': query,
        'response': response[:200] + "..." if len(response) > 200 else response,
        'timestamp': datetime.now().strftime('%H:%M')
    }

    st.session_state.chat_history.append(entry)

    # Keep only last 5
    if len(st.session_state.chat_history) > 5:
        st.session_state.chat_history = st.session_state.chat_history[-5:]


def render_chat_history():
    """Show recent chat history"""
    if st.session_state.chat_history:
        with st.expander("📜 Recent Queries", expanded=False):
            for chat in reversed(st.session_state.chat_history[-3:]):
                st.markdown(f"**Q:** {chat['query'][:50]}...")
                st.markdown(f"**A:** {chat['response']}")
                st.caption(f"🕒 {chat['timestamp']}")
                st.divider()


def tools_section():
    """Medical tools"""
    st.markdown("## 🧰 Medical Tools")

    tool = st.selectbox(
        "Select Tool",
        ["💊 Drug Interactions", "🏷️ Medical Codes"]
    )

    if "Drug" in tool:
        drug_tool()
    elif "Medical" in tool:
        terminology_tool()


def drug_tool():
    """Drug interaction checker"""
    st.markdown("### 💊 Drug Interaction Checker")

    medications = st.text_area(
        "Enter medications (one per line):",
        placeholder="warfarin\naspirin\nmetformin"
    )

    if medications and st.button("🔍 Check Interactions"):
        try:
            med_list = [m.strip() for m in medications.split('\n') if m.strip()]

            if len(med_list) < 2:
                st.warning("Enter at least 2 medications")
                return

            analysis = st.session_state.drug_checker.analyze_medication_list(
                '\n'.join(med_list)
            )

            # Display results
            interactions = analysis.get('interactions', [])

            if interactions:
                st.markdown("#### ⚠️ Interactions Found")
                for interaction in interactions:
                    severity = interaction.get('severity', 'unknown')
                    if severity == 'major':
                        st.error(f"🚨 **{interaction['drug1']} + {interaction['drug2']}:** {interaction['description']}")
                    else:
                        st.warning(
                            f"⚠️ **{interaction['drug1']} + {interaction['drug2']}:** {interaction['description']}")
            else:
                st.success("✅ No major interactions found")

        except Exception as e:
            st.error(f"❌ Analysis failed: {e}")


def terminology_tool():
    """Medical terminology mapper"""
    st.markdown("### 🏷️ Medical Code Mapper")

    medical_text = st.text_area(
        "Enter medical text:",
        placeholder="Patient with diabetes mellitus and hypertension..."
    )

    if medical_text and st.button("🔍 Map Codes"):
        try:
            analysis = st.session_state.terminology_mapper.analyze_medical_text(medical_text)

            # Show results
            terms = analysis.get('terms_found', [])
            if terms:
                st.markdown("#### 🏷️ Found Terms")
                for term in terms:
                    st.markdown(f"**{term['term']}** ({term['category']})")
                    if term.get('icd10'):
                        st.caption(f"ICD-10: {term['icd10']}")
            else:
                st.info("No medical terms found")

        except Exception as e:
            st.error(f"❌ Analysis failed: {e}")


def main():
    """Main application"""
    # Initialize
    init_session_state()

    # Render UI
    render_header()
    render_sidebar()

    # Setup components
    setup_components()

    # Main tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📄 Documents", "🧰 Tools"])

    with tab1:
        chat_section()

    with tab2:
        document_section()

    with tab3:
        tools_section()

    # Footer
    st.markdown("---")
    st.caption("⚠️ For healthcare professionals only. Always verify information.")


if __name__ == "__main__":
    main()