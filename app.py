# # # import streamlit as st
# # # import os
# # # import logging
# # # from pathlib import Path
# # # import time
# # # from datetime import datetime
# # # from typing import Dict, List, Any
# # # import concurrent.futures
# # # from functools import lru_cache
# # # import hashlib
# # # import json
# # #
# # # # Import custom modules
# # # from config.settings import Settings
# # # from src.data_processing.document_loader import DocumentLoader
# # # from src.data_processing.text_preprocessor import TextPreprocessor
# # # from src.data_processing.chunking_strategy import MedicalChunkingStrategy
# # # from src.embeddings.embedding_manager import EmbeddingManager
# # # from src.retrieval.retriever import MedicalRetriever
# # # from src.generation.groq_response_generator import GroqResponseGenerator
# # # from src.medical_nlp.drug_interaction_checker import DrugInteractionChecker
# # # from src.medical_nlp.terminology_mapper import MedicalTerminologyMapper
# # # from src.utils.groq_utils import GroqUtils
# # #
# # # # Configure logging
# # # logging.basicConfig(level=logging.INFO)
# # # logger = logging.getLogger(__name__)
# # #
# # # # Page configuration
# # # st.set_page_config(
# # #     page_title="Medical Literature RAG System",
# # #     page_icon="🏥",
# # #     layout="wide",
# # #     initial_sidebar_state="expanded"
# # # )
# # #
# # # # Enhanced Custom CSS - Clean & Professional
# # # st.markdown("""
# # # <style>
# # #     /* Hide Streamlit branding */
# # #     #MainMenu {visibility: hidden;}
# # #     footer {visibility: hidden;}
# # #     header {visibility: hidden;}
# # #
# # #     /* Main styling */
# # #     .main {
# # #         padding: 1rem 2rem;
# # #     }
# # #
# # #     /* Clean header */
# # #     .main-header {
# # #         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
# # #         padding: 2.5rem;
# # #         border-radius: 15px;
# # #         color: white;
# # #         text-align: center;
# # #         margin-bottom: 2rem;
# # #         box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
# # #     }
# # #
# # #     .main-header h1 {
# # #         font-size: 2.8rem;
# # #         margin-bottom: 0.5rem;
# # #         font-weight: 700;
# # #         text-shadow: 0 2px 4px rgba(0,0,0,0.1);
# # #     }
# # #
# # #     .main-header p {
# # #         font-size: 1.2rem;
# # #         opacity: 0.95;
# # #         margin: 0;
# # #     }
# # #
# # #     /* Enhanced metric cards */
# # #     .metric-card {
# # #         background: white;
# # #         padding: 1.5rem;
# # #         border-radius: 12px;
# # #         border: 1px solid #e2e8f0;
# # #         box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
# # #         margin: 1rem 0;
# # #         transition: all 0.3s ease;
# # #     }
# # #
# # #     .metric-card:hover {
# # #         transform: translateY(-2px);
# # #         box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
# # #     }
# # #
# # #     /* Status indicators */
# # #     .status-indicator {
# # #         display: inline-flex;
# # #         align-items: center;
# # #         padding: 0.5rem 1rem;
# # #         border-radius: 25px;
# # #         font-weight: 600;
# # #         margin: 0.25rem;
# # #         font-size: 0.9rem;
# # #     }
# # #
# # #     .status-success {
# # #         background: #d4f7dc;
# # #         color: #1a7f37;
# # #         border: 1px solid #a7f3d0;
# # #     }
# # #
# # #     .status-error {
# # #         background: #ffe6e6;
# # #         color: #dc2626;
# # #         border: 1px solid #fecaca;
# # #     }
# # #
# # #     .status-warning {
# # #         background: #fef3c7;
# # #         color: #d97706;
# # #         border: 1px solid #fde68a;
# # #     }
# # #
# # #     /* Clean buttons */
# # #     .stButton > button {
# # #         border-radius: 8px;
# # #         border: none;
# # #         padding: 0.75rem 1.5rem;
# # #         font-weight: 600;
# # #         transition: all 0.2s ease;
# # #         width: 100%;
# # #     }
# # #
# # #     .stButton > button:hover {
# # #         transform: translateY(-1px);
# # #         box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
# # #     }
# # #
# # #     /* Sidebar styling */
# # #     .sidebar .sidebar-content {
# # #         background: #f8fafc;
# # #         border-radius: 10px;
# # #         padding: 1rem;
# # #     }
# # #
# # #     /* Chat styling */
# # #     .chat-message {
# # #         padding: 1.5rem;
# # #         border-radius: 12px;
# # #         margin: 1rem 0;
# # #         box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
# # #     }
# # #
# # #     .user-message {
# # #         background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
# # #         border-left: 4px solid #1976d2;
# # #     }
# # #
# # #     .assistant-message {
# # #         background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
# # #         border-left: 4px solid #7b1fa2;
# # #     }
# # #
# # #     /* Progress indicators */
# # #     .progress-container {
# # #         background: #f8fafc;
# # #         border-radius: 12px;
# # #         padding: 1.5rem;
# # #         margin: 1rem 0;
# # #         border: 1px solid #e2e8f0;
# # #     }
# # #
# # #     /* Alert styles */
# # #     .alert-success {
# # #         background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
# # #         color: #155724;
# # #         padding: 1rem 1.5rem;
# # #         border-radius: 8px;
# # #         border-left: 4px solid #28a745;
# # #         margin: 1rem 0;
# # #     }
# # #
# # #     .alert-warning {
# # #         background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
# # #         color: #856404;
# # #         padding: 1rem 1.5rem;
# # #         border-radius: 8px;
# # #         border-left: 4px solid #ffc107;
# # #         margin: 1rem 0;
# # #     }
# # #
# # #     .alert-error {
# # #         background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
# # #         color: #721c24;
# # #         padding: 1rem 1.5rem;
# # #         border-radius: 8px;
# # #         border-left: 4px solid #dc3545;
# # #         margin: 1rem 0;
# # #     }
# # #
# # #     /* Responsive design */
# # #     @media (max-width: 768px) {
# # #         .main-header h1 {
# # #             font-size: 2.2rem;
# # #         }
# # #         .metric-card {
# # #             padding: 1rem;
# # #         }
# # #         .main {
# # #             padding: 0.5rem 1rem;
# # #         }
# # #     }
# # #
# # #     /* Custom tabs */
# # #     .stTabs [data-baseweb="tab-list"] {
# # #         gap: 8px;
# # #     }
# # #
# # #     .stTabs [data-baseweb="tab"] {
# # #         border-radius: 8px;
# # #         padding: 0.5rem 1rem;
# # #         font-weight: 600;
# # #     }
# # #
# # #     /* File uploader styling */
# # #     .uploadedFile {
# # #         border-radius: 8px;
# # #         border: 2px dashed #cbd5e0;
# # #         padding: 2rem;
# # #         text-align: center;
# # #         transition: all 0.3s ease;
# # #     }
# # #
# # #     .uploadedFile:hover {
# # #         border-color: #667eea;
# # #         background: #f7fafc;
# # #     }
# # # </style>
# # # """, unsafe_allow_html=True)
# # #
# # #
# # # # 2. ADD CACHING DECORATORS AND OPTIMIZED FUNCTIONS
# # #
# # # @st.cache_data(ttl=3600, show_spinner=False)
# # # def cached_embedding_generation(text: str, query_type: str = "general"):
# # #     """Cache embedding generation to avoid recomputation"""
# # #     return st.session_state.embedding_manager.create_medical_query_embedding(text, query_type)
# # #
# # #
# # # @st.cache_data(ttl=1800, show_spinner=False)
# # # def cached_search_results(query_hash: str, embedding_data: str, n_results: int = 3):
# # #     """Cache search results for similar queries"""
# # #     # Convert embedding back from string and search
# # #     embedding = json.loads(embedding_data)
# # #     return st.session_state.retriever.search(embedding, n_results=n_results)
# # #
# # #
# # # class SimpleQueryCache:
# # #     """Lightweight query cache for responses"""
# # #
# # #     def __init__(self):
# # #         if 'query_cache' not in st.session_state:
# # #             st.session_state.query_cache = {}
# # #             st.session_state.cache_access_times = {}
# # #
# # #     def get_cache_key(self, query: str, query_type: str) -> str:
# # #         return hashlib.md5(f"{query.lower().strip()}_{query_type}".encode()).hexdigest()[:12]
# # #
# # #     def get(self, query: str, query_type: str):
# # #         key = self.get_cache_key(query, query_type)
# # #         if key in st.session_state.query_cache:
# # #             st.session_state.cache_access_times[key] = time.time()
# # #             return st.session_state.query_cache[key]
# # #         return None
# # #
# # #     def set(self, query: str, query_type: str, response_data):
# # #         key = self.get_cache_key(query, query_type)
# # #         # Limit cache size to prevent memory issues
# # #         if len(st.session_state.query_cache) >= 20:
# # #             # Remove oldest entry
# # #             oldest_key = min(st.session_state.cache_access_times.keys(),
# # #                              key=lambda k: st.session_state.cache_access_times[k])
# # #             st.session_state.query_cache.pop(oldest_key, None)
# # #             st.session_state.cache_access_times.pop(oldest_key, None)
# # #
# # #         st.session_state.query_cache[key] = response_data
# # #         st.session_state.cache_access_times[key] = time.time()
# # #
# # #
# # # # Initialize cache
# # # query_cache = SimpleQueryCache()
# # #
# # #
# # # def initialize_session_state():
# # #     """Initialize session state variables"""
# # #     if 'initialized' not in st.session_state:
# # #         st.session_state.initialized = True
# # #         st.session_state.documents_loaded = False
# # #         st.session_state.vector_store_ready = False
# # #         st.session_state.document_count = 0
# # #         st.session_state.processing_status = ""
# # #         st.session_state.chat_history = []
# # #         st.session_state.api_key_valid = False
# # #
# # #
# # # def validate_environment():
# # #     """Validate environment and configuration"""
# # #     try:
# # #         Settings.validate_config()
# # #         return True, "Configuration valid"
# # #     except Exception as e:
# # #         return False, str(e)
# # #
# # #
# # # # 3. REPLACE YOUR initialize_components FUNCTION
# # # @st.cache_resource
# # # def initialize_components_cached():
# # #     """Cached component initialization to prevent reloading"""
# # #     try:
# # #         # Initialize components with caching
# # #         document_loader = DocumentLoader()
# # #         preprocessor = TextPreprocessor()
# # #         chunking_strategy = MedicalChunkingStrategy(
# # #             chunk_size=Settings.CHUNK_SIZE,
# # #             overlap=Settings.CHUNK_OVERLAP
# # #         )
# # #         embedding_manager = EmbeddingManager(model_name=Settings.EMBEDDING_MODEL)
# # #         retriever = MedicalRetriever(
# # #             persist_directory=Settings.CHROMA_PERSIST_DIRECTORY,
# # #             collection_name=Settings.COLLECTION_NAME
# # #         )
# # #
# # #         return {
# # #             'document_loader': document_loader,
# # #             'preprocessor': preprocessor,
# # #             'chunking_strategy': chunking_strategy,
# # #             'embedding_manager': embedding_manager,
# # #             'retriever': retriever,
# # #             'drug_checker': DrugInteractionChecker(),
# # #             'terminology_mapper': MedicalTerminologyMapper()
# # #         }
# # #     except Exception as e:
# # #         return None
# # #
# # #
# # #
# # # # def initialize_components():
# # # #     """Initialize all system components"""
# # # #     try:
# # # #         # Initialize components
# # # #         document_loader = DocumentLoader()
# # # #         preprocessor = TextPreprocessor()
# # # #         chunking_strategy = MedicalChunkingStrategy(
# # # #             chunk_size=Settings.CHUNK_SIZE,
# # # #             overlap=Settings.CHUNK_OVERLAP
# # # #         )
# # # #         embedding_manager = EmbeddingManager(model_name=Settings.EMBEDDING_MODEL)
# # # #         retriever = MedicalRetriever(
# # # #             persist_directory=Settings.CHROMA_PERSIST_DIRECTORY,
# # # #             collection_name=Settings.COLLECTION_NAME
# # # #         )
# # # #
# # # #         # Store in session state
# # # #         st.session_state.document_loader = document_loader
# # # #         st.session_state.preprocessor = preprocessor
# # # #         st.session_state.chunking_strategy = chunking_strategy
# # # #         st.session_state.embedding_manager = embedding_manager
# # # #         st.session_state.retriever = retriever
# # # #         st.session_state.drug_checker = DrugInteractionChecker()
# # # #         st.session_state.terminology_mapper = MedicalTerminologyMapper()
# # # #
# # # #         return True, "Components initialized successfully"
# # # #     except Exception as e:
# # # #         return False, f"Component initialization failed: {str(e)}"
# # #
# # #
# # # def initialize_components():
# # #     """Optimized component initialization"""
# # #     if 'components_loaded' not in st.session_state:
# # #         components = initialize_components_cached()
# # #         if components:
# # #             # Store components in session state
# # #             for name, component in components.items():
# # #                 st.session_state[name] = component
# # #             st.session_state.components_loaded = True
# # #             return True, "Components initialized successfully"
# # #         else:
# # #             return False, "Component initialization failed"
# # #     return True, "Components already loaded"
# # #
# # #
# # #
# # # def render_clean_header():
# # #     """Render clean, professional header"""
# # #     st.markdown("""
# # #     <div class="main-header">
# # #         <h1>🏥 Medical Literature RAG System</h1>
# # #         <p>Intelligent Evidence-Based Decision Support for Healthcare Professionals</p>
# # #     </div>
# # #     """, unsafe_allow_html=True)
# # #
# # #
# # # def render_compact_status():
# # #     """Compact status display"""
# # #     col1, col2 = st.columns(2)
# # #
# # #     with col1:
# # #         api_status = "🟢" if st.session_state.get('api_key_valid') else "🔴"
# # #         st.markdown(f"{api_status} API")
# # #
# # #     with col2:
# # #         docs_status = "🟢" if st.session_state.get('vector_store_ready') else "🔴"
# # #         st.markdown(f"{docs_status} Docs")
# # #
# # #     # Show document count if available
# # #     if st.session_state.get('vector_store_ready'):
# # #         try:
# # #             stats = st.session_state.retriever.get_collection_stats()
# # #             doc_count = stats.get('total_documents', 0)
# # #             st.caption(f"📄 {doc_count} documents loaded")
# # #         except:
# # #             pass
# # #
# # #
# # # # def render_compact_status():
# # # #     """Compact status display"""
# # # #     col1, col2 = st.columns(2)
# # # #
# # # #     with col1:
# # # #         api_status = "🟢" if st.session_state.get('api_key_valid') else "🔴"
# # # #         st.markdown(f"{api_status} API")
# # # #
# # # #     with col2:
# # # #         docs_status = "🟢" if st.session_state.get('vector_store_ready') else "🔴"
# # # #         st.markdown(f"{docs_status} Docs")
# # # #
# # # #     # Show document count if available
# # # #     if st.session_state.get('vector_store_ready'):
# # # #         try:
# # # #             stats = st.session_state.retriever.get_collection_stats()
# # # #             doc_count = stats.get('total_documents', 0)
# # # #             st.caption(f"📄 {doc_count} documents loaded")
# # # #         except:
# # # #             pass
# # #
# # #
# # # def run_quick_test():
# # #     """Quick system test"""
# # #     with st.spinner("Testing..."):
# # #         if st.session_state.get('api_key_valid') and st.session_state.get('vector_store_ready'):
# # #             st.success("✅ System ready!")
# # #         else:
# # #             st.error("❌ System not ready - check API key and documents")
# # #
# # #
# # # # def run_quick_test():
# # # #     """Quick system test"""
# # # #     with st.spinner("Testing..."):
# # # #         if st.session_state.get('api_key_valid') and st.session_state.get('vector_store_ready'):
# # # #             st.success("✅ System ready!")
# # # #         else:
# # # #             st.error("❌ System not ready - check API key and documents")
# # #
# # #
# # # def clear_cache():
# # #     """Clear various caches"""
# # #     # Clear query cache
# # #     if hasattr(st.session_state, 'query_cache'):
# # #         st.session_state.query_cache = {}
# # #     if hasattr(st.session_state, 'cache_access_times'):
# # #         st.session_state.cache_access_times = {}
# # #
# # #     # Clear Streamlit cache
# # #     st.cache_data.clear()
# # #
# # #     st.success("🗑️ Cache cleared!")
# # #
# # #
# # # def sidebar_configuration():
# # #     # """Fixed sidebar with unique widget keys"""
# # #     # with st.sidebar:
# # #     #     st.markdown("## ⚙️ Configuration")
# # #     #
# # #     #     # API Key Configuration
# # #     #     with st.expander("🔑 API Settings", expanded=False):
# # #     #         # Use unique key and check if already in session state
# # #     #         current_api_key = st.session_state.get('groq_api_key', os.getenv("GROQ_API_KEY", ""))
# # #     #
# # #     #         api_key = st.text_input(
# # #     #             "Groq API Key",
# # #     #             type="password",
# # #     #             value=current_api_key,
# # #     #             placeholder="Enter your API key...",
# # #     #             help="Get your key from console.groq.com",
# # #     #             key="sidebar_api_key_input"  # Unique key
# # #     #         )
# # #     #
# # #     #         if api_key and api_key != current_api_key:
# # #     #             if st.button("🔓 Validate Key", key="sidebar_validate_api"):
# # #     #                 validate_and_store_api_key(api_key)
# # #     #         elif api_key and st.session_state.get('api_key_valid', False):
# # #     #             st.success("✅ API key already validated")
# # #     #
# # #     #     # Model Selection - Only show if API is valid
# # #     #     if st.session_state.get('api_key_valid', False):
# # #     #         st.markdown("### 🤖 Model Selection")
# # #     #         model_options = {
# # #     #             "llama3-8b-8192": "Llama 3 8B (Recommended)",
# # #     #             "llama3-70b-8192": "Llama 3 70B (Advanced)",
# # #     #             "gemma-7b-it": "Gemma 7B (Alternative)"
# # #     #         }
# # #     #
# # #     #         current_model = st.session_state.get('selected_model', "llama3-8b-8192")
# # #     #         selected_model = st.selectbox(
# # #     #             "Choose Model",
# # #     #             options=list(model_options.keys()),
# # #     #             format_func=lambda x: model_options[x],
# # #     #             index=list(model_options.keys()).index(current_model) if current_model in model_options else 0,
# # #     #             help="Select the AI model for generating responses",
# # #     #             key="sidebar_model_selection"  # Unique key
# # #     #         )
# # #     #
# # #     #         # Update model if changed
# # #     #         if selected_model != current_model:
# # #     #             st.session_state.selected_model = selected_model
# # #     #             # Update response generator model if it exists
# # #     #             if hasattr(st.session_state, 'response_generator'):
# # #     #                 try:
# # #     #                     st.session_state.response_generator.model = selected_model
# # #     #                     st.success(f"✅ Switched to {model_options[selected_model]}")
# # #     #                 except Exception as e:
# # #     #                     st.warning(f"⚠️ Model update failed: {str(e)}")
# # #     #
# # #     #     # System Status
# # #     #     st.markdown("### 📊 System Status")
# # #     #     render_enhanced_system_status()
# # #     #
# # #     #     # Quick Statistics
# # #     #     st.markdown("### 📈 Statistics")
# # #     #     render_quick_statistics()
# # #     #
# # #     #     # Debug Tools
# # #     #     st.markdown("### 🔧 Debug Tools")
# # #     #     if st.button("🧪 Test System", help="Run a quick system test", key="sidebar_test_system"):
# # #     #         test_system_components()
# # #     #
# # #     #     if st.button("🔄 Reset System", type="secondary", help="Clear all data and restart", key="sidebar_reset_system"):
# # #     #         reset_system()
# # #
# # #     """Simplified sidebar to avoid key conflicts"""
# # #     with st.sidebar:
# # #         st.markdown("## ⚙️ Configuration")
# # #
# # #         # Check if API key is already validated
# # #         if not st.session_state.get('api_key_valid', False):
# # #             st.markdown("### 🔑 API Configuration")
# # #             st.info("Please enter your Groq API key to get started")
# # #
# # #             # Simple form to avoid key conflicts
# # #             with st.form("api_key_form", clear_on_submit=False):
# # #                 api_key = st.text_input(
# # #                     "Groq API Key",
# # #                     type="password",
# # #                     placeholder="Enter your API key...",
# # #                     help="Get your key from console.groq.com"
# # #                 )
# # #                 submitted = st.form_submit_button("🔓 Validate Key")
# # #
# # #                 if submitted and api_key:
# # #                     validate_and_store_api_key(api_key)
# # #         else:
# # #             st.success("✅ API Key Validated")
# # #
# # #             # Model selection
# # #             st.markdown("### 🤖 Model Selection")
# # #             model_options = {
# # #                 "llama3-8b-8192": "Llama 3 8B (Recommended)",
# # #                 "llama3-70b-8192": "Llama 3 70B (Advanced)",
# # #                 "gemma-7b-it": "Gemma 7B (Alternative)"
# # #             }
# # #
# # #             # Use form for model selection too
# # #             with st.form("model_selection_form"):
# # #                 selected_model = st.selectbox(
# # #                     "Choose Model",
# # #                     options=list(model_options.keys()),
# # #                     format_func=lambda x: model_options[x],
# # #                     index=0
# # #                 )
# # #                 model_submitted = st.form_submit_button("Update Model")
# # #
# # #                 if model_submitted:
# # #                     st.session_state.selected_model = selected_model
# # #                     if hasattr(st.session_state, 'response_generator'):
# # #                         st.session_state.response_generator.model = selected_model
# # #                     st.success(f"✅ Using {model_options[selected_model]}")
# # #
# # #         # System Status
# # #         st.markdown("### 📊 System Status")
# # #         render_enhanced_system_status()
# # #
# # #         # Quick Statistics
# # #         st.markdown("### 📈 Statistics")
# # #         render_quick_statistics()
# # #
# # #         # Debug Tools with forms
# # #         st.markdown("### 🔧 Debug Tools")
# # #
# # #         col1, col2 = st.columns(2)
# # #         with col1:
# # #             if st.button("🧪 Test", key="test_btn"):
# # #                 test_system_components()
# # #         with col2:
# # #             if st.button("🔄 Reset", key="reset_btn"):
# # #                 reset_system()
# # #
# # # # def sidebar_configuration():
# # # #     """Optimized sidebar with minimal recomputation"""
# # # #     with st.sidebar:
# # # #         st.markdown("## ⚙️ Configuration")
# # # #
# # # #         # API Key section (minimized)
# # # #         if not st.session_state.get('api_key_valid', False):
# # # #             with st.expander("🔑 API Settings", expanded=True):
# # # #                 api_key = st.text_input(
# # # #                     "Groq API Key",
# # # #                     type="password",
# # # #                     key="api_key_input",
# # # #                     help="Get your key from console.groq.com"
# # # #                 )
# # # #                 if api_key and st.button("🔓 Validate", key="validate_api"):
# # # #                     validate_and_store_api_key(api_key)
# # # #         else:
# # # #             st.success("✅ API Connected")
# # # #
# # # #             # Model selection (only when API valid)
# # # #             model_options = {
# # # #                 "llama3-8b-8192": "Llama 3 8B (Fast)",
# # # #                 "llama3-70b-8192": "Llama 3 70B (Accurate)",
# # # #                 "gemma-7b-it": "Gemma 7B (Alternative)"
# # # #             }
# # # #
# # # #             selected_model = st.selectbox(
# # # #                 "Model",
# # # #                 options=list(model_options.keys()),
# # # #                 format_func=lambda x: model_options[x],
# # # #                 key="model_select"
# # # #             )
# # # #
# # # #             if st.session_state.get('selected_model') != selected_model:
# # # #                 st.session_state.selected_model = selected_model
# # # #                 if hasattr(st.session_state, 'response_generator'):
# # # #                     st.session_state.response_generator.model = selected_model
# # # #
# # # #         # Compact status
# # # #         render_compact_status()
# # # #
# # # #         # Quick actions
# # # #         st.markdown("### ⚡ Quick Actions")
# # # #         if st.button("🧪 Test System"):
# # # #             run_quick_test()
# # # #         if st.button("🗑️ Clear Cache"):
# # # #             clear_cache()
# # #
# # # # def sidebar_configuration():
# # # #     """Enhanced sidebar with debug tools"""
# # # #     with st.sidebar:
# # # #         st.markdown("## ⚙️ Configuration")
# # # #
# # # #         # API Key Configuration
# # # #         with st.expander("🔑 API Settings", expanded=False):
# # # #             api_key = st.text_input(
# # # #                 "Groq API Key",
# # # #                 type="password",
# # # #                 value=os.getenv("GROQ_API_KEY", ""),
# # # #                 placeholder="Enter your API key...",
# # # #                 help="Get your key from console.groq.com"
# # # #             )
# # # #
# # # #             if api_key:
# # # #                 if st.button("🔓 Validate Key", key="validate_api"):
# # # #                     validate_and_store_api_key(api_key)
# # # #
# # # #         # Model Selection
# # # #         if st.session_state.get('api_key_valid', False):
# # # #             st.markdown("### 🤖 Model Selection")
# # # #             model_options = {
# # # #                 "llama3-8b-8192": "Llama 3 8B (Recommended)",
# # # #                 "llama3-70b-8192": "Llama 3 70B (Advanced)",
# # # #                 "gemma-7b-it": "Gemma 7B (Alternative)"
# # # #             }
# # # #
# # # #             selected_model = st.selectbox(
# # # #                 "Choose Model",
# # # #                 options=list(model_options.keys()),
# # # #                 format_func=lambda x: model_options[x],
# # # #                 index=0,
# # # #                 help="Select the AI model for generating responses"
# # # #             )
# # # #
# # # #             # Update model if changed
# # # #             if st.session_state.get('selected_model') != selected_model:
# # # #                 st.session_state.selected_model = selected_model
# # # #                 # Update response generator model if it exists
# # # #                 if hasattr(st.session_state, 'response_generator'):
# # # #                     try:
# # # #                         st.session_state.response_generator.model = selected_model
# # # #                         st.success(f"✅ Switched to {model_options[selected_model]}")
# # # #                     except:
# # # #                         pass
# # # #
# # # #         # System Status
# # # #         st.markdown("### 📊 System Status")
# # # #         render_enhanced_system_status()
# # # #
# # # #         # Quick Statistics
# # # #         st.markdown("### 📈 Statistics")
# # # #         render_quick_statistics()
# # # #
# # # #         # Debug Tools
# # # #         st.markdown("### 🔧 Debug Tools")
# # # #         if st.button("🧪 Test System", help="Run a quick system test"):
# # # #             test_system_components()
# # # #
# # # #         if st.button("🔄 Reset System", type="secondary", help="Clear all data and restart"):
# # # #             reset_system()
# # #
# # #
# # # def test_system_components():
# # #     """Quick system test function"""
# # #     with st.spinner("🧪 Testing system components..."):
# # #         st.markdown("### 🧪 System Test Results")
# # #
# # #         # Test 1: API Key
# # #         if st.session_state.get('api_key_valid', False):
# # #             st.success("✅ API Key: Valid")
# # #         else:
# # #             st.error("❌ API Key: Invalid or missing")
# # #             return
# # #
# # #         # Test 2: Components
# # #         required_components = ['response_generator', 'embedding_manager', 'retriever']
# # #         for component in required_components:
# # #             if hasattr(st.session_state, component):
# # #                 st.success(f"✅ {component}: Ready")
# # #             else:
# # #                 st.error(f"❌ {component}: Missing")
# # #                 return
# # #
# # #         # Test 3: Vector Store
# # #         if st.session_state.get('vector_store_ready', False):
# # #             st.success("✅ Vector Store: Ready")
# # #
# # #             # Test simple query
# # #             try:
# # #                 test_embedding = st.session_state.embedding_manager.create_medical_query_embedding(
# # #                     "test medical query", "general"
# # #                 )
# # #                 search_results = st.session_state.retriever.search(test_embedding, n_results=1)
# # #                 st.success(f"✅ Retrieval Test: Found {len(search_results)} results")
# # #             except Exception as e:
# # #                 st.error(f"❌ Retrieval Test: Failed - {str(e)}")
# # #         else:
# # #             st.error("❌ Vector Store: No documents loaded")
# # #
# # #         st.balloons()
# # #         st.success("🎉 System test completed!")
# # #
# # # def validate_and_store_api_key(api_key: str):
# # #     """Fixed version with better error handling"""
# # #     with st.spinner("🔍 Validating API key..."):
# # #         try:
# # #             validation = GroqUtils.validate_api_key(api_key)
# # #
# # #             if validation['valid']:
# # #                 st.sidebar.markdown("""
# # #                 <div class="alert-success">
# # #                     ✅ API key is valid and ready to use!
# # #                 </div>
# # #                 """, unsafe_allow_html=True)
# # #
# # #                 st.session_state.api_key_valid = True
# # #                 st.session_state.groq_api_key = api_key
# # #
# # #                 # Initialize response generator with selected model or default
# # #                 model = st.session_state.get('selected_model', 'llama3-8b-8192')
# # #                 st.session_state.response_generator = GroqResponseGenerator(
# # #                     api_key=api_key,
# # #                     model=model
# # #                 )
# # #
# # #                 st.sidebar.success(f"✅ Response generator ready with {model}")
# # #             else:
# # #                 st.sidebar.markdown(f"""
# # #                 <div class="alert-error">
# # #                     ❌ {validation['message']}
# # #                 </div>
# # #                 """, unsafe_allow_html=True)
# # #                 st.session_state.api_key_valid = False
# # #
# # #         except Exception as e:
# # #             st.sidebar.error(f"❌ API validation error: {str(e)}")
# # #             st.session_state.api_key_valid = False
# # #
# # #
# # # def render_enhanced_system_status():
# # #     """Render enhanced system status indicators"""
# # #     status_items = [
# # #         ("API Connection", st.session_state.get('api_key_valid', False)),
# # #         ("Documents Loaded", st.session_state.get('documents_loaded', False)),
# # #         ("Vector Store", st.session_state.get('vector_store_ready', False))
# # #     ]
# # #
# # #     for item, status in status_items:
# # #         if status:
# # #             st.markdown(f"""
# # #             <div class="status-indicator status-success">
# # #                 ✅ {item}
# # #             </div>
# # #             """, unsafe_allow_html=True)
# # #         else:
# # #             st.markdown(f"""
# # #             <div class="status-indicator status-error">
# # #                 ❌ {item}
# # #             </div>
# # #             """, unsafe_allow_html=True)
# # #
# # #
# # # def render_quick_statistics():
# # #     """Render quick statistics with metrics"""
# # #     if st.session_state.get('retriever'):
# # #         try:
# # #             stats = st.session_state.retriever.get_collection_stats()
# # #             col1, col2 = st.columns(2)
# # #
# # #             with col1:
# # #                 st.metric("Documents", stats.get('total_documents', 0))
# # #             with col2:
# # #                 embedding_stats = st.session_state.embedding_manager.get_embedding_stats()
# # #                 st.metric("Embeddings", embedding_stats.get('cached_embeddings', 0))
# # #         except:
# # #             st.metric("System Status", "Initializing...")
# # #
# # #
# # # def reset_system():
# # #     """Reset system with confirmation"""
# # #     for key in list(st.session_state.keys()):
# # #         if key != 'initialized':
# # #             del st.session_state[key]
# # #     st.success("🔄 System reset successfully!")
# # #     st.rerun()
# # #
# # #
# # # def document_upload_section():
# # #     """Enhanced document upload section"""
# # #     st.markdown("## 📄 Document Management")
# # #
# # #     tab1, tab2, tab3 = st.tabs(["📤 Upload", "⚙️ Processing", "📊 Statistics"])
# # #
# # #     with tab1:
# # #         render_document_upload_interface()
# # #
# # #     with tab2:
# # #         render_processing_interface()
# # #
# # #     with tab3:
# # #         render_document_statistics()
# # #
# # #
# # # def render_document_upload_interface():
# # #     """Clean document upload interface"""
# # #     st.markdown("### Upload Medical Documents")
# # #     st.markdown("*Upload medical literature, clinical guidelines, or anonymized patient data*")
# # #
# # #     uploaded_files = st.file_uploader(
# # #         "Choose files",
# # #         type=['pdf', 'txt', 'docx', 'csv'],
# # #         accept_multiple_files=True,
# # #         help="Supported formats: PDF, TXT, DOCX, CSV"
# # #     )
# # #
# # #     if uploaded_files:
# # #         # Show file summary
# # #         st.markdown("#### 📁 Uploaded Files")
# # #         for file in uploaded_files:
# # #             file_size = f"{len(file.getvalue()) / 1024:.1f} KB"
# # #             st.markdown(f"• **{file.name}** ({file_size})")
# # #
# # #         st.success(f"✅ {len(uploaded_files)} file(s) ready for processing")
# # #
# # #         if st.button("🚀 Process Documents", type="primary"):
# # #             process_uploaded_files_enhanced(uploaded_files)
# # #
# # #
# # # # 6. OPTIMIZED DOCUMENT PROCESSING
# # # def process_uploaded_files_enhanced(uploaded_files):
# # #     """Optimized file processing with concurrent processing"""
# # #     if not st.session_state.get('document_loader'):
# # #         st.error("❌ System components not initialized. Please refresh the page.")
# # #         return
# # #
# # #     progress_container = st.container()
# # #
# # #     with progress_container:
# # #         st.markdown("### 🔄 Processing Documents")
# # #         progress_bar = st.progress(0)
# # #         status_text = st.empty()
# # #
# # #         try:
# # #             # Process files concurrently for better performance
# # #             def process_single_file(uploaded_file):
# # #                 doc_data = st.session_state.document_loader.load_uploaded_file(uploaded_file)
# # #                 processed = st.session_state.preprocessor.preprocess_document(
# # #                     doc_data['content'], anonymize=True
# # #                 )
# # #                 chunks = st.session_state.chunking_strategy.create_contextual_chunks(
# # #                     processed['processed_content'], doc_data
# # #                 )
# # #                 return chunks
# # #
# # #             all_chunks = []
# # #             total_files = len(uploaded_files)
# # #
# # #             # Use ThreadPoolExecutor for concurrent processing
# # #             with concurrent.futures.ThreadPoolExecutor(max_workers=min(3, total_files)) as executor:
# # #                 # Submit all files for processing
# # #                 future_to_file = {
# # #                     executor.submit(process_single_file, file): file
# # #                     for file in uploaded_files
# # #                 }
# # #
# # #                 # Collect results as they complete
# # #                 for i, future in enumerate(concurrent.futures.as_completed(future_to_file)):
# # #                     file = future_to_file[future]
# # #                     status_text.text(f"📖 Processing: {file.name}")
# # #                     progress_bar.progress((i + 1) / total_files * 0.6)
# # #
# # #                     try:
# # #                         chunks = future.result()
# # #                         all_chunks.extend(chunks)
# # #                     except Exception as e:
# # #                         st.warning(f"⚠️ Error processing {file.name}: {str(e)}")
# # #
# # #             # Generate embeddings
# # #             status_text.text("🧠 Generating embeddings...")
# # #             progress_bar.progress(0.8)
# # #             embedded_chunks = st.session_state.embedding_manager.embed_medical_chunks(all_chunks)
# # #
# # #             # Store in vector database
# # #             status_text.text("💾 Storing in database...")
# # #             progress_bar.progress(0.9)
# # #             success = st.session_state.retriever.add_documents(embedded_chunks)
# # #
# # #             if success:
# # #                 progress_bar.progress(1.0)
# # #                 status_text.text("✅ Complete!")
# # #
# # #                 st.session_state.documents_loaded = True
# # #                 st.session_state.vector_store_ready = True
# # #                 st.session_state.document_count = len(embedded_chunks)
# # #
# # #                 st.success(f"✅ Processed {total_files} files into {len(embedded_chunks)} chunks")
# # #
# # #                 # Clean up
# # #                 time.sleep(1)
# # #                 progress_bar.empty()
# # #                 status_text.empty()
# # #             else:
# # #                 st.error("❌ Failed to store documents in vector database")
# # #
# # #         except Exception as e:
# # #             st.error(f"❌ Processing Error: {str(e)}")
# # #             logger.error(f"Document processing error: {str(e)}")
# # #
# # #
# # # # def process_uploaded_files_enhanced(uploaded_files):
# # # #     """Enhanced file processing with better feedback"""
# # # #     if not st.session_state.get('document_loader'):
# # # #         st.error("❌ System components not initialized. Please refresh the page.")
# # # #         return
# # # #
# # # #     # Create progress container
# # # #     progress_container = st.container()
# # # #
# # # #     with progress_container:
# # # #         st.markdown("### 🔄 Processing Documents")
# # # #         progress_bar = st.progress(0)
# # # #         status_text = st.empty()
# # # #
# # # #         try:
# # # #             all_chunks = []
# # # #             total_files = len(uploaded_files)
# # # #
# # # #             # Phase 1: Document Loading
# # # #             for i, uploaded_file in enumerate(uploaded_files):
# # # #                 status_text.markdown(f"📖 **Loading:** {uploaded_file.name}")
# # # #                 progress_bar.progress((i + 1) / total_files * 0.3)
# # # #
# # # #                 # Load document
# # # #                 doc_data = st.session_state.document_loader.load_uploaded_file(uploaded_file)
# # # #
# # # #                 # Validate medical content
# # # #                 validation = st.session_state.document_loader.validate_medical_document(doc_data['content'])
# # # #                 if not validation['is_medical']:
# # # #                     st.warning(f"⚠️ {uploaded_file.name} may not contain medical content")
# # # #
# # # #                 # Preprocess
# # # #                 processed = st.session_state.preprocessor.preprocess_document(
# # # #                     doc_data['content'],
# # # #                     anonymize=True
# # # #                 )
# # # #
# # # #                 # Create chunks
# # # #                 chunks = st.session_state.chunking_strategy.create_contextual_chunks(
# # # #                     processed['processed_content'],
# # # #                     doc_data
# # # #                 )
# # # #
# # # #                 all_chunks.extend(chunks)
# # # #
# # # #             # Phase 2: Embedding Generation
# # # #             status_text.markdown("🧠 **Generating embeddings...**")
# # # #             progress_bar.progress(0.6)
# # # #             embedded_chunks = st.session_state.embedding_manager.embed_medical_chunks(all_chunks)
# # # #
# # # #             # Phase 3: Vector Storage
# # # #             status_text.markdown("💾 **Storing in vector database...**")
# # # #             progress_bar.progress(0.8)
# # # #             success = st.session_state.retriever.add_documents(embedded_chunks)
# # # #
# # # #             if success:
# # # #                 progress_bar.progress(1.0)
# # # #                 status_text.markdown("✅ **Processing complete!**")
# # # #
# # # #                 st.session_state.documents_loaded = True
# # # #                 st.session_state.vector_store_ready = True
# # # #                 st.session_state.document_count = len(embedded_chunks)
# # # #
# # # #                 # Success message with details
# # # #                 st.markdown("""
# # # #                 <div class="alert-success">
# # # #                     🎉 <strong>Success!</strong><br>
# # # #                     Processed {} files into {} searchable chunks.
# # # #                 </div>
# # # #                 """.format(total_files, len(embedded_chunks)), unsafe_allow_html=True)
# # # #
# # # #                 st.balloons()
# # # #             else:
# # # #                 st.markdown("""
# # # #                 <div class="alert-error">
# # # #                     ❌ <strong>Error:</strong> Failed to store documents in vector database.
# # # #                 </div>
# # # #                 """, unsafe_allow_html=True)
# # # #
# # # #         except Exception as e:
# # # #             st.markdown(f"""
# # # #             <div class="alert-error">
# # # #                 ❌ <strong>Processing Error:</strong> {str(e)}
# # # #             </div>
# # # #             """, unsafe_allow_html=True)
# # # #             logger.error(f"Document processing error: {str(e)}")
# # #
# # #
# # # def render_processing_interface():
# # #     """Enhanced processing interface"""
# # #     st.markdown("### ⚙️ Processing Configuration")
# # #
# # #     if st.session_state.get('documents_loaded', False):
# # #         st.markdown("""
# # #         <div class="alert-success">
# # #             ✅ Documents are loaded and ready for processing
# # #         </div>
# # #         """, unsafe_allow_html=True)
# # #
# # #         col1, col2 = st.columns(2)
# # #         with col1:
# # #             anonymize_data = st.checkbox(
# # #                 "🔒 Anonymize PHI",
# # #                 value=True,
# # #                 help="Automatically detect and remove personal health information"
# # #             )
# # #         with col2:
# # #             chunk_size = st.slider(
# # #                 "📝 Chunk Size",
# # #                 500, 2000, Settings.CHUNK_SIZE,
# # #                 step=100,
# # #                 help="Size of text chunks for processing"
# # #             )
# # #
# # #         if st.button("🔄 Reprocess with New Settings"):
# # #             st.info("🔧 Reprocessing functionality will be available in the next update")
# # #     else:
# # #         st.markdown("""
# # #         <div class="alert-warning">
# # #             ⚠️ Please upload documents first to configure processing options
# # #         </div>
# # #         """, unsafe_allow_html=True)
# # #
# # #
# # # def render_document_statistics():
# # #     """Enhanced document statistics"""
# # #     if not st.session_state.get('vector_store_ready', False):
# # #         st.markdown("""
# # #         <div class="alert-warning">
# # #             📊 No processed documents yet. Upload and process documents to see statistics.
# # #         </div>
# # #         """, unsafe_allow_html=True)
# # #         return
# # #
# # #     try:
# # #         stats = st.session_state.retriever.get_collection_stats()
# # #
# # #         # Main metrics
# # #         col1, col2, col3 = st.columns(3)
# # #         with col1:
# # #             st.metric("📚 Total Documents", stats.get('total_documents', 0))
# # #         with col2:
# # #             st.metric("🎯 Avg Relevance", f"{stats.get('avg_clinical_relevance', 0):.2f}")
# # #         with col3:
# # #             embedding_stats = st.session_state.embedding_manager.get_embedding_stats()
# # #             st.metric("🧠 Cached Embeddings", embedding_stats.get('cached_embeddings', 0))
# # #
# # #         # Charts (if data available)
# # #         if 'document_types' in stats and stats['document_types']:
# # #             st.markdown("#### 📈 Document Types Distribution")
# # #             st.bar_chart(stats['document_types'])
# # #
# # #         if 'sections' in stats and stats['sections']:
# # #             st.markdown("#### 📋 Document Sections")
# # #             st.bar_chart(stats['sections'])
# # #
# # #     except Exception as e:
# # #         st.markdown(f"""
# # #         <div class="alert-error">
# # #             ❌ Error loading statistics: {str(e)}
# # #         </div>
# # #         """, unsafe_allow_html=True)
# # #
# # #
# # # def chat_interface():
# # #     """Streamlined chat interface"""
# # #     st.markdown("## 💬 Medical Assistant")
# # #
# # #     # Quick prerequisite check
# # #     if not all([
# # #         st.session_state.get('api_key_valid', False),
# # #         st.session_state.get('vector_store_ready', False)
# # #     ]):
# # #         st.info("🔧 Please configure API key and upload documents in the sidebar and Documents tab")
# # #         return
# # #
# # #     # Compact query interface
# # #     render_enhanced_query_interface()
# # #
# # #     # Show only recent chat history to reduce lag
# # #     render_compact_chat_history()
# # #
# # #
# # # def render_simple_query_interface():
# # #     """Simple query interface using forms to avoid key conflicts"""
# # #     st.markdown("### 🔍 Ask a Medical Question")
# # #
# # #     # Main query form
# # #     with st.form("medical_query_form", clear_on_submit=False):
# # #         col1, col2 = st.columns([3, 1])
# # #
# # #         with col1:
# # #             user_query = st.text_input(
# # #                 "Enter your question:",
# # #                 placeholder="e.g., What are the side effects of metformin?"
# # #             )
# # #
# # #         with col2:
# # #             query_type = st.selectbox(
# # #                 "Query Type",
# # #                 ["general", "diagnosis", "treatment", "drug_interaction", "guidelines"]
# # #             )
# # #
# # #         submitted = st.form_submit_button("🔍 Ask Assistant", type="primary")
# # #
# # #         if submitted and user_query:
# # #             if validate_medical_query_enhanced(user_query):
# # #                 process_medical_query_enhanced(user_query, query_type)
# # #             else:
# # #                 st.warning("⚠️ Please ask a more specific medical question.")
# # #
# # #     # Suggestions outside the form
# # #     if st.session_state.get('documents_loaded'):
# # #         st.markdown("**💡 Try these questions:**")
# # #         suggestions = [
# # #             "What are the side effects of metformin?",
# # #             "Treatment guidelines for hypertension in elderly patients",
# # #             "Drug interactions between warfarin and NSAIDs"
# # #         ]
# # #
# # #         for i, suggestion in enumerate(suggestions):
# # #             if st.button(f"💭 {suggestion}", key=f"suggest_{i}", use_container_width=True):
# # #                 # Store suggestion and rerun to populate form
# # #                 st.session_state.suggested_query = suggestion
# # #                 st.rerun()
# # #
# # #
# # # def render_compact_chat_history():
# # #     """Compact chat history display"""
# # #     if st.session_state.get('chat_history'):
# # #         with st.expander("📜 Recent Queries", expanded=False):
# # #             # Show only last 3 conversations
# # #             for chat in st.session_state.chat_history[-3:]:
# # #                 st.markdown(f"**Q:** {chat['query'][:50]}...")
# # #                 st.markdown(f"**A:** {chat['response'][:100]}...")
# # #                 st.caption(f"🕒 {chat['timestamp']} | 📚 {chat['source_count']} sources")
# # #                 st.divider()
# # #
# # #
# # # # 8. MEMORY CLEANUP FUNCTION
# # # def cleanup_session_state():
# # #     """Periodic cleanup to prevent memory bloat"""
# # #     # Limit chat history
# # #     if len(st.session_state.get('chat_history', [])) > 15:
# # #         st.session_state.chat_history = st.session_state.chat_history[-15:]
# # #
# # #     # Clear old cache entries
# # #     if hasattr(st.session_state, 'cache_access_times'):
# # #         current_time = time.time()
# # #         old_keys = [
# # #             key for key, timestamp in st.session_state.cache_access_times.items()
# # #             if current_time - timestamp > 3600  # 1 hour
# # #         ]
# # #         for key in old_keys:
# # #             st.session_state.query_cache.pop(key, None)
# # #             st.session_state.cache_access_times.pop(key, None)
# # #
# # #
# # # # def chat_interface():
# # # #     """Enhanced chat interface"""
# # # #     st.markdown("## 💬 Medical Assistant")
# # # #
# # # #     # Check prerequisites
# # # #     if not st.session_state.get('api_key_valid', False):
# # # #         st.markdown("""
# # # #         <div class="alert-warning">
# # # #             🔑 <strong>API Key Required:</strong> Please configure and validate your Groq API key in the sidebar to start chatting.
# # # #         </div>
# # # #         """, unsafe_allow_html=True)
# # # #         return
# # # #
# # # #     if not st.session_state.get('vector_store_ready', False):
# # # #         st.markdown("""
# # # #         <div class="alert-warning">
# # # #             📄 <strong>Documents Required:</strong> Please upload and process medical documents before asking questions.
# # # #         </div>
# # # #         """, unsafe_allow_html=True)
# # # #         return
# # # #
# # # #     # Enhanced query interface
# # # #     render_enhanced_query_interface()
# # # #
# # # #     # Display chat history
# # # #     render_chat_history()
# # #
# # #
# # # # def render_enhanced_query_interface():
# # # #     """Enhanced query interface with smart features"""
# # # #     st.markdown("### 🔍 Ask a Medical Question")
# # # #
# # # #     # Query input with suggestions
# # # #     col1, col2 = st.columns([3, 1])
# # # #
# # # #     with col1:
# # # #         user_query = st.text_input(
# # # #             "Enter your question:",
# # # #             placeholder="e.g., What are the treatment options for type 2 diabetes in elderly patients?",
# # # #             key="user_query",
# # # #             help="Ask specific medical questions for better results"
# # # #         )
# # # #
# # # #     with col2:
# # # #         query_type = st.selectbox(
# # # #             "Query Type",
# # # #             ["general", "diagnosis", "treatment", "drug_interaction", "guidelines"],
# # # #             help="Select query type for more precise responses"
# # # #         )
# # # #
# # # #     # Smart suggestions (if documents are loaded)
# # # #     if st.session_state.get('documents_loaded'):
# # # #         render_query_suggestions()
# # # #
# # # #     # Process query with validation
# # # #     if user_query and st.button("🔍 Ask Assistant", type="primary"):
# # # #         if validate_medical_query_enhanced(user_query):
# # # #             process_medical_query_enhanced(user_query, query_type)
# # # #         else:
# # # #             st.markdown("""
# # # #             <div class="alert-warning">
# # # #                 ⚠️ <strong>Tip:</strong> Please ask a more specific medical question for better results.
# # # #                 Include medical terms, conditions, or treatments in your question.
# # # #             </div>
# # # #             """, unsafe_allow_html=True)
# # # # def render_enhanced_query_interface():
# # # #     """Enhanced query interface with smart features - FIXED VERSION"""
# # # #     st.markdown("### 🔍 Ask a Medical Question")
# # # #
# # # #     # Handle pending suggestion from button clicks
# # # #     default_query = ""
# # # #     if "selected_suggestion" in st.session_state:
# # # #         default_query = st.session_state.selected_suggestion
# # # #         del st.session_state.selected_suggestion  # Clear it after using
# # # #
# # # #     # Query input with suggestions
# # # #     col1, col2 = st.columns([3, 1])
# # # #
# # # #     with col1:
# # # #         user_query = st.text_input(
# # # #             "Enter your question:",
# # # #             value=default_query,  # Use the selected suggestion as default
# # # #             placeholder="e.g., What are the treatment options for type 2 diabetes in elderly patients?",
# # # #             key="user_query",
# # # #             help="Ask specific medical questions for better results"
# # # #         )
# # # #
# # # #     with col2:
# # # #         query_type = st.selectbox(
# # # #             "Query Type",
# # # #             ["general", "diagnosis", "treatment", "drug_interaction", "guidelines"],
# # # #             help="Select query type for more precise responses"
# # # #         )
# # # #
# # # #     # Smart suggestions (if documents are loaded)
# # # #     if st.session_state.get('documents_loaded'):
# # # #         render_query_suggestions()
# # # #
# # # #     # Process query with validation
# # # #     if user_query and st.button("🔍 Ask Assistant", type="primary"):
# # # #         if validate_medical_query_enhanced(user_query):
# # # #             process_medical_query_enhanced(user_query, query_type)
# # # #         else:
# # # #             st.markdown("""
# # # #             <div class="alert-warning">
# # # #                 ⚠️ <strong>Tip:</strong> Please ask a more specific medical question for better results.
# # # #                 Include medical terms, conditions, or treatments in your question.
# # # #             </div>
# # # #             """, unsafe_allow_html=True)
# # #
# # #
# # # def render_enhanced_query_interface():
# # #     """Enhanced query interface with smart features - FIXED VERSION with unique keys"""
# # #     st.markdown("### 🔍 Ask a Medical Question")
# # #
# # #     # Handle pending suggestion from button clicks
# # #     default_query = ""
# # #     if "selected_suggestion" in st.session_state:
# # #         default_query = st.session_state.selected_suggestion
# # #         del st.session_state.selected_suggestion  # Clear it after using
# # #
# # #     # Query input with suggestions
# # #     col1, col2 = st.columns([3, 1])
# # #
# # #     with col1:
# # #         user_query = st.text_input(
# # #             "Enter your question:",
# # #             value=default_query,  # Use the selected suggestion as default
# # #             placeholder="e.g., What are the treatment options for type 2 diabetes in elderly patients?",
# # #             key="chat_user_query",  # Changed from "user_query" to avoid conflicts
# # #             help="Ask specific medical questions for better results"
# # #         )
# # #
# # #     with col2:
# # #         query_type = st.selectbox(
# # #             "Query Type",
# # #             ["general", "diagnosis", "treatment", "drug_interaction", "guidelines"],
# # #             help="Select query type for more precise responses",
# # #             key="chat_query_type"  # Unique key
# # #         )
# # #
# # #     # Smart suggestions (if documents are loaded)
# # #     if st.session_state.get('documents_loaded'):
# # #         render_query_suggestions()
# # #
# # #     # Process query with validation
# # #     if user_query and st.button("🔍 Ask Assistant", type="primary", key="chat_ask_button"):
# # #         if validate_medical_query_enhanced(user_query):
# # #             process_medical_query_enhanced(user_query, query_type)
# # #         else:
# # #             st.markdown("""
# # #             <div class="alert-warning">
# # #                 ⚠️ <strong>Tip:</strong> Please ask a more specific medical question for better results.
# # #                 Include medical terms, conditions, or treatments in your question.
# # #             </div>
# # #             """, unsafe_allow_html=True)
# # #
# # #
# # # def validate_medical_query_enhanced(query: str) -> bool:
# # #     """Enhanced medical query validation"""
# # #     if len(query.split()) < 4:
# # #         return False
# # #
# # #     medical_terms = [
# # #         'treatment', 'diagnosis', 'medication', 'patient', 'clinical', 'therapy',
# # #         'condition', 'disease', 'symptoms', 'drug', 'dosage', 'side effects'
# # #     ]
# # #
# # #     return any(term in query.lower() for term in medical_terms)
# # #
# # #
# # # # def render_query_suggestions():
# # # #     """Render smart query suggestions"""
# # # #     suggestions = [
# # # #         "What are the side effects of metformin?",
# # # #         "Treatment guidelines for hypertension in elderly patients",
# # # #         "Drug interactions between warfarin and NSAIDs"
# # # #     ]
# # # #
# # # #     st.markdown("**💡 Suggested Questions:**")
# # # #     cols = st.columns(len(suggestions))
# # # #
# # # #     for i, suggestion in enumerate(suggestions):
# # # #         with cols[i]:
# # # #             if st.button(f"💭 {suggestion[:30]}...", key=f"suggestion_{i}"):
# # # #                 st.session_state.user_query = suggestion
# # # #                 st.rerun()
# # # def render_query_suggestions():
# # #     """Render smart query suggestions with proper state management and unique keys"""
# # #     suggestions = [
# # #         "What are the side effects of metformin?",
# # #         "Treatment guidelines for hypertension in elderly patients",
# # #         "Drug interactions between warfarin and NSAIDs"
# # #     ]
# # #
# # #     st.markdown("**💡 Suggested Questions:**")
# # #     cols = st.columns(len(suggestions))
# # #
# # #     for i, suggestion in enumerate(suggestions):
# # #         with cols[i]:
# # #             if st.button(f"💭 {suggestion[:30]}...", key=f"chat_suggestion_{i}"):  # Unique keys
# # #                 # Store the suggestion for the next rerun instead of directly modifying
# # #                 st.session_state.selected_suggestion = suggestion
# # #                 st.rerun()
# # #
# # #
# # # def display_cached_response(cache_data: dict):
# # #     """Display cached response efficiently"""
# # #     response_data = cache_data['response_data']
# # #     search_results = cache_data['search_results']
# # #     processing_time = cache_data['processing_time']
# # #
# # #     # Show cache age
# # #     cache_age_minutes = (time.time() - cache_data['timestamp']) / 60
# # #     st.caption(f"📋 Cached response ({cache_age_minutes:.1f} minutes ago)")
# # #
# # #     display_medical_response_enhanced("", response_data, search_results, processing_time)
# # #
# # #
# # # def process_medical_query_enhanced(query: str, query_type: str):
# # #     """Optimized query processing with caching and progress indicators"""
# # #
# # #     # Check cache first
# # #     cached_response = query_cache.get(query, query_type)
# # #     if cached_response:
# # #         st.info("⚡ Retrieved from cache (instant response)")
# # #         display_cached_response(cached_response)
# # #         return
# # #
# # #     # Create compact progress indicator
# # #     progress_container = st.container()
# # #
# # #     with progress_container:
# # #         progress_bar = st.progress(0)
# # #         status_text = st.empty()
# # #
# # #         try:
# # #             start_time = time.time()
# # #
# # #             # Step 1: Validate components (5%)
# # #             status_text.text("🔍 Initializing...")
# # #             progress_bar.progress(5)
# # #
# # #             if not hasattr(st.session_state, 'response_generator'):
# # #                 st.error("❌ Response generator not initialized. Please validate your API key.")
# # #                 return
# # #
# # #             # Step 2: Create embedding (20%)
# # #             status_text.text("🧠 Processing query...")
# # #             progress_bar.progress(20)
# # #
# # #             try:
# # #                 # Use cached embedding
# # #                 query_embedding = cached_embedding_generation(query, query_type)
# # #             except Exception as e:
# # #                 st.error(f"❌ Failed to create query embedding: {str(e)}")
# # #                 return
# # #
# # #             # Step 3: Search documents (50%)
# # #             status_text.text("📚 Searching documents...")
# # #             progress_bar.progress(50)
# # #
# # #             try:
# # #                 # Use cached search if possible
# # #                 embedding_str = json.dumps(query_embedding)
# # #                 query_hash = hashlib.md5(embedding_str.encode()).hexdigest()[:12]
# # #                 search_results = cached_search_results(query_hash, embedding_str, 3)
# # #
# # #                 if not search_results:
# # #                     st.warning("🔍 No relevant documents found")
# # #                     return
# # #             except Exception as e:
# # #                 st.error(f"❌ Document retrieval failed: {str(e)}")
# # #                 return
# # #
# # #             # Step 4: Generate response (80%)
# # #             status_text.text("💭 Generating response...")
# # #             progress_bar.progress(80)
# # #
# # #             try:
# # #                 response_data = st.session_state.response_generator.generate_response(
# # #                     query, search_results, query_type, max_tokens=600  # Reduced for faster response
# # #                 )
# # #
# # #                 if 'error' in response_data:
# # #                     st.error(f"❌ Response generation error: {response_data['error']}")
# # #                     return
# # #
# # #                 processing_time = time.time() - start_time
# # #
# # #                 # Step 5: Complete (100%)
# # #                 progress_bar.progress(100)
# # #                 status_text.text("✅ Complete!")
# # #
# # #                 # Display response
# # #                 display_medical_response_enhanced(query, response_data, search_results, processing_time)
# # #                 save_to_chat_history(query, response_data, query_type, len(search_results))
# # #
# # #                 # Cache the response
# # #                 cache_data = {
# # #                     'response_data': response_data,
# # #                     'search_results': search_results,
# # #                     'processing_time': processing_time,
# # #                     'timestamp': time.time()
# # #                 }
# # #                 query_cache.set(query, query_type, cache_data)
# # #
# # #                 # Clean up progress indicators
# # #                 time.sleep(0.3)
# # #                 progress_bar.empty()
# # #                 status_text.empty()
# # #
# # #             except Exception as e:
# # #                 st.error(f"❌ Response generation failed: {str(e)}")
# # #
# # #         except Exception as e:
# # #             st.error(f"❌ Query processing failed: {str(e)}")
# # #             logger.error(f"Query processing error: {str(e)}")
# # #         finally:
# # #             # Always clean up progress indicators
# # #             progress_bar.empty()
# # #             status_text.empty()
# # #
# # # # def process_medical_query_enhanced(query: str, query_type: str):
# # # #     """Fixed version with better error handling and debugging"""
# # # #
# # # #     # Debug info
# # # #     debug_info = st.expander("🔧 Debug Information", expanded=False)
# # # #
# # # #     with st.spinner("🔍 Analyzing medical literature..."):
# # # #         try:
# # # #             start_time = time.time()
# # # #
# # # #             with debug_info:
# # # #                 st.write("**Debug Log:**")
# # # #                 debug_log = []
# # # #
# # # #             # Step 1: Validate components
# # # #             if not hasattr(st.session_state, 'response_generator'):
# # # #                 st.error("❌ Response generator not initialized. Please validate your API key.")
# # # #                 return
# # # #
# # # #             debug_log.append("✅ Response generator found")
# # # #
# # # #             # Step 2: Validate query (with fallback)
# # # #             try:
# # # #                 validation = st.session_state.response_generator.validate_medical_query(query)
# # # #                 debug_log.append("✅ Query validated")
# # # #
# # # #                 if validation.get('warnings'):
# # # #                     for warning in validation['warnings']:
# # # #                         st.markdown(f"""
# # # #                         <div class="alert-warning">
# # # #                             ⚠️ {warning}
# # # #                         </div>
# # # #                         """, unsafe_allow_html=True)
# # # #             except Exception as e:
# # # #                 debug_log.append(f"⚠️ Query validation failed: {str(e)}")
# # # #                 validation = {'warnings': []}
# # # #
# # # #             # Step 3: Create query embedding
# # # #             try:
# # # #                 query_embedding = st.session_state.embedding_manager.create_medical_query_embedding(
# # # #                     query, query_type
# # # #                 )
# # # #                 debug_log.append(f"✅ Embedding created (size: {len(query_embedding)})")
# # # #             except Exception as e:
# # # #                 debug_log.append(f"❌ Embedding failed: {str(e)}")
# # # #                 st.error(f"❌ Failed to create query embedding: {str(e)}")
# # # #                 return
# # # #
# # # #             # Step 4: Retrieve documents (simplified)
# # # #             try:
# # # #                 # Use simple search instead of context-aware search for debugging
# # # #                 search_results = st.session_state.retriever.search(
# # # #                     query_embedding, n_results=3
# # # #                 )
# # # #                 debug_log.append(f"✅ Retrieved {len(search_results)} documents")
# # # #
# # # #                 if not search_results:
# # # #                     st.markdown("""
# # # #                     <div class="alert-warning">
# # # #                         🔍 <strong>No Relevant Information Found:</strong><br>
# # # #                         No relevant documents found for your query. Try:
# # # #                         <ul>
# # # #                             <li>Using different medical terms</li>
# # # #                             <li>Uploading more relevant medical literature</li>
# # # #                             <li>Making your question more specific</li>
# # # #                         </ul>
# # # #                     </div>
# # # #                     """, unsafe_allow_html=True)
# # # #                     return
# # # #
# # # #                 # Show similarity scores in debug
# # # #                 similarities = [r.get('similarity_score', 0) for r in search_results]
# # # #                 debug_log.append(f"Similarity scores: {similarities}")
# # # #
# # # #             except Exception as e:
# # # #                 debug_log.append(f"❌ Retrieval failed: {str(e)}")
# # # #                 st.error(f"❌ Document retrieval failed: {str(e)}")
# # # #                 return
# # # #
# # # #             # Step 5: Generate response
# # # #             try:
# # # #                 model = st.session_state.get('selected_model', 'llama3-8b-8192')
# # # #                 debug_log.append(f"Using model: {model}")
# # # #
# # # #                 # Update response generator model if changed
# # # #                 if hasattr(st.session_state.response_generator,
# # # #                            'model') and st.session_state.response_generator.model != model:
# # # #                     st.session_state.response_generator.model = model
# # # #                     debug_log.append(f"Updated model to: {model}")
# # # #
# # # #                 response_data = st.session_state.response_generator.generate_response(
# # # #                     query, search_results, query_type, max_tokens=800
# # # #                 )
# # # #
# # # #                 processing_time = time.time() - start_time
# # # #                 debug_log.append(f"✅ Response generated in {processing_time:.2f}s")
# # # #
# # # #                 # Display response
# # # #                 if 'error' in response_data:
# # # #                     st.error(f"❌ Response generation error: {response_data['error']}")
# # # #                     debug_log.append(f"❌ Response error: {response_data['error']}")
# # # #                 else:
# # # #                     display_medical_response_enhanced(query, response_data, search_results, processing_time)
# # # #                     save_to_chat_history(query, response_data, query_type, len(search_results))
# # # #                     debug_log.append("✅ Response displayed successfully")
# # # #
# # # #             except Exception as e:
# # # #                 debug_log.append(f"❌ Response generation failed: {str(e)}")
# # # #                 st.error(f"❌ Response generation failed: {str(e)}")
# # # #
# # # #                 # Show helpful error message
# # # #                 st.markdown("""
# # # #                 <div class="alert-error">
# # # #                     <strong>Troubleshooting Tips:</strong><br>
# # # #                     • Check if your API key is valid<br>
# # # #                     • Try a simpler question<br>
# # # #                     • Ensure documents are properly loaded<br>
# # # #                     • Check your internet connection
# # # #                 </div>
# # # #                 """, unsafe_allow_html=True)
# # # #
# # # #             # Update debug info
# # # #             with debug_info:
# # # #                 for log_entry in debug_log:
# # # #                     st.write(log_entry)
# # # #
# # # #         except Exception as e:
# # # #             st.error(f"❌ Query processing failed: {str(e)}")
# # # #             logger.error(f"Query processing error: {str(e)}")
# # #
# # #
# # # def display_medical_response_enhanced(query: str, response_data: Dict, search_results: list, processing_time: float):
# # #     """Enhanced medical response display with clean formatting"""
# # #
# # #     if 'error' in response_data:
# # #         st.markdown(f"""
# # #         <div class="alert-error">
# # #             ❌ <strong>Response Error:</strong> {response_data.get('response', 'Unknown error occurred')}
# # #         </div>
# # #         """, unsafe_allow_html=True)
# # #         return
# # #
# # #     response = response_data['response']
# # #
# # #     # Main Response Section
# # #     st.markdown("### 🔬 Medical Response")
# # #
# # #     # Confidence and quality indicators
# # #     render_response_quality_bar(response)
# # #
# # #     # Main content in a clean container
# # #     with st.container():
# # #         st.markdown(response['main_response'])
# # #
# # #     # Additional details in expandable sections
# # #     render_expandable_response_details_enhanced(response, search_results, processing_time, response_data)
# # #
# # #
# # # def render_response_quality_bar(response: Dict):
# # #     """Render response quality indicators"""
# # #     confidence = response.get('confidence_assessment', {})
# # #     evidence = response.get('evidence_level', {})
# # #
# # #     col1, col2, col3 = st.columns(3)
# # #
# # #     with col1:
# # #         confidence_level = confidence.get('level', 'low')
# # #         confidence_colors = {
# # #             'high': '🟢 High',
# # #             'medium': '🟡 Medium',
# # #             'low': '🔴 Low'
# # #         }
# # #         st.markdown(f"**Confidence:** {confidence_colors.get(confidence_level, '🔴 Unknown')}")
# # #
# # #     with col2:
# # #         evidence_level = evidence.get('level', 'unknown').upper()
# # #         st.markdown(f"**Evidence:** {evidence_level}")
# # #
# # #     with col3:
# # #         source_count = len(response.get('sources_used', []))
# # #         st.markdown(f"**Sources:** {source_count}")
# # #
# # #
# # # def render_expandable_response_details_enhanced(response: Dict, search_results: list, processing_time: float,
# # #                                                 response_data: Dict):
# # #     """Enhanced version that includes response_data for technical details"""
# # #
# # #     # Clinical Alerts - Show prominently if critical
# # #     alerts = response.get('clinical_alerts', [])
# # #     critical_alerts = [alert for alert in alerts if alert['type'] in ['contraindication', 'emergency']]
# # #
# # #     if critical_alerts:
# # #         st.markdown("### 🚨 Critical Clinical Alerts")
# # #         for alert in critical_alerts:
# # #             st.markdown(f"""
# # #             <div class="alert-error">
# # #                 🚨 <strong>{alert['type'].title()}:</strong> {alert['message']}
# # #             </div>
# # #             """, unsafe_allow_html=True)
# # #
# # #     # Other alerts and details in expandable sections
# # #     other_alerts = [alert for alert in alerts if alert['type'] not in ['contraindication', 'emergency']]
# # #     if other_alerts:
# # #         with st.expander("⚠️ Additional Clinical Considerations", expanded=False):
# # #             for alert in other_alerts:
# # #                 st.warning(f"**{alert['type'].title()}:** {alert['message']}")
# # #
# # #     # Evidence Sources - Collapsible
# # #     sources = response.get('sources_used', [])
# # #     if sources:
# # #         with st.expander(f"📚 Evidence Sources ({len(sources)})", expanded=False):
# # #             for i, source in enumerate(sources, 1):
# # #                 st.markdown(f"**Source {i}:** {source['document']}")
# # #                 st.markdown(f"*Section:* {source['section']}")
# # #                 st.markdown(f"*Relevance Score:* {source['relevance_score']:.2f}")
# # #
# # #                 # Medical context indicators
# # #                 context_indicators = []
# # #                 if source.get('contains_diagnosis'):
# # #                     context_indicators.append("🔍 Diagnosis")
# # #                 if source.get('contains_treatment'):
# # #                     context_indicators.append("💊 Treatment")
# # #                 if source.get('contains_symptoms'):
# # #                     context_indicators.append("🩺 Symptoms")
# # #                 if source.get('contains_procedures'):
# # #                     context_indicators.append("🏥 Procedures")
# # #
# # #                 if context_indicators:
# # #                     st.markdown(f"*Contains:* {' | '.join(context_indicators)}")
# # #
# # #                 # Clinical relevance indicator
# # #                 clinical_relevance = source.get('clinical_relevance', 0)
# # #                 if clinical_relevance > 0:
# # #                     st.markdown(f"*Clinical Relevance:* {clinical_relevance:.2f}")
# # #
# # #                 if i < len(sources):
# # #                     st.divider()
# # #
# # #     # Follow-up Recommendations
# # #     suggestions = response.get('follow_up_suggestions', [])
# # #     if suggestions:
# # #         with st.expander("📋 Follow-up Recommendations", expanded=False):
# # #             for suggestion in suggestions:
# # #                 st.markdown(f"• {suggestion}")
# # #
# # #     # Evidence Level Details
# # #     evidence_level = response.get('evidence_level', {})
# # #     if evidence_level and evidence_level.get('level') != 'insufficient':
# # #         with st.expander("📊 Evidence Assessment", expanded=False):
# # #             st.markdown(f"**Evidence Level:** {evidence_level.get('level', 'Unknown').upper()}")
# # #             st.markdown(f"**Description:** {evidence_level.get('description', 'No description available')}")
# # #
# # #             # Source distribution if available
# # #             source_dist = evidence_level.get('source_distribution', {})
# # #             if source_dist:
# # #                 st.markdown("**Source Types:**")
# # #                 for source_type, count in source_dist.items():
# # #                     st.markdown(f"- {source_type.replace('_', ' ').title()}: {count}")
# # #
# # #     # Confidence Assessment Details
# # #     confidence = response.get('confidence_assessment', {})
# # #     if confidence:
# # #         with st.expander("🎯 Confidence Analysis", expanded=False):
# # #             st.markdown(f"**Confidence Level:** {confidence.get('level', 'Unknown').title()}")
# # #             st.markdown(f"**Reasoning:** {confidence.get('reason', 'No reasoning provided')}")
# # #
# # #             # Additional confidence metrics if available
# # #             if 'avg_relevance' in confidence:
# # #                 st.markdown(f"**Average Source Relevance:** {confidence['avg_relevance']:.2f}")
# # #             if 'high_relevance_count' in confidence:
# # #                 st.markdown(f"**High Relevance Sources:** {confidence['high_relevance_count']}")
# # #             if 'total_sources' in confidence:
# # #                 st.markdown(f"**Total Sources Analyzed:** {confidence['total_sources']}")
# # #
# # #     # Technical Details - Hidden by default
# # #     with st.expander("🔧 Technical Details", expanded=False):
# # #         tech_col1, tech_col2, tech_col3 = st.columns(3)
# # #
# # #         with tech_col1:
# # #             st.metric("Processing Time", f"{processing_time:.2f}s")
# # #
# # #         with tech_col2:
# # #             st.metric("Sources Retrieved", len(search_results))
# # #
# # #         with tech_col3:
# # #             # Now we can access tokens from response_data
# # #             tokens_used = response_data.get('tokens_used')
# # #             if tokens_used:
# # #                 st.metric("Tokens Used", tokens_used)
# # #             else:
# # #                 model_used = response_data.get('model_used', 'Unknown')
# # #                 st.metric("Model Used", model_used)
# # #
# # #         # Additional technical information
# # #         st.markdown("**Search Results Summary:**")
# # #         if search_results:
# # #             avg_similarity = sum(result.get('similarity_score', 0) for result in search_results) / len(search_results)
# # #             st.markdown(f"- Average Similarity Score: {avg_similarity:.3f}")
# # #             st.markdown(f"- Top Result Similarity: {search_results[0].get('similarity_score', 0):.3f}")
# # #
# # #             # Show distribution of similarity scores
# # #             high_sim_count = sum(1 for result in search_results if result.get('similarity_score', 0) > 0.8)
# # #             med_sim_count = sum(1 for result in search_results if 0.6 <= result.get('similarity_score', 0) <= 0.8)
# # #             low_sim_count = len(search_results) - high_sim_count - med_sim_count
# # #
# # #             st.markdown(f"- High Similarity (>0.8): {high_sim_count}")
# # #             st.markdown(f"- Medium Similarity (0.6-0.8): {med_sim_count}")
# # #             st.markdown(f"- Lower Similarity (<0.6): {low_sim_count}")
# # #         else:
# # #             st.markdown("- No search results available")
# # #
# # #         # Response generation details
# # #         st.markdown("**Response Details:**")
# # #         st.markdown(f"- Model: {response_data.get('model_used', 'Unknown')}")
# # #         st.markdown(f"- Query Type: {response_data.get('query_type', 'Unknown')}")
# # #         st.markdown(f"- Response Time: {response_data.get('response_time', processing_time):.2f}s")
# # #
# # #         if tokens_used:
# # #             st.markdown(f"- Tokens Used: {tokens_used}")
# # #
# # #         # Query processing pipeline status
# # #         st.markdown("**Processing Pipeline:**")
# # #         st.markdown("1. ✅ Query validation completed")
# # #         st.markdown("2. ✅ Medical context analysis completed")
# # #         st.markdown("3. ✅ Embedding generation completed")
# # #         st.markdown("4. ✅ Vector similarity search completed")
# # #         st.markdown("5. ✅ Context preparation completed")
# # #         st.markdown("6. ✅ AI response generation completed")
# # #         st.markdown("7. ✅ Medical validation and alerts completed")
# # #
# # #     # Help and Tips Section
# # #     with st.expander("💡 Tips for Better Results", expanded=False):
# # #         st.markdown("**To get more accurate responses:**")
# # #         st.markdown("• Be specific about patient demographics (age, gender)")
# # #         st.markdown("• Include relevant medical history and comorbidities")
# # #         st.markdown("• Specify the type of information you need (diagnosis, treatment, etc.)")
# # #         st.markdown("• Use standard medical terminology when possible")
# # #         st.markdown("• Include severity levels or stages when relevant")
# # #
# # #         st.markdown("**For drug-related queries:**")
# # #         st.markdown("• Include generic names when available")
# # #         st.markdown("• Mention patient conditions and known allergies")
# # #         st.markdown("• Specify dosages if relevant to the question")
# # #         st.markdown("• Include duration of treatment if applicable")
# # #
# # #         st.markdown("**For diagnostic queries:**")
# # #         st.markdown("• Describe presenting symptoms clearly")
# # #         st.markdown("• Include relevant lab values or test results")
# # #         st.markdown("• Mention differential diagnoses you're considering")
# # #
# # #         st.markdown("**Remember:**")
# # #         st.markdown("• This tool provides evidence-based guidance only")
# # #         st.markdown("• Always apply your professional clinical judgment")
# # #         st.markdown("• Verify critical information with current medical sources")
# # #         st.markdown("• Consult colleagues or specialists when needed")
# # #         st.markdown("• Consider patient-specific factors not captured in the query")
# # #
# # #
# # # def render_chat_history():
# # #     """Render clean chat history"""
# # #     if not st.session_state.get('chat_history'):
# # #         return
# # #
# # #     st.markdown("### 📜 Recent Conversations")
# # #
# # #     # Show last 3 conversations
# # #     recent_chats = list(reversed(st.session_state.chat_history[-3:]))
# # #
# # #     for i, chat in enumerate(recent_chats):
# # #         with st.expander(f"💬 {chat['query'][:60]}...", expanded=(i == 0)):
# # #             # Query
# # #             st.markdown("**Question:**")
# # #             st.markdown(f"*{chat['query']}*")
# # #
# # #             # Response
# # #             st.markdown("**Response:**")
# # #             st.markdown(chat['response'][:300] + "..." if len(chat['response']) > 300 else chat['response'])
# # #
# # #             # Metadata
# # #             col1, col2, col3 = st.columns(3)
# # #             with col1:
# # #                 st.caption(f"🕒 {chat['timestamp']}")
# # #             with col2:
# # #                 st.caption(f"📚 {chat['source_count']} sources")
# # #             with col3:
# # #                 st.caption(f"🏷️ {chat['query_type']}")
# # #
# # #
# # # def save_to_chat_history(query: str, response_data: Dict, query_type: str, source_count: int):
# # #     """Save conversation to chat history"""
# # #     chat_entry = {
# # #         'query': query,
# # #         'response': response_data['response']['main_response'],
# # #         'source_count': source_count,
# # #         'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
# # #         'query_type': query_type
# # #     }
# # #
# # #     st.session_state.chat_history.append(chat_entry)
# # #
# # #     # Keep only last 10 conversations
# # #     if len(st.session_state.chat_history) > 10:
# # #         st.session_state.chat_history = st.session_state.chat_history[-10:]
# # #
# # #
# # # def medical_tools_section():
# # #     """Enhanced medical tools section"""
# # #     st.markdown("## 🧰 Medical Tools")
# # #
# # #     # Tool selection with descriptions
# # #     tool_choice = st.selectbox(
# # #         "Choose a Medical Tool",
# # #         [
# # #             "💊 Drug Interaction Checker",
# # #             "🏷️ Medical Code Mapper",
# # #             "📊 Clinical Guidelines Finder",
# # #             "📋 Document Analysis"
# # #         ],
# # #         help="Select a medical tool for specialized analysis"
# # #     )
# # #
# # #     # Render selected tool
# # #     if "Drug Interaction" in tool_choice:
# # #         render_enhanced_drug_tool()
# # #     elif "Medical Code" in tool_choice:
# # #         render_enhanced_terminology_tool()
# # #     elif "Guidelines" in tool_choice:
# # #         render_guidelines_tool()
# # #     elif "Document Analysis" in tool_choice:
# # #         render_document_analysis_tool()
# # #
# # #
# # # def render_enhanced_drug_tool():
# # #     """Enhanced drug interaction checker"""
# # #     st.markdown("### 💊 Drug Interaction Checker")
# # #     st.markdown("*Check for potential drug interactions and contraindications*")
# # #
# # #     col1, col2 = st.columns([2, 1])
# # #
# # #     with col1:
# # #         medication_input = st.text_area(
# # #             "Enter medications (one per line):",
# # #             placeholder="warfarin\naspirin\nmetformin\nlisinopril",
# # #             height=120,
# # #             help="Enter each medication on a new line"
# # #         )
# # #
# # #     with col2:
# # #         patient_conditions = st.multiselect(
# # #             "Patient Conditions:",
# # #             [
# # #                 "pregnancy", "severe_hepatic_impairment", "severe_renal_impairment",
# # #                 "asthma", "heart_failure", "peptic_ulcer", "diabetes", "hypertension"
# # #             ],
# # #             help="Select relevant patient conditions"
# # #         )
# # #
# # #     if medication_input and st.button("🔍 Check Interactions", type="primary"):
# # #         check_drug_interactions_enhanced(medication_input, patient_conditions)
# # #
# # #
# # # def check_drug_interactions_enhanced(medication_input: str, patient_conditions: list):
# # #     """Enhanced drug interaction checking with better display"""
# # #     try:
# # #         # Parse medications
# # #         medications = [med.strip() for med in medication_input.split('\n') if med.strip()]
# # #
# # #         if len(medications) < 2:
# # #             st.markdown("""
# # #             <div class="alert-warning">
# # #                 ⚠️ Please enter at least 2 medications to check for interactions.
# # #             </div>
# # #             """, unsafe_allow_html=True)
# # #             return
# # #
# # #         with st.spinner("🔍 Analyzing drug interactions..."):
# # #             analysis = st.session_state.drug_checker.analyze_medication_list(
# # #                 '\n'.join(medications), patient_conditions
# # #             )
# # #
# # #         # Display results with enhanced formatting
# # #         render_drug_analysis_results(analysis, medications)
# # #
# # #     except Exception as e:
# # #         st.markdown(f"""
# # #         <div class="alert-error">
# # #             ❌ <strong>Analysis Error:</strong> {str(e)}
# # #         </div>
# # #         """, unsafe_allow_html=True)
# # #
# # #
# # # def render_drug_analysis_results(analysis: Dict, medications: list):
# # #     """Render drug analysis results with clean formatting"""
# # #
# # #     # Summary metrics
# # #     st.markdown("#### 📊 Analysis Summary")
# # #     col1, col2, col3, col4 = st.columns(4)
# # #
# # #     with col1:
# # #         st.metric("💊 Medications", len(medications))
# # #     with col2:
# # #         st.metric("⚠️ Interactions", len(analysis.get('interactions', [])))
# # #     with col3:
# # #         st.metric("🚫 Contraindications", len(analysis.get('contraindications', [])))
# # #     with col4:
# # #         risk_level = analysis.get('risk_assessment', {}).get('risk_level', 'unknown')
# # #         risk_colors = {'high': '🔴', 'moderate': '🟡', 'low': '🟢', 'minimal': '🟢'}
# # #         st.metric("🎯 Risk Level", f"{risk_colors.get(risk_level, '⚪')} {risk_level.title()}")
# # #
# # #     # Interactions
# # #     interactions = analysis.get('interactions', [])
# # #     if interactions:
# # #         st.markdown("#### ⚠️ Drug Interactions Found")
# # #
# # #         # Group by severity
# # #         major_interactions = [i for i in interactions if i.get('severity') == 'major']
# # #         moderate_interactions = [i for i in interactions if i.get('severity') == 'moderate']
# # #         minor_interactions = [i for i in interactions if i.get('severity') == 'minor']
# # #
# # #         # Major interactions (always visible)
# # #         if major_interactions:
# # #             st.markdown("**🚨 Major Interactions (Immediate Attention Required)**")
# # #             for interaction in major_interactions:
# # #                 st.markdown(f"""
# # #                 <div class="alert-error">
# # #                     <strong>{interaction.get('drug1', 'Drug A')} + {interaction.get('drug2', 'Drug B')}</strong><br>
# # #                     <em>Risk:</em> {interaction.get('description', 'Significant interaction')}<br>
# # #                     <em>Management:</em> {interaction.get('management', 'Consult healthcare provider')}
# # #                 </div>
# # #                 """, unsafe_allow_html=True)
# # #
# # #         # Moderate interactions (expandable)
# # #         if moderate_interactions:
# # #             with st.expander(f"⚠️ Moderate Interactions ({len(moderate_interactions)})", expanded=False):
# # #                 for interaction in moderate_interactions:
# # #                     st.warning(
# # #                         f"**{interaction.get('drug1')} + {interaction.get('drug2')}:** {interaction.get('description')}")
# # #
# # #         # Minor interactions (expandable)
# # #         if minor_interactions:
# # #             with st.expander(f"ℹ️ Minor Interactions ({len(minor_interactions)})", expanded=False):
# # #                 for interaction in minor_interactions:
# # #                     st.info(
# # #                         f"**{interaction.get('drug1')} + {interaction.get('drug2')}:** {interaction.get('description')}")
# # #
# # #     # Recommendations
# # #     recommendations = analysis.get('recommendations', [])
# # #     if recommendations:
# # #         st.markdown("#### 📋 Clinical Recommendations")
# # #         for rec in recommendations:
# # #             st.markdown(f"• {rec}")
# # #
# # #     # No interactions found
# # #     if not interactions:
# # #         st.markdown("""
# # #         <div class="alert-success">
# # #             ✅ <strong>No Significant Interactions Found</strong><br>
# # #             No major drug interactions detected with the current medication list.
# # #         </div>
# # #         """, unsafe_allow_html=True)
# # #
# # #
# # # def render_enhanced_terminology_tool():
# # #     """Enhanced medical terminology mapping tool"""
# # #     st.markdown("### 🏷️ Medical Terminology Mapper")
# # #     st.markdown("*Map medical text to standard codes (ICD-10, SNOMED-CT)*")
# # #
# # #     medical_text = st.text_area(
# # #         "Enter medical text for analysis:",
# # #         placeholder="Patient presents with acute myocardial infarction, diabetes mellitus type 2, and hypertension...",
# # #         height=120,
# # #         help="Enter clinical text containing medical conditions and procedures"
# # #     )
# # #
# # #     if medical_text and st.button("🔍 Analyze Terminology", type="primary"):
# # #         analyze_medical_terminology_enhanced(medical_text)
# # #
# # #
# # # def analyze_medical_terminology_enhanced(medical_text: str):
# # #     """Enhanced medical terminology analysis"""
# # #     try:
# # #         with st.spinner("🔍 Analyzing medical terminology..."):
# # #             analysis = st.session_state.terminology_mapper.create_coding_summary(medical_text)
# # #
# # #         # Display results
# # #         render_terminology_analysis_results(analysis)
# # #
# # #     except Exception as e:
# # #         st.markdown(f"""
# # #         <div class="alert-error">
# # #             ❌ <strong>Analysis Error:</strong> {str(e)}
# # #         </div>
# # #         """, unsafe_allow_html=True)
# # #
# # #
# # # def render_terminology_analysis_results(analysis: Dict):
# # #     """Render terminology analysis results"""
# # #
# # #     # Summary statistics
# # #     stats = analysis.get('statistics', {})
# # #     col1, col2, col3 = st.columns(3)
# # #
# # #     with col1:
# # #         st.metric("🏷️ Terms Found", stats.get('total_terms_found', 0))
# # #     with col2:
# # #         st.metric("🔢 Codes Mapped", stats.get('total_codes_mapped', 0))
# # #     with col3:
# # #         coverage = stats.get('mapping_coverage', 0)
# # #         st.metric("📊 Coverage", f"{coverage:.1%}")
# # #
# # #     # Standardized text
# # #     standardized_text = analysis.get('text_analysis', {}).get('standardized_text', '')
# # #     if standardized_text:
# # #         st.markdown("#### 📝 Standardized Text")
# # #         st.info(standardized_text)
# # #
# # #     # Extracted terms
# # #     extracted_terms = analysis.get('text_analysis', {}).get('extracted_terms', [])
# # #     if extracted_terms:
# # #         with st.expander(f"🏷️ Extracted Medical Terms ({len(extracted_terms)})", expanded=True):
# # #             for term in extracted_terms:
# # #                 st.markdown(f"**{term.get('original')}** → {term.get('standardized')} *({term.get('type')})*")
# # #
# # #     # Medical codes
# # #     codes_by_system = analysis.get('codes_by_system', {})
# # #     if codes_by_system:
# # #         st.markdown("#### 🔢 Medical Codes")
# # #         for system, codes in codes_by_system.items():
# # #             with st.expander(f"{system} Codes ({len(codes)})", expanded=True):
# # #                 for code in codes:
# # #                     st.markdown(f"**{code.get('code')}:** {code.get('description')}")
# # #
# # #
# # # def render_guidelines_tool():
# # #     """Clinical guidelines finder tool"""
# # #     st.markdown("### 📊 Clinical Guidelines Finder")
# # #     st.markdown("*Find relevant clinical guidelines for medical conditions*")
# # #
# # #     condition = st.text_input(
# # #         "Enter medical condition:",
# # #         placeholder="e.g., diabetes, hypertension, heart failure",
# # #         help="Enter a medical condition to find relevant guidelines"
# # #     )
# # #
# # #     if condition and st.button("🔍 Find Guidelines", type="primary"):
# # #         st.info("🔧 Guidelines finder will be available in the next update")
# # #
# # #
# # # def render_document_analysis_tool():
# # #     """Document analysis tool"""
# # #     st.markdown("### 📋 Document Analysis")
# # #
# # #     if not st.session_state.get('vector_store_ready', False):
# # #         st.markdown("""
# # #         <div class="alert-warning">
# # #             📊 Please upload and process documents first to use document analysis.
# # #         </div>
# # #         """, unsafe_allow_html=True)
# # #         return
# # #
# # #     analysis_options = [
# # #         "📊 Collection Overview",
# # #         "📈 Document Types Analysis",
# # #         "🔍 Medical Content Analysis",
# # #         "📋 Quality Assessment"
# # #     ]
# # #
# # #     selected_analysis = st.selectbox("Choose Analysis Type", analysis_options)
# # #
# # #     if st.button("📊 Run Analysis", type="primary"):
# # #         run_document_analysis(selected_analysis)
# # #
# # #
# # # def run_document_analysis(analysis_type: str):
# # #     """Run document analysis based on selected type"""
# # #     try:
# # #         with st.spinner("📊 Running analysis..."):
# # #             if "Collection Overview" in analysis_type:
# # #                 stats = st.session_state.retriever.get_collection_stats()
# # #                 st.markdown("#### 📊 Collection Statistics")
# # #                 st.json(stats)
# # #
# # #             elif "Document Types" in analysis_type:
# # #                 st.info("📈 Detailed document type analysis coming in next update")
# # #
# # #             elif "Medical Content" in analysis_type:
# # #                 st.info("🔍 Medical content analysis coming in next update")
# # #
# # #             elif "Quality Assessment" in analysis_type:
# # #                 st.info("📋 Quality assessment features coming in next update")
# # #
# # #     except Exception as e:
# # #         st.markdown(f"""
# # #         <div class="alert-error">
# # #             ❌ <strong>Analysis Error:</strong> {str(e)}
# # #         </div>
# # #         """, unsafe_allow_html=True)
# # #
# # #
# # # # def main():
# # # #     """Main application function with enhanced layout"""
# # # #     # Initialize session state
# # # #     initialize_session_state()
# # # #
# # # #     # Render clean header
# # # #     render_clean_header()
# # # #
# # # #     # Sidebar configuration
# # # #     sidebar_configuration()
# # # #
# # # #     # Initialize components if not done
# # # #     if not st.session_state.get('components_initialized', False):
# # # #         success, message = initialize_components()
# # # #         if success:
# # # #             st.session_state.components_initialized = True
# # # #         else:
# # # #             st.markdown(f"""
# # # #             <div class="alert-error">
# # # #                 ❌ <strong>Initialization Failed:</strong> {message}
# # # #             </div>
# # # #             """, unsafe_allow_html=True)
# # # #             return
# # # #
# # # #     # Main content with enhanced tabs
# # # #     tab1, tab2, tab3 = st.tabs(["📄 Documents", "💬 Chat", "🧰 Tools"])
# # # #
# # # #     with tab1:
# # # #         document_upload_section()
# # # #
# # # #     with tab2:
# # # #         chat_interface()
# # # #
# # # #     with tab3:
# # # #         medical_tools_section()
# # # #
# # # #     # Enhanced footer
# # # #     render_enhanced_footer()
# # #
# # #
# # # def sidebar_configuration_optimized():
# # #     """Optimized sidebar with minimal recomputation and unique keys"""
# # #     with st.sidebar:
# # #         st.markdown("## ⚙️ Configuration")
# # #
# # #         # API Key section (minimized)
# # #         if not st.session_state.get('api_key_valid', False):
# # #             with st.expander("🔑 API Settings", expanded=True):
# # #                 api_key = st.text_input(
# # #                     "Groq API Key",
# # #                     type="password",
# # #                     key="optimized_api_key_input",  # Unique key
# # #                     help="Get your key from console.groq.com"
# # #                 )
# # #                 if api_key and st.button("🔓 Validate", key="optimized_validate_api"):
# # #                     validate_and_store_api_key(api_key)
# # #         else:
# # #             st.success("✅ API Connected")
# # #
# # #             # Model selection (only when API valid)
# # #             model_options = {
# # #                 "llama3-8b-8192": "Llama 3 8B (Fast)",
# # #                 "llama3-70b-8192": "Llama 3 70B (Accurate)",
# # #                 "gemma-7b-it": "Gemma 7B (Alternative)"
# # #             }
# # #
# # #             selected_model = st.selectbox(
# # #                 "Model",
# # #                 options=list(model_options.keys()),
# # #                 format_func=lambda x: model_options[x],
# # #                 key="optimized_model_select"  # Unique key
# # #             )
# # #
# # #             if st.session_state.get('selected_model') != selected_model:
# # #                 st.session_state.selected_model = selected_model
# # #                 if hasattr(st.session_state, 'response_generator'):
# # #                     st.session_state.response_generator.model = selected_model
# # #
# # #         # Compact status
# # #         render_compact_status()
# # #
# # #         # Quick actions
# # #         st.markdown("### ⚡ Quick Actions")
# # #         if st.button("🧪 Test System", key="optimized_test_system"):
# # #             run_quick_test()
# # #         if st.button("🗑️ Clear Cache", key="optimized_clear_cache"):
# # #             clear_cache()
# # #
# # # def main():
# # #     """Main application function with fixed widget keys"""
# # #     # Initialize session state
# # #     initialize_session_state()
# # #
# # #     # Render clean header
# # #     render_clean_header()
# # #
# # #     # Use the simple sidebar to avoid conflicts
# # #     sidebar_configuration_optimized()  # or sidebar_configuration() if you want the complex version
# # #
# # #     # Initialize components if not done
# # #     if not st.session_state.get('components_initialized', False):
# # #         success, message = initialize_components()
# # #         if success:
# # #             st.session_state.components_initialized = True
# # #         else:
# # #             st.markdown(f"""
# # #             <div class="alert-error">
# # #                 ❌ <strong>Initialization Failed:</strong> {message}
# # #             </div>
# # #             """, unsafe_allow_html=True)
# # #             return
# # #
# # #     # Main content with enhanced tabs
# # #     tab1, tab2, tab3 = st.tabs(["💬 Chat", "📄 Documents", "🧰 Tools"])
# # #
# # #     with tab1:
# # #         # Check prerequisites first
# # #         if not st.session_state.get('api_key_valid', False):
# # #             st.info("🔑 Please configure your API key in the sidebar to start chatting.")
# # #         elif not st.session_state.get('vector_store_ready', False):
# # #             st.info("📄 Please upload documents in the Documents tab to start chatting.")
# # #         else:
# # #             render_simple_query_interface()  # Use simple version
# # #             render_chat_history()
# # #
# # #     with tab2:
# # #         document_upload_section()
# # #
# # #     with tab3:
# # #         medical_tools_section()
# # #
# # #     # Enhanced footer
# # #     render_enhanced_footer()
# # #
# # # # def main():
# # # #     """Optimized main function"""
# # # #     # Initialize session state
# # # #     initialize_session_state()
# # # #
# # # #     # Periodic cleanup (every 20 page loads)
# # # #     page_loads = st.session_state.get('page_loads', 0) + 1
# # # #     st.session_state.page_loads = page_loads
# # # #     if page_loads % 20 == 0:
# # # #         cleanup_session_state()
# # # #
# # # #     # Render header
# # # #     render_clean_header()
# # # #
# # # #     # Optimized sidebar
# # # #     sidebar_configuration()
# # # #
# # # #     # Initialize components with caching
# # # #     if not st.session_state.get('components_loaded', False):
# # # #         with st.spinner("🚀 Loading system components..."):
# # # #             success, message = initialize_components()
# # # #             if not success:
# # # #                 st.error(f"❌ {message}")
# # # #                 return
# # # #
# # # #     # Main content - make Chat the default tab for better UX
# # # #     tab1, tab2, tab3 = st.tabs(["💬 Chat", "📄 Documents", "🧰 Tools"])
# # # #
# # # #     with tab1:
# # # #         chat_interface()
# # # #
# # # #     with tab2:
# # # #         document_upload_section()
# # # #
# # # #     with tab3:
# # # #         medical_tools_section()
# # # #
# # # #     # Compact footer
# # # #     render_compact_footer()
# # #
# # #
# # # def render_compact_footer():
# # #     """Compact footer"""
# # #     st.markdown("---")
# # #     st.caption("⚠️ For healthcare professionals only. Always verify information and apply clinical judgment.")
# # #
# # #
# # # def render_enhanced_footer():
# # #     """Render enhanced footer with important disclaimers"""
# # #     st.markdown("---")
# # #     st.markdown("""
# # #     <div style="text-align: center; padding: 2rem; background: #f8fafc; border-radius: 10px; margin-top: 2rem;">
# # #         <div style="color: #e53e3e; font-weight: 600; margin-bottom: 1rem;">
# # #             ⚠️ <strong>Professional Use Only - Important Disclaimers</strong>
# # #         </div>
# # #         <div style="color: #4a5568; line-height: 1.6;">
# # #             <p style="margin-bottom: 0.5rem;">
# # #                 🏥 <strong>For Healthcare Professionals Only:</strong> This system is designed to assist qualified healthcare professionals with evidence-based information.
# # #             </p>
# # #             <p style="margin-bottom: 0.5rem;">
# # #                 🧠 <strong>Clinical Judgment Required:</strong> Always apply professional clinical judgment and consult current medical guidelines.
# # #             </p>
# # #             <p style="margin-bottom: 0.5rem;">
# # #                 🚨 <strong>Not for Emergencies:</strong> Do not use for emergency medical situations. Seek immediate professional medical attention.
# # #             </p>
# # #             <p style="margin-bottom: 1rem;">
# # #                 📚 <strong>Verify Information:</strong> Always cross-reference with current medical literature and institutional protocols.
# # #             </p>
# # #         </div>
# # #         <div style="color: #718096; font-size: 0.9rem;">
# # #             Medical Literature RAG System | Powered by Advanced AI | Built with Streamlit, Groq & ChromaDB
# # #         </div>
# # #     </div>
# # #     """, unsafe_allow_html=True)
# # #
# # #
# # # if __name__ == "__main__":
# # #     main()
# #
# # import streamlit as st
# # import os
# # import time
# # import hashlib
# # import json
# # from datetime import datetime
# # from typing import Dict, List, Any
# #
# # # Import custom modules
# # from config.settings import Settings
# # from src.data_processing.document_loader import DocumentLoader
# # from src.data_processing.text_preprocessor import TextPreprocessor
# # from src.data_processing.chunking_strategy import MedicalChunkingStrategy
# # from src.embeddings.embedding_manager import EmbeddingManager
# # from src.retrieval.retriever import MedicalRetriever
# # from src.generation.groq_response_generator import GroqResponseGenerator
# # from src.medical_nlp.drug_interaction_checker import DrugInteractionChecker
# # from src.medical_nlp.terminology_mapper import MedicalTerminologyMapper
# # from src.utils.groq_utils import GroqUtils
# #
# # # Page configuration
# # st.set_page_config(
# #     page_title="Medical RAG System",
# #     page_icon="🏥",
# #     layout="wide"
# # )
# #
# # # Minimal CSS for clean look
# # st.markdown("""
# # <style>
# #     #MainMenu {visibility: hidden;}
# #     footer {visibility: hidden;}
# #     header {visibility: hidden;}
# #
# #     .main-header {
# #         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
# #         padding: 2rem;
# #         border-radius: 10px;
# #         color: white;
# #         text-align: center;
# #         margin-bottom: 1rem;
# #     }
# #
# #     .status-ok { color: #22c55e; }
# #     .status-error { color: #ef4444; }
# #
# #     .stButton > button {
# #         width: 100%;
# #         border-radius: 6px;
# #     }
# # </style>
# # """, unsafe_allow_html=True)
# #
# #
# # # Simple cache for queries
# # class QueryCache:
# #     def __init__(self):
# #         if 'query_cache' not in st.session_state:
# #             st.session_state.query_cache = {}
# #
# #     def get_key(self, query: str) -> str:
# #         return hashlib.md5(query.lower().encode()).hexdigest()[:10]
# #
# #     def get(self, query: str):
# #         key = self.get_key(query)
# #         return st.session_state.query_cache.get(key)
# #
# #     def set(self, query: str, response):
# #         key = self.get_key(query)
# #         # Keep only 10 cached responses
# #         if len(st.session_state.query_cache) >= 10:
# #             oldest = list(st.session_state.query_cache.keys())[0]
# #             del st.session_state.query_cache[oldest]
# #         st.session_state.query_cache[key] = response
# #
# #
# # query_cache = QueryCache()
# #
# #
# # def init_session_state():
# #     """Initialize session state"""
# #     defaults = {
# #         'api_key_valid': False,
# #         'docs_loaded': False,
# #         'chat_history': [],
# #         'components_ready': False
# #     }
# #     for key, value in defaults.items():
# #         if key not in st.session_state:
# #             st.session_state[key] = value
# #
# #
# # @st.cache_resource
# # def init_components():
# #     """Initialize system components with caching"""
# #     try:
# #         return {
# #             'document_loader': DocumentLoader(),
# #             'preprocessor': TextPreprocessor(),
# #             'chunking_strategy': MedicalChunkingStrategy(
# #                 chunk_size=Settings.CHUNK_SIZE,
# #                 overlap=Settings.CHUNK_OVERLAP
# #             ),
# #             'embedding_manager': EmbeddingManager(model_name=Settings.EMBEDDING_MODEL),
# #             'retriever': MedicalRetriever(
# #                 persist_directory=Settings.CHROMA_PERSIST_DIRECTORY,
# #                 collection_name=Settings.COLLECTION_NAME
# #             ),
# #             'drug_checker': DrugInteractionChecker(),
# #             'terminology_mapper': MedicalTerminologyMapper()
# #         }
# #     except Exception as e:
# #         st.error(f"Component initialization failed: {e}")
# #         return None
# #
# #
# # def setup_components():
# #     """Setup components in session state"""
# #     if not st.session_state.components_ready:
# #         components = init_components()
# #         if components:
# #             for name, component in components.items():
# #                 st.session_state[name] = component
# #             st.session_state.components_ready = True
# #
# #
# # def render_header():
# #     """Simple header"""
# #     st.markdown("""
# #     <div class="main-header">
# #         <h1>🏥 Medical Literature RAG</h1>
# #         <p>AI-Powered Clinical Decision Support</p>
# #     </div>
# #     """, unsafe_allow_html=True)
# #
# #
# # def render_sidebar():
# #     """Compact sidebar"""
# #     with st.sidebar:
# #         st.markdown("## ⚙️ Setup")
# #
# #         # API Key
# #         if not st.session_state.api_key_valid:
# #             with st.form("api_form"):
# #                 api_key = st.text_input("Groq API Key", type="password")
# #                 if st.form_submit_button("Validate"):
# #                     validate_api_key(api_key)
# #         else:
# #             st.success("✅ API Ready")
# #
# #             # Model selection
# #             models = {
# #                 "llama3-8b-8192": "Llama 3 8B (Fast)",
# #                 "llama3-70b-8192": "Llama 3 70B (Accurate)"
# #             }
# #
# #             selected = st.selectbox("Model", list(models.keys()), format_func=lambda x: models[x])
# #
# #             if st.session_state.get('selected_model') != selected:
# #                 st.session_state.selected_model = selected
# #                 if hasattr(st.session_state, 'response_generator'):
# #                     st.session_state.response_generator.model = selected
# #
# #         # Status
# #         st.markdown("### 📊 Status")
# #         api_status = "✅" if st.session_state.api_key_valid else "❌"
# #         docs_status = "✅" if st.session_state.docs_loaded else "❌"
# #         st.markdown(f"API: {api_status} | Docs: {docs_status}")
# #
# #         if st.session_state.docs_loaded:
# #             try:
# #                 stats = st.session_state.retriever.get_stats()
# #                 st.caption(f"📄 {stats.get('total_documents', 0)} documents")
# #             except:
# #                 pass
# #
# #         # Quick actions
# #         if st.button("🧪 Test"):
# #             test_system()
# #         if st.button("🔄 Reset"):
# #             reset_system()
# #
# #
# # def validate_api_key(api_key: str):
# #     """Validate API key"""
# #     if not api_key:
# #         st.error("Please enter API key")
# #         return
# #
# #     with st.spinner("Validating..."):
# #         validation = GroqUtils.validate_api_key(api_key)
# #
# #         if validation['valid']:
# #             st.session_state.api_key_valid = True
# #             st.session_state.groq_api_key = api_key
# #             st.session_state.response_generator = GroqResponseGenerator(
# #                 api_key=api_key,
# #                 model="llama3-8b-8192"
# #             )
# #             st.success("✅ API key validated")
# #             st.rerun()
# #         else:
# #             st.error(f"❌ {validation['message']}")
# #
# #
# # def test_system():
# #     """Quick system test"""
# #     if st.session_state.api_key_valid and st.session_state.docs_loaded:
# #         st.success("✅ System ready!")
# #     else:
# #         st.error("❌ Setup incomplete")
# #
# #
# # def reset_system():
# #     """Reset system"""
# #     for key in list(st.session_state.keys()):
# #         if key not in ['query_cache']:
# #             del st.session_state[key]
# #     st.rerun()
# #
# #
# # def document_section():
# #     """Document upload and processing"""
# #     st.markdown("## 📄 Documents")
# #
# #     uploaded_files = st.file_uploader(
# #         "Upload medical documents",
# #         type=['pdf', 'txt', 'docx'],
# #         accept_multiple_files=True
# #     )
# #
# #     if uploaded_files:
# #         st.success(f"✅ {len(uploaded_files)} files selected")
# #
# #         if st.button("🚀 Process Documents", type="primary"):
# #             process_documents(uploaded_files)
# #
# #
# # def process_documents(files):
# #     """Process uploaded documents"""
# #     if not st.session_state.components_ready:
# #         st.error("Components not ready")
# #         return
# #
# #     progress = st.progress(0)
# #     status = st.empty()
# #
# #     try:
# #         all_chunks = []
# #
# #         # Process files
# #         for i, file in enumerate(files):
# #             status.text(f"Processing: {file.name}")
# #             progress.progress((i + 1) / len(files) * 0.6)
# #
# #             doc_data = st.session_state.document_loader.load_uploaded_file(file)
# #             processed = st.session_state.preprocessor.preprocess_document(
# #                 doc_data['content'], anonymize=True
# #             )
# #             chunks = st.session_state.chunking_strategy.create_contextual_chunks(
# #                 processed['processed_content'], doc_data
# #             )
# #             all_chunks.extend(chunks)
# #
# #         # Generate embeddings
# #         status.text("Generating embeddings...")
# #         progress.progress(0.8)
# #         embedded_chunks = st.session_state.embedding_manager.embed_medical_chunks(all_chunks)
# #
# #         # Store in database
# #         status.text("Storing in database...")
# #         progress.progress(0.9)
# #         success = st.session_state.retriever.add_documents(embedded_chunks)
# #
# #         if success:
# #             progress.progress(1.0)
# #             st.session_state.docs_loaded = True
# #             st.success(f"✅ Processed {len(files)} files into {len(embedded_chunks)} chunks")
# #         else:
# #             st.error("❌ Failed to store documents")
# #
# #     except Exception as e:
# #         st.error(f"❌ Processing error: {e}")
# #     finally:
# #         progress.empty()
# #         status.empty()
# #
# #
# # def chat_section():
# #     """Main chat interface"""
# #     st.markdown("## 💬 Medical Assistant")
# #
# #     # Prerequisites check
# #     if not st.session_state.api_key_valid:
# #         st.info("🔑 Configure API key in sidebar")
# #         return
# #     if not st.session_state.docs_loaded:
# #         st.info("📄 Upload documents first")
# #         return
# #
# #     # Query interface
# #     with st.form("query_form"):
# #         col1, col2 = st.columns([3, 1])
# #
# #         with col1:
# #             query = st.text_input(
# #                 "Ask a medical question:",
# #                 placeholder="What are the side effects of metformin?"
# #             )
# #
# #         with col2:
# #             query_type = st.selectbox(
# #                 "Type",
# #                 ["general", "diagnosis", "treatment", "drug_interaction"]
# #             )
# #
# #         submitted = st.form_submit_button("🔍 Ask", type="primary")
# #
# #         if submitted and query:
# #             process_query(query, query_type)
# #
# #     # Suggestions
# #     if st.session_state.docs_loaded:
# #         st.markdown("**💡 Try asking:**")
# #         suggestions = [
# #             "Side effects of metformin",
# #             "Hypertension treatment guidelines",
# #             "Warfarin drug interactions"
# #         ]
# #
# #         cols = st.columns(len(suggestions))
# #         for i, suggestion in enumerate(suggestions):
# #             with cols[i]:
# #                 if st.button(f"💭 {suggestion[:20]}...", key=f"suggest_{i}"):
# #                     process_query(suggestion, "general")
# #
# #     # Chat history
# #     render_chat_history()
# #
# #
# # def process_query(query: str, query_type: str):
# #     """Process medical query"""
# #     # Check cache first
# #     cached = query_cache.get(query)
# #     if cached:
# #         st.info("⚡ From cache")
# #         display_response(cached['response'], cached['sources'])
# #         return
# #
# #     with st.spinner("🔍 Searching medical literature..."):
# #         try:
# #             start_time = time.time()
# #
# #             # Create embedding
# #             embedding = st.session_state.embedding_manager.create_medical_query_embedding(
# #                 query, query_type
# #             )
# #
# #             # Search documents
# #             results = st.session_state.retriever.search(embedding, n_results=3)
# #
# #             if not results:
# #                 st.warning("🔍 No relevant documents found")
# #                 return
# #
# #             # Generate response
# #             response_data = st.session_state.response_generator.generate_response(
# #                 query, results, query_type, max_tokens=600
# #             )
# #
# #             processing_time = time.time() - start_time
# #
# #             if 'error' not in response_data:
# #                 display_response(response_data['response'], results)
# #                 save_to_history(query, response_data['response'])
# #
# #                 # Cache response
# #                 query_cache.set(query, {
# #                     'response': response_data['response'],
# #                     'sources': results,
# #                     'timestamp': time.time()
# #                 })
# #             else:
# #                 st.error(f"❌ {response_data['response']}")
# #
# #         except Exception as e:
# #             st.error(f"❌ Query failed: {e}")
# #
# #
# # def display_response(response: str, sources: list):
# #     """Display medical response"""
# #     # Main response
# #     st.markdown("### 🔬 Response")
# #     st.markdown(response)
# #
# #     # Sources (expandable)
# #     if sources:
# #         with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
# #             for i, source in enumerate(sources, 1):
# #                 st.markdown(f"**Source {i}:** {source.get('metadata', {}).get('source', 'Unknown')}")
# #                 st.markdown(f"*Similarity:* {source.get('similarity_score', 0):.2f}")
# #                 if i < len(sources):
# #                     st.divider()
# #
# #
# # def save_to_history(query: str, response: str):
# #     """Save to chat history"""
# #     entry = {
# #         'query': query,
# #         'response': response[:200] + "..." if len(response) > 200 else response,
# #         'timestamp': datetime.now().strftime('%H:%M')
# #     }
# #
# #     st.session_state.chat_history.append(entry)
# #
# #     # Keep only last 5
# #     if len(st.session_state.chat_history) > 5:
# #         st.session_state.chat_history = st.session_state.chat_history[-5:]
# #
# #
# # def render_chat_history():
# #     """Show recent chat history"""
# #     if st.session_state.chat_history:
# #         with st.expander("📜 Recent Queries", expanded=False):
# #             for chat in reversed(st.session_state.chat_history[-3:]):
# #                 st.markdown(f"**Q:** {chat['query'][:50]}...")
# #                 st.markdown(f"**A:** {chat['response']}")
# #                 st.caption(f"🕒 {chat['timestamp']}")
# #                 st.divider()
# #
# #
# # def tools_section():
# #     """Medical tools"""
# #     st.markdown("## 🧰 Medical Tools")
# #
# #     tool = st.selectbox(
# #         "Select Tool",
# #         ["💊 Drug Interactions", "🏷️ Medical Codes"]
# #     )
# #
# #     if "Drug" in tool:
# #         drug_tool()
# #     elif "Medical" in tool:
# #         terminology_tool()
# #
# #
# # def drug_tool():
# #     """Drug interaction checker"""
# #     st.markdown("### 💊 Drug Interaction Checker")
# #
# #     medications = st.text_area(
# #         "Enter medications (one per line):",
# #         placeholder="warfarin\naspirin\nmetformin"
# #     )
# #
# #     if medications and st.button("🔍 Check Interactions"):
# #         try:
# #             med_list = [m.strip() for m in medications.split('\n') if m.strip()]
# #
# #             if len(med_list) < 2:
# #                 st.warning("Enter at least 2 medications")
# #                 return
# #
# #             analysis = st.session_state.drug_checker.analyze_medication_list(
# #                 '\n'.join(med_list)
# #             )
# #
# #             # Display results
# #             interactions = analysis.get('interactions', [])
# #
# #             if interactions:
# #                 st.markdown("#### ⚠️ Interactions Found")
# #                 for interaction in interactions:
# #                     severity = interaction.get('severity', 'unknown')
# #                     if severity == 'major':
# #                         st.error(f"🚨 **{interaction['drug1']} + {interaction['drug2']}:** {interaction['description']}")
# #                     else:
# #                         st.warning(
# #                             f"⚠️ **{interaction['drug1']} + {interaction['drug2']}:** {interaction['description']}")
# #             else:
# #                 st.success("✅ No major interactions found")
# #
# #         except Exception as e:
# #             st.error(f"❌ Analysis failed: {e}")
# #
# #
# # def terminology_tool():
# #     """Medical terminology mapper"""
# #     st.markdown("### 🏷️ Medical Code Mapper")
# #
# #     medical_text = st.text_area(
# #         "Enter medical text:",
# #         placeholder="Patient with diabetes mellitus and hypertension..."
# #     )
# #
# #     if medical_text and st.button("🔍 Map Codes"):
# #         try:
# #             analysis = st.session_state.terminology_mapper.analyze_medical_text(medical_text)
# #
# #             # Show results
# #             terms = analysis.get('terms_found', [])
# #             if terms:
# #                 st.markdown("#### 🏷️ Found Terms")
# #                 for term in terms:
# #                     st.markdown(f"**{term['term']}** ({term['category']})")
# #                     if term.get('icd10'):
# #                         st.caption(f"ICD-10: {term['icd10']}")
# #             else:
# #                 st.info("No medical terms found")
# #
# #         except Exception as e:
# #             st.error(f"❌ Analysis failed: {e}")
# #
# #
# # def main():
# #     """Main application"""
# #     # Initialize
# #     init_session_state()
# #
# #     # Render UI
# #     render_header()
# #     render_sidebar()
# #
# #     # Setup components
# #     setup_components()
# #
# #     # Main tabs
# #     tab1, tab2, tab3 = st.tabs(["💬 Chat", "📄 Documents", "🧰 Tools"])
# #
# #     with tab1:
# #         chat_section()
# #
# #     with tab2:
# #         document_section()
# #
# #     with tab3:
# #         tools_section()
# #
# #     # Footer
# #     st.markdown("---")
# #     st.caption("⚠️ For healthcare professionals only. Always verify information.")
# #
# #
# # if __name__ == "__main__":
# #     main()
#
# import streamlit as st
# import os
# import time
# import hashlib
# import json
# from datetime import datetime
# from typing import Dict, List, Any
#
# # Import custom modules
# from config.settings import Settings
# from src.data_processing.document_loader import DocumentLoader
# from src.data_processing.text_preprocessor import TextPreprocessor
# from src.data_processing.chunking_strategy import MedicalChunkingStrategy
# from src.embeddings.embedding_manager import EmbeddingManager
# from src.retrieval.retriever import MedicalRetriever
# from src.generation.groq_response_generator import GroqResponseGenerator
# from src.medical_nlp.drug_interaction_checker import DrugInteractionChecker
# from src.medical_nlp.terminology_mapper import MedicalTerminologyMapper
# from src.utils.groq_utils import GroqUtils
#
# # Page configuration
# st.set_page_config(
#     page_title="Medical RAG System",
#     page_icon="🏥",
#     layout="wide"
# )
#
# # Minimal CSS for clean look
# st.markdown("""
# <style>
#     #MainMenu {visibility: hidden;}
#     footer {visibility: hidden;}
#     header {visibility: hidden;}
#
#     .main-header {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         padding: 2rem;
#         border-radius: 10px;
#         color: white;
#         text-align: center;
#         margin-bottom: 1rem;
#     }
#
#     .status-ok { color: #22c55e; }
#     .status-error { color: #ef4444; }
#
#     .stButton > button {
#         width: 100%;
#         border-radius: 6px;
#     }
# </style>
# """, unsafe_allow_html=True)
#
#
# # Simple cache for queries
# class QueryCache:
#     def __init__(self):
#         if 'query_cache' not in st.session_state:
#             st.session_state.query_cache = {}
#
#     def get_key(self, query: str) -> str:
#         return hashlib.md5(query.lower().encode()).hexdigest()[:10]
#
#     def get(self, query: str):
#         key = self.get_key(query)
#         return st.session_state.query_cache.get(key)
#
#     def set(self, query: str, response):
#         key = self.get_key(query)
#         # Keep only 10 cached responses
#         if len(st.session_state.query_cache) >= 10:
#             oldest = list(st.session_state.query_cache.keys())[0]
#             del st.session_state.query_cache[oldest]
#         st.session_state.query_cache[key] = response
#
#
# query_cache = QueryCache()
#
#
# def init_session_state():
#     """Initialize session state"""
#     defaults = {
#         'api_key_valid': False,
#         'docs_loaded': False,
#         'chat_history': [],
#         'components_ready': False
#     }
#     for key, value in defaults.items():
#         if key not in st.session_state:
#             st.session_state[key] = value
#
#
# @st.cache_resource
# def init_components():
#     """Initialize system components with caching"""
#     try:
#         # Create a compatibility wrapper for the optimized chunking strategy
#         class CompatibleMedicalChunker:
#             """Wrapper to make optimized chunking strategy compatible"""
#
#             def __init__(self, chunk_size=1000, chunk_overlap=200):
#                 try:
#                     # Try optimized version first (uses 'overlap')
#                     self.chunker = MedicalChunkingStrategy(
#                         chunk_size=chunk_size,
#                         overlap=chunk_overlap  # optimized version parameter
#                     )
#                     self.strategy_type = "optimized"
#                 except Exception:
#                     try:
#                         # Try original version (uses 'chunk_overlap')
#                         self.chunker = MedicalChunkingStrategy(
#                             chunk_size=chunk_size,
#                             chunk_overlap=chunk_overlap
#                         )
#                         self.strategy_type = "original"
#                     except Exception:
#                         # Create minimal chunker
#                         self.chunker = None
#                         self.strategy_type = "manual"
#
#                 # Set compatibility attributes
#                 self.chunk_size = chunk_size
#                 self.chunk_overlap = chunk_overlap
#
#                 # Add section_patterns attribute for compatibility
#                 if hasattr(self.chunker, 'section_patterns'):
#                     self.section_patterns = self.chunker.section_patterns
#                 else:
#                     # Provide basic section patterns
#                     self.section_patterns = [
#                         r'^(ABSTRACT|INTRODUCTION|METHODS?|RESULTS?|DISCUSSION|CONCLUSION)S?\s*:?',
#                         r'^(BACKGROUND|OBJECTIVE|TREATMENT|DIAGNOSIS)S?\s*:?'
#                     ]
#
#             def create_contextual_chunks(self, text, metadata=None):
#                 """Main chunking method with multiple fallbacks"""
#                 if self.chunker is None:
#                     return self._manual_chunk(text, metadata)
#
#                 try:
#                     # Try optimized method
#                     if hasattr(self.chunker, 'create_contextual_chunks'):
#                         return self.chunker.create_contextual_chunks(text, metadata or {})
#                     elif hasattr(self.chunker, 'semantic_chunking'):
#                         # Convert optimized output to expected format
#                         chunks = self.chunker.semantic_chunking(text)
#                         return self._convert_chunks(chunks, metadata)
#                     else:
#                         return self._manual_chunk(text, metadata)
#                 except Exception as e:
#                     st.warning(f"Advanced chunking failed: {e}")
#                     return self._manual_chunk(text, metadata)
#
#             def create_chunks(self, text, metadata=None):
#                 """Alias for create_contextual_chunks"""
#                 return self.create_contextual_chunks(text, metadata)
#
#             def _convert_chunks(self, chunks, metadata):
#                 """Convert optimized chunk format to expected format"""
#                 converted = []
#                 for i, chunk in enumerate(chunks):
#                     converted_chunk = {
#                         'content': chunk.get('content', ''),
#                         'chunk_id': f"chunk_{i}",
#                         'source': metadata.get('filename', 'unknown') if metadata else 'unknown',
#                         'document_type': metadata.get('type', 'general') if metadata else 'general',
#                         'metadata': metadata or {},
#                         'section': chunk.get('section', 'unknown'),
#                         'chunk_type': chunk.get('chunk_type', 'unknown'),
#                         'size': chunk.get('size', len(chunk.get('content', '')))
#                     }
#                     converted.append(converted_chunk)
#                 return converted
#
#             def _manual_chunk(self, text, metadata):
#                 """Manual fallback chunking"""
#                 words = text.split()
#                 chunks = []
#                 chunk_size = 800  # Slightly smaller for safety
#
#                 for i in range(0, len(words), chunk_size):
#                     chunk_words = words[i:i + chunk_size]
#                     chunk_text = ' '.join(chunk_words)
#
#                     chunks.append({
#                         'content': chunk_text,
#                         'chunk_id': f"manual_chunk_{i // chunk_size}",
#                         'source': metadata.get('filename', 'unknown') if metadata else 'unknown',
#                         'document_type': metadata.get('type', 'general') if metadata else 'general',
#                         'metadata': metadata or {},
#                         'section': 'unknown',
#                         'chunk_type': 'manual',
#                         'size': len(chunk_text)
#                     })
#
#                 return chunks
#
#         # Use the compatibility wrapper
#         chunking_strategy = CompatibleMedicalChunker(
#             Settings.CHUNK_SIZE,
#             Settings.CHUNK_OVERLAP
#         )
#
#         return {
#             'document_loader': DocumentLoader(),
#             'preprocessor': TextPreprocessor(),
#             'chunking_strategy': chunking_strategy,
#             'embedding_manager': EmbeddingManager(model_name=Settings.EMBEDDING_MODEL),
#             'retriever': MedicalRetriever(
#                 persist_directory=Settings.CHROMA_PERSIST_DIRECTORY,
#                 collection_name=Settings.COLLECTION_NAME
#             ),
#             'drug_checker': DrugInteractionChecker(),
#             'terminology_mapper': MedicalTerminologyMapper()
#         }
#
#     except Exception as e:
#         st.error(f"All initialization methods failed: {e}")
#         return None
#
#
# def setup_components():
#     """Setup components in session state"""
#     if not st.session_state.components_ready:
#         with st.spinner("🚀 Loading system components..."):
#             components = init_components()
#             if components:
#                 for name, component in components.items():
#                     st.session_state[name] = component
#                 st.session_state.components_ready = True
#                 st.success("✅ Components loaded successfully")
#             else:
#                 st.error("❌ Failed to initialize components")
#                 st.info("Please check that all modules are properly installed and imported")
#
#
# def render_header():
#     """Simple header"""
#     st.markdown("""
#     <div class="main-header">
#         <h1>🏥 Medical Literature RAG</h1>
#         <p>AI-Powered Clinical Decision Support</p>
#     </div>
#     """, unsafe_allow_html=True)
#
#
# def render_sidebar():
#     """Compact sidebar"""
#     with st.sidebar:
#         st.markdown("## ⚙️ Setup")
#
#         # API Key
#         if not st.session_state.api_key_valid:
#             with st.form("api_form"):
#                 api_key = st.text_input("Groq API Key", type="password")
#                 if st.form_submit_button("Validate"):
#                     validate_api_key(api_key)
#         else:
#             st.success("✅ API Ready")
#
#             # Model selection
#             models = {
#                 "llama3-8b-8192": "Llama 3 8B (Fast)",
#                 "llama3-70b-8192": "Llama 3 70B (Accurate)"
#             }
#
#             selected = st.selectbox("Model", list(models.keys()), format_func=lambda x: models[x])
#
#             if st.session_state.get('selected_model') != selected:
#                 st.session_state.selected_model = selected
#                 if hasattr(st.session_state, 'response_generator'):
#                     st.session_state.response_generator.model = selected
#
#         # Status
#         st.markdown("### 📊 Status")
#         api_status = "✅" if st.session_state.api_key_valid else "❌"
#         docs_status = "✅" if st.session_state.docs_loaded else "❌"
#         st.markdown(f"API: {api_status} | Docs: {docs_status}")
#
#         if st.session_state.docs_loaded:
#             try:
#                 stats = st.session_state.retriever.get_stats()
#                 st.caption(f"📄 {stats.get('total_documents', 0)} documents")
#             except:
#                 pass
#
#         # Quick actions
#         if st.button("🧪 Test"):
#             test_system()
#         if st.button("🔄 Reset"):
#             reset_system()
#
#
# def validate_api_key(api_key: str):
#     """Validate API key"""
#     if not api_key:
#         st.error("Please enter API key")
#         return
#
#     with st.spinner("Validating..."):
#         validation = GroqUtils.validate_api_key(api_key)
#
#         if validation['valid']:
#             st.session_state.api_key_valid = True
#             st.session_state.groq_api_key = api_key
#             st.session_state.response_generator = GroqResponseGenerator(
#                 api_key=api_key,
#                 model="llama3-8b-8192"
#             )
#             st.success("✅ API key validated")
#             st.rerun()
#         else:
#             st.error(f"❌ {validation['message']}")
#
#
# def test_system():
#     """Quick system test"""
#     if st.session_state.api_key_valid and st.session_state.docs_loaded:
#         st.success("✅ System ready!")
#     else:
#         st.error("❌ Setup incomplete")
#
#
# def reset_system():
#     """Reset system"""
#     for key in list(st.session_state.keys()):
#         if key not in ['query_cache']:
#             del st.session_state[key]
#     st.rerun()
#
#
# def document_section():
#     """Document upload and processing"""
#     st.markdown("## 📄 Documents")
#
#     uploaded_files = st.file_uploader(
#         "Upload medical documents",
#         type=['pdf', 'txt', 'docx'],
#         accept_multiple_files=True
#     )
#
#     if uploaded_files:
#         st.success(f"✅ {len(uploaded_files)} files selected")
#
#         if st.button("🚀 Process Documents", type="primary"):
#             process_documents(uploaded_files)
#
#
# def process_documents(files):
#     """Process uploaded documents with proper file handling"""
#     if not st.session_state.components_ready:
#         st.error("Components not ready")
#         return
#
#     progress = st.progress(0)
#     status = st.empty()
#
#     try:
#         all_chunks = []
#
#         # Process files
#         for i, file in enumerate(files):
#             status.text(f"Processing: {file.name}")
#             progress.progress((i + 1) / len(files) * 0.6)
#
#             try:
#                 # Read file content directly from uploaded file object
#                 file_content = None
#
#                 if file.type == "application/pdf":
#                     # For PDF files, try to read content directly
#                     try:
#                         # Reset file pointer
#                         file.seek(0)
#
#                         # Create a simple document data structure
#                         doc_data = {
#                             'content': f"PDF content from {file.name}. Size: {len(file.getvalue())} bytes.",
#                             'source': file.name,
#                             'document_type': 'pdf',
#                             'metadata': {'filename': file.name, 'size': len(file.getvalue())}
#                         }
#
#                         # Try to extract text if PDF reader is available
#                         try:
#                             import PyPDF2
#                             import io
#
#                             file.seek(0)
#                             pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.getvalue()))
#                             text_content = ""
#                             for page in pdf_reader.pages:
#                                 text_content += page.extract_text() + "\n"
#
#                             if text_content.strip():
#                                 doc_data['content'] = text_content
#
#                         except ImportError:
#                             st.warning(f"PyPDF2 not available. Using filename for {file.name}")
#                         except Exception as pdf_error:
#                             st.warning(f"PDF reading failed for {file.name}: {pdf_error}")
#
#                     except Exception as e:
#                         st.warning(f"Error reading PDF {file.name}: {e}")
#                         continue
#
#                 elif file.type == "text/plain":
#                     # For text files
#                     try:
#                         file.seek(0)
#                         content = file.getvalue().decode('utf-8')
#                         doc_data = {
#                             'content': content,
#                             'source': file.name,
#                             'document_type': 'text',
#                             'metadata': {'filename': file.name, 'size': len(content)}
#                         }
#                     except Exception as e:
#                         st.warning(f"Error reading text file {file.name}: {e}")
#                         continue
#
#                 else:
#                     # For other file types, create basic structure
#                     file.seek(0)
#                     doc_data = {
#                         'content': f"Document: {file.name}. Content type: {file.type}. Size: {len(file.getvalue())} bytes.",
#                         'source': file.name,
#                         'document_type': 'unknown',
#                         'metadata': {'filename': file.name, 'type': file.type}
#                     }
#
#                 # Preprocess content
#                 try:
#                     processed = st.session_state.preprocessor.preprocess_document(
#                         doc_data['content'], anonymize=True
#                     )
#                 except Exception as preprocess_error:
#                     st.warning(f"Preprocessing failed for {file.name}: {preprocess_error}")
#                     # Use original content if preprocessing fails
#                     processed = {'processed_content': doc_data['content']}
#
#                 # Create chunks - try multiple method names to ensure compatibility
#                 try:
#                     # First try the optimized method
#                     if hasattr(st.session_state.chunking_strategy, 'create_contextual_chunks'):
#                         chunks = st.session_state.chunking_strategy.create_contextual_chunks(
#                             processed['processed_content'], doc_data
#                         )
#                     elif hasattr(st.session_state.chunking_strategy, 'create_chunks'):
#                         chunks = st.session_state.chunking_strategy.create_chunks(
#                             processed['processed_content'], doc_data
#                         )
#                     elif hasattr(st.session_state.chunking_strategy, 'semantic_chunking'):
#                         # For optimized chunking strategy
#                         base_chunks = st.session_state.chunking_strategy.semantic_chunking(
#                             processed['processed_content']
#                         )
#                         # Convert to expected format
#                         chunks = []
#                         for i, chunk in enumerate(base_chunks):
#                             chunk_data = {
#                                 'content': chunk.get('content', ''),
#                                 'chunk_id': f"{file.name}_chunk_{i}",
#                                 'source': file.name,
#                                 'document_type': doc_data.get('document_type', 'general'),
#                                 'metadata': doc_data.get('metadata', {}),
#                                 'section': chunk.get('section', 'unknown'),
#                                 'chunk_type': chunk.get('chunk_type', 'unknown')
#                             }
#                             chunks.append(chunk_data)
#                     else:
#                         # Manual chunking as absolute fallback
#                         words = processed['processed_content'].split()
#                         chunk_size = 1000
#                         chunks = []
#                         for j in range(0, len(words), chunk_size):
#                             chunk_text = ' '.join(words[j:j + chunk_size])
#                             chunks.append({
#                                 'content': chunk_text,
#                                 'chunk_id': f"{file.name}_chunk_{j // chunk_size}",
#                                 'source': file.name,
#                                 'document_type': doc_data.get('document_type', 'general'),
#                                 'metadata': doc_data.get('metadata', {})
#                             })
#
#                     all_chunks.extend(chunks)
#
#                 except Exception as chunk_error:
#                     st.warning(f"Chunking failed for {file.name}: {chunk_error}")
#                     # Create single chunk as fallback
#                     chunks = [{
#                         'content': processed['processed_content'][:2000],  # Limit size
#                         'chunk_id': f"{file.name}_single_chunk",
#                         'source': file.name,
#                         'document_type': doc_data.get('document_type', 'general'),
#                         'metadata': doc_data.get('metadata', {})
#                     }]
#                     all_chunks.extend(chunks)
#
#             except Exception as file_error:
#                 st.warning(f"Skipping {file.name} due to error: {file_error}")
#                 continue
#
#         if not all_chunks:
#             st.error("❌ No content could be extracted from uploaded files")
#             return
#
#         # Generate embeddings
#         status.text("Generating embeddings...")
#         progress.progress(0.8)
#
#         try:
#             embedded_chunks = st.session_state.embedding_manager.embed_medical_chunks(all_chunks)
#         except Exception as embed_error:
#             st.error(f"❌ Embedding generation failed: {embed_error}")
#             return
#
#         # Store in database
#         status.text("Storing in database...")
#         progress.progress(0.9)
#
#         try:
#             success = st.session_state.retriever.add_documents(embedded_chunks)
#
#             if success:
#                 progress.progress(1.0)
#                 st.session_state.docs_loaded = True
#                 st.success(f"✅ Processed {len(files)} files into {len(embedded_chunks)} chunks")
#             else:
#                 st.error("❌ Failed to store documents in database")
#         except Exception as store_error:
#             st.error(f"❌ Document storage failed: {store_error}")
#
#     except Exception as e:
#         st.error(f"❌ Processing error: {e}")
#     finally:
#         progress.empty()
#         status.empty()
#
#
# def chat_section():
#     """Main chat interface"""
#     st.markdown("## 💬 Medical Assistant")
#
#     # Prerequisites check
#     if not st.session_state.api_key_valid:
#         st.info("🔑 Configure API key in sidebar")
#         return
#     if not st.session_state.docs_loaded:
#         st.info("📄 Upload documents first")
#         return
#
#     # Query interface
#     with st.form("query_form"):
#         col1, col2 = st.columns([3, 1])
#
#         with col1:
#             query = st.text_input(
#                 "Ask a medical question:",
#                 placeholder="What are the side effects of metformin?"
#             )
#
#         with col2:
#             query_type = st.selectbox(
#                 "Type",
#                 ["general", "diagnosis", "treatment", "drug_interaction"]
#             )
#
#         submitted = st.form_submit_button("🔍 Ask", type="primary")
#
#         if submitted and query:
#             process_query(query, query_type)
#
#     # Suggestions
#     if st.session_state.docs_loaded:
#         st.markdown("**💡 Try asking:**")
#         suggestions = [
#             "Side effects of metformin",
#             "Hypertension treatment guidelines",
#             "Warfarin drug interactions"
#         ]
#
#         cols = st.columns(len(suggestions))
#         for i, suggestion in enumerate(suggestions):
#             with cols[i]:
#                 if st.button(f"💭 {suggestion[:20]}...", key=f"suggest_{i}"):
#                     process_query(suggestion, "general")
#
#     # Chat history
#     render_chat_history()
#
#
# def process_query(query: str, query_type: str):
#     """Process medical query"""
#     # Check cache first
#     cached = query_cache.get(query)
#     if cached:
#         st.info("⚡ From cache")
#         display_response(cached['response'], cached['sources'])
#         return
#
#     with st.spinner("🔍 Searching medical literature..."):
#         try:
#             start_time = time.time()
#
#             # Create embedding
#             embedding = st.session_state.embedding_manager.create_medical_query_embedding(
#                 query, query_type
#             )
#
#             # Search documents
#             results = st.session_state.retriever.search(embedding, n_results=3)
#
#             if not results:
#                 st.warning("🔍 No relevant documents found")
#                 return
#
#             # Generate response
#             response_data = st.session_state.response_generator.generate_response(
#                 query, results, query_type, max_tokens=600
#             )
#
#             processing_time = time.time() - start_time
#
#             if 'error' not in response_data:
#                 display_response(response_data['response'], results)
#                 save_to_history(query, response_data['response'])
#
#                 # Cache response
#                 query_cache.set(query, {
#                     'response': response_data['response'],
#                     'sources': results,
#                     'timestamp': time.time()
#                 })
#             else:
#                 st.error(f"❌ {response_data['response']}")
#
#         except Exception as e:
#             st.error(f"❌ Query failed: {e}")
#
#
# def display_response(response: str, sources: list):
#     """Display medical response"""
#     # Main response
#     st.markdown("### 🔬 Response")
#     st.markdown(response)
#
#     # Sources (expandable)
#     if sources:
#         with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
#             for i, source in enumerate(sources, 1):
#                 st.markdown(f"**Source {i}:** {source.get('metadata', {}).get('source', 'Unknown')}")
#                 st.markdown(f"*Similarity:* {source.get('similarity_score', 0):.2f}")
#                 if i < len(sources):
#                     st.divider()
#
#
# def save_to_history(query: str, response: str):
#     """Save to chat history"""
#     entry = {
#         'query': query,
#         'response': response[:200] + "..." if len(response) > 200 else response,
#         'timestamp': datetime.now().strftime('%H:%M')
#     }
#
#     st.session_state.chat_history.append(entry)
#
#     # Keep only last 5
#     if len(st.session_state.chat_history) > 5:
#         st.session_state.chat_history = st.session_state.chat_history[-5:]
#
#
# def render_chat_history():
#     """Show recent chat history"""
#     if st.session_state.chat_history:
#         with st.expander("📜 Recent Queries", expanded=False):
#             for chat in reversed(st.session_state.chat_history[-3:]):
#                 st.markdown(f"**Q:** {chat['query'][:50]}...")
#                 st.markdown(f"**A:** {chat['response']}")
#                 st.caption(f"🕒 {chat['timestamp']}")
#                 st.divider()
#
#
# def tools_section():
#     """Medical tools"""
#     st.markdown("## 🧰 Medical Tools")
#
#     tool = st.selectbox(
#         "Select Tool",
#         ["💊 Drug Interactions", "🏷️ Medical Codes"]
#     )
#
#     if "Drug" in tool:
#         drug_tool()
#     elif "Medical" in tool:
#         terminology_tool()
#
#
# def drug_tool():
#     """Drug interaction checker"""
#     st.markdown("### 💊 Drug Interaction Checker")
#
#     medications = st.text_area(
#         "Enter medications (one per line):",
#         placeholder="warfarin\naspirin\nmetformin"
#     )
#
#     if medications and st.button("🔍 Check Interactions"):
#         try:
#             med_list = [m.strip() for m in medications.split('\n') if m.strip()]
#
#             if len(med_list) < 2:
#                 st.warning("Enter at least 2 medications")
#                 return
#
#             analysis = st.session_state.drug_checker.analyze_medication_list(
#                 '\n'.join(med_list)
#             )
#
#             # Display results
#             interactions = analysis.get('interactions', [])
#
#             if interactions:
#                 st.markdown("#### ⚠️ Interactions Found")
#                 for interaction in interactions:
#                     severity = interaction.get('severity', 'unknown')
#                     if severity == 'major':
#                         st.error(f"🚨 **{interaction['drug1']} + {interaction['drug2']}:** {interaction['description']}")
#                     else:
#                         st.warning(
#                             f"⚠️ **{interaction['drug1']} + {interaction['drug2']}:** {interaction['description']}")
#             else:
#                 st.success("✅ No major interactions found")
#
#         except Exception as e:
#             st.error(f"❌ Analysis failed: {e}")
#
#
# def terminology_tool():
#     """Medical terminology mapper"""
#     st.markdown("### 🏷️ Medical Code Mapper")
#
#     medical_text = st.text_area(
#         "Enter medical text:",
#         placeholder="Patient with diabetes mellitus and hypertension..."
#     )
#
#     if medical_text and st.button("🔍 Map Codes"):
#         try:
#             analysis = st.session_state.terminology_mapper.analyze_medical_text(medical_text)
#
#             # Show results
#             terms = analysis.get('terms_found', [])
#             if terms:
#                 st.markdown("#### 🏷️ Found Terms")
#                 for term in terms:
#                     st.markdown(f"**{term['term']}** ({term['category']})")
#                     if term.get('icd10'):
#                         st.caption(f"ICD-10: {term['icd10']}")
#             else:
#                 st.info("No medical terms found")
#
#         except Exception as e:
#             st.error(f"❌ Analysis failed: {e}")
#
#
# def main():
#     """Main application"""
#     # Initialize
#     init_session_state()
#
#     # Render UI
#     render_header()
#     render_sidebar()
#
#     # Setup components
#     setup_components()
#
#     # Main tabs
#     tab1, tab2, tab3 = st.tabs(["💬 Chat", "📄 Documents", "🧰 Tools"])
#
#     with tab1:
#         chat_section()
#
#     with tab2:
#         document_section()
#
#     with tab3:
#         tools_section()
#
#     # Footer
#     st.markdown("---")
#     st.caption("⚠️ For healthcare professionals only. Always verify information.")
#
#
# if __name__ == "__main__":
#     main()

# import streamlit as st
# import os
# import time
# import re
# import hashlib
# import json
# from datetime import datetime
# from typing import Dict, List, Any
#
# # Import custom modules
# from config.settings import Settings
# from src.data_processing.document_loader import DocumentLoader
# from src.data_processing.text_preprocessor import TextPreprocessor
# from src.data_processing.chunking_strategy import MedicalChunkingStrategy
# from src.embeddings.embedding_manager import EmbeddingManager
# from src.retrieval.retriever import MedicalRetriever
# from src.generation.groq_response_generator import GroqResponseGenerator
# from src.medical_nlp.drug_interaction_checker import DrugInteractionChecker
# from src.medical_nlp.terminology_mapper import MedicalTerminologyMapper
# from src.utils.groq_utils import GroqUtils
#
# # Page configuration
# st.set_page_config(
#     page_title="Medical RAG System",
#     page_icon="🏥",
#     layout="wide"
# )
#
# # Minimal CSS for clean look
# st.markdown("""
# <style>
#     #MainMenu {visibility: hidden;}
#     footer {visibility: hidden;}
#     header {visibility: hidden;}
#
#     .main-header {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         padding: 2rem;
#         border-radius: 10px;
#         color: white;
#         text-align: center;
#         margin-bottom: 1rem;
#     }
#
#     .status-ok { color: #22c55e; }
#     .status-error { color: #ef4444; }
#
#     .stButton > button {
#         width: 100%;
#         border-radius: 6px;
#     }
# </style>
# """, unsafe_allow_html=True)
#
#
# # Simple cache for queries
# class QueryCache:
#     def __init__(self):
#         if 'query_cache' not in st.session_state:
#             st.session_state.query_cache = {}
#
#     def get_key(self, query: str) -> str:
#         return hashlib.md5(query.lower().encode()).hexdigest()[:10]
#
#     def get(self, query: str):
#         key = self.get_key(query)
#         return st.session_state.query_cache.get(key)
#
#     def set(self, query: str, response):
#         key = self.get_key(query)
#         # Keep only 10 cached responses
#         if len(st.session_state.query_cache) >= 10:
#             oldest = list(st.session_state.query_cache.keys())[0]
#             del st.session_state.query_cache[oldest]
#         st.session_state.query_cache[key] = response
#
#
# query_cache = QueryCache()
#
#
# def init_session_state():
#     """Initialize session state"""
#     defaults = {
#         'api_key_valid': False,
#         'docs_loaded': False,
#         'chat_history': [],
#         'components_ready': False
#     }
#     for key, value in defaults.items():
#         if key not in st.session_state:
#             st.session_state[key] = value
#
#
# @st.cache_resource
# def init_components():
#     """Initialize system components with caching"""
#     try:
#         # Create a compatibility wrapper for the optimized chunking strategy
#         class CompatibleMedicalChunker:
#             """Wrapper to make optimized chunking strategy compatible"""
#
#             def __init__(self, chunk_size=1000, chunk_overlap=200):
#                 try:
#                     # Try optimized version first (uses 'overlap')
#                     self.chunker = MedicalChunkingStrategy(
#                         chunk_size=chunk_size,
#                         overlap=chunk_overlap  # optimized version parameter
#                     )
#                     self.strategy_type = "optimized"
#                 except Exception:
#                     try:
#                         # Try original version (uses 'chunk_overlap')
#                         self.chunker = MedicalChunkingStrategy(
#                             chunk_size=chunk_size,
#                             chunk_overlap=chunk_overlap
#                         )
#                         self.strategy_type = "original"
#                     except Exception:
#                         # Create minimal chunker
#                         self.chunker = None
#                         self.strategy_type = "manual"
#
#                 # Set compatibility attributes
#                 self.chunk_size = chunk_size
#                 self.chunk_overlap = chunk_overlap
#
#                 # Add section_patterns attribute for compatibility
#                 if hasattr(self.chunker, 'section_patterns'):
#                     self.section_patterns = self.chunker.section_patterns
#                 else:
#                     # Provide basic section patterns
#                     self.section_patterns = [
#                         r'^(ABSTRACT|INTRODUCTION|METHODS?|RESULTS?|DISCUSSION|CONCLUSION)S?\s*:?',
#                         r'^(BACKGROUND|OBJECTIVE|TREATMENT|DIAGNOSIS)S?\s*:?'
#                     ]
#
#             def create_contextual_chunks(self, text, metadata=None):
#                 """Main chunking method with multiple fallbacks"""
#                 if self.chunker is None:
#                     return self._manual_chunk(text, metadata)
#
#                 try:
#                     # Try optimized method
#                     if hasattr(self.chunker, 'create_contextual_chunks'):
#                         return self.chunker.create_contextual_chunks(text, metadata or {})
#                     elif hasattr(self.chunker, 'semantic_chunking'):
#                         # Convert optimized output to expected format
#                         chunks = self.chunker.semantic_chunking(text)
#                         return self._convert_chunks(chunks, metadata)
#                     else:
#                         return self._manual_chunk(text, metadata)
#                 except Exception as e:
#                     st.warning(f"Advanced chunking failed: {e}")
#                     return self._manual_chunk(text, metadata)
#
#             def create_chunks(self, text, metadata=None):
#                 """Alias for create_contextual_chunks"""
#                 return self.create_contextual_chunks(text, metadata)
#
#             def _convert_chunks(self, chunks, metadata):
#                 """Convert optimized chunk format to expected format"""
#                 converted = []
#                 for i, chunk in enumerate(chunks):
#                     converted_chunk = {
#                         'content': chunk.get('content', ''),
#                         'chunk_id': f"chunk_{i}",
#                         'source': metadata.get('filename', 'unknown') if metadata else 'unknown',
#                         'document_type': metadata.get('type', 'general') if metadata else 'general',
#                         'metadata': metadata or {},
#                         'section': chunk.get('section', 'unknown'),
#                         'chunk_type': chunk.get('chunk_type', 'unknown'),
#                         'size': chunk.get('size', len(chunk.get('content', '')))
#                     }
#                     converted.append(converted_chunk)
#                 return converted
#
#             def _manual_chunk(self, text, metadata):
#                 """Manual fallback chunking"""
#                 words = text.split()
#                 chunks = []
#                 chunk_size = 800  # Slightly smaller for safety
#
#                 for i in range(0, len(words), chunk_size):
#                     chunk_words = words[i:i + chunk_size]
#                     chunk_text = ' '.join(chunk_words)
#
#                     chunks.append({
#                         'content': chunk_text,
#                         'chunk_id': f"manual_chunk_{i // chunk_size}",
#                         'source': metadata.get('filename', 'unknown') if metadata else 'unknown',
#                         'document_type': metadata.get('type', 'general') if metadata else 'general',
#                         'metadata': metadata or {},
#                         'section': 'unknown',
#                         'chunk_type': 'manual',
#                         'size': len(chunk_text)
#                     })
#
#                 return chunks
#
#         # Use the compatibility wrapper
#         chunking_strategy = CompatibleMedicalChunker(
#             Settings.CHUNK_SIZE,
#             Settings.CHUNK_OVERLAP
#         )
#
#         return {
#             'document_loader': DocumentLoader(),
#             'preprocessor': TextPreprocessor(),
#             'chunking_strategy': chunking_strategy,
#             'embedding_manager': EmbeddingManager(model_name=Settings.EMBEDDING_MODEL),
#             'retriever': MedicalRetriever(
#                 persist_directory=Settings.CHROMA_PERSIST_DIRECTORY,
#                 collection_name=Settings.COLLECTION_NAME
#             ),
#             'drug_checker': DrugInteractionChecker(),
#             'terminology_mapper': MedicalTerminologyMapper()
#         }
#
#     except Exception as e:
#         st.error(f"All initialization methods failed: {e}")
#         return None
#
#
# def setup_components():
#     """Setup components in session state"""
#     if not st.session_state.components_ready:
#         with st.spinner("🚀 Loading system components..."):
#             components = init_components()
#             if components:
#                 for name, component in components.items():
#                     st.session_state[name] = component
#                 st.session_state.components_ready = True
#                 st.success("✅ Components loaded successfully")
#             else:
#                 st.error("❌ Failed to initialize components")
#                 st.info("Please check that all modules are properly installed and imported")
#
#
# def render_header():
#     """Simple header"""
#     st.markdown("""
#     <div class="main-header">
#         <h1>🏥 Medical Literature RAG</h1>
#         <p>AI-Powered Clinical Decision Support</p>
#     </div>
#     """, unsafe_allow_html=True)
#
#
# def render_sidebar():
#     """Compact sidebar"""
#     with st.sidebar:
#         st.markdown("## ⚙️ Setup")
#
#         # API Key
#         if not st.session_state.api_key_valid:
#             with st.form("api_form"):
#                 api_key = st.text_input("Groq API Key", type="password")
#                 if st.form_submit_button("Validate"):
#                     validate_api_key(api_key)
#         else:
#             st.success("✅ API Ready")
#
#             # Model selection
#             models = {
#                 "llama3-8b-8192": "Llama 3 8B (Fast)",
#                 "llama3-70b-8192": "Llama 3 70B (Accurate)"
#             }
#
#             selected = st.selectbox("Model", list(models.keys()), format_func=lambda x: models[x])
#
#             if st.session_state.get('selected_model') != selected:
#                 st.session_state.selected_model = selected
#                 if hasattr(st.session_state, 'response_generator'):
#                     st.session_state.response_generator.model = selected
#
#         # Status
#         st.markdown("### 📊 Status")
#         api_status = "✅" if st.session_state.api_key_valid else "❌"
#         docs_status = "✅" if st.session_state.docs_loaded else "❌"
#         st.markdown(f"API: {api_status} | Docs: {docs_status}")
#
#         if st.session_state.docs_loaded:
#             try:
#                 stats = st.session_state.retriever.get_stats()
#                 st.caption(f"📄 {stats.get('total_documents', 0)} documents")
#             except:
#                 pass
#
#         # Quick actions
#         if st.button("🧪 Test"):
#             test_system()
#         if st.button("🔄 Reset"):
#             reset_system()
#
#
# def validate_api_key(api_key: str):
#     """Validate API key"""
#     if not api_key:
#         st.error("Please enter API key")
#         return
#
#     with st.spinner("Validating..."):
#         validation = GroqUtils.validate_api_key(api_key)
#
#         if validation['valid']:
#             st.session_state.api_key_valid = True
#             st.session_state.groq_api_key = api_key
#             st.session_state.response_generator = GroqResponseGenerator(
#                 api_key=api_key,
#                 model="llama3-8b-8192"
#             )
#             st.success("✅ API key validated")
#             st.rerun()
#         else:
#             st.error(f"❌ {validation['message']}")
#
#
# def test_system():
#     """Quick system test"""
#     if st.session_state.api_key_valid and st.session_state.docs_loaded:
#         st.success("✅ System ready!")
#     else:
#         st.error("❌ Setup incomplete")
#
#
# def reset_system():
#     """Reset system"""
#     for key in list(st.session_state.keys()):
#         if key not in ['query_cache']:
#             del st.session_state[key]
#     st.rerun()
#
#
# def document_section():
#     """Document upload and processing"""
#     st.markdown("## 📄 Documents")
#
#     uploaded_files = st.file_uploader(
#         "Upload medical documents",
#         type=['pdf', 'txt', 'docx'],
#         accept_multiple_files=True
#     )
#
#     if uploaded_files:
#         st.success(f"✅ {len(uploaded_files)} files selected")
#
#         if st.button("🚀 Process Documents", type="primary"):
#             process_documents(uploaded_files)
#
#
# def process_documents(files):
#     """Process uploaded documents with proper file handling"""
#     if not st.session_state.components_ready:
#         st.error("Components not ready")
#         return
#
#     progress = st.progress(0)
#     status = st.empty()
#
#     try:
#         all_chunks = []
#
#         # Process files
#         for i, file in enumerate(files):
#             status.text(f"Processing: {file.name}")
#             progress.progress((i + 1) / len(files) * 0.6)
#
#             try:
#                 # Read file content directly from uploaded file object
#                 file_content = None
#
#                 if file.type == "application/pdf":
#                     # For PDF files, try to read content directly
#                     try:
#                         # Reset file pointer
#                         file.seek(0)
#
#                         # Create a simple document data structure
#                         doc_data = {
#                             'content': f"PDF content from {file.name}. Size: {len(file.getvalue())} bytes.",
#                             'source': file.name,
#                             'document_type': 'pdf',
#                             'metadata': {'filename': file.name, 'size': len(file.getvalue())}
#                         }
#
#                         # Try to extract text if PDF reader is available
#                         try:
#                             import PyPDF2
#                             import io
#
#                             file.seek(0)
#                             pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.getvalue()))
#                             text_content = ""
#                             for page in pdf_reader.pages:
#                                 text_content += page.extract_text() + "\n"
#
#                             if text_content.strip():
#                                 doc_data['content'] = text_content
#
#                         except ImportError:
#                             st.warning(f"PyPDF2 not available. Using filename for {file.name}")
#                         except Exception as pdf_error:
#                             st.warning(f"PDF reading failed for {file.name}: {pdf_error}")
#
#                     except Exception as e:
#                         st.warning(f"Error reading PDF {file.name}: {e}")
#                         continue
#
#                 elif file.type == "text/plain":
#                     # For text files
#                     try:
#                         file.seek(0)
#                         content = file.getvalue().decode('utf-8')
#                         doc_data = {
#                             'content': content,
#                             'source': file.name,
#                             'document_type': 'text',
#                             'metadata': {'filename': file.name, 'size': len(content)}
#                         }
#                     except Exception as e:
#                         st.warning(f"Error reading text file {file.name}: {e}")
#                         continue
#
#                 else:
#                     # For other file types, create basic structure
#                     file.seek(0)
#                     doc_data = {
#                         'content': f"Document: {file.name}. Content type: {file.type}. Size: {len(file.getvalue())} bytes.",
#                         'source': file.name,
#                         'document_type': 'unknown',
#                         'metadata': {'filename': file.name, 'type': file.type}
#                     }
#
#                 # Preprocess content
#                 try:
#                     processed = st.session_state.preprocessor.preprocess_document(
#                         doc_data['content'], anonymize=True
#                     )
#                 except Exception as preprocess_error:
#                     st.warning(f"Preprocessing failed for {file.name}: {preprocess_error}")
#                     # Use original content if preprocessing fails
#                     processed = {'processed_content': doc_data['content']}
#
#                 # Create chunks - try multiple method names to ensure compatibility
#                 try:
#                     # First try the optimized method
#                     if hasattr(st.session_state.chunking_strategy, 'create_contextual_chunks'):
#                         chunks = st.session_state.chunking_strategy.create_contextual_chunks(
#                             processed['processed_content'], doc_data
#                         )
#                     elif hasattr(st.session_state.chunking_strategy, 'create_chunks'):
#                         chunks = st.session_state.chunking_strategy.create_chunks(
#                             processed['processed_content'], doc_data
#                         )
#                     elif hasattr(st.session_state.chunking_strategy, 'semantic_chunking'):
#                         # For optimized chunking strategy
#                         base_chunks = st.session_state.chunking_strategy.semantic_chunking(
#                             processed['processed_content']
#                         )
#                         # Convert to expected format
#                         chunks = []
#                         for i, chunk in enumerate(base_chunks):
#                             chunk_data = {
#                                 'content': chunk.get('content', ''),
#                                 'chunk_id': f"{file.name}_chunk_{i}",
#                                 'source': file.name,
#                                 'document_type': doc_data.get('document_type', 'general'),
#                                 'metadata': doc_data.get('metadata', {}),
#                                 'section': chunk.get('section', 'unknown'),
#                                 'chunk_type': chunk.get('chunk_type', 'unknown')
#                             }
#                             chunks.append(chunk_data)
#                     else:
#                         # Manual chunking as absolute fallback
#                         words = processed['processed_content'].split()
#                         chunk_size = 1000
#                         chunks = []
#                         for j in range(0, len(words), chunk_size):
#                             chunk_text = ' '.join(words[j:j + chunk_size])
#                             chunks.append({
#                                 'content': chunk_text,
#                                 'chunk_id': f"{file.name}_chunk_{j // chunk_size}",
#                                 'source': file.name,
#                                 'document_type': doc_data.get('document_type', 'general'),
#                                 'metadata': doc_data.get('metadata', {})
#                             })
#
#                     all_chunks.extend(chunks)
#
#                 except Exception as chunk_error:
#                     st.warning(f"Chunking failed for {file.name}: {chunk_error}")
#                     # Create single chunk as fallback
#                     chunks = [{
#                         'content': processed['processed_content'][:2000],  # Limit size
#                         'chunk_id': f"{file.name}_single_chunk",
#                         'source': file.name,
#                         'document_type': doc_data.get('document_type', 'general'),
#                         'metadata': doc_data.get('metadata', {})
#                     }]
#                     all_chunks.extend(chunks)
#
#             except Exception as file_error:
#                 st.warning(f"Skipping {file.name} due to error: {file_error}")
#                 continue
#
#         if not all_chunks:
#             st.error("❌ No content could be extracted from uploaded files")
#             return
#
#         # Generate embeddings
#         status.text("Generating embeddings...")
#         progress.progress(0.8)
#
#         try:
#             embedded_chunks = st.session_state.embedding_manager.embed_medical_chunks(all_chunks)
#         except Exception as embed_error:
#             st.error(f"❌ Embedding generation failed: {embed_error}")
#             return
#
#         # Store in database
#         status.text("Storing in database...")
#         progress.progress(0.9)
#
#         try:
#             success = st.session_state.retriever.add_documents(embedded_chunks)
#
#             if success:
#                 progress.progress(1.0)
#                 st.session_state.docs_loaded = True
#                 st.success(f"✅ Processed {len(files)} files into {len(embedded_chunks)} chunks")
#             else:
#                 st.error("❌ Failed to store documents in database")
#         except Exception as store_error:
#             st.error(f"❌ Document storage failed: {store_error}")
#
#     except Exception as e:
#         st.error(f"❌ Processing error: {e}")
#     finally:
#         progress.empty()
#         status.empty()
#
#
# def chat_section():
#     """Main chat interface"""
#     st.markdown("## 💬 Medical Assistant")
#
#     # Prerequisites check
#     if not st.session_state.api_key_valid:
#         st.info("🔑 Configure API key in sidebar")
#         return
#     if not st.session_state.docs_loaded:
#         st.info("📄 Upload documents first")
#         return
#
#     # Query interface
#     with st.form("query_form"):
#         col1, col2 = st.columns([3, 1])
#
#         with col1:
#             query = st.text_input(
#                 "Ask a medical question:",
#                 placeholder="What are the side effects of metformin?"
#             )
#
#         with col2:
#             query_type = st.selectbox(
#                 "Type",
#                 ["general", "diagnosis", "treatment", "drug_interaction"]
#             )
#
#         submitted = st.form_submit_button("🔍 Ask", type="primary")
#
#         if submitted and query:
#             process_query(query, query_type)
#
#     # Suggestions
#     if st.session_state.docs_loaded:
#         st.markdown("**💡 Try asking:**")
#         suggestions = [
#             "Side effects of metformin",
#             "Hypertension treatment guidelines",
#             "Warfarin drug interactions"
#         ]
#
#         cols = st.columns(len(suggestions))
#         for i, suggestion in enumerate(suggestions):
#             with cols[i]:
#                 if st.button(f"💭 {suggestion[:20]}...", key=f"suggest_{i}"):
#                     process_query(suggestion, "general")
#
#     # Chat history
#     render_chat_history()
#
#
# def process_query(query: str, query_type: str):
#     """Process medical query"""
#     # Check cache first
#     cached = query_cache.get(query)
#     if cached:
#         st.info("⚡ From cache")
#         display_response(cached['response'], cached['sources'])
#         return
#
#     with st.spinner("🔍 Searching medical literature..."):
#         try:
#             start_time = time.time()
#
#             # Create embedding
#             embedding = st.session_state.embedding_manager.create_medical_query_embedding(
#                 query, query_type
#             )
#
#             # Search documents
#             results = st.session_state.retriever.search(embedding, n_results=3)
#
#             if not results:
#                 st.warning("🔍 No relevant documents found")
#                 return
#
#             # Generate response
#             response_data = st.session_state.response_generator.generate_response(
#                 query, results, query_type, max_tokens=600
#             )
#
#             processing_time = time.time() - start_time
#
#             if 'error' not in response_data:
#                 display_response(response_data['response'], results)
#                 save_to_history(query, response_data['response'])
#
#                 # Cache response
#                 query_cache.set(query, {
#                     'response': response_data['response'],
#                     'sources': results,
#                     'timestamp': time.time()
#                 })
#             else:
#                 st.error(f"❌ {response_data['response']}")
#
#         except Exception as e:
#             st.error(f"❌ Query failed: {e}")
#
#
# def display_response(response: str, sources: list):
#     """Display medical response with professional formatting"""
#
#     # Parse and structure the response
#     formatted_response = format_medical_response(response)
#
#     # Main response with better styling
#     st.markdown("### 🔬 Medical Analysis")
#
#     # Executive Summary (if available)
#     if 'summary' in formatted_response:
#         with st.container():
#             st.markdown("#### 📋 **Executive Summary**")
#             st.info(formatted_response['summary'])
#
#     # Main Content
#     if 'main_content' in formatted_response:
#         st.markdown("#### 📖 **Detailed Analysis**")
#
#         # Format main content with proper sections
#         for section in formatted_response['main_content']:
#             if section['type'] == 'heading':
#                 st.markdown(f"**{section['content']}**")
#             elif section['type'] == 'paragraph':
#                 st.markdown(section['content'])
#             elif section['type'] == 'list':
#                 for item in section['items']:
#                     st.markdown(f"• {item}")
#             elif section['type'] == 'numbered_list':
#                 for i, item in enumerate(section['items'], 1):
#                     st.markdown(f"{i}. {item}")
#
#     # Clinical Recommendations (highlighted)
#     if 'recommendations' in formatted_response:
#         st.markdown("#### 💊 **Clinical Recommendations**")
#
#         # Color-code recommendations by priority
#         for i, rec in enumerate(formatted_response['recommendations'], 1):
#             if 'first-line' in rec.lower() or 'emergency' in rec.lower():
#                 st.success(f"**{i}.** {rec}")
#             elif 'avoid' in rec.lower() or 'contraindicated' in rec.lower():
#                 st.error(f"**{i}.** {rec}")
#             else:
#                 st.info(f"**{i}.** {rec}")
#
#     # Limitations (if present)
#     if 'limitations' in formatted_response:
#         with st.expander("⚠️ **Limitations & Considerations**", expanded=False):
#             for limitation in formatted_response['limitations']:
#                 st.warning(f"• {limitation}")
#
#     # Sources with better formatting
#     if sources:
#         with st.expander(f"📚 **Evidence Sources** ({len(sources)})", expanded=False):
#             for i, source in enumerate(sources, 1):
#                 col1, col2 = st.columns([3, 1])
#
#                 with col1:
#                     source_name = source.get('metadata', {}).get('source', f'Source {i}')
#                     st.markdown(f"**{i}. {source_name}**")
#
#                     # Show excerpt if available
#                     if 'content' in source:
#                         excerpt = source['content'][:150] + "..." if len(source['content']) > 150 else source['content']
#                         st.caption(f"*Excerpt:* {excerpt}")
#
#                 with col2:
#                     similarity = source.get('similarity_score', 0)
#                     if similarity > 0.8:
#                         st.success(f"Relevance: {similarity:.2f}")
#                     elif similarity > 0.6:
#                         st.warning(f"Relevance: {similarity:.2f}")
#                     else:
#                         st.info(f"Relevance: {similarity:.2f}")
#
#                 if i < len(sources):
#                     st.divider()
#
#
# def format_medical_response(response_text: str) -> dict:
#     """Parse and structure medical response text for better formatting"""
#
#     # Initialize structure
#     formatted = {
#         'main_content': [],
#         'recommendations': [],
#         'limitations': []
#     }
#
#     # Split into paragraphs
#     paragraphs = [p.strip() for p in response_text.split('\n\n') if p.strip()]
#
#     current_section = 'main_content'
#
#     for paragraph in paragraphs:
#         # Detect section headers
#         if paragraph.startswith('Actionable Recommendations:'):
#             current_section = 'recommendations'
#             continue
#         elif paragraph.startswith('Limitations:'):
#             current_section = 'limitations'
#             continue
#         elif paragraph.startswith('Based on'):
#             # This is likely a summary
#             formatted['summary'] = paragraph
#             continue
#
#         # Process content based on current section
#         if current_section == 'recommendations':
#             # Parse numbered recommendations
#             if re.match(r'^\d+\.', paragraph):
#                 # Remove number and clean up
#                 rec_text = re.sub(r'^\d+\.\s*', '', paragraph)
#                 formatted['recommendations'].append(rec_text)
#             else:
#                 # Look for bullet points
#                 lines = paragraph.split('\n')
#                 for line in lines:
#                     if line.strip().startswith('•') or line.strip().startswith('-'):
#                         rec_text = re.sub(r'^[•\-]\s*', '', line.strip())
#                         formatted['recommendations'].append(rec_text)
#
#         elif current_section == 'limitations':
#             # Parse limitations
#             if paragraph.startswith('•'):
#                 lines = paragraph.split('\n')
#                 for line in lines:
#                     if line.strip().startswith('•'):
#                         limitation = re.sub(r'^•\s*', '', line.strip())
#                         formatted['limitations'].append(limitation)
#             else:
#                 formatted['limitations'].append(paragraph)
#
#         else:
#             # Main content processing
#             if paragraph.startswith('Source'):
#                 # This is source information
#                 formatted['main_content'].append({
#                     'type': 'paragraph',
#                     'content': paragraph
#                 })
#             elif re.match(r'^[A-Z][^.]*:$', paragraph):
#                 # This looks like a heading
#                 formatted['main_content'].append({
#                     'type': 'heading',
#                     'content': paragraph.rstrip(':')
#                 })
#             else:
#                 formatted['main_content'].append({
#                     'type': 'paragraph',
#                     'content': paragraph
#                 })
#
#     return formatted
#
#
# def save_to_history(query: str, response: str):
#     """Save to chat history"""
#     entry = {
#         'query': query,
#         'response': response[:200] + "..." if len(response) > 200 else response,
#         'timestamp': datetime.now().strftime('%H:%M')
#     }
#
#     st.session_state.chat_history.append(entry)
#
#     # Keep only last 5
#     if len(st.session_state.chat_history) > 5:
#         st.session_state.chat_history = st.session_state.chat_history[-5:]
#
#
# def render_chat_history():
#     """Show recent chat history"""
#     if st.session_state.chat_history:
#         with st.expander("📜 Recent Queries", expanded=False):
#             for chat in reversed(st.session_state.chat_history[-3:]):
#                 st.markdown(f"**Q:** {chat['query'][:50]}...")
#                 st.markdown(f"**A:** {chat['response']}")
#                 st.caption(f"🕒 {chat['timestamp']}")
#                 st.divider()
#
#
# def tools_section():
#     """Medical tools"""
#     st.markdown("## 🧰 Medical Tools")
#
#     tool = st.selectbox(
#         "Select Tool",
#         ["💊 Drug Interactions", "🏷️ Medical Codes"]
#     )
#
#     if "Drug" in tool:
#         drug_tool()
#     elif "Medical" in tool:
#         terminology_tool()
#
#
# def drug_tool():
#     """Drug interaction checker"""
#     st.markdown("### 💊 Drug Interaction Checker")
#
#     medications = st.text_area(
#         "Enter medications (one per line):",
#         placeholder="warfarin\naspirin\nmetformin"
#     )
#
#     if medications and st.button("🔍 Check Interactions"):
#         try:
#             med_list = [m.strip() for m in medications.split('\n') if m.strip()]
#
#             if len(med_list) < 2:
#                 st.warning("Enter at least 2 medications")
#                 return
#
#             analysis = st.session_state.drug_checker.analyze_medication_list(
#                 '\n'.join(med_list)
#             )
#
#             # Display results
#             interactions = analysis.get('interactions', [])
#
#             if interactions:
#                 st.markdown("#### ⚠️ Interactions Found")
#                 for interaction in interactions:
#                     severity = interaction.get('severity', 'unknown')
#                     if severity == 'major':
#                         st.error(f"🚨 **{interaction['drug1']} + {interaction['drug2']}:** {interaction['description']}")
#                     else:
#                         st.warning(
#                             f"⚠️ **{interaction['drug1']} + {interaction['drug2']}:** {interaction['description']}")
#             else:
#                 st.success("✅ No major interactions found")
#
#         except Exception as e:
#             st.error(f"❌ Analysis failed: {e}")
#
#
# def terminology_tool():
#     """Medical terminology mapper"""
#     st.markdown("### 🏷️ Medical Code Mapper")
#
#     medical_text = st.text_area(
#         "Enter medical text:",
#         placeholder="Patient with diabetes mellitus and hypertension..."
#     )
#
#     if medical_text and st.button("🔍 Map Codes"):
#         try:
#             analysis = st.session_state.terminology_mapper.analyze_medical_text(medical_text)
#
#             # Show results
#             terms = analysis.get('terms_found', [])
#             if terms:
#                 st.markdown("#### 🏷️ Found Terms")
#                 for term in terms:
#                     st.markdown(f"**{term['term']}** ({term['category']})")
#                     if term.get('icd10'):
#                         st.caption(f"ICD-10: {term['icd10']}")
#             else:
#                 st.info("No medical terms found")
#
#         except Exception as e:
#             st.error(f"❌ Analysis failed: {e}")
#
#
# def main():
#     """Main application"""
#     # Initialize
#     init_session_state()
#
#     # Render UI
#     render_header()
#     render_sidebar()
#
#     # Setup components
#     setup_components()
#
#     # Main tabs
#     tab1, tab2, tab3 = st.tabs(["💬 Chat", "📄 Documents", "🧰 Tools"])
#
#     with tab1:
#         chat_section()
#
#     with tab2:
#         document_section()
#
#     with tab3:
#         tools_section()
#
#     # Footer
#     st.markdown("---")
#     st.caption("⚠️ For healthcare professionals only. Always verify information.")
#
#
# # if __name__ == "__main__":
# #     main(), paragraph):
# #     # This looks like a heading
# #     formatted['main_content'].append({
# #         'type': 'heading',
# #         'content': paragraph.rstrip(':')
# #     })
# #     elif '•' in paragraph or paragraph.count('\n') > 2:
# #     # This might be a list
# #     lines = paragraph.split('\n')
# #     items = []
# #     regular_text = []
# #
# #     for line in lines:
# #         line = line.strip()
# #     if line.startswith('•') or line.startswith('-'):
# #         items.append(re.sub(r'^[•\-]\s*', '', line))
# #     elif line:
# #         regular_text.append(line)
# #
# #     if items:
# #         if
# #     regular_text:
# #     formatted['main_content'].append({
# #         'type': 'paragraph',
# #         'content': ' '.join(regular_text)
# #     })
# #     formatted['main_content'].append({
# #         'type': 'list',
# #         'items': items
# #     })
# #     else:
# #     formatted['main_content'].append({
# #         'type': 'paragraph',
# #         'content': paragraph
# #     })
# #     else:
# #     # Regular paragraph
# #     formatted['main_content'].append({
# #         'type': 'paragraph',
# #         'content': paragraph
# #     })
# #
# #     return formatted
#
# def process_paragraph(paragraph, formatted):
#     # This looks like a heading
#     if paragraph.endswith(':'):
#         formatted['main_content'].append({
#             'type': 'heading',
#             'content': paragraph.rstrip(':')
#         })
#
#     # This might be a list
#     elif '•' in paragraph or paragraph.count('\n') > 2:
#         lines = paragraph.split('\n')
#         items = []
#         regular_text = []
#
#         for line in lines:
#             line = line.strip()
#             if line.startswith('•') or line.startswith('-'):
#                 items.append(re.sub(r'^[•\-]\s*', '', line))
#             elif line:
#                 regular_text.append(line)
#
#         if items:
#             if regular_text:
#                 formatted['main_content'].append({
#                     'type': 'paragraph',
#                     'content': ' '.join(regular_text)
#                 })
#             formatted['main_content'].append({
#                 'type': 'list',
#                 'items': items
#             })
#         else:
#             formatted['main_content'].append({
#                 'type': 'paragraph',
#                 'content': paragraph
#             })
#
#     # Regular paragraph
#     else:
#         formatted['main_content'].append({
#             'type': 'paragraph',
#             'content': paragraph
#         })
#
#     return formatted
#
#
# def save_to_history(query: str, response: str):
#     """Save to chat history"""
#     entry = {
#         'query': query,
#         'response': response[:200] + "..." if len(response) > 200 else response,
#         'timestamp': datetime.now().strftime('%H:%M')
#     }
#
#     st.session_state.chat_history.append(entry)
#
#     # Keep only last 5
#     if len(st.session_state.chat_history) > 5:
#         st.session_state.chat_history = st.session_state.chat_history[-5:]
#
#
# def render_chat_history():
#     """Show recent chat history"""
#     if st.session_state.chat_history:
#         with st.expander("📜 Recent Queries", expanded=False):
#             for chat in reversed(st.session_state.chat_history[-3:]):
#                 st.markdown(f"**Q:** {chat['query'][:50]}...")
#                 st.markdown(f"**A:** {chat['response']}")
#                 st.caption(f"🕒 {chat['timestamp']}")
#                 st.divider()
#
#
# def tools_section():
#     """Medical tools"""
#     st.markdown("## 🧰 Medical Tools")
#
#     tool = st.selectbox(
#         "Select Tool",
#         ["💊 Drug Interactions", "🏷️ Medical Codes"]
#     )
#
#     if "Drug" in tool:
#         drug_tool()
#     elif "Medical" in tool:
#         terminology_tool()
#
#
# def drug_tool():
#     """Drug interaction checker"""
#     st.markdown("### 💊 Drug Interaction Checker")
#
#     medications = st.text_area(
#         "Enter medications (one per line):",
#         placeholder="warfarin\naspirin\nmetformin"
#     )
#
#     if medications and st.button("🔍 Check Interactions"):
#         try:
#             med_list = [m.strip() for m in medications.split('\n') if m.strip()]
#
#             if len(med_list) < 2:
#                 st.warning("Enter at least 2 medications")
#                 return
#
#             analysis = st.session_state.drug_checker.analyze_medication_list(
#                 '\n'.join(med_list)
#             )
#
#             # Display results
#             interactions = analysis.get('interactions', [])
#
#             if interactions:
#                 st.markdown("#### ⚠️ Interactions Found")
#                 for interaction in interactions:
#                     severity = interaction.get('severity', 'unknown')
#                     if severity == 'major':
#                         st.error(f"🚨 **{interaction['drug1']} + {interaction['drug2']}:** {interaction['description']}")
#                     else:
#                         st.warning(
#                             f"⚠️ **{interaction['drug1']} + {interaction['drug2']}:** {interaction['description']}")
#             else:
#                 st.success("✅ No major interactions found")
#
#         except Exception as e:
#             st.error(f"❌ Analysis failed: {e}")
#
#
# def terminology_tool():
#     """Medical terminology mapper"""
#     st.markdown("### 🏷️ Medical Code Mapper")
#
#     medical_text = st.text_area(
#         "Enter medical text:",
#         placeholder="Patient with diabetes mellitus and hypertension..."
#     )
#
#     if medical_text and st.button("🔍 Map Codes"):
#         try:
#             analysis = st.session_state.terminology_mapper.analyze_medical_text(medical_text)
#
#             # Show results
#             terms = analysis.get('terms_found', [])
#             if terms:
#                 st.markdown("#### 🏷️ Found Terms")
#                 for term in terms:
#                     st.markdown(f"**{term['term']}** ({term['category']})")
#                     if term.get('icd10'):
#                         st.caption(f"ICD-10: {term['icd10']}")
#             else:
#                 st.info("No medical terms found")
#
#         except Exception as e:
#             st.error(f"❌ Analysis failed: {e}")
#
#
# def main():
#     """Main application"""
#     # Initialize
#     init_session_state()
#
#     # Render UI
#     render_header()
#     render_sidebar()
#
#     # Setup components
#     setup_components()
#
#     # Main tabs
#     tab1, tab2, tab3 = st.tabs(["💬 Chat", "📄 Documents", "🧰 Tools"])
#
#     with tab1:
#         chat_section()
#
#     with tab2:
#         document_section()
#
#     with tab3:
#         tools_section()
#
#     # Footer
#     st.markdown("---")
#     st.caption("⚠️ For healthcare professionals only. Always verify information.")
#
#
# if __name__ == "__main__":
#     main()

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
from src.embeddings.embedding_manager import EmbeddingManager
from src.retrieval.retriever import MedicalRetriever
from src.generation.groq_response_generator import GroqResponseGenerator
from src.medical_nlp.drug_interaction_checker import DrugInteractionChecker
from src.medical_nlp.terminology_mapper import MedicalTerminologyMapper
from src.utils.groq_utils import GroqUtils

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
                st.success("✅ API key automatically loaded from environment")
        except Exception as e:
            st.warning(f"Environment API key validation failed: {e}")
    # If no API key or validation fails, the existing manual input will handle it

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
                persist_directory=Settings.CHROMA_PERSIST_DIRECTORY,
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

        # Quick actions
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧪 Test", key="test_system"):
                test_system()
        with col2:
            if st.button("🔄 Reset", key="reset_system"):
                reset_system()


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