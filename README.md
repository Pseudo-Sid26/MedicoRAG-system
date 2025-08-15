# 🩺 MedicoRAG System

<div align="center">

![MedicoRAG Banner](https://img.shields.io/badge/MedicoRAG-AI%20Powered%20Medical%20Assistant-blue?style=for-the-badge&logo=medical-cross)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=flat-square&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Database-orange?style=flat-square)](https://faiss.ai/)
[![Deploy](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/pseudo-sid26/medicorag-system/main/app.py)

**MedicoRAG** is an AI-powered **Retrieval-Augmented Generation (RAG)** system designed to assist healthcare professionals by processing **medical journals, clinical guidelines, and patient data** (anonymized) to deliver **accurate, evidence-based answers** quickly.

🚀 **[Deploy to Streamlit Cloud Now!](https://share.streamlit.io/pseudo-sid26/medicorag-system/main/app.py)**

⚠️ **Disclaimer:** For healthcare professionals only. Always verify AI-generated information with authoritative sources.

[🚀 Features](#-features) • [☁️ Deploy Now](#️-cloud-deployment) • [📦 Installation](#-installation) • [📖 How It Works](#-how-it-works) • [🤝 Contributing](#-contributing)

</div>

---

## 📋 Table of Contents

- [🚀 Features](#-features)
- [🛠 Tech Stack](#-tech-stack)
- [📂 Project Structure](#-project-structure)
- [📦 Installation](#-installation)
- [☁️ Cloud Deployment](#️-cloud-deployment)
- [📖 How It Works](#-how-it-works)
- [📌 Example Use Cases](#-example-use-cases)
- [🎯 Demo Screenshots](#-demo-screenshots)
- [🔧 Configuration](#-configuration)
- [🤝 Contributing](#-contributing)

---

## 🚀 Features

<table>
<tr>
<td width="50%">

### 📚 **Document Processing**
- Upload and analyze PDFs, research papers
- OCR support for scanned documents
- Clinical guidelines integration
- Multi-format document support

</td>
<td width="50%">

### 💬 **Interactive Chat**
- Natural language medical queries
- Contextually relevant answers
- Citation tracking and sources
- Real-time response generation

</td>
</tr>
<tr>
<td width="50%">

### 🧠 **RAG-Powered AI**
- Vector search capabilities
- Large language model integration
- Semantic document retrieval
- Evidence-based responses

</td>
<td width="50%">

### 📊 **Data Support**
- Anonymized patient data handling
- Structured and unstructured data
- Multi-source search capabilities
- Secure data processing

</td>
</tr>
</table>

---

## 🛠 Tech Stack

| Category | Technology | Purpose |
|----------|------------|---------|
| **Frontend** | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white) | Interactive web interface |
| **Backend** | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) | Core application logic |
| **AI/ML** | ![Groq](https://img.shields.io/badge/Groq-LLM%20API-green?style=flat-square) | Fast language model inference |
| **Database** | ![FAISS](https://img.shields.io/badge/FAISS-Vector%20DB-orange?style=flat-square) | Vector embeddings storage |
| **Embeddings** | ![TF-IDF](https://img.shields.io/badge/TF--IDF-Custom%20Implementation-blue?style=flat-square) | Text vectorization |
| **Processing** | ![NLP](https://img.shields.io/badge/NLP-Medical%20Processing-red?style=flat-square) | Document text extraction |

---

## ☁️ Cloud Deployment

### Deploy to Streamlit Cloud

#### Prerequisites
- GitHub account
- Streamlit Cloud account (free)
- Groq API key

#### Quick Deploy Steps

1. **🍴 Fork this repository**
   - Click the "Fork" button on GitHub
   - Choose your account as the destination

2. **🔗 Connect to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"

3. **⚙️ Configure deployment**
   ```
   Repository: your-username/MedicoRAG-system
   Branch: main
   Main file path: app.py
   ```

4. **🔐 Set up secrets**
   
   In Streamlit Cloud, go to **App settings > Secrets** and add:
   ```toml
   # Required secrets
   GROQ_API_KEY = "your_groq_api_key_here"
   
   # Optional secrets (with defaults)
   GROQ_MODEL = "llama3-8b-8192"
   EMBEDDING_MODEL = "tfidf"
   VECTOR_STORE_DIRECTORY = "./vectordb_store"
   COLLECTION_NAME = "medical_documents"
   CHUNK_SIZE = "1000"
   CHUNK_OVERLAP = "200"
   TOP_K_RETRIEVAL = "5"
   MAX_RESPONSE_TOKENS = "800"
   RESPONSE_TEMPERATURE = "0.1"
   LOG_LEVEL = "INFO"
   ```

5. **🚀 Deploy**
   - Click "Deploy!"
   - Wait for the app to build and deploy
   - Your app will be live at: `https://your-app-name.streamlit.app`

#### 🎯 Get Your Groq API Key

1. Visit [console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key and add it to Streamlit secrets

#### 📋 Deployment Checklist

- [ ] Repository forked to your GitHub
- [ ] Streamlit Cloud account created
- [ ] Groq API key obtained
- [ ] Secrets configured in Streamlit Cloud
- [ ] App deployed and running
- [ ] Test document upload functionality
- [ ] Test medical query functionality

#### 🔧 Troubleshooting Deployment

**Common Issues:**

| Issue | Solution |
|-------|----------|
| **Build fails** | Check `requirements.txt` for compatibility |
| **API errors** | Verify `GROQ_API_KEY` in secrets |
| **Memory issues** | Restart the app in Streamlit Cloud |
| **Slow performance** | Check vector store initialization |

**Performance Tips:**
- Vector store persists between sessions
- First query may take longer (model loading)
- Subsequent queries are faster
- Consider upgrading to Streamlit Cloud Pro for better performance

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- Git
- Virtual environment (recommended)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/Pseudo-Sid26/MedicoRAG-system.git
   cd MedicoRAG-system
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configurations
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Open in browser**
   ```
   Navigate to: http://localhost:8501
   ```

---

## 📖 How It Works

```mermaid
graph TD
    A[📄 Upload Medical Documents] --> B[🔍 Document Processing]
    B --> C[🧠 Text Extraction & Chunking]
    C --> D[🔢 Generate Embeddings]
    D --> E[💾 Store in Vector DB]
    E --> F[💬 User Query]
    F --> G[🔍 Semantic Search]
    G --> H[📊 Retrieve Relevant Chunks]
    H --> I[🤖 LLM Processing]
    I --> J[✅ Generate Response with Citations]
```

### Step-by-Step Process

1. **📤 Document Upload**: Healthcare professionals upload medical documents (PDFs, research papers, clinical guidelines)

2. **🔧 Processing Pipeline**: Documents are processed using OCR (if needed), text extraction, and chunking

3. **🔢 Embedding Generation**: Text chunks are converted to vector embeddings using custom TF-IDF implementation

4. **💾 Vector Storage**: Embeddings are stored in FAISS for efficient similarity search

5. **💬 Query Processing**: Users ask medical questions in natural language

6. **🔍 Retrieval**: System performs semantic search to find most relevant document chunks using FAISS

7. **🤖 Answer Generation**: Groq LLM generates evidence-based responses with proper citations

---

## 📌 Example Use Cases

<div align="center">

| Use Case | Description | Example Query |
|----------|-------------|---------------|
| 🩻 **Imaging Protocols** | Review radiology guidelines | "What is the standard protocol for chest CT scanning?" |
| 💊 **Drug Dosage** | Check medication guidelines | "What is the recommended dosage of metformin for type 2 diabetes?" |
| 📑 **Clinical Trials** | Summarize research findings | "Summarize recent trials on immunotherapy for lung cancer" |
| 📋 **Case Analysis** | Interpret patient data | "Analyze symptoms: fever, cough, shortness of breath" |

</div>

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# API Keys (Required for deployment)
GROQ_API_KEY=your_groq_api_key_here

# Model Configuration
GROQ_MODEL=llama3-8b-8192
EMBEDDING_MODEL=tfidf

# Database Configuration
VECTOR_STORE_DIRECTORY=./vectordb_store
COLLECTION_NAME=medical_documents

# Application Settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RETRIEVAL=5
MAX_RESPONSE_TOKENS=800
RESPONSE_TEMPERATURE=0.1

# Processing Settings
MAX_FILE_SIZE=104857600
LOG_LEVEL=INFO
```

### Customization Options

- **Document Types**: Configure supported file formats
- **Chunk Size**: Adjust text chunking parameters
- **Model Selection**: Choose different LLM and embedding models
- **UI Themes**: Customize Streamlit interface

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### How to Contribute

1. **🍴 Fork the repository**
2. **🌿 Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **💻 Make your changes**
4. **✅ Commit your changes**
   ```bash
   git commit -m 'Add some amazing feature'
   ```
5. **📤 Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
6. **🔄 Open a Pull Request**

### Contribution Guidelines

- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation as needed
- Ensure HIPAA compliance for medical data handling

### Areas for Contribution

- 🐛 Bug fixes and improvements
- 📚 Documentation enhancements
- 🧪 Additional test coverage
- 🎨 UI/UX improvements
- 🔧 New feature development

---

<div align="center">

### 📞 Contact & Support

**Developed by:** [Siddhesh Chavan](https://github.com/Pseudo-Sid26)

[![GitHub followers](https://img.shields.io/github/followers/Pseudo-Sid26?style=social)](https://github.com/Pseudo-Sid26)
[![GitHub stars](https://img.shields.io/github/stars/Pseudo-Sid26/MedicoRAG-system?style=social)](https://github.com/Pseudo-Sid26/MedicoRAG-system)

**⭐ Star this repository if you find it helpful!**

---

*Made with ❤️ for the healthcare community*

</div>
