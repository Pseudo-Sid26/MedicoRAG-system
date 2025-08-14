#!/usr/bin/env python3
"""
Setup script for Medical Literature RAG System
Handles installation, configuration, and initial setup
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import argparse


def run_command(command, check=True):
    """Run a shell command and handle errors"""
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")


def create_virtual_environment():
    """Create virtual environment"""
    print("Creating virtual environment...")
    venv_path = Path("medical_rag_env")

    if venv_path.exists():
        print("Virtual environment already exists")
        return True

    return run_command(f"{sys.executable} -m venv medical_rag_env")


def activate_virtual_environment():
    """Get activation command for virtual environment"""
    if os.name == 'nt':  # Windows
        return "medical_rag_env\\Scripts\\activate"
    else:  # Unix/Linux/MacOS
        return "source medical_rag_env/bin/activate"


def install_requirements():
    """Install Python requirements"""
    print("Installing Python requirements...")

    # Get the correct pip path
    if os.name == 'nt':  # Windows
        pip_path = "medical_rag_env\\Scripts\\pip"
    else:  # Unix/Linux/MacOS
        pip_path = "medical_rag_env/bin/pip"

    # Upgrade pip first
    if not run_command(f"{pip_path} install --upgrade pip"):
        return False

    # Install requirements
    return run_command(f"{pip_path} install -r requirements.txt")


def download_spacy_model():
    """Download required spaCy model"""
    print("Downloading spaCy English model...")

    if os.name == 'nt':  # Windows
        python_path = "medical_rag_env\\Scripts\\python"
    else:  # Unix/Linux/MacOS
        python_path = "medical_rag_env/bin/python"

    return run_command(f"{python_path} -m spacy download en_core_web_sm")


def create_directories():
    """Create necessary directories"""
    print("Creating directory structure...")

    directories = [
        "vector_store",
        "logs",
        "data",
        "data/uploads",
        "data/processed",
        "tests"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

    return True


def create_env_file():
    """Create .env file from template"""
    print("Creating environment configuration...")

    env_template = """# Medical RAG System Environment Variables

# Groq API Configuration (REQUIRED)
GROQ_API_KEY=your_groq_api_key_here

# HuggingFace Configuration (Optional)
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# Application Configuration
APP_ENV=development
DEBUG=true

# Vector Store Configuration
CHROMA_PERSIST_DIRECTORY=./vector_store
COLLECTION_NAME=medical_documents

# Model Configuration
GROQ_MODEL=mixtral-8x7b-32768
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Chunking Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Retrieval Configuration
TOP_K_RETRIEVAL=5
SIMILARITY_THRESHOLD=0.7

# Security Configuration
MAX_FILE_SIZE=104857600  # 100MB
ALLOWED_EXTENSIONS=.pdf,.txt,.docx,.csv

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=./logs/medical_rag.log
"""

    env_path = Path(".env")
    if not env_path.exists():
        with open(env_path, "w") as f:
            f.write(env_template)
        print("✅ Created .env file")
        print("⚠️  Please edit .env file and add your Groq API key")
    else:
        print("✅ .env file already exists")

    return True


def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
medical_rag_env/
venv/
env/

# Environment Variables
.env
.env.local

# Vector Store
vector_store/

# Logs
logs/
*.log

# Data
data/uploads/*
data/processed/*
!data/uploads/.gitkeep
!data/processed/.gitkeep

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# Cache
.cache/
*.cache

# Temporary files
*.tmp
*.temp
"""

    gitignore_path = Path(".gitignore")
    if not gitignore_path.exists():
        with open(gitignore_path, "w") as f:
            f.write(gitignore_content)
        print("✅ Created .gitignore file")

    return True


def create_placeholder_files():
    """Create placeholder files for empty directories"""
    placeholders = [
        "data/uploads/.gitkeep",
        "data/processed/.gitkeep",
        "logs/.gitkeep"
    ]

    for placeholder in placeholders:
        placeholder_path = Path(placeholder)
        placeholder_path.parent.mkdir(parents=True, exist_ok=True)
        placeholder_path.touch()

    return True


def verify_installation():
    """Verify that installation was successful"""
    print("\nVerifying installation...")

    # Check if virtual environment exists
    venv_path = Path("medical_rag_env")
    if not venv_path.exists():
        print("❌ Virtual environment not found")
        return False

    # Check if .env file exists
    if not Path(".env").exists():
        print("❌ .env file not found")
        return False

    # Check if key directories exist
    required_dirs = ["vector_store", "logs", "data"]
    for directory in required_dirs:
        if not Path(directory).exists():
            print(f"❌ Directory {directory} not found")
            return False

    print("✅ Installation verification passed")
    return True


def print_next_steps():
    """Print next steps for the user"""
    activation_cmd = activate_virtual_environment()

    print(f"""
🎉 Installation completed successfully!

Next steps:
1. Activate the virtual environment:
   {activation_cmd}

2. Edit the .env file and add your Groq API key:
   - Get a Groq API key from: https://console.groq.com/
   - Edit .env file and replace 'your_groq_api_key_here' with your actual key

3. Run the application:
   streamlit run app.py

4. Open your browser and go to: http://localhost:8501

📚 Documentation:
   - README.md: Complete setup and usage guide
   - config/settings.py: Configuration options
   - src/: Source code documentation

⚠️  Important:
   - This system is for healthcare professionals only
   - Always validate AI responses with current medical literature
   - Do not use for emergency medical situations

🔧 Troubleshooting:
   - Check logs in the logs/ directory
   - Verify API key configuration in .env
   - Ensure Python 3.8+ is installed
   - Check internet connection for model downloads

Happy coding! 🏥✨
""")


def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Setup Medical Literature RAG System")
    parser.add_argument("--skip-venv", action="store_true", help="Skip virtual environment creation")
    parser.add_argument("--skip-models", action="store_true", help="Skip model downloads")
    args = parser.parse_args()

    print("🏥 Medical Literature RAG System Setup")
    print("=" * 50)

    # Check Python version
    check_python_version()

    # Create virtual environment
    if not args.skip_venv:
        if not create_virtual_environment():
            print("❌ Failed to create virtual environment")
            sys.exit(1)

    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements")
        sys.exit(1)

    # Download models
    if not args.skip_models:
        if not download_spacy_model():
            print("⚠️  Warning: Failed to download spaCy model. You can download it later.")

    # Create directories
    if not create_directories():
        print("❌ Failed to create directories")
        sys.exit(1)

    # Create configuration files
    if not create_env_file():
        print("❌ Failed to create .env file")
        sys.exit(1)

    if not create_gitignore():
        print("⚠️  Warning: Failed to create .gitignore file")

    if not create_placeholder_files():
        print("⚠️  Warning: Failed to create placeholder files")

    # Verify installation
    if not verify_installation():
        print("❌ Installation verification failed")
        sys.exit(1)

    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    main()