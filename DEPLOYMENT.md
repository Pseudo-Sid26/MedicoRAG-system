# 🚀 Streamlit Cloud Deployment Guide

This guide will help you deploy the MedicoRAG system to Streamlit Cloud.

## Prerequisites

- GitHub account
- Streamlit Cloud account (free at [share.streamlit.io](https://share.streamlit.io))
- Groq API key (free at [console.groq.com](https://console.groq.com))

## Step-by-Step Deployment

### 1. Fork the Repository

1. Go to the main repository page

3. Select your GitHub account as the destination

### 2. Get Your Groq API Key

1. Visit [console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Navigate to the "API Keys" section
4. Click "Create API Key"
5. Copy the generated key (you'll need this later)

### 3. Connect to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your forked repository
5. Set the branch to `main`
6. Set the main file path to `app.py`

### 4. Configure Secrets

In the Streamlit Cloud app settings, go to the "Secrets" tab and add:

```toml
GROQ_API_KEY = "your_actual_groq_api_key_here"
```

Optional secrets (these have defaults, but you can customize):

```toml
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

### 5. Deploy

1. Click "Deploy!" in Streamlit Cloud
2. Wait for the build to complete (usually 2-5 minutes)
3. Your app will be available at `https://your-app-name.streamlit.app`

## Testing Your Deployment

1. **Upload Test**: Try uploading a PDF document
2. **Query Test**: Ask a medical question
3. **Response Test**: Verify you get appropriate responses

## Troubleshooting

### GROQ_API_KEY Error Fix

If you get the error: `ValueError: GROQ_API_KEY is required`, follow these steps:

1. **Access Your App Dashboard**:
   - Go to your deployed app URL
   - Click the hamburger menu (≡) in the bottom-right corner
   - Select "Manage app"

2. **Configure Secrets**:
   - In the app dashboard, click on "Settings" in the left sidebar
   - Click on the "Secrets" tab
   - Add your API key in TOML format:
   ```toml
   GROQ_API_KEY = "gsk_your_actual_groq_api_key_here"
   ```

3. **Save and Restart**:
   - Click "Save" 
   - The app will automatically restart (takes 1-2 minutes)
   - The error should be resolved

### Common Issues

| Problem | Solution |
|---------|----------|
| `GROQ_API_KEY is required` | Follow the steps above to add API key to secrets |
| Build fails | Check requirements.txt compatibility |
| API errors | Verify GROQ_API_KEY is valid and has quota |
| Memory issues | Restart app in Streamlit Cloud dashboard |
| Slow loading | First load initializes models, subsequent loads are faster |

### Performance Notes

- The vector store persists between user sessions
- First query may take 30-60 seconds (model initialization)
- Subsequent queries are much faster
- Consider Streamlit Cloud Pro for better performance

## Security Notes

- Never commit API keys to your repository
- Use Streamlit Cloud secrets for all sensitive information
- The app is designed for healthcare professionals only

## Support

If you encounter issues:

1. Check the Streamlit Cloud logs
2. Verify all secrets are correctly configured
3. Ensure your Groq API key is valid and has quota remaining
4. Create an issue in the GitHub repository

## Updating Your Deployment

To update your deployed app:

1. Make changes to your forked repository
2. Push changes to the `main` branch
3. Streamlit Cloud will automatically redeploy

Happy deploying! 🎉
