import os
import streamlit as st
from dotenv import load_dotenv

# Load local environment if running locally
load_dotenv()

# Check local environment first, then look into Streamlit Secrets
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    try:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    except Exception:
        GROQ_API_KEY = None

# Inject it directly into the system climate so Langchain can find it instantly
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "llama-3.3-70b-versatile"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
