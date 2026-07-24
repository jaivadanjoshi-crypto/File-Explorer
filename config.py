import os

from dotenv import load_dotenv

# Load variables from a local .env file (used only when running locally).
load_dotenv()


def _get_secret(name: str, default=None):
    """Read a secret from the environment first, then Streamlit secrets.

    Locally, values come from the .env file via os.getenv.
    On Streamlit Community Cloud there is no .env file, so we fall back
    to st.secrets, which is populated from Settings -> Secrets.
    """
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st

        return st.secrets[name]
    except Exception:
        return default


# NOTE: the variable name must have NO leading/trailing spaces.
GROQ_API_KEY = _get_secret("GROQ_API_KEY")

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "llama-3.3-70b-versatile"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
