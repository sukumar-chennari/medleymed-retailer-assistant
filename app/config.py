import os

from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
TEXT_MODEL = os.environ.get("OLLAMA_TEXT_MODEL", "llama3.2")
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_VISION_MODEL = os.environ.get("GEMINI_VISION_MODEL", "gemini-3.6-flash")

SMTP_HOST = os.environ.get("SMTP_HOST")
SMTP_PORT = os.environ.get("SMTP_PORT")
SMTP_USER = os.environ.get("SMTP_USER")
SMTP_PASS = os.environ.get("SMTP_PASS")
SMTP_FROM_NAME = os.environ.get("SMTP_FROM_NAME", "MedleyMed Orders")

SMTP_CONFIGURED = all([SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS])
