import os
from dotenv import load_dotenv

load_dotenv()

# ============================================
# SimultanePro - Konfigürasyonu (SECURE)
# ============================================

ALLOW_LOCAL = os.getenv("ALLOW_LOCAL", "false").lower() == "true"

# --- Deepgram ---
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# --- OpenAI ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODELS = ["gpt-4o-mini", "gpt-4o"]

# --- Anthropic ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODELS = ["claude-haiku-4-5-20251001", "claude-sonnet-4-6"]

# --- Whisper ---
WHISPER_MODEL = "base"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

# --- Ollama ---
OLLAMA_MODEL = "mistral"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# --- Kimlik Doğrulama ---
AUTH_USERNAME = os.getenv("AUTH_USERNAME", "admin")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD", "")
SECRET_KEY = os.getenv("SECRET_KEY", "changeme-please-set-in-env")

# --- Sunucu ---
HOST = "0.0.0.0"
PORT = int(os.getenv("PORT", "8001"))

# --- Diller ---
STT_LANGUAGES = [
    {"code": "tr", "name": "Türkçe"},
    {"code": "en-US", "name": "İngilizce"},
    {"code": "de", "name": "Almanca"},
    {"code": "fr", "name": "Fransızca"},
    {"code": "es", "name": "İspanyolca"},
    {"code": "it", "name": "İtalyanca"},
    {"code": "ar", "name": "Arapça"},
]
TRANSLATION_LANGUAGES = [
    {"code": "tr", "name": "Türkçe"},
    {"code": "en", "name": "İngilizce"},
    {"code": "de", "name": "Almanca"},
    {"code": "fr", "name": "Fransızca"},
    {"code": "es", "name": "İspanyolca"},
    {"code": "it", "name": "İtalyanca"},
    {"code": "ar", "name": "Arapça"},
]
