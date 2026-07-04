# config.py
import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

class Config:
    # OpenRouter Configuration
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    
    # Model Selection
    STT_MODEL: str = os.getenv("STT_MODEL", "openai/whisper-large-v3")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "openai/gpt-4o-mini:nitro")
    TTS_MODEL: str = os.getenv("TTS_MODEL", "openai/gpt-4o-mini-tts-2025-12-15")
    TTS_VOICE: str = os.getenv("TTS_VOICE", "alloy")
    
    # LLM Settings
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "150"))
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    
    # VAD Settings
    VAD_THRESHOLD: float = float(os.getenv("VAD_THRESHOLD", "0.5"))
    VAD_SAMPLE_RATE: int = int(os.getenv("VAD_SAMPLE_RATE", "16000"))
    
    # ASR Settings
    ASR_CHUNK_DURATION: float = float(os.getenv("ASR_CHUNK_DURATION", "2.0"))
    
    # Vector Search
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "3"))
    
    # Redis
    REDIS_URL: Optional[str] = os.getenv("REDIS_URL")
    CACHE_ENABLED: bool = os.getenv("CACHE_ENABLED", "true").lower() == "true"
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))
    
    # Performance
    MAX_CONCURRENT_USERS: int = int(os.getenv("MAX_CONCURRENT_USERS", "10"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # API Settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR: str = os.path.join(BASE_DIR, "models")
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    STATIC_DIR: str = os.path.join(BASE_DIR, "static")
    LOGS_DIR: str = os.path.join(BASE_DIR, "logs")
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if not cls.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY is required")
        return True

config = Config()