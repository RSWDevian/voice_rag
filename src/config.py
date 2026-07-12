# config.py
import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

class Config:
    # OpenRouter Configuration
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_TIMEOUT: int = int(os.getenv("OPENROUTER_TIMEOUT", "30"))
    
    # STT Engine Selection
    # "v1" -> src/stt.py (OpenRouter API), "v2" -> src/stt_v2.py (local Moonshine)
    STT_ENGINE: str = os.getenv("STT_ENGINE", "v1")

    # Model Selection
    STT_MODEL: str = os.getenv("STT_MODEL", "google/gemini-2.5-flash")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "openai/gpt-4o-mini:nitro")
    ELEVENLABS_API_KEY: Optional[str] = os.getenv("ELEVENLABS_API_KEY")
    ELEVENLABS_VOICE_ID: Optional[str] = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
    ELEVENLABS_MODEL_ID: str = os.getenv("ELEVENLABS_MODEL_ID", "eleven_turbo_v2")
    TTS_MODEL: str = os.getenv("TTS_MODEL", "openai/gpt-4o-mini-tts-2025-12-15")
    TTS_VOICE: str = os.getenv("TTS_VOICE", "alloy")
    TTS_RESPONSE_FORMAT: str = os.getenv("TTS_RESPONSE_FORMAT", "mp3")
    TTS_SPEED: float = float(os.getenv("TTS_SPEED", "1.0"))
    TTS_ENABLED: bool = os.getenv("TTS_ENABLED", "true").lower() == "true"

    # TTS Engine Selection
    # "v1" -> src/tts.py (ElevenLabs API), "v2" -> src/tts_v2.py (local Piper)
    TTS_ENGINE: str = os.getenv("TTS_ENGINE", "v2")
    PIPER_VOICE: str = os.getenv("PIPER_VOICE", "en_US-lessac-medium")
    PIPER_USE_CUDA: bool = os.getenv("PIPER_USE_CUDA", "false").lower() == "true"
    
    # LLM Engine Selection
    # "v1" -> src/llm.py (OpenRouter API), "v2" -> src/llm_v2.py (local Ollama)
    LLM_ENGINE: str = os.getenv("LLM_ENGINE", "v1")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "ibm/granite3.1-moe:1b")

    # LLM Settings
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "80"))
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    
    # VAD Settings
    VAD_THRESHOLD: float = float(os.getenv("VAD_THRESHOLD", "0.5"))
    VAD_SAMPLE_RATE: int = int(os.getenv("VAD_SAMPLE_RATE", "16000"))
    
    # ASR Settings
    ASR_CHUNK_DURATION: float = float(os.getenv("ASR_CHUNK_DURATION", "0.5"))
    
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