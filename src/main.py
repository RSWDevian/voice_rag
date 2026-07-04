from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import staticfiles
from fastapi.responses import HTMLResponse
import asyncio
import json

from src.config import config
# from src.pipeline import Pipeline
from src.utils.logger import setup_logger

app = FastAPI(title="Streaming RAG Voice Assistant", version="1.0.0")
logger = setup_logger()
