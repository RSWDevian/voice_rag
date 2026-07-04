# src/llm.py
"""
LLM Integration using OpenRouter Unified API
Supports streaming responses with low latency
"""

import json
import time
import asyncio
from typing import AsyncGenerator, Dict, List, Any, Optional, Union

import httpx

from src.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OpenRouterLLM:
    """
    OpenRouter LLM client with streaming support
    Optimized for low latency with model routing and caching
    """
    
    def __init__(self):
        """Initialize OpenRouter LLM client"""
        self.api_key = config.OPENROUTER_API_KEY
        self.base_url = config.OPENROUTER_BASE_URL
        self.model = config.LLM_MODEL
        self.max_tokens = config.LLM_MAX_TOKENS
        self.temperature = config.LLM_TEMPERATURE
        
        # Additional parameters
        self.top_p = getattr(config, 'LLM_TOP_P', 0.9)
        self.frequency_penalty = getattr(config, 'LLM_FREQUENCY_PENALTY', 0.0)
        self.presence_penalty = getattr(config, 'LLM_PRESENCE_PENALTY', 0.0)
        self.timeout = getattr(config, 'API_TIMEOUT', 10.0)
        
        # Performance tracking
        self.total_requests = 0
        self.total_latency_ms = 0.0
        self.total_tokens = 0
        
        # Create HTTP client with connection pooling
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            limits=httpx.Limits(
                max_keepalive_connections=10,
                max_connections=20,
                keepalive_expiry=30.0
            )
        )
        
        logger.info(f"LLM initialized: model={self.model}, max_tokens={self.max_tokens}")
    
    async def stream_response(
        self, 
        query: Dict[str, Any], 
        context: List[str],
        system_prompt: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream LLM response with low latency
        
        Args:
            query: Query dictionary containing text and metadata
            context: List of context strings from retrieval
            system_prompt: Optional custom system prompt
            
        Yields:
            str: Response chunks as they arrive
        """
        try:
            start_time = time.time()
            
            # Build prompt
            prompt = self._build_prompt(query, context)
            
            # Prepare messages
            messages = [
                {
                    "role": "system", 
                    "content": system_prompt or self._get_default_system_prompt()
                },
                {"role": "user", "content": prompt}
            ]
            
            # Prepare request
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "frequency_penalty": self.frequency_penalty,
                "presence_penalty": self.presence_penalty,
                "stream": True
            }
            
            # Track tokens if available
            if hasattr(config, 'LLM_TOP_K'):
                payload["top_k"] = config.LLM_TOP_K
            
            logger.debug(f"LLM request: {prompt[:100]}...")
            
            # Make streaming request
            async with self.client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost:8000",
                    "X-Title": "Streaming RAG Demo"
                },
                json=payload
            ) as response:
                
                # Check for errors
                if response.status_code != 200:
                    error_text = await response.aread()
                    logger.error(f"LLM API error: {response.status_code} - {error_text}")
                    yield f"Error: {response.status_code}"
                    return
                
                # Process stream
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    
                    data = line[6:]  # Remove "data: " prefix
                    
                    if data == "[DONE]":
                        break
                    
                    try:
                        chunk = json.loads(data)
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            
                            if content:
                                # Track tokens
                                self.total_tokens += 1
                                yield content
                                
                    except json.JSONDecodeError as e:
                        logger.debug(f"JSON decode error: {e}")
                        continue
                
                # Track performance
                latency_ms = (time.time() - start_time) * 1000
                self.total_requests += 1
                self.total_latency_ms += latency_ms
                
                # Log occasional stats
                if self.total_requests % 10 == 0:
                    avg_latency = self.total_latency_ms / self.total_requests
                    logger.debug(f"LLM stats: avg_latency={avg_latency:.2f}ms, "
                               f"requests={self.total_requests}")
                
                logger.debug(f"LLM response complete: {latency_ms:.0f}ms, "
                           f"tokens={self.total_tokens}")
                
        except httpx.TimeoutException:
            logger.error("LLM API timeout")
            yield "I'm having trouble connecting. Please try again."
        except Exception as e:
            logger.error(f"LLM streaming error: {e}")
            yield f"Error: {str(e)}"
    
    async def complete(
        self, 
        query: Dict[str, Any], 
        context: List[str]
    ) -> str:
        """
        Get complete response (non-streaming)
        
        Args:
            query: Query dictionary
            context: List of context strings
            
        Returns:
            str: Complete response text
        """
        response_chunks = []
        async for chunk in self.stream_response(query, context):
            response_chunks.append(chunk)
        return ''.join(response_chunks)
    
    def _build_prompt(self, query: Dict[str, Any], context: List[str]) -> str:
        """
        Build prompt from query and context
        
        Args:
            query: Query dictionary
            context: List of context strings
            
        Returns:
            str: Formatted prompt
        """
        # Build context string
        context_str = ""
        if context:
            # Join context with separators
            context_parts = []
            for i, ctx in enumerate(context, 1):
                if ctx and ctx.strip():
                    context_parts.append(f"{i}. {ctx.strip()}")
            
            if context_parts:
                context_str = "\n".join(context_parts)
        
        # Get query text
        query_text = query.get("text", "")
        intent = query.get("intent", "general")
        
        # Build prompt with structure
        prompt_parts = []
        
        # Add context if available
        if context_str:
            prompt_parts.append(f"Context:\n{context_str}\n")
        
        # Add query with intent
        if intent in ["question", "knowledge_query"]:
            prompt_parts.append(f"Question: {query_text}")
        elif intent in ["command", "tool_execution"]:
            prompt_parts.append(f"Command: {query_text}")
        else:
            prompt_parts.append(f"Query: {query_text}")
        
        # Add instruction for response
        prompt_parts.append("\nProvide a brief, direct answer (1-2 sentences).")
        
        return "\n".join(prompt_parts)
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt"""
        return """You are a helpful AI assistant with access to retrieved context. 
            Follow these guidelines:
            1. Base your answers on the provided context when available
            2. Keep responses brief (1-2 sentences) and direct
            3. If the context doesn't contain the answer, say so clearly
            4. Be concise - avoid unnecessary words
            5. Use natural, conversational language"""
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics"""
        if self.total_requests == 0:
            return {
                "total_requests": 0,
                "avg_latency_ms": 0.0,
                "total_latency_ms": 0.0,
                "total_tokens": 0,
                "avg_tokens_per_request": 0
            }
        
        return {
            "total_requests": self.total_requests,
            "avg_latency_ms": self.total_latency_ms / self.total_requests,
            "total_latency_ms": self.total_latency_ms,
            "total_tokens": self.total_tokens,
            "avg_tokens_per_request": self.total_tokens / self.total_requests
        }
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.total_requests = 0
        self.total_latency_ms = 0.0
        self.total_tokens = 0
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()
    
    def __del__(self):
        """Cleanup on deletion"""
        if hasattr(self, 'client'):
            try:
                asyncio.create_task(self.client.aclose())
            except:
                pass


class StreamingLLM:
    """
    Streaming LLM wrapper with buffering and callbacks
    Wraps OpenRouterLLM with additional streaming features
    """
    
    def __init__(self):
        """Initialize streaming LLM wrapper"""
        self.llm = OpenRouterLLM()
        self.buffer = []
        self.buffer_size = 0
        self.max_buffer_chars = 100  # Flush buffer when this many chars accumulated
        
        # Callbacks
        self.on_token_callbacks = []
        self.on_chunk_callbacks = []
        self.on_complete_callbacks = []
        
        logger.info("Streaming LLM initialized")
    
    async def stream_response(
        self,
        query: Dict[str, Any],
        context: List[str],
        system_prompt: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream response with buffering and callbacks
        
        Args:
            query: Query dictionary
            context: List of context strings
            system_prompt: Optional custom system prompt
            
        Yields:
            str: Response chunks (possibly batched)
        """
        chunk_buffer = []
        first_chunk = True
        start_time = time.time()
        
        async for token in self.llm.stream_response(query, context, system_prompt):
            # Trigger token callbacks
            for callback in self.on_token_callbacks:
                try:
                    await callback(token)
                except Exception as e:
                    logger.debug(f"Token callback error: {e}")
            
            # Buffer token
            chunk_buffer.append(token)
            self.buffer.append(token)
            self.buffer_size += len(token)
            
            # Yield chunk when buffer is full or on first token
            if len(chunk_buffer) >= 3 or self.buffer_size >= self.max_buffer_chars:
                chunk = ''.join(chunk_buffer)
                
                # Trigger chunk callbacks
                for callback in self.on_chunk_callbacks:
                    try:
                        await callback(chunk)
                    except Exception as e:
                        logger.debug(f"Chunk callback error: {e}")
                
                yield chunk
                
                # Reset chunk buffer
                chunk_buffer = []
                self.buffer_size = 0
        
        # Yield remaining chunk
        if chunk_buffer:
            chunk = ''.join(chunk_buffer)
            
            # Trigger chunk callbacks
            for callback in self.on_chunk_callbacks:
                try:
                    await callback(chunk)
                except Exception as e:
                    logger.debug(f"Chunk callback error: {e}")
            
            yield chunk
        
        # Trigger complete callbacks
        full_response = ''.join(self.buffer)
        for callback in self.on_complete_callbacks:
            try:
                await callback(full_response)
            except Exception as e:
                logger.debug(f"Complete callback error: {e}")
        
        # Log completion
        total_time = (time.time() - start_time) * 1000
        logger.debug(f"Stream complete: {len(full_response)} chars in {total_time:.0f}ms")
        
        # Reset buffer
        self.buffer = []
        self.buffer_size = 0
    
    async def complete(
        self,
        query: Dict[str, Any],
        context: List[str],
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Get complete response
        
        Args:
            query: Query dictionary
            context: List of context strings
            system_prompt: Optional custom system prompt
            
        Returns:
            str: Complete response text
        """
        response_parts = []
        async for chunk in self.stream_response(query, context, system_prompt):
            response_parts.append(chunk)
        return ''.join(response_parts)
    
    def add_token_callback(self, callback):
        """Add callback for each token"""
        self.on_token_callbacks.append(callback)
    
    def add_chunk_callback(self, callback):
        """Add callback for each chunk"""
        self.on_chunk_callbacks.append(callback)
    
    def add_complete_callback(self, callback):
        """Add callback for completion"""
        self.on_complete_callbacks.append(callback)
    
    def clear_callbacks(self):
        """Clear all callbacks"""
        self.on_token_callbacks.clear()
        self.on_chunk_callbacks.clear()
        self.on_complete_callbacks.clear()
    
    async def close(self):
        """Close underlying LLM client"""
        await self.llm.close()
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics"""
        return self.llm.get_performance_stats()


# Singleton instance
_llm_instance = None


def get_llm() -> StreamingLLM:
    """Get or create global LLM instance"""
    global _llm_instance
    
    if _llm_instance is None:
        _llm_instance = StreamingLLM()
    
    return _llm_instance


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_llm():
        """Test LLM streaming"""
        llm = StreamingLLM()
        
        # Test query
        query = {
            "text": "What is the capital of France?",
            "intent": "question",
            "keywords": ["capital", "France"]
        }
        context = [
            "France is a country in Western Europe.",
            "Paris is the capital and largest city of France."
        ]
        
        print("Testing LLM Streaming:")
        print("=" * 60)
        
        # Stream response
        response_chunks = []
        async for chunk in llm.stream_response(query, context):
            print(f"Chunk: {chunk}")
            response_chunks.append(chunk)
        
        full_response = ''.join(response_chunks)
        print(f"\nFull response: {full_response}")
        print(f"\nPerformance: {llm.get_performance_stats()}")
        
        await llm.close()
    
    asyncio.run(test_llm())