# src/query_builder.py
"""
Query Builder - Constructs optimized queries for retrieval
Supports multiple query types and context management
"""

import re
import time
from typing import List, Dict, Any, Optional, Tuple
from collections import deque

from src.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class QueryBuilder:
    """
    Builds optimized queries for retrieval with context management
    Supports multiple query types and entity extraction
    """
    
    def __init__(self, max_context: int = 3):
        """Initialize query builder with context window"""
        self.context_window = deque(maxlen=max_context * 2)  # Store Q&A pairs
        self.max_context = max_context
        self.stopwords = self._load_stopwords()
        
        logger.info(f"Query builder initialized: max_context={max_context}")
    
    def _load_stopwords(self) -> set:
        """Load stopwords for keyword extraction"""
        return {
            'what', 'how', 'why', 'when', 'where', 'who', 'which', 'whom',
            'is', 'are', 'was', 'were', 'am', 'be', 'been', 'being',
            'the', 'a', 'an', 'of', 'to', 'for', 'with', 'on', 'at', 'from',
            'by', 'in', 'into', 'through', 'during', 'including', 'without',
            'do', 'does', 'did', 'done', 'doing', 'has', 'have', 'had',
            'can', 'could', 'will', 'would', 'should', 'may', 'might', 'must'
        }
    
    def build_query(self, transcript: str, intent: str, speaker_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Build structured query from transcript and intent
        
        Args:
            transcript: Transcribed text
            intent: Classified intent
            speaker_id: Optional speaker identifier
            
        Returns:
            Dict with query components
        """
        start_time = time.time()
        
        # Clean and normalize transcript
        cleaned_text = self._clean_text(transcript)
        
        # Extract keywords and entities
        keywords = self._extract_keywords(cleaned_text)
        entities = self._extract_entities(cleaned_text)
        
        # Build context
        context = self._get_context()
        
        query = {
            "text": cleaned_text,
            "raw_text": transcript,
            "intent": intent,
            "keywords": keywords,
            "entities": entities,
            "context": context,
            "speaker_id": speaker_id,
            "timestamp": time.time(),
            "query_type": self._determine_query_type(intent, keywords)
        }
        
        # Log query building
        latency_ms = (time.time() - start_time) * 1000
        logger.debug(f"Query built: '{cleaned_text[:30]}...' ({latency_ms:.2f}ms)")
        
        return query
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove special characters (keep alphanumeric and punctuation)
        text = re.sub(r'[^a-zA-Z0-9\s\.\,\?\']', '', text)
        
        return text.strip()
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        words = text.lower().split()
        
        # Filter stopwords and short words
        keywords = [
            w for w in words 
            if w not in self.stopwords 
            and len(w) > 2
            and not w.isdigit()
        ]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for w in keywords:
            if w not in seen:
                seen.add(w)
                unique_keywords.append(w)
        
        return unique_keywords[:10]  # Limit to top 10
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text
        Simple implementation - can be replaced with NER
        """
        entities = {
            "dates": [],
            "numbers": [],
            "locations": [],
            "names": []
        }
        
        # Extract dates (simple pattern)
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # 12/31/2023
            r'\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{2,4}',
            r'(today|tomorrow|yesterday|monday|tuesday|wednesday|thursday|friday|saturday|sunday)'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["dates"].extend(matches)
        
        # Extract numbers
        numbers = re.findall(r'\b\d+\.?\d*\b', text)
        entities["numbers"] = numbers[:5]  # Limit
        
        # Extract potential names (capitalized words not at start of sentence)
        words = text.split()
        for i, word in enumerate(words):
            if (word[0].isupper() and len(word) > 1 and 
                i > 0 and words[i-1][-1] in '.!?'):
                entities["names"].append(word)
        
        return {k: list(set(v)) for k, v in entities.items()}
    
    def _determine_query_type(self, intent: str, keywords: List[str]) -> str:
        """Determine type of query based on intent and keywords"""
        query_type_map = {
            "question": "factual",
            "command": "action",
            "greeting": "social",
            "farewell": "social",
            "knowledge_query": "knowledge",
            "tool_execution": "action",
            "data_query": "data",
            "graph_query": "relationship"
        }
        
        return query_type_map.get(intent, "general")
    
    def _get_context(self) -> str:
        """Get recent conversation context"""
        if not self.context_window:
            return ""
        
        # Get last N context items
        context_items = list(self.context_window)[-self.max_context:]
        return " ".join(context_items)
    
    def update_context(self, query: Dict[str, Any], response: str):
        """
        Update context window with latest interaction
        
        Args:
            query: Query dictionary
            response: Response text
        """
        if query and response:
            context_entry = f"Q: {query['text']} A: {response[:100]}"
            self.context_window.append(context_entry)
            logger.debug(f"Context updated: {len(self.context_window)} entries")
    
    def get_conversation_history(self, max_turns: Optional[int] = None) -> List[str]:
        """
        Get conversation history
        
        Args:
            max_turns: Maximum number of turns to return
            
        Returns:
            List of conversation turns
        """
        if max_turns:
            return list(self.context_window)[-max_turns:]
        return list(self.context_window)
    
    def clear_context(self):
        """Clear conversation context"""
        self.context_window.clear()
        logger.info("Context cleared")
    
    def expand_query(self, query: Dict[str, Any]) -> List[str]:
        """
        Generate multiple query variants for better retrieval
        
        Args:
            query: Original query dictionary
            
        Returns:
            List of query variations
        """
        variants = []
        text = query["text"]
        keywords = query["keywords"]
        
        # Original query
        variants.append(text)
        
        # Keyword-only query
        if keywords:
            variants.append(" ".join(keywords[:5]))
        
        # Query with context
        if query.get("context"):
            variants.append(f"{query['context']} {text}")
        
        # Entity-focused query
        entities = query.get("entities", {})
        if entities.get("names"):
            variants.append(f"{' '.join(entities['names'])} {text}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_variants = []
        for v in variants:
            if v not in seen:
                seen.add(v)
                unique_variants.append(v)
        
        return unique_variants[:5]  # Limit to 5 variants
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics"""
        return {
            "context_size": len(self.context_window),
            "max_context": self.max_context,
            "stopwords_count": len(self.stopwords)
        }


# Singleton instance
_query_builder_instance = None


def get_query_builder() -> QueryBuilder:
    """Get or create global query builder instance"""
    global _query_builder_instance
    
    if _query_builder_instance is None:
        _query_builder_instance = QueryBuilder()
    
    return _query_builder_instance


# Example usage
if __name__ == "__main__":
    # Test query builder
    qb = QueryBuilder()
    
    # Test queries
    test_queries = [
        ("What is the weather like in New York today?", "question"),
        ("Search for documents about artificial intelligence", "command"),
        ("Show me the latest sales data", "data_query")
    ]
    
    print("Query Builder Test:")
    print("=" * 60)
    
    for text, intent in test_queries:
        query = qb.build_query(text, intent)
        
        print(f"\nInput: '{text}'")
        print(f"Intent: {intent}")
        print(f"Keywords: {query['keywords']}")
        print(f"Entities: {query['entities']}")
        print(f"Query Type: {query['query_type']}")
        
        # Test expansion
        variants = qb.expand_query(query)
        print(f"Variants: {variants}")
        
        # Update context
        qb.update_context(query, f"Test response to: {text[:30]}")
    
    print(f"\nContext History: {qb.get_conversation_history()}")
    print(f"Performance: {qb.get_performance_stats()}")