"""
Intent classification with Modular Service Mapping.
Lighweight and fast for low latency inference.
"""

import time
import json
from typing import List, Tuple, Dict, Optional, Any
from enum import Enum
from dataclasses import dataclass, field
from transformers import pipeline

from src.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class IntentType(Enum):
    """Standard intent types"""
    QUESTION = "question"
    COMMAND = "command"

    GREETING = "greeting"
    FAREWELL = "farewell"
    CLARIFICATION = "clarification"
    CONFIRMATIION = "confirmation"
    GENERAL = "general"

class ServiceType(Enum):
    """Standard service types"""
    VECTOR_SEARCH = "vector_search"
    SQL_DATABASE = "sql_database"
    MCP_SERVER = "mcp_server"
    TOOL_FUNCTION = "tool_function"

@dataclass
class ServiceRoute:
    """Configuration for routing an intent to a service"""
    service_type: ServiceType
    service_name: str
    priority: int = 1
    parameters: Dict[str, Any] = field(default_factory=dict)
    fallback_chain: List[str] = field(default_factory=list)
    timeout_ms: int = 500
    requires_auth: bool = False


@dataclass
class IntentMapping:
    """Mapping from intent to service route"""
    intent: str
    primary_service: ServiceRoute
    fallback_services: List[ServiceRoute] = field(default_factory=list)
    confidence_threshold: float = 0.6
    entity_extraction: bool = False
    context_required: bool = False

class ModularIntentClassifier:
    """
    Enhanced intent classifier with modular service mapping
    Routes intents to appropriate services (MCP, KG, Vector, etc.)
    """
    
    def __init__(self):
        """Initialize with modular service routing"""
        self._model = None
        self._model_loaded = False
        
        # Performance tracking
        self.total_predictions = 0
        self.total_latency_ms = 0.0
        
        # Intent to service mappings
        self.intent_service_map: Dict[str, IntentMapping] = {}
        self._load_intent_mappings()
        
        # Keyword-based intent detection with service hints
        self.keyword_rules: Dict[str, Dict] = {}
        self._load_keyword_rules()
        
        # Load model
        self._load_model()
        
        logger.info(f"Modular intent classifier initialized: model_loaded={self._model_loaded}, "
                   f"mappings={len(self.intent_service_map)}")
    
    def _load_intent_mappings(self):
        """Load intent to service mappings from configuration"""
        
        # Define service mappings for different intents
        mappings = {
            "knowledge_query": IntentMapping(
                intent="knowledge_query",
                primary_service=ServiceRoute(
                    service_type=ServiceType.VECTOR_SEARCH,
                    service_name="document_vector_store",
                    priority=1,
                    parameters={"top_k": 3, "similarity_threshold": 0.7}
                ),
                fallback_services=[
                    ServiceRoute(
                        service_type=ServiceType.KNOWLEDGE_GRAPH,
                        service_name="neo4j_knowledge_graph",
                        priority=2
                    ),
                    ServiceRoute(
                        service_type=ServiceType.SQL_DATABASE,
                        service_name="postgres_knowledge",
                        priority=3
                    )
                ],
                confidence_threshold=0.6,
                entity_extraction=True
            ),
            
            "data_query": IntentMapping(
                intent="data_query",
                primary_service=ServiceRoute(
                    service_type=ServiceType.SQL_DATABASE,
                    service_name="postgres_analytics",
                    priority=1,
                    parameters={"limit": 10}
                ),
                fallback_services=[
                    ServiceRoute(
                        service_type=ServiceType.VECTOR_SEARCH,
                        service_name="document_vector_store",
                        priority=2
                    )
                ],
                entity_extraction=True
            ),
            
            "tool_execution": IntentMapping(
                intent="tool_execution",
                primary_service=ServiceRoute(
                    service_type=ServiceType.MCP_SERVER,
                    service_name="mcp_tool_server",
                    priority=1,
                    parameters={}
                ),
                fallback_services=[
                    ServiceRoute(
                        service_type=ServiceType.TOOL_FUNCTION,
                        service_name="local_function_tools",
                        priority=2
                    ),
                    ServiceRoute(
                        service_type=ServiceType.API_CALL,
                        service_name="external_api",
                        priority=3
                    )
                ],
                confidence_threshold=0.5,
                context_required=True
            ),
            
            "graph_query": IntentMapping(
                intent="graph_query",
                primary_service=ServiceRoute(
                    service_type=ServiceType.KNOWLEDGE_GRAPH,
                    service_name="neo4j_knowledge_graph",
                    priority=1,
                    parameters={"depth": 2}
                ),
                fallback_services=[
                    ServiceRoute(
                        service_type=ServiceType.VECTOR_SEARCH,
                        service_name="document_vector_store",
                        priority=2
                    )
                ]
            ),
            
            "api_call": IntentMapping(
                intent="api_call",
                primary_service=ServiceRoute(
                    service_type=ServiceType.API_CALL,
                    service_name="external_api_gateway",
                    priority=1,
                    parameters={"method": "GET"}
                ),
                fallback_services=[
                    ServiceRoute(
                        service_type=ServiceType.MCP_SERVER,
                        service_name="mcp_api_server",
                        priority=2
                    )
                ]
            ),
            
            "general": IntentMapping(
                intent="general",
                primary_service=ServiceRoute(
                    service_type=ServiceType.FALLBACK,
                    service_name="llm_fallback",
                    priority=1
                ),
                confidence_threshold=0.3
            )
        }
        
        self.intent_service_map = mappings
    
    def _load_keyword_rules(self):
        """Load keyword-based intent detection rules with service hints"""
        
        self.keyword_rules = {
            "weather": {
                "keywords": ["weather", "temperature", "forecast", "rain", "sunny"],
                "intent": "knowledge_query",
                "service_hint": ServiceType.API_CALL,
                "confidence": 0.9
            },
            "search": {
                "keywords": ["search", "find", "look up", "retrieve", "get information"],
                "intent": "knowledge_query",
                "service_hint": ServiceType.VECTOR_SEARCH,
                "confidence": 0.8
            },
            "calculate": {
                "keywords": ["calculate", "compute", "sum", "average", "count"],
                "intent": "data_query",
                "service_hint": ServiceType.SQL_DATABASE,
                "confidence": 0.85
            },
            "execute": {
                "keywords": ["run", "execute", "perform", "do", "start"],
                "intent": "tool_execution",
                "service_hint": ServiceType.MCP_SERVER,
                "confidence": 0.75
            },
            "graph": {
                "keywords": ["relationship", "connection", "network", "linked", "connected"],
                "intent": "graph_query",
                "service_hint": ServiceType.KNOWLEDGE_GRAPH,
                "confidence": 0.8
            },
            "api": {
                "keywords": ["api", "endpoint", "call", "fetch", "retrieve from"],
                "intent": "api_call",
                "service_hint": ServiceType.API_CALL,
                "confidence": 0.85
            }
        }
    
    def _load_model(self):
        """Load DistilBERT model for intent classification"""
        try:
            start_time = time.time()
            
            self._model = pipeline(
                "text-classification",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1,
                model_kwargs={"torchscript": True},
                framework="pt"
            )
            
            self._model_loaded = True
            load_time = (time.time() - start_time) * 1000
            logger.info(f"Intent classifier loaded in {load_time:.0f}ms")
            
        except Exception as e:
            logger.warning(f"Failed to load intent classifier: {e}")
            self._model_loaded = False
    
    def classify(self, text: str) -> Tuple[str, ServiceRoute, float]:
        """
        Classify intent and return corresponding service route
        
        Args:
            text: Input text
            
        Returns:
            Tuple[str, ServiceRoute, float]: (intent_type, service_route, confidence)
        """
        if not text or not text.strip():
            return "general", self.intent_service_map["general"].primary_service, 0.0
        
        try:
            start_time = time.time()
            text_trimmed = text[:100]
            
            # 1. Check keyword rules first (fastest)
            keyword_intent, keyword_confidence, keyword_service = self._match_keywords(text_trimmed)
            
            if keyword_confidence > 0.8:
                # High confidence keyword match - use directly
                intent = keyword_intent
                service_route = keyword_service
                confidence = keyword_confidence
                logger.debug(f"Keyword match: {intent} ({confidence:.2f})")
            
            # 2. Use model-based classification
            elif self._model_loaded:
                result = self._model(text_trimmed)[0]
                label = result['label']
                model_confidence = result['score']
                
                # Map model output to intent
                intent = self._map_model_to_intent(label)
                confidence = model_confidence
                
                # Get service route for intent
                service_route = self._get_service_route(intent)
                
                # If model confidence is low but keyword match exists, use keyword
                if confidence < 0.6 and keyword_confidence > 0.5:
                    intent = keyword_intent
                    service_route = keyword_service
                    confidence = max(confidence, keyword_confidence)
                
                logger.debug(f"Model prediction: {intent} ({confidence:.2f})")
            
            # 3. Fallback to keyword-only
            else:
                intent = keyword_intent if keyword_confidence > 0.3 else "general"
                service_route = keyword_service if keyword_confidence > 0.3 else self.intent_service_map["general"].primary_service
                confidence = max(keyword_confidence, 0.3)
            
            # Track performance
            latency_ms = (time.time() - start_time) * 1000
            self.total_predictions += 1
            self.total_latency_ms += latency_ms
            
            return intent, service_route, confidence
            
        except Exception as e:
            logger.warning(f"Intent classification error: {e}")
            return "general", self.intent_service_map["general"].primary_service, 0.0
    
    def _match_keywords(self, text: str) -> Tuple[str, float, ServiceRoute]:
        """Match keywords to determine intent and service route"""
        text_lower = text.lower()
        best_match = "general"
        best_confidence = 0.0
        best_service = self.intent_service_map["general"].primary_service
        
        for rule_name, rule in self.keyword_rules.items():
            confidence = 0.0
            matched_keywords = 0
            
            for keyword in rule["keywords"]:
                if keyword in text_lower:
                    matched_keywords += 1
            
            if matched_keywords > 0:
                # Calculate confidence based on matched keywords ratio
                confidence = (matched_keywords / len(rule["keywords"])) * rule["confidence"]
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = rule["intent"]
                    
                    # Get service route for this intent
                    intent_mapping = self.intent_service_map.get(rule["intent"])
                    if intent_mapping:
                        best_service = intent_mapping.primary_service
                    else:
                        # Use service hint if available
                        service_type = rule.get("service_hint")
                        if service_type:
                            best_service = ServiceRoute(
                                service_type=service_type,
                                service_name=f"{service_type.value}_service",
                                priority=1
                            )
        
        return best_match, best_confidence, best_service
    
    def _map_model_to_intent(self, model_label: str) -> str:
        """Map model output to intent type"""
        mapping = {
            "POSITIVE": "knowledge_query",
            "NEGATIVE": "tool_execution",
            "NEUTRAL": "general"
        }
        return mapping.get(model_label, "general")
    
    def _get_service_route(self, intent: str) -> ServiceRoute:
        """Get service route for an intent"""
        intent_mapping = self.intent_service_map.get(intent)
        if intent_mapping:
            return intent_mapping.primary_service
        
        # Fallback to general
        return self.intent_service_map["general"].primary_service
    
    def get_routing_chain(self, intent: str) -> List[ServiceRoute]:
        """
        Get full routing chain for an intent (primary + fallbacks)
        
        Args:
            intent: Intent type
            
        Returns:
            List[ServiceRoute]: Ordered list of services to try
        """
        intent_mapping = self.intent_service_map.get(intent)
        if not intent_mapping:
            return [self.intent_service_map["general"].primary_service]
        
        routing_chain = [intent_mapping.primary_service]
        routing_chain.extend(intent_mapping.fallback_services)
        
        # Sort by priority
        routing_chain.sort(key=lambda x: x.priority)
        
        return routing_chain
    
    def get_service_requirements(self, service_route: ServiceRoute) -> Dict[str, Any]:
        """
        Get requirements for a service (entities, context, etc.)
        
        Args:
            service_route: Service route configuration
            
        Returns:
            Dict: Service requirements
        """
        requirements = {
            "service_type": service_route.service_type.value,
            "service_name": service_route.service_name,
            "parameters": service_route.parameters,
            "timeout_ms": service_route.timeout_ms,
            "requires_auth": service_route.requires_auth
        }
        
        # Add intent-specific requirements
        for intent_mapping in self.intent_service_map.values():
            if intent_mapping.primary_service.service_name == service_route.service_name:
                requirements["entity_extraction"] = intent_mapping.entity_extraction
                requirements["context_required"] = intent_mapping.context_required
                requirements["confidence_threshold"] = intent_mapping.confidence_threshold
                break
        
        return requirements
    
    def add_custom_mapping(self, intent: str, mapping: IntentMapping):
        """
        Add custom intent-to-service mapping
        
        Args:
            intent: Intent name
            mapping: IntentMapping configuration
        """
        self.intent_service_map[intent] = mapping
        logger.info(f"Added custom mapping for intent: {intent}")
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics"""
        if self.total_predictions == 0:
            return {
                "total_predictions": 0,
                "avg_latency_ms": 0.0,
                "total_latency_ms": 0.0,
                "model_loaded": self._model_loaded,
                "mappings_count": len(self.intent_service_map)
            }
        
        return {
            "total_predictions": self.total_predictions,
            "avg_latency_ms": self.total_latency_ms / self.total_predictions,
            "total_latency_ms": self.total_latency_ms,
            "model_loaded": self._model_loaded,
            "mappings_count": len(self.intent_service_map)
        }


# Singleton instance
_intent_instance = None


def get_intent_classifier() -> ModularIntentClassifier:
    """Get or create global intent classifier instance"""
    global _intent_instance
    
    if _intent_instance is None:
        _intent_instance = ModularIntentClassifier()
    
    return _intent_instance


# Example usage
if __name__ == "__main__":
    classifier = get_intent_classifier()
    
    test_queries = [
        "What is the weather today?",
        "Search for documents about AI",
        "Calculate the average sales",
        "Execute the data processing pipeline",
        "Show me relationships in the knowledge graph",
        "Call the weather API",
        "Hello, how are you?"
    ]
    
    print("Modular Intent Classification Test:")
    print("=" * 60)
    
    for query in test_queries:
        intent, service, confidence = classifier.classify(query)
        routing_chain = classifier.get_routing_chain(intent)
        
        print(f"\nQuery: '{query}'")
        print(f"  Intent: {intent}")
        print(f"  Confidence: {confidence:.2f}")
        print(f"  Primary Service: {service.service_type.value} -> {service.service_name}")
        
        if len(routing_chain) > 1:
            print(f"  Fallback Chain: {[s.service_name for s in routing_chain[1:]]}")
        
        requirements = classifier.get_service_requirements(service)
        if requirements.get("entity_extraction"):
            print(f"  Requires Entity Extraction: Yes")
        if requirements.get("context_required"):
            print(f"  Requires Context: Yes")
    
    print("\nPerformance Stats:")
    print(classifier.get_performance_stats())
