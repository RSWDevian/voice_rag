# src/metrics.py
"""
Metrics Collection and Performance Tracking
Monitors latency, throughput, and error rates for all pipeline components
"""

import time
import json
from typing import Dict, List, Any, Optional, Union
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MetricSnapshot:
    """Snapshot of metrics at a point in time"""
    timestamp: float
    component: str
    metric_type: str
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """
    Collects and tracks performance metrics for all pipeline components
    Supports multiple metric types with statistical analysis
    """
    
    # Metric types
    METRIC_LATENCY = "latency"
    METRIC_THROUGHPUT = "throughput"
    METRIC_ERROR = "error"
    METRIC_SUCCESS = "success"
    METRIC_COUNT = "count"
    METRIC_SIZE = "size"
    METRIC_DURATION = "duration"
    
    def __init__(self, max_samples: int = 1000):
        """
        Initialize metrics collector
        
        Args:
            max_samples: Maximum number of samples to keep per metric
        """
        self.max_samples = max_samples
        
        # Storage for metrics
        self.metrics: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=max_samples)))
        
        # Component-specific metrics
        self.component_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Session tracking
        self.session_start = time.time()
        self.session_id = None
        self.sessions: List[Dict[str, Any]] = []
        
        # Error tracking
        self.errors: List[Dict[str, Any]] = []
        self.error_counts: Dict[str, int] = defaultdict(int)
        
        # Aggregated statistics
        self.aggregated_stats: Dict[str, Dict[str, float]] = {}
        
        logger.info(f"Metrics collector initialized: max_samples={max_samples}")
    
    def record_latency(self, component: str, latency_ms: float, metadata: Optional[Dict] = None):
        """
        Record latency metric for a component
        
        Args:
            component: Component name (e.g., 'vad', 'stt', 'llm')
            latency_ms: Latency in milliseconds
            metadata: Additional metadata to store
        """
        self._record_metric(component, self.METRIC_LATENCY, latency_ms, metadata)
        
        # Update component stats
        if component not in self.component_metrics:
            self.component_metrics[component] = {
                "total_latency": 0.0,
                "count": 0,
                "max_latency": 0.0,
                "min_latency": float('inf')
            }
        
        comp_stats = self.component_metrics[component]
        comp_stats["total_latency"] += latency_ms
        comp_stats["count"] += 1
        comp_stats["max_latency"] = max(comp_stats["max_latency"], latency_ms)
        comp_stats["min_latency"] = min(comp_stats["min_latency"], latency_ms)
    
    def record_throughput(self, component: str, items_per_second: float, metadata: Optional[Dict] = None):
        """
        Record throughput metric
        
        Args:
            component: Component name
            items_per_second: Throughput rate
            metadata: Additional metadata
        """
        self._record_metric(component, self.METRIC_THROUGHPUT, items_per_second, metadata)
    
    def record_error(self, component: str, error: Union[str, Exception], metadata: Optional[Dict] = None):
        """
        Record error occurrence
        
        Args:
            component: Component name
            error: Error message or exception
            metadata: Additional metadata
        """
        error_str = str(error)
        self.error_counts[error_str] += 1
        
        error_record = {
            "timestamp": time.time(),
            "component": component,
            "error": error_str,
            "metadata": metadata or {}
        }
        self.errors.append(error_record)
        
        # Keep only last 100 errors
        if len(self.errors) > 100:
            self.errors = self.errors[-100:]
        
        self._record_metric(component, self.METRIC_ERROR, 1.0, {"error": error_str})
        
        logger.debug(f"Error recorded: {component} - {error_str[:50]}")
    
    def record_success(self, component: str, metadata: Optional[Dict] = None):
        """
        Record successful operation
        
        Args:
            component: Component name
            metadata: Additional metadata
        """
        self._record_metric(component, self.METRIC_SUCCESS, 1.0, metadata)
    
    def record_count(self, component: str, count: int, metadata: Optional[Dict] = None):
        """
        Record count metric
        
        Args:
            component: Component name
            count: Count value
            metadata: Additional metadata
        """
        self._record_metric(component, self.METRIC_COUNT, count, metadata)
    
    def record_size(self, component: str, size_bytes: int, metadata: Optional[Dict] = None):
        """
        Record size metric (e.g., audio bytes, text length)
        
        Args:
            component: Component name
            size_bytes: Size in bytes
            metadata: Additional metadata
        """
        self._record_metric(component, self.METRIC_SIZE, size_bytes, metadata)
    
    def _record_metric(self, component: str, metric_type: str, value: float, metadata: Optional[Dict] = None):
        """
        Internal method to record a metric
        
        Args:
            component: Component name
            metric_type: Type of metric
            value: Metric value
            metadata: Additional metadata
        """
        # Store metric
        self.metrics[component][metric_type].append(value)
        
        # Create snapshot
        snapshot = MetricSnapshot(
            timestamp=time.time(),
            component=component,
            metric_type=metric_type,
            value=value,
            metadata=metadata or {}
        )
        
        # Update aggregated stats
        self._update_aggregated_stats(component, metric_type, value)
    
    def _update_aggregated_stats(self, component: str, metric_type: str, value: float):
        """
        Update aggregated statistics
        
        Args:
            component: Component name
            metric_type: Type of metric
            value: Metric value
        """
        key = f"{component}_{metric_type}"
        
        if key not in self.aggregated_stats:
            self.aggregated_stats[key] = {
                "count": 0,
                "sum": 0.0,
                "min": float('inf'),
                "max": float('-inf'),
                "avg": 0.0
            }
        
        stats = self.aggregated_stats[key]
        stats["count"] += 1
        stats["sum"] += value
        stats["min"] = min(stats["min"], value)
        stats["max"] = max(stats["max"], value)
        stats["avg"] = stats["sum"] / stats["count"]
    
    def get_latency_stats(self, component: Optional[str] = None) -> Dict[str, Any]:
        """
        Get latency statistics
        
        Args:
            component: Component name (None for all)
            
        Returns:
            Dict: Latency statistics
        """
        if component:
            return self._calculate_stats(self.metrics[component].get(self.METRIC_LATENCY, []))
        
        result = {}
        for comp in self.metrics:
            latency_values = self.metrics[comp].get(self.METRIC_LATENCY, [])
            if latency_values:
                result[comp] = self._calculate_stats(latency_values)
        
        return result
    
    def get_component_stats(self, component: str) -> Dict[str, Any]:
        """
        Get comprehensive stats for a component
        
        Args:
            component: Component name
            
        Returns:
            Dict: Component statistics
        """
        stats = {
            "component": component,
            "latency": self.get_latency_stats(component),
            "throughput": self._calculate_stats(self.metrics[component].get(self.METRIC_THROUGHPUT, [])),
            "errors": self.error_counts,
            "success_rate": self._get_success_rate(component),
            "total_operations": len(self.metrics[component].get(self.METRIC_LATENCY, [])),
            "component_metrics": self.component_metrics.get(component, {})
        }
        
        # Add percentile values if available
        latency_values = self.metrics[component].get(self.METRIC_LATENCY, [])
        if latency_values:
            stats["percentiles"] = self._calculate_percentiles(latency_values)
        
        return stats
    
    def get_all_stats(self) -> Dict[str, Any]:
        """
        Get all metrics and statistics
        
        Returns:
            Dict: Complete metrics snapshot
        """
        return {
            "timestamp": time.time(),
            "session_start": self.session_start,
            "session_duration": time.time() - self.session_start,
            "components": {
                comp: self.get_component_stats(comp)
                for comp in self.metrics
            },
            "aggregated": self.aggregated_stats,
            "error_count": len(self.errors),
            "error_summary": dict(self.error_counts),
            "total_metrics": sum(
                sum(len(values) for values in comp_metrics.values())
                for comp_metrics in self.metrics.values()
            )
        }
    
    def get_latest_latency(self, component: str) -> float:
        """
        Get the latest latency recorded for a component
        
        Args:
            component: Component name
            
        Returns:
            float: Latest latency in milliseconds, or 0 if none
        """
        latency_values = self.metrics[component].get(self.METRIC_LATENCY, [])
        if latency_values:
            return latency_values[-1]
        return 0.0
    
    def get_average_latency(self, component: str) -> float:
        """
        Get average latency for a component
        
        Args:
            component: Component name
            
        Returns:
            float: Average latency in milliseconds
        """
        latency_values = self.metrics[component].get(self.METRIC_LATENCY, [])
        if not latency_values:
            return 0.0
        return sum(latency_values) / len(latency_values)
    
    def _calculate_stats(self, values: List[float]) -> Dict[str, Any]:
        """
        Calculate statistics from a list of values
        
        Args:
            values: List of numeric values
            
        Returns:
            Dict: Statistics
        """
        if not values:
            return {
                "count": 0,
                "avg": 0.0,
                "min": 0.0,
                "max": 0.0,
                "sum": 0.0
            }
        
        return {
            "count": len(values),
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "sum": sum(values)
        }
    
    def _calculate_percentiles(self, values: List[float]) -> Dict[str, float]:
        """
        Calculate percentiles from values
        
        Args:
            values: List of numeric values
            
        Returns:
            Dict: Percentile values
        """
        if not values:
            return {"p50": 0, "p90": 0, "p95": 0, "p99": 0}
        
        sorted_values = sorted(values)
        return {
            "p50": np.percentile(sorted_values, 50),
            "p90": np.percentile(sorted_values, 90),
            "p95": np.percentile(sorted_values, 95),
            "p99": np.percentile(sorted_values, 99)
        }
    
    def _get_success_rate(self, component: str) -> float:
        """
        Calculate success rate for a component
        
        Args:
            component: Component name
            
        Returns:
            float: Success rate (0-1)
        """
        successes = len(self.metrics[component].get(self.METRIC_SUCCESS, []))
        errors = len(self.metrics[component].get(self.METRIC_ERROR, []))
        total = successes + errors
        
        if total == 0:
            return 1.0  # No operations = 100% success
        
        return successes / total
    
    def start_session(self, session_id: Optional[str] = None):
        """
        Start a new session
        
        Args:
            session_id: Optional session identifier
        """
        self.session_id = session_id or f"session_{int(time.time())}"
        self.session_start = time.time()
        self.sessions.append({
            "session_id": self.session_id,
            "start_time": self.session_start,
            "metrics": {}
        })
        logger.info(f"Session started: {self.session_id}")
    
    def end_session(self):
        """
        End the current session
        """
        if self.sessions:
            current_session = self.sessions[-1]
            current_session["end_time"] = time.time()
            current_session["duration"] = current_session["end_time"] - current_session["start_time"]
            current_session["metrics"] = self.get_all_stats()
            logger.info(f"Session ended: {self.session_id}")
    
    def reset_session(self):
        """
        Reset session metrics
        """
        self.component_metrics = defaultdict(dict)
        self.error_counts = defaultdict(int)
        self.errors = []
        self.aggregated_stats = {}
        
        # Reset metrics storage
        for component in self.metrics:
            for metric_type in self.metrics[component]:
                self.metrics[component][metric_type].clear()
        
        logger.debug("Session metrics reset")
    
    def reset_all(self):
        """
        Reset all metrics
        """
        self.metrics = defaultdict(lambda: defaultdict(lambda: deque(maxlen=self.max_samples)))
        self.component_metrics = defaultdict(dict)
        self.errors = []
        self.error_counts = defaultdict(int)
        self.aggregated_stats = {}
        self.session_start = time.time()
        self.sessions = []
        
        logger.info("All metrics reset")
    
    def export_metrics(self, format: str = "json") -> Union[str, Dict]:
        """
        Export metrics in specified format
        
        Args:
            format: Export format ('json' or 'dict')
            
        Returns:
            Union[str, Dict]: Exported metrics
        """
        data = {
            "timestamp": time.time(),
            "session_id": self.session_id,
            "session_start": self.session_start,
            "session_duration": time.time() - self.session_start,
            "metrics": self.get_all_stats(),
            "sessions": self.sessions[-5:],  # Last 5 sessions
            "errors": self.errors[-10:]  # Last 10 errors
        }
        
        if format == "json":
            return json.dumps(data, indent=2, default=str)
        return data
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status based on metrics
        
        Returns:
            Dict: Health status
        """
        status = {
            "status": "healthy",
            "checks": {},
            "timestamp": time.time()
        }
        
        # Check each component
        for component in self.metrics:
            comp_status = "healthy"
            issues = []
            
            # Check error rate
            success_rate = self._get_success_rate(component)
            if success_rate < 0.9:
                comp_status = "warning"
                issues.append(f"Low success rate: {success_rate:.2%}")
            
            # Check latency
            avg_latency = self.get_average_latency(component)
            if avg_latency > 500:  # 500ms threshold
                comp_status = "warning" if comp_status == "healthy" else "critical"
                issues.append(f"High average latency: {avg_latency:.0f}ms")
            
            # Check if we have recent data
            latency_values = self.metrics[component].get(self.METRIC_LATENCY, [])
            if not latency_values and component not in ['vad', 'stt']:
                comp_status = "unknown"
                issues.append("No recent data")
            
            status["checks"][component] = {
                "status": comp_status,
                "issues": issues,
                "success_rate": success_rate,
                "avg_latency": avg_latency,
                "total_operations": len(latency_values)
            }
            
            if comp_status in ["warning", "critical"]:
                status["status"] = "degraded" if status["status"] == "healthy" else "critical"
        
        return status


# Singleton instance
_metrics_instance = None


def get_metrics_collector() -> MetricsCollector:
    """
    Get or create global metrics collector instance
    
    Returns:
        MetricsCollector: Global metrics collector
    """
    global _metrics_instance
    
    if _metrics_instance is None:
        _metrics_instance = MetricsCollector()
    
    return _metrics_instance


# Example usage
if __name__ == "__main__":
    # Test metrics collector
    metrics = get_metrics_collector()
    
    print("Testing Metrics Collector:")
    print("=" * 60)
    
    # Simulate recording metrics
    for i in range(100):
        # VAD latency: 15-25ms
        metrics.record_latency("vad", 15 + np.random.randn() * 5)
        
        # STT latency: 150-250ms
        metrics.record_latency("stt", 200 + np.random.randn() * 30)
        
        # LLM latency: 120-200ms
        metrics.record_latency("llm", 160 + np.random.randn() * 25)
        
        # TTS latency: 100-150ms
        metrics.record_latency("tts", 125 + np.random.randn() * 20)
        
        # Record some successes
        if i % 10 != 0:  # 90% success rate
            metrics.record_success("llm")
        else:
            metrics.record_error("llm", "Timeout error")
        
        # Record throughput
        metrics.record_throughput("stt", 5 + np.random.randn())
        
        # Record sizes
        metrics.record_size("audio", 480 * 2)  # 30ms audio
    
    # Get stats
    print("Latency Stats:")
    latency_stats = metrics.get_latency_stats()
    for component, stats in latency_stats.items():
        print(f"  {component}: avg={stats['avg']:.1f}ms, min={stats['min']:.1f}ms, max={stats['max']:.1f}ms")
    
    print("\nComponent Stats (LLM):")
    llm_stats = metrics.get_component_stats("llm")
    print(f"  Success Rate: {llm_stats['success_rate']:.2%}")
    print(f"  Total Operations: {llm_stats['total_operations']}")
    print(f"  Errors: {llm_stats['errors']}")
    if "percentiles" in llm_stats:
        print(f"  Percentiles: p50={llm_stats['percentiles']['p50']:.1f}ms, "
              f"p95={llm_stats['percentiles']['p95']:.1f}ms, "
              f"p99={llm_stats['percentiles']['p99']:.1f}ms")
    
    print("\nHealth Status:")
    health = metrics.get_health_status()
    print(f"  Overall Status: {health['status']}")
    for component, check in health['checks'].items():
        print(f"  {component}: {check['status']} - {', '.join(check['issues']) if check['issues'] else 'OK'}")
    
    print("\nAll Stats (summary):")
    all_stats = metrics.get_all_stats()
    print(f"  Session Duration: {all_stats['session_duration']:.1f}s")
    print(f"  Error Count: {all_stats['error_count']}")
    print(f"  Total Metrics: {all_stats['total_metrics']}")
    
    # Export metrics
    exported = metrics.export_metrics("json")
    print(f"\nExported metrics size: {len(exported)} bytes")