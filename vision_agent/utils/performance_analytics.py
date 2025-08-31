"""
Advanced Performance Monitoring & Analytics System
Real-time dashboards, predictive failure detection, and comprehensive metrics.
"""

import asyncio
import json
import logging
import time
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Tuple
from enum import Enum
from functools import wraps
import numpy as np


class MetricType(str, Enum):
    """Types of performance metrics."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    RESOURCE_USAGE = "resource_usage"
    USER_SATISFACTION = "user_satisfaction"
    COST = "cost"


class AlertLevel(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """Individual performance metric data point."""
    metric_type: MetricType
    value: float
    timestamp: float
    agent_name: Optional[str] = None
    tool_name: Optional[str] = None
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class Alert:
    """Performance alert."""
    level: AlertLevel
    message: str
    timestamp: float
    metric_type: MetricType
    value: float
    threshold: float
    agent_name: Optional[str] = None
    resolved: bool = False


@dataclass
class PerformanceSummary:
    """Performance summary for a time period."""
    start_time: float
    end_time: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_per_second: float
    error_rate: float
    avg_cost_per_request: float


class FailurePrediction:
    """
    ML-based failure prediction system.
    Analyzes patterns to predict potential failures before they occur.
    """
    
    def __init__(self, prediction_window: int = 100):
        self.prediction_window = prediction_window
        self.failure_patterns: List[Dict[str, Any]] = []
        self.feature_history: deque = deque(maxlen=prediction_window)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def extract_features(self, context: Dict[str, Any]) -> np.ndarray:
        """Extract features for failure prediction."""
        features = [
            context.get('cpu_usage', 0) / 100,
            context.get('memory_usage', 0) / 100,
            context.get('active_tasks', 0) / 20,  # Normalize to max 20 tasks
            context.get('queue_length', 0) / 50,  # Normalize to max 50 queue
            context.get('recent_error_rate', 0),
            context.get('avg_latency', 0) / 10000,  # Normalize to 10s max
            context.get('disk_io', 0) / 100,
            len(context.get('recent_failures', [])) / 10  # Normalize to 10 recent failures
        ]
        
        return np.array(features, dtype=np.float32)
    
    def record_failure(self, context: Dict[str, Any], failure_type: str):
        """Record a failure event for learning."""
        features = self.extract_features(context)
        
        failure_pattern = {
            "timestamp": time.time(),
            "features": features.tolist(),
            "failure_type": failure_type,
            "context": context
        }
        
        self.failure_patterns.append(failure_pattern)
        
        # Keep only recent patterns
        if len(self.failure_patterns) > 1000:
            self.failure_patterns = self.failure_patterns[-500:]
    
    def predict_failure_probability(self, context: Dict[str, Any]) -> Tuple[float, List[str]]:
        """
        Predict probability of failure and potential causes.
        
        Args:
            context: Current system context
            
        Returns:
            Tuple of (failure_probability, potential_causes)
        """
        if len(self.failure_patterns) < 10:
            return 0.0, []
        
        current_features = self.extract_features(context)
        self.feature_history.append(current_features)
        
        # Simple pattern matching (can be enhanced with ML models)
        failure_probability = 0.0
        potential_causes = []
        
        # Check for resource exhaustion patterns
        if current_features[0] > 0.9:  # High CPU
            failure_probability += 0.3
            potential_causes.append("High CPU usage")
        
        if current_features[1] > 0.9:  # High memory
            failure_probability += 0.3
            potential_causes.append("High memory usage")
        
        if current_features[2] > 0.8:  # Too many active tasks
            failure_probability += 0.2
            potential_causes.append("Task overload")
        
        if current_features[4] > 0.1:  # High error rate
            failure_probability += 0.4
            potential_causes.append("High recent error rate")
        
        # Check for trending issues
        if len(self.feature_history) >= 5:
            recent_features = np.array(list(self.feature_history)[-5:])
            
            # Check for increasing trends
            for i, feature_name in enumerate(['cpu', 'memory', 'tasks', 'queue', 'errors', 'latency']):
                trend = np.polyfit(range(5), recent_features[:, i], 1)[0]
                if trend > 0.02:  # Increasing trend
                    failure_probability += min(trend * 5, 0.2)
                    potential_causes.append(f"Increasing {feature_name} trend")
        
        # Historical pattern matching
        similarity_threshold = 0.8
        for pattern in self.failure_patterns[-50:]:  # Check recent patterns
            pattern_features = np.array(pattern["features"])
            similarity = self._calculate_feature_similarity(current_features, pattern_features)
            
            if similarity > similarity_threshold:
                failure_probability += 0.1 * similarity
                potential_causes.append(f"Similar to {pattern['failure_type']} pattern")
        
        return min(failure_probability, 1.0), potential_causes
    
    def _calculate_feature_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate similarity between feature vectors."""
        try:
            # Euclidean distance normalized to [0, 1]
            distance = np.linalg.norm(features1 - features2)
            max_distance = np.sqrt(len(features1))  # Maximum possible distance
            similarity = 1 - (distance / max_distance)
            return max(0.0, similarity)
        except Exception:
            return 0.0


class PerformanceAnalytics:
    """
    Comprehensive performance analytics and monitoring system.
    """
    
    def __init__(self, 
                 metrics_retention_hours: int = 168,  # 1 week
                 alert_thresholds: Optional[Dict[str, float]] = None):
        self.metrics_retention_hours = metrics_retention_hours
        
        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            "latency_p95_ms": 5000,     # 5 seconds
            "error_rate": 0.05,         # 5%
            "cpu_usage": 0.85,          # 85%
            "memory_usage": 0.90,       # 90%
            "queue_length": 100,        # 100 items
            "cost_per_hour": 10.0       # $10/hour
        }
        
        # Data storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.alerts: List[Alert] = []
        self.performance_summaries: Dict[str, PerformanceSummary] = {}
        
        # Agent-specific metrics
        self.agent_metrics: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=1000))
        )
        
        # Tool-specific metrics
        self.tool_metrics: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=1000))
        )
        
        # Real-time subscribers
        self.metric_subscribers: List[Callable[[PerformanceMetric], None]] = []
        self.alert_subscribers: List[Callable[[Alert], None]] = []
        
        self.failure_predictor = FailurePrediction()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self._monitoring_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
    
    async def start(self):
        """Start the performance analytics system."""
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Performance Analytics System started")
    
    async def stop(self):
        """Stop the performance analytics system."""
        self._shutdown_event.set()
        if self._monitoring_task:
            await self._monitoring_task
        self.logger.info("Performance Analytics System stopped")
    
    def record_metric(self, metric: PerformanceMetric):
        """Record a performance metric."""
        # Store in general metrics
        metric_key = f"{metric.metric_type.value}"
        if metric.agent_name:
            metric_key += f"_{metric.agent_name}"
        if metric.tool_name:
            metric_key += f"_{metric.tool_name}"
        
        self.metrics[metric_key].append(metric)
        
        # Store in agent-specific metrics
        if metric.agent_name:
            self.agent_metrics[metric.agent_name][metric.metric_type.value].append(metric)
        
        # Store in tool-specific metrics
        if metric.tool_name:
            self.tool_metrics[metric.tool_name][metric.metric_type.value].append(metric)
        
        # Check for alerts
        asyncio.create_task(self._check_alerts(metric))
        
        # Notify subscribers
        for subscriber in self.metric_subscribers:
            try:
                asyncio.create_task(subscriber(metric))
            except Exception as e:
                self.logger.warning(f"Metric subscriber error: {e}")
    
    async def _check_alerts(self, metric: PerformanceMetric):
        """Check if metric triggers any alerts."""
        alerts_triggered = []
        
        # Latency alerts
        if metric.metric_type == MetricType.LATENCY:
            if metric.value > self.alert_thresholds.get("latency_p95_ms", 5000):
                alert = Alert(
                    level=AlertLevel.WARNING if metric.value < 10000 else AlertLevel.ERROR,
                    message=f"High latency detected: {metric.value:.2f}ms",
                    timestamp=metric.timestamp,
                    metric_type=metric.metric_type,
                    value=metric.value,
                    threshold=self.alert_thresholds["latency_p95_ms"],
                    agent_name=metric.agent_name
                )
                alerts_triggered.append(alert)
        
        # Error rate alerts
        elif metric.metric_type == MetricType.ERROR_RATE:
            if metric.value > self.alert_thresholds.get("error_rate", 0.05):
                alert = Alert(
                    level=AlertLevel.ERROR,
                    message=f"High error rate: {metric.value:.2%}",
                    timestamp=metric.timestamp,
                    metric_type=metric.metric_type,
                    value=metric.value,
                    threshold=self.alert_thresholds["error_rate"],
                    agent_name=metric.agent_name
                )
                alerts_triggered.append(alert)
        
        # Resource usage alerts
        elif metric.metric_type == MetricType.RESOURCE_USAGE:
            cpu_threshold = self.alert_thresholds.get("cpu_usage", 0.85)
            memory_threshold = self.alert_thresholds.get("memory_usage", 0.90)
            
            if metric.value > cpu_threshold:
                alert = Alert(
                    level=AlertLevel.WARNING,
                    message=f"High CPU usage: {metric.value:.1%}",
                    timestamp=metric.timestamp,
                    metric_type=metric.metric_type,
                    value=metric.value,
                    threshold=cpu_threshold
                )
                alerts_triggered.append(alert)
        
        # Store and notify alerts
        for alert in alerts_triggered:
            self.alerts.append(alert)
            
            # Notify alert subscribers
            for subscriber in self.alert_subscribers:
                try:
                    asyncio.create_task(subscriber(alert))
                except Exception as e:
                    self.logger.warning(f"Alert subscriber error: {e}")
        
        # Predict potential failures
        if alerts_triggered:
            await self._run_failure_prediction()
    
    async def _run_failure_prediction(self):
        """Run failure prediction analysis."""
        try:
            # Get current system context
            context = await self._get_system_context()
            
            failure_prob, causes = self.failure_predictor.predict_failure_probability(context)
            
            if failure_prob > 0.8:
                alert = Alert(
                    level=AlertLevel.CRITICAL,
                    message=f"High failure probability: {failure_prob:.1%}. Causes: {', '.join(causes)}",
                    timestamp=time.time(),
                    metric_type=MetricType.ERROR_RATE,
                    value=failure_prob,
                    threshold=0.8
                )
                
                self.alerts.append(alert)
                self.logger.critical(alert.message)
                
                # Trigger proactive intervention
                await self._trigger_proactive_intervention(causes)
                
        except Exception as e:
            self.logger.error(f"Failure prediction error: {e}")
    
    async def _get_system_context(self) -> Dict[str, Any]:
        """Get current system context for failure prediction."""
        # Get recent metrics
        recent_latency = self._get_recent_values(MetricType.LATENCY, minutes=10)
        recent_errors = self._get_recent_values(MetricType.ERROR_RATE, minutes=10)
        
        return {
            "cpu_usage": 0,  # Would integrate with resource manager
            "memory_usage": 0,
            "active_tasks": 0,
            "queue_length": 0,
            "recent_error_rate": statistics.mean(recent_errors) if recent_errors else 0,
            "avg_latency": statistics.mean(recent_latency) if recent_latency else 0,
            "recent_failures": [alert for alert in self.alerts[-10:] if alert.level in [AlertLevel.ERROR, AlertLevel.CRITICAL]]
        }
    
    def _get_recent_values(self, metric_type: MetricType, minutes: int = 10) -> List[float]:
        """Get recent metric values."""
        cutoff_time = time.time() - (minutes * 60)
        values = []
        
        for metric_key, metric_deque in self.metrics.items():
            if metric_type.value in metric_key:
                for metric in metric_deque:
                    if metric.timestamp > cutoff_time:
                        values.append(metric.value)
        
        return values
    
    async def _trigger_proactive_intervention(self, causes: List[str]):
        """Trigger proactive interventions based on predicted failure causes."""
        interventions = []
        
        if "High CPU usage" in causes:
            interventions.append("Reduce concurrency limits")
        
        if "High memory usage" in causes:
            interventions.append("Clear caches and free memory")
        
        if "Task overload" in causes:
            interventions.append("Pause new task acceptance")
        
        if "High recent error rate" in causes:
            interventions.append("Switch to fallback algorithms")
        
        self.logger.info(f"Proactive interventions triggered: {interventions}")
        
        # Here you would implement actual intervention logic
        # For now, just log the interventions
    
    def calculate_performance_summary(self, 
                                    start_time: float,
                                    end_time: float,
                                    agent_name: Optional[str] = None) -> PerformanceSummary:
        """Calculate performance summary for a time period."""
        # Filter metrics by time and agent
        filtered_metrics = []
        for metric_deque in self.metrics.values():
            for metric in metric_deque:
                if start_time <= metric.timestamp <= end_time:
                    if agent_name is None or metric.agent_name == agent_name:
                        filtered_metrics.append(metric)
        
        if not filtered_metrics:
            return PerformanceSummary(
                start_time=start_time,
                end_time=end_time,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                avg_latency_ms=0,
                p95_latency_ms=0,
                p99_latency_ms=0,
                throughput_per_second=0,
                error_rate=0,
                avg_cost_per_request=0
            )
        
        # Extract latency metrics
        latency_metrics = [m.value for m in filtered_metrics if m.metric_type == MetricType.LATENCY]
        error_metrics = [m.value for m in filtered_metrics if m.metric_type == MetricType.ERROR_RATE]
        cost_metrics = [m.value for m in filtered_metrics if m.metric_type == MetricType.COST]
        
        # Calculate statistics
        total_requests = len(latency_metrics)
        failed_requests = sum(1 for m in filtered_metrics if m.metric_type == MetricType.ERROR_RATE and m.value > 0)
        successful_requests = total_requests - failed_requests
        
        avg_latency = statistics.mean(latency_metrics) if latency_metrics else 0
        p95_latency = np.percentile(latency_metrics, 95) if latency_metrics else 0
        p99_latency = np.percentile(latency_metrics, 99) if latency_metrics else 0
        
        duration_hours = (end_time - start_time) / 3600
        throughput = total_requests / duration_hours if duration_hours > 0 else 0
        
        error_rate = statistics.mean(error_metrics) if error_metrics else 0
        avg_cost = statistics.mean(cost_metrics) if cost_metrics else 0
        
        return PerformanceSummary(
            start_time=start_time,
            end_time=end_time,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_latency_ms=avg_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            throughput_per_second=throughput / 3600 if throughput > 0 else 0,
            error_rate=error_rate,
            avg_cost_per_request=avg_cost
        )
    
    async def _monitoring_loop(self):
        """Main monitoring loop for real-time analytics."""
        while not self._shutdown_event.is_set():
            try:
                # Clean up old metrics
                cutoff_time = time.time() - (self.metrics_retention_hours * 3600)
                await self._cleanup_old_metrics(cutoff_time)
                
                # Generate hourly summary
                now = time.time()
                hour_start = now - 3600  # Last hour
                
                summary = self.calculate_performance_summary(hour_start, now)
                summary_key = f"hourly_{int(hour_start)}"
                self.performance_summaries[summary_key] = summary
                
                # Log summary
                self.logger.info(
                    f"Hourly summary: {summary.total_requests} requests, "
                    f"avg latency: {summary.avg_latency_ms:.2f}ms, "
                    f"error rate: {summary.error_rate:.2%}"
                )
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(300)
    
    async def _cleanup_old_metrics(self, cutoff_time: float):
        """Remove metrics older than retention period."""
        removed_count = 0
        
        for metric_key, metric_deque in self.metrics.items():
            # Remove old metrics
            while metric_deque and metric_deque[0].timestamp < cutoff_time:
                metric_deque.popleft()
                removed_count += 1
        
        # Clean up old alerts
        self.alerts = [
            alert for alert in self.alerts
            if alert.timestamp > cutoff_time
        ]
        
        if removed_count > 0:
            self.logger.debug(f"Cleaned up {removed_count} old metrics")
    
    def subscribe_to_metrics(self, callback: Callable[[PerformanceMetric], None]):
        """Subscribe to real-time metric updates."""
        self.metric_subscribers.append(callback)
    
    def subscribe_to_alerts(self, callback: Callable[[Alert], None]):
        """Subscribe to real-time alert notifications."""
        self.alert_subscribers.append(callback)
    
    def get_real_time_dashboard_data(self) -> Dict[str, Any]:
        """Get data for real-time performance dashboard."""
        now = time.time()
        last_hour = now - 3600
        last_day = now - 86400
        
        # Recent metrics
        recent_latency = self._get_recent_values(MetricType.LATENCY, minutes=30)
        recent_errors = self._get_recent_values(MetricType.ERROR_RATE, minutes=30)
        recent_costs = self._get_recent_values(MetricType.COST, minutes=30)
        
        # Agent performance
        agent_performance = {}
        for agent_name, agent_metrics in self.agent_metrics.items():
            latency_values = [m.value for m in agent_metrics.get("latency", []) if m.timestamp > last_hour]
            agent_performance[agent_name] = {
                "avg_latency": statistics.mean(latency_values) if latency_values else 0,
                "request_count": len(latency_values),
                "status": "healthy" if statistics.mean(latency_values) < 5000 else "degraded" if latency_values else "inactive"
            }
        
        # Active alerts
        active_alerts = [alert for alert in self.alerts[-50:] if not alert.resolved]
        
        return {
            "timestamp": now,
            "current_metrics": {
                "avg_latency_ms": statistics.mean(recent_latency) if recent_latency else 0,
                "error_rate": statistics.mean(recent_errors) if recent_errors else 0,
                "cost_per_hour": sum(recent_costs) if recent_costs else 0,
                "throughput_per_minute": len(recent_latency)  # Simplified
            },
            "agent_performance": agent_performance,
            "active_alerts": [asdict(alert) for alert in active_alerts],
            "system_health": self._calculate_system_health(),
            "predictions": {
                "failure_probability": 0.0,  # Would integrate with failure predictor
                "estimated_causes": []
            }
        }
    
    def _calculate_system_health(self) -> str:
        """Calculate overall system health status."""
        recent_errors = self._get_recent_values(MetricType.ERROR_RATE, minutes=10)
        recent_latency = self._get_recent_values(MetricType.LATENCY, minutes=10)
        
        error_rate = statistics.mean(recent_errors) if recent_errors else 0
        avg_latency = statistics.mean(recent_latency) if recent_latency else 0
        
        # Simple health calculation
        if error_rate > 0.1 or avg_latency > 10000:
            return "unhealthy"
        elif error_rate > 0.05 or avg_latency > 5000:
            return "degraded"
        else:
            return "healthy"
    
    def get_agent_analytics(self, agent_name: str) -> Dict[str, Any]:
        """Get detailed analytics for a specific agent."""
        if agent_name not in self.agent_metrics:
            return {"error": "Agent not found"}
        
        agent_data = self.agent_metrics[agent_name]
        
        # Calculate statistics
        latency_values = [m.value for m in agent_data.get("latency", [])]
        error_values = [m.value for m in agent_data.get("error_rate", [])]
        
        return {
            "agent_name": agent_name,
            "total_requests": len(latency_values),
            "avg_latency_ms": statistics.mean(latency_values) if latency_values else 0,
            "p95_latency_ms": np.percentile(latency_values, 95) if latency_values else 0,
            "error_rate": statistics.mean(error_values) if error_values else 0,
            "success_rate": 1 - (statistics.mean(error_values) if error_values else 0),
            "performance_trend": self._calculate_trend(latency_values[-50:]) if len(latency_values) > 10 else "stable"
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate performance trend."""
        if len(values) < 5:
            return "stable"
        
        # Simple linear regression slope
        x = list(range(len(values)))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 100:  # Latency increasing
            return "degrading"
        elif slope < -100:  # Latency decreasing
            return "improving"
        else:
            return "stable"


# Global performance analytics instance
performance_analytics = PerformanceAnalytics()


# Decorator for automatic performance tracking
def track_performance(metric_type: MetricType = MetricType.LATENCY):
    """Decorator to automatically track performance metrics."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            agent_name = getattr(args[0], '__class__', {}).get('__name__') if args else None
            
            try:
                result = await func(*args, **kwargs)
                
                # Record success metric
                execution_time = (time.time() - start_time) * 1000  # Convert to ms
                metric = PerformanceMetric(
                    metric_type=metric_type,
                    value=execution_time,
                    timestamp=time.time(),
                    agent_name=agent_name,
                    tool_name=func.__name__
                )
                performance_analytics.record_metric(metric)
                
                return result
                
            except Exception as e:
                # Record error metric
                error_metric = PerformanceMetric(
                    metric_type=MetricType.ERROR_RATE,
                    value=1.0,  # Error occurred
                    timestamp=time.time(),
                    agent_name=agent_name,
                    tool_name=func.__name__
                )
                performance_analytics.record_metric(error_metric)
                
                # Record failure for prediction
                context = await performance_analytics._get_system_context()
                performance_analytics.failure_predictor.record_failure(
                    context, str(type(e).__name__)
                )
                
                raise
        
        return wrapper
    return decorator


async def initialize_performance_analytics():
    """Initialize the performance analytics system."""
    await performance_analytics.start()
    logging.getLogger('PerformanceAnalytics').info("Performance Analytics System initialized")


async def shutdown_performance_analytics():
    """Shutdown the performance analytics system."""
    await performance_analytics.stop()
