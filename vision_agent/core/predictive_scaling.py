"""
Predictive Resource Scaler - ML-based resource prediction and scaling
Provides proactive resource management based on workload forecasting.
"""

import asyncio
import numpy as np
import psutil
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class ResourceMetrics:
    """Resource usage metrics at a point in time."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    disk_io_read: int
    disk_io_write: int
    network_io_sent: int
    network_io_recv: int
    active_connections: int
    queue_size: int
    response_time: float

@dataclass
class WorkloadFeatures:
    """Features that influence resource requirements."""
    query_complexity: float  # 0-1 scale
    time_of_day: int  # 0-23
    day_of_week: int  # 0-6
    user_count: int
    avg_query_length: float
    concurrent_requests: int
    cache_hit_rate: float
    agent_types_active: int

class TimeSeriesPredictor:
    """Advanced time series prediction for resource usage."""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics_history: deque = deque(maxlen=history_size)
        self.feature_history: deque = deque(maxlen=history_size)
        
        # ML models for different resources
        self.cpu_model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.memory_model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.response_time_model = RandomForestRegressor(n_estimators=50, random_state=42)
        
        self.scaler = StandardScaler()
        self.is_trained = False
        self.min_samples_for_training = 50
        
    def add_observation(self, metrics: ResourceMetrics, features: WorkloadFeatures):
        """Add a new observation to the training data."""
        self.metrics_history.append(metrics)
        self.feature_history.append(features)
        
        # Retrain if we have enough samples
        if len(self.metrics_history) >= self.min_samples_for_training:
            asyncio.create_task(self._retrain_models())
    
    async def _retrain_models(self):
        """Retrain prediction models with latest data."""
        try:
            if len(self.metrics_history) < self.min_samples_for_training:
                return
            
            # Prepare training data
            X = self._prepare_features()
            
            # Prepare targets
            y_cpu = np.array([m.cpu_percent for m in self.metrics_history])
            y_memory = np.array([m.memory_percent for m in self.metrics_history])
            y_response_time = np.array([m.response_time for m in self.metrics_history])
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train models
            self.cpu_model.fit(X_scaled, y_cpu)
            self.memory_model.fit(X_scaled, y_memory)
            self.response_time_model.fit(X_scaled, y_response_time)
            
            self.is_trained = True
            logger.info(f"Retrained prediction models with {len(self.metrics_history)} samples")
            
        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
    
    def _prepare_features(self) -> np.ndarray:
        """Prepare feature matrix from history."""
        features = []
        
        for f in self.feature_history:
            feature_vector = [
                f.query_complexity,
                f.time_of_day / 23.0,  # Normalize
                f.day_of_week / 6.0,   # Normalize
                np.log1p(f.user_count),  # Log transform
                np.log1p(f.avg_query_length),
                f.concurrent_requests,
                f.cache_hit_rate,
                f.agent_types_active
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    async def forecast(self, horizon_minutes: int = 5, 
                     features: Optional[WorkloadFeatures] = None) -> Dict[str, float]:
        """Forecast resource usage for the next horizon_minutes."""
        if not self.is_trained:
            # Return current usage if not trained
            current_metrics = await self._get_current_metrics()
            return {
                'cpu_percent': current_metrics.cpu_percent,
                'memory_percent': current_metrics.memory_percent,
                'response_time': current_metrics.response_time,
                'confidence': 0.1  # Low confidence without training
            }
        
        try:
            # Use provided features or estimate current features
            if features is None:
                features = await self._estimate_current_features()
            
            # Prepare feature vector
            feature_vector = np.array([[
                features.query_complexity,
                features.time_of_day / 23.0,
                features.day_of_week / 6.0,
                np.log1p(features.user_count),
                np.log1p(features.avg_query_length),
                features.concurrent_requests,
                features.cache_hit_rate,
                features.agent_types_active
            ]])
            
            # Scale features
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            # Make predictions
            cpu_pred = self.cpu_model.predict(feature_vector_scaled)[0]
            memory_pred = self.memory_model.predict(feature_vector_scaled)[0]
            response_time_pred = self.response_time_model.predict(feature_vector_scaled)[0]
            
            # Calculate confidence based on model performance
            confidence = self._calculate_prediction_confidence()
            
            return {
                'cpu_percent': max(0, min(100, cpu_pred)),
                'memory_percent': max(0, min(100, memory_pred)),
                'response_time': max(0, response_time_pred),
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Forecasting failed: {e}")
            # Fallback to current metrics
            current_metrics = await self._get_current_metrics()
            return {
                'cpu_percent': current_metrics.cpu_percent,
                'memory_percent': current_metrics.memory_percent,
                'response_time': current_metrics.response_time,
                'confidence': 0.1
            }
    
    def _calculate_prediction_confidence(self) -> float:
        """Calculate confidence in predictions based on model performance."""
        if not self.is_trained or len(self.metrics_history) < 20:
            return 0.1
        
        # Simple confidence based on training data size and recency
        data_size_factor = min(1.0, len(self.metrics_history) / 200)
        recency_factor = 0.9  # Assume 90% confidence in recency
        
        return data_size_factor * recency_factor
    
    async def _get_current_metrics(self) -> ResourceMetrics:
        """Get current system resource metrics."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()
        net_io = psutil.net_io_counters()
        
        return ResourceMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_io_read=disk_io.read_bytes if disk_io else 0,
            disk_io_write=disk_io.write_bytes if disk_io else 0,
            network_io_sent=net_io.bytes_sent if net_io else 0,
            network_io_recv=net_io.bytes_recv if net_io else 0,
            active_connections=len(psutil.net_connections()),
            queue_size=0,  # Would be updated by server
            response_time=0.1  # Default baseline
        )
    
    async def _estimate_current_features(self) -> WorkloadFeatures:
        """Estimate current workload features."""
        current_time = time.localtime()
        
        return WorkloadFeatures(
            query_complexity=0.5,  # Default medium complexity
            time_of_day=current_time.tm_hour,
            day_of_week=current_time.tm_wday,
            user_count=1,  # Estimate
            avg_query_length=100.0,  # Default
            concurrent_requests=1,
            cache_hit_rate=0.3,  # Conservative estimate
            agent_types_active=4  # All agents
        )

class ResourceOptimizer:
    """Optimize resource allocation based on predictions."""
    
    def __init__(self):
        self.current_capacity = {
            'cpu_cores': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'max_concurrent_requests': 100
        }
        self.scaling_history: List[Tuple[float, Dict[str, float]]] = []
        
    async def scale_up(self, target_capacity: float, resource_type: str = 'cpu'):
        """Scale up resources proactively."""
        current_usage = psutil.cpu_percent() if resource_type == 'cpu' else psutil.virtual_memory().percent
        
        if target_capacity > current_usage * 1.5:
            # Significant scaling needed
            scaling_factor = min(2.0, target_capacity / max(current_usage, 10))
            
            # Simulate scaling (in production, this would trigger actual scaling)
            new_capacity = {
                'max_concurrent_requests': int(
                    self.current_capacity['max_concurrent_requests'] * scaling_factor
                ),
                'worker_processes': int(
                    max(1, self.current_capacity.get('worker_processes', 1) * scaling_factor)
                ),
                'cache_size': int(
                    self.current_capacity.get('cache_size', 1000) * scaling_factor
                )
            }
            
            self.scaling_history.append((time.time(), new_capacity))
            logger.info(f"Scaled up {resource_type} capacity by {scaling_factor:.1f}x to handle "
                       f"predicted load of {target_capacity:.1f}%")
            
            return new_capacity
        
        return self.current_capacity
    
    async def scale_down(self, target_capacity: float, resource_type: str = 'cpu'):
        """Scale down resources to save costs."""
        current_usage = psutil.cpu_percent() if resource_type == 'cpu' else psutil.virtual_memory().percent
        
        if target_capacity < current_usage * 0.5:
            # Safe to scale down
            scaling_factor = max(0.5, target_capacity / max(current_usage, 10))
            
            new_capacity = {
                'max_concurrent_requests': max(10, int(
                    self.current_capacity['max_concurrent_requests'] * scaling_factor
                )),
                'worker_processes': max(1, int(
                    self.current_capacity.get('worker_processes', 1) * scaling_factor
                )),
                'cache_size': max(100, int(
                    self.current_capacity.get('cache_size', 1000) * scaling_factor
                ))
            }
            
            self.scaling_history.append((time.time(), new_capacity))
            logger.info(f"Scaled down {resource_type} capacity by {1/scaling_factor:.1f}x "
                       f"for predicted load of {target_capacity:.1f}%")
            
            return new_capacity
        
        return self.current_capacity

class PredictiveScaler:
    """Main predictive scaling orchestrator."""
    
    def __init__(self, prediction_interval: int = 60):
        self.load_predictor = TimeSeriesPredictor()
        self.resource_optimizer = ResourceOptimizer()
        self.prediction_interval = prediction_interval
        self.is_running = False
        self.scaling_task: Optional[asyncio.Task] = None
        
        # Scaling thresholds
        self.scale_up_threshold = 0.8  # Scale up if predicted load > 80% capacity
        self.scale_down_threshold = 0.3  # Scale down if predicted load < 30% capacity
        self.scale_up_factor = 1.5
        self.scale_down_factor = 0.7
        
    async def start_predictive_scaling(self):
        """Start the predictive scaling loop."""
        if self.is_running:
            logger.warning("Predictive scaling already running")
            return
        
        self.is_running = True
        self.scaling_task = asyncio.create_task(self._scaling_loop())
        logger.info("Started predictive resource scaling")
    
    async def stop_predictive_scaling(self):
        """Stop the predictive scaling loop."""
        self.is_running = False
        if self.scaling_task:
            self.scaling_task.cancel()
            try:
                await self.scaling_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped predictive resource scaling")
    
    async def _scaling_loop(self):
        """Main scaling loop that runs continuously."""
        while self.is_running:
            try:
                await self.predict_and_scale()
                await asyncio.sleep(self.prediction_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scaling loop error: {e}")
                await asyncio.sleep(self.prediction_interval)
    
    async def predict_and_scale(self):
        """Predict load and scale resources accordingly."""
        try:
            # Get current metrics
            current_metrics = await self.load_predictor._get_current_metrics()
            current_features = await self.load_predictor._estimate_current_features()
            
            # Add observation for learning
            self.load_predictor.add_observation(current_metrics, current_features)
            
            # Predict load 5 minutes ahead
            predicted_load = await self.load_predictor.forecast(
                horizon_minutes=5,
                features=current_features
            )
            
            # Make scaling decisions
            await self._make_scaling_decision(predicted_load, current_metrics)
            
        except Exception as e:
            logger.error(f"Predict and scale failed: {e}")
    
    async def _make_scaling_decision(self, predicted_load: Dict[str, float], 
                                   current_metrics: ResourceMetrics):
        """Make intelligent scaling decisions based on predictions."""
        pred_cpu = predicted_load['cpu_percent']
        pred_memory = predicted_load['memory_percent']
        confidence = predicted_load['confidence']
        
        # Only make scaling decisions if confidence is high enough
        if confidence < 0.5:
            logger.debug(f"Skipping scaling due to low confidence: {confidence:.2f}")
            return
        
        current_cpu = current_metrics.cpu_percent
        current_memory = current_metrics.memory_percent
        
        # CPU-based scaling decisions
        if pred_cpu > self.scale_up_threshold * 100:
            logger.info(f"Predicted CPU load {pred_cpu:.1f}% exceeds threshold "
                       f"{self.scale_up_threshold*100:.1f}% - scaling up")
            
            await self.resource_optimizer.scale_up(
                target_capacity=pred_cpu,
                resource_type='cpu'
            )
            
        elif pred_cpu < self.scale_down_threshold * 100 and current_cpu < 50:
            logger.info(f"Predicted CPU load {pred_cpu:.1f}% below threshold "
                       f"{self.scale_down_threshold*100:.1f}% - scaling down")
            
            await self.resource_optimizer.scale_down(
                target_capacity=pred_cpu,
                resource_type='cpu'
            )
        
        # Memory-based scaling decisions
        if pred_memory > 85:  # High memory usage threshold
            logger.info(f"Predicted memory usage {pred_memory:.1f}% is high - optimizing")
            
            await self._optimize_memory_usage()
            
        # Response time-based scaling
        pred_response_time = predicted_load.get('response_time', 0)
        if pred_response_time > 2.0:  # 2 second threshold
            logger.info(f"Predicted response time {pred_response_time:.2f}s is high - scaling up")
            
            await self.resource_optimizer.scale_up(
                target_capacity=current_cpu * 1.5,
                resource_type='cpu'
            )
    
    async def _optimize_memory_usage(self):
        """Optimize memory usage when high usage is predicted."""
        # Trigger garbage collection
        import gc
        gc.collect()
        
        # Clear caches if available
        try:
            # This would integrate with the semantic cache manager
            logger.info("Triggered memory optimization and cache cleanup")
        except Exception as e:
            logger.warning(f"Memory optimization failed: {e}")
    
    def get_scaling_history(self) -> List[Dict[str, any]]:
        """Get history of scaling decisions."""
        history = []
        for timestamp, capacity in self.resource_optimizer.scaling_history:
            history.append({
                'timestamp': timestamp,
                'capacity': capacity,
                'time_ago_minutes': (time.time() - timestamp) / 60
            })
        return history
    
    def get_prediction_accuracy(self) -> Dict[str, float]:
        """Calculate prediction accuracy metrics."""
        if len(self.load_predictor.metrics_history) < 10:
            return {'accuracy': 0.0, 'sample_size': 0}
        
        # Simple accuracy calculation (would be more sophisticated in production)
        recent_metrics = list(self.load_predictor.metrics_history)[-10:]
        
        cpu_accuracy = 1.0 - np.std([m.cpu_percent for m in recent_metrics]) / 100
        memory_accuracy = 1.0 - np.std([m.memory_percent for m in recent_metrics]) / 100
        
        return {
            'cpu_accuracy': max(0, cpu_accuracy),
            'memory_accuracy': max(0, memory_accuracy),
            'overall_accuracy': (cpu_accuracy + memory_accuracy) / 2,
            'sample_size': len(recent_metrics),
            'confidence': self.load_predictor._calculate_prediction_confidence()
        }

class WorkloadCharacterizer:
    """Characterize workload patterns for better prediction."""
    
    def __init__(self):
        self.query_complexity_analyzer = QueryComplexityAnalyzer()
        self.pattern_detector = WorkloadPatternDetector()
        
    async def characterize_current_workload(self, recent_queries: List[str],
                                          active_agents: List[str]) -> WorkloadFeatures:
        """Analyze current workload characteristics."""
        try:
            # Analyze query complexity
            complexity_scores = [
                await self.query_complexity_analyzer.analyze_complexity(query)
                for query in recent_queries[-10:]  # Last 10 queries
            ]
            avg_complexity = np.mean(complexity_scores) if complexity_scores else 0.5
            
            # Calculate average query length
            avg_length = np.mean([len(q) for q in recent_queries]) if recent_queries else 100
            
            # Current time features
            current_time = time.localtime()
            
            return WorkloadFeatures(
                query_complexity=avg_complexity,
                time_of_day=current_time.tm_hour,
                day_of_week=current_time.tm_wday,
                user_count=1,  # Would be tracked by server
                avg_query_length=avg_length,
                concurrent_requests=len(recent_queries),
                cache_hit_rate=0.3,  # Would be tracked by cache manager
                agent_types_active=len(active_agents)
            )
            
        except Exception as e:
            logger.error(f"Workload characterization failed: {e}")
            return await self.load_predictor._estimate_current_features()

class QueryComplexityAnalyzer:
    """Analyze the complexity of user queries."""
    
    def __init__(self):
        self.complexity_cache: Dict[str, float] = {}
        
    async def analyze_complexity(self, query: str) -> float:
        """Analyze query complexity on a 0-1 scale."""
        if query in self.complexity_cache:
            return self.complexity_cache[query]
        
        # Multi-factor complexity analysis
        factors = {
            'length': min(1.0, len(query) / 500),  # Length factor
            'words': min(1.0, len(query.split()) / 50),  # Word count factor
            'questions': min(1.0, query.count('?') / 3),  # Question complexity
            'technical_terms': self._count_technical_terms(query) / 10,
            'nested_requests': self._count_nested_requests(query) / 5,
            'multi_modal': self._detect_multimodal_requests(query)
        }
        
        # Weighted combination
        weights = {
            'length': 0.1,
            'words': 0.15,
            'questions': 0.2,
            'technical_terms': 0.25,
            'nested_requests': 0.2,
            'multi_modal': 0.1
        }
        
        complexity = sum(factors[key] * weights[key] for key in factors)
        complexity = min(1.0, complexity)  # Cap at 1.0
        
        self.complexity_cache[query] = complexity
        return complexity
    
    def _count_technical_terms(self, query: str) -> int:
        """Count technical terms that indicate complexity."""
        technical_terms = [
            'algorithm', 'model', 'neural', 'machine learning', 'ai', 'api',
            'database', 'optimization', 'performance', 'analysis', 'classification',
            'detection', 'recognition', 'processing', 'framework', 'architecture'
        ]
        
        query_lower = query.lower()
        return sum(1 for term in technical_terms if term in query_lower)
    
    def _count_nested_requests(self, query: str) -> int:
        """Count nested or chained requests."""
        indicators = ['then', 'after', 'next', 'also', 'additionally', 'furthermore']
        query_lower = query.lower()
        return sum(1 for indicator in indicators if indicator in query_lower)
    
    def _detect_multimodal_requests(self, query: str) -> float:
        """Detect if query involves multiple modalities."""
        modalities = ['image', 'video', 'audio', 'text', 'face', 'object', 'classify']
        query_lower = query.lower()
        
        detected_modalities = sum(1 for modality in modalities if modality in query_lower)
        return min(1.0, detected_modalities / 3)  # Normalize to 0-1

class WorkloadPatternDetector:
    """Detect recurring workload patterns."""
    
    def __init__(self):
        self.hourly_patterns: Dict[int, List[float]] = defaultdict(list)
        self.daily_patterns: Dict[int, List[float]] = defaultdict(list)
        
    def add_workload_sample(self, features: WorkloadFeatures, load: float):
        """Add a workload sample for pattern detection."""
        self.hourly_patterns[features.time_of_day].append(load)
        self.daily_patterns[features.day_of_week].append(load)
        
        # Keep only recent samples
        for hour_list in self.hourly_patterns.values():
            if len(hour_list) > 50:
                hour_list[:] = hour_list[-50:]
        
        for day_list in self.daily_patterns.values():
            if len(day_list) > 20:
                day_list[:] = day_list[-20:]
    
    def get_expected_load(self, time_of_day: int, day_of_week: int) -> float:
        """Get expected load based on historical patterns."""
        hourly_avg = np.mean(self.hourly_patterns[time_of_day]) if self.hourly_patterns[time_of_day] else 50.0
        daily_avg = np.mean(self.daily_patterns[day_of_week]) if self.daily_patterns[day_of_week] else 50.0
        
        # Weighted average of hourly and daily patterns
        return hourly_avg * 0.7 + daily_avg * 0.3
