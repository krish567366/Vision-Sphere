"""
Enhanced Async Base Agent with Advanced Performance Patterns
Integration of all enterprise-grade patterns: resource management, semantic caching,
speculative execution, performance analytics, and reliability.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, AsyncGenerator, Union, Callable
from functools import wraps
import numpy as np

# Import all advanced systems
from .caching import AsyncFileCache, async_file_cache
from .semantic_cache import smart_semantic_cache, semantic_cache_manager
from .resource_manager import adaptive_resource_control, TaskComplexity, resource_manager
from .speculative_execution import enable_speculation, speculative_runner, cost_optimizer
from .performance_analytics import track_performance, MetricType, performance_analytics
from .reliability import reliable_tool, reliability_manager, FallbackStrategy
from .tracing import TraceManager, SpanType, trace_manager
from .streaming import StreamEventManager, EventType, event_manager


@dataclass
class EnhancedProcessingResult:
    """Enhanced processing result with comprehensive metadata."""
    # Core results
    primary_result: Any
    confidence: float
    processing_time_ms: float
    
    # Performance metadata
    resource_usage: Dict[str, float]
    cache_hit: bool
    speculative_data: Optional[Dict[str, Any]] = None
    fallback_used: bool = False
    
    # Tracing metadata
    trace_id: str = ""
    span_id: str = ""
    
    # Quality metrics
    accuracy_estimate: Optional[float] = None
    completeness_score: Optional[float] = None
    
    # Cost tracking
    estimated_cost: Optional[float] = None
    model_used: Optional[str] = None


class EnhancedAsyncBaseAgent(ABC):
    """
    Next-generation async base agent with enterprise-grade performance patterns.
    
    Features:
    - Adaptive resource management with dynamic concurrency
    - ML-based semantic caching with intelligent invalidation
    - Speculative tool execution for reduced latency
    - Comprehensive performance analytics and monitoring
    - Circuit breaker pattern with graceful degradation
    - Advanced tracing and observability
    - Real-time streaming capabilities
    """
    
    def __init__(self, 
                 agent_name: str,
                 max_concurrency: int = 10,
                 enable_speculative_execution: bool = True,
                 enable_semantic_caching: bool = True,
                 cost_budget_per_request: float = 0.10):
        self.agent_name = agent_name
        self.max_concurrency = max_concurrency
        self.enable_speculative_execution = enable_speculative_execution
        self.enable_semantic_caching = enable_semantic_caching
        self.cost_budget_per_request = cost_budget_per_request
        
        # Performance tracking
        self.request_count = 0
        self.total_processing_time = 0.0
        self.error_count = 0
        
        # Agent-specific configuration
        self.agent_config = self._get_agent_config()
        
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{agent_name}")
        
        # Initialize advanced systems integration
        self._initialize_advanced_systems()
    
    def _initialize_advanced_systems(self):
        """Initialize integration with advanced systems."""
        # Register tools for speculative execution
        if self.enable_speculative_execution:
            self._register_speculative_tools()
        
        # Register fallback strategies
        self._register_fallback_strategies()
        
        # Register health checks
        reliability_manager.register_health_check(
            self.agent_name, 
            self._health_check
        )
    
    def _get_agent_config(self) -> Dict[str, Any]:
        """Get agent-specific configuration."""
        return {
            "complexity_mapping": {
                "simple": TaskComplexity.SIMPLE,
                "medium": TaskComplexity.MEDIUM,
                "complex": TaskComplexity.COMPLEX,
                "intensive": TaskComplexity.INTENSIVE
            },
            "default_complexity": TaskComplexity.MEDIUM,
            "semantic_cache_tags": [self.agent_name, "vision", "ai"],
            "fallback_confidence": 0.3
        }
    
    @abstractmethod
    async def _process_core(self, input_data: Any, **kwargs) -> Any:
        """Core processing logic to be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _extract_semantic_key(self, input_data: Any) -> str:
        """Extract semantic key for caching (to be implemented by subclasses)."""
        pass
    
    @abstractmethod
    def _estimate_task_complexity(self, input_data: Any) -> TaskComplexity:
        """Estimate task complexity (to be implemented by subclasses)."""
        pass
    
    async def process(self, 
                     input_data: Any,
                     trace_id: Optional[str] = None,
                     **kwargs) -> EnhancedProcessingResult:
        """
        Main processing entry point with all enterprise patterns.
        
        Args:
            input_data: Input data to process
            trace_id: Optional trace ID for correlation
            **kwargs: Additional processing parameters
            
        Returns:
            Enhanced processing result with comprehensive metadata
        """
        start_time = time.time()
        self.request_count += 1
        
        # Start tracing
        processing_span = await trace_manager.start_span(
            f"{self.agent_name}.process",
            SpanType.AGENT,
            trace_id=trace_id,
            attributes={
                "agent.name": self.agent_name,
                "input.type": type(input_data).__name__,
                "request.id": str(self.request_count)
            }
        )
        
        try:
            # Estimate task complexity
            complexity = self._estimate_task_complexity(input_data)
            processing_span.set_attribute("task.complexity", complexity.value)
            
            # Cost optimization
            estimated_tokens = len(str(input_data))
            optimal_model = cost_optimizer.route_request(
                complexity_score=self._complexity_to_score(complexity),
                budget_per_request=self.cost_budget_per_request,
                estimated_tokens=estimated_tokens
            )
            processing_span.set_attribute("model.selected", optimal_model)
            
            # Execute with all patterns integrated
            result = await self._execute_with_patterns(
                input_data=input_data,
                complexity=complexity,
                trace_id=processing_span.trace_id,
                span_id=processing_span.span_id,
                **kwargs
            )
            
            # Calculate final metrics
            processing_time_ms = (time.time() - start_time) * 1000
            self.total_processing_time += processing_time_ms / 1000
            
            # Record performance metrics
            self._record_performance_metrics(processing_time_ms, True)
            
            # Create enhanced result
            enhanced_result = EnhancedProcessingResult(
                primary_result=result,
                confidence=self._calculate_confidence(result),
                processing_time_ms=processing_time_ms,
                resource_usage=await self._get_resource_usage(),
                cache_hit=getattr(result, '_cache_hit', False),
                speculative_data=getattr(result, 'speculative_data', None),
                fallback_used=getattr(result, '_fallback_used', False),
                trace_id=processing_span.trace_id,
                span_id=processing_span.span_id,
                estimated_cost=self._estimate_processing_cost(processing_time_ms, optimal_model),
                model_used=optimal_model
            )
            
            # End span successfully
            await trace_manager.end_span(processing_span.span_id, "SUCCESS")
            
            return enhanced_result
            
        except Exception as e:
            self.error_count += 1
            self._record_performance_metrics((time.time() - start_time) * 1000, False)
            
            # End span with error
            await trace_manager.end_span(processing_span.span_id, "ERROR", str(e))
            
            # Try graceful degradation
            try:
                fallback_result = await self._handle_processing_failure(input_data, e, **kwargs)
                
                enhanced_result = EnhancedProcessingResult(
                    primary_result=fallback_result,
                    confidence=self.agent_config["fallback_confidence"],
                    processing_time_ms=(time.time() - start_time) * 1000,
                    resource_usage=await self._get_resource_usage(),
                    cache_hit=False,
                    fallback_used=True,
                    trace_id=processing_span.trace_id,
                    span_id=processing_span.span_id
                )
                
                return enhanced_result
                
            except Exception as fallback_error:
                self.logger.error(f"Both primary and fallback processing failed: {fallback_error}")
                raise
    
    async def _execute_with_patterns(self,
                                   input_data: Any,
                                   complexity: TaskComplexity,
                                   trace_id: str,
                                   span_id: str,
                                   **kwargs) -> Any:
        """Execute processing with all performance patterns."""
        
        # 1. Adaptive Resource Management
        async with resource_manager.acquire_slot(
            f"{self.agent_name}_{int(time.time() * 1000000)}", 
            complexity
        ):
            # 2. Semantic Caching (if enabled)
            if self.enable_semantic_caching:
                semantic_key = self._extract_semantic_key(input_data)
                cache_key = f"{self.agent_name}_{hash(str(input_data))}"
                
                cached_result = await semantic_cache_manager.get(
                    cache_key,
                    semantic_query=semantic_key,
                    tags=self.agent_config["semantic_cache_tags"]
                )
                
                if cached_result is not None:
                    cached_result._cache_hit = True
                    return cached_result
            
            # 3. Speculative Execution (if enabled)
            if self.enable_speculative_execution:
                async def core_execution():
                    return await self._execute_with_reliability(input_data, **kwargs)
                
                result = await speculative_runner.run_with_speculation(
                    query=str(input_data)[:200],  # First 200 chars as query
                    main_execution_func=core_execution
                )
            else:
                result = await self._execute_with_reliability(input_data, **kwargs)
            
            # 4. Cache successful results
            if self.enable_semantic_caching and not getattr(result, '_fallback_used', False):
                semantic_key = self._extract_semantic_key(input_data)
                cache_key = f"{self.agent_name}_{hash(str(input_data))}"
                
                await semantic_cache_manager.set(
                    cache_key,
                    result,
                    semantic_key=semantic_key,
                    tags=self.agent_config["semantic_cache_tags"]
                )
            
            return result
    
    async def _execute_with_reliability(self, input_data: Any, **kwargs) -> Any:
        """Execute core processing with reliability protection."""
        return await reliability_manager.execute_with_reliability(
            tool_name=f"{self.agent_name}_core",
            tool_func=self._process_core,
            *[input_data],
            fallback_strategies=self._get_fallback_strategies(),
            **kwargs
        )
    
    def _get_fallback_strategies(self) -> Dict[FallbackStrategy, Callable]:
        """Get fallback strategies for this agent."""
        # Override in subclasses for specific fallbacks
        return {
            FallbackStrategy.DEFAULT_RESPONSE: self._default_fallback
        }
    
    async def _default_fallback(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """Default fallback implementation."""
        return {
            "result": None,
            "message": f"{self.agent_name} temporarily unavailable",
            "fallback": True,
            "confidence": 0.0
        }
    
    def _register_speculative_tools(self):
        """Register tools for speculative execution."""
        # Override in subclasses to register specific tools
        pass
    
    def _register_fallback_strategies(self):
        """Register fallback strategies."""
        # Override in subclasses for specific strategies
        pass
    
    async def _health_check(self) -> bool:
        """Health check for this agent."""
        try:
            # Simple health check - override for specific checks
            test_data = np.zeros((100, 100, 3), dtype=np.uint8)  # Small test image
            
            start_time = time.time()
            result = await self._process_core(test_data)
            processing_time = time.time() - start_time
            
            # Check if result is valid (not just a string error)
            if isinstance(result, str):
                # If result is a string, it's likely an error message
                self.logger.warning(f"Health check returned string result: {result}")
                return False
            
            # Healthy if processing completes within reasonable time and returns valid result
            return processing_time < 5.0 and result is not None
            
        except Exception as e:
            self.logger.warning(f"Health check failed: {e}")
            return False
    
    def _complexity_to_score(self, complexity: TaskComplexity) -> float:
        """Convert task complexity to score (0-1)."""
        mapping = {
            TaskComplexity.SIMPLE: 0.2,
            TaskComplexity.MEDIUM: 0.5,
            TaskComplexity.COMPLEX: 0.8,
            TaskComplexity.INTENSIVE: 1.0
        }
        return mapping.get(complexity, 0.5)
    
    def _calculate_confidence(self, result: Any) -> float:
        """Calculate confidence score for result."""
        # Override in subclasses for specific confidence calculation
        if hasattr(result, 'confidence'):
            return float(result.confidence)
        elif hasattr(result, 'score'):
            return float(result.score)
        elif getattr(result, '_fallback_used', False):
            return 0.3
        else:
            return 0.8  # Default confidence
    
    async def _get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage."""
        # Simplified - would integrate with actual monitoring
        try:
            import psutil
            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "active_tasks": len(resource_manager.active_tasks)
            }
        except ImportError:
            return {"cpu_percent": 0, "memory_percent": 0, "active_tasks": 0}
    
    def _estimate_processing_cost(self, processing_time_ms: float, model_used: str) -> float:
        """Estimate processing cost."""
        # Simplified cost calculation
        base_cost = 0.001  # $0.001 base cost
        time_factor = processing_time_ms / 1000 / 60  # Cost per minute
        
        model_multipliers = {
            "gpt-4o-mini": 1.0,
            "gpt-4o": 3.0,
            "o1-preview": 10.0
        }
        
        multiplier = model_multipliers.get(model_used, 1.0)
        return base_cost * time_factor * multiplier
    
    def _record_performance_metrics(self, processing_time_ms: float, success: bool):
        """Record performance metrics."""
        from .performance_analytics import PerformanceMetric, MetricType
        
        # Latency metric
        latency_metric = PerformanceMetric(
            metric_type=MetricType.LATENCY,
            value=processing_time_ms,
            timestamp=time.time(),
            agent_name=self.agent_name
        )
        performance_analytics.record_metric(latency_metric)
        
        # Error rate metric
        error_metric = PerformanceMetric(
            metric_type=MetricType.ERROR_RATE,
            value=0.0 if success else 1.0,
            timestamp=time.time(),
            agent_name=self.agent_name
        )
        performance_analytics.record_metric(error_metric)
    
    async def _handle_processing_failure(self, 
                                       input_data: Any, 
                                       error: Exception,
                                       **kwargs) -> Any:
        """Handle processing failure with graceful degradation."""
        return await reliability_manager.degradation_manager.handle_tool_failure(
            tool_name=f"{self.agent_name}_core",
            error=error,
            original_args=(input_data,),
            original_kwargs=kwargs
        )
    
    # Streaming capabilities
    async def stream_process(self, 
                           input_stream: AsyncGenerator[Any, None],
                           **kwargs) -> AsyncGenerator[EnhancedProcessingResult, None]:
        """
        Process streaming input with real-time results.
        
        Args:
            input_stream: Async generator of input data
            **kwargs: Processing parameters
            
        Yields:
            Enhanced processing results
        """
        async for input_data in input_stream:
            try:
                result = await self.process(input_data, **kwargs)
                
                # Publish streaming event
                await event_manager.publish_event(
                    EventType.PROCESSING_UPDATE,
                    {
                        "agent_name": self.agent_name,
                        "result": result.primary_result,
                        "confidence": result.confidence,
                        "processing_time_ms": result.processing_time_ms
                    }
                )
                
                yield result
                
            except Exception as e:
                self.logger.error(f"Stream processing error: {e}")
                
                # Yield error result
                error_result = EnhancedProcessingResult(
                    primary_result={"error": str(e)},
                    confidence=0.0,
                    processing_time_ms=0.0,
                    resource_usage={},
                    cache_hit=False,
                    fallback_used=True
                )
                yield error_result
    
    # Batch processing with optimization
    async def batch_process(self, 
                           input_batch: List[Any],
                           batch_size: int = 10,
                           **kwargs) -> List[EnhancedProcessingResult]:
        """
        Process batch of inputs with optimization.
        
        Args:
            input_batch: List of input data
            batch_size: Processing batch size
            **kwargs: Processing parameters
            
        Returns:
            List of enhanced processing results
        """
        results = []
        
        # Process in chunks to avoid overwhelming the system
        for i in range(0, len(input_batch), batch_size):
            chunk = input_batch[i:i + batch_size]
            
            # Process chunk concurrently
            chunk_tasks = [
                self.process(input_data, **kwargs)
                for input_data in chunk
            ]
            
            chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
            
            # Handle exceptions in results
            for result in chunk_results:
                if isinstance(result, Exception):
                    error_result = EnhancedProcessingResult(
                        primary_result={"error": str(result)},
                        confidence=0.0,
                        processing_time_ms=0.0,
                        resource_usage={},
                        cache_hit=False,
                        fallback_used=True
                    )
                    results.append(error_result)
                else:
                    results.append(result)
        
        return results
    
    # Performance monitoring
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get comprehensive agent statistics."""
        avg_processing_time = 0.0
        if self.request_count > 0:
            avg_processing_time = self.total_processing_time / self.request_count
        
        error_rate = 0.0
        if self.request_count > 0:
            error_rate = self.error_count / self.request_count
        
        return {
            "agent_name": self.agent_name,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "success_rate": 1.0 - error_rate,
            "avg_processing_time_seconds": avg_processing_time,
            "performance_analytics": performance_analytics.get_agent_analytics(self.agent_name),
            "reliability_status": reliability_manager.get_reliability_status(),
            "resource_stats": resource_manager.get_resource_stats(),
            "cache_stats": semantic_cache_manager.get_stats(),
            "speculation_stats": speculative_runner.get_stats()
        }
    
    async def optimize_performance(self):
        """Run performance optimization routines."""
        self.logger.info(f"Running performance optimization for {self.agent_name}")
        
        # Clear low-relevance cache entries
        await semantic_cache_manager._cleanup_low_relevance_entries(0)
        
        # Update resource allocation based on recent performance
        current_metrics = resource_manager.get_current_metrics()
        if current_metrics.overall_load > 0.8:
            self.logger.warning("High system load detected, reducing concurrency")
            # Could adjust agent-specific settings here
        
        self.logger.info("Performance optimization completed")
    
    # Context managers for advanced patterns
    @asynccontextmanager
    async def processing_context(self, input_data: Any):
        """Context manager for processing with automatic resource management."""
        complexity = self._estimate_task_complexity(input_data)
        task_id = f"{self.agent_name}_{int(time.time() * 1000000)}"
        
        async with resource_manager.acquire_slot(task_id, complexity):
            yield
    
    @asynccontextmanager 
    async def traced_processing(self, operation_name: str, trace_id: Optional[str] = None):
        """Context manager for traced operations."""
        span = await trace_manager.start_span(
            f"{self.agent_name}.{operation_name}",
            SpanType.PROCESSING,
            trace_id=trace_id
        )
        
        try:
            yield span
        finally:
            await trace_manager.end_span(span.span_id)


# Enhanced decorator that combines all patterns
def enhanced_agent_method(
    complexity: TaskComplexity = TaskComplexity.MEDIUM,
    enable_caching: bool = True,
    enable_speculation: bool = True,
    semantic_tags: Optional[List[str]] = None
):
    """
    Decorator that applies all enterprise patterns to agent methods.
    
    Args:
        complexity: Task complexity level
        enable_caching: Enable semantic caching
        enable_speculation: Enable speculative execution
        semantic_tags: Semantic cache tags
    """
    def decorator(func):
        # Apply decorators in order
        enhanced_func = func
        
        # 1. Performance tracking
        enhanced_func = track_performance(MetricType.LATENCY)(enhanced_func)
        
        # 2. Reliability protection
        enhanced_func = reliable_tool(func.__name__)(enhanced_func)
        
        # 3. Resource management
        enhanced_func = adaptive_resource_control(complexity)(enhanced_func)
        
        # 4. Semantic caching
        if enable_caching:
            enhanced_func = smart_semantic_cache(
                tags=semantic_tags or ["vision", "ai"]
            )(enhanced_func)
        
        # 5. Speculative execution
        if enable_speculation:
            from .speculative_execution import enable_speculation as speculation_decorator
            enhanced_func = speculation_decorator()(enhanced_func)
        
        return enhanced_func
    
    return decorator


# Global initialization function
async def initialize_enhanced_systems():
    """Initialize all enhanced systems."""
    from .resource_manager import initialize_resource_manager
    from .semantic_cache import initialize_semantic_cache
    from .speculative_execution import initialize_speculative_system
    from .performance_analytics import initialize_performance_analytics
    from .reliability import initialize_reliability_system
    
    # Initialize all systems
    await initialize_resource_manager()
    await initialize_semantic_cache()
    await initialize_speculative_system()
    await initialize_performance_analytics()
    await initialize_reliability_system()
    
    logging.getLogger('EnhancedSystems').info(
        "🚀 All enhanced enterprise systems initialized successfully!"
    )


async def shutdown_enhanced_systems():
    """Shutdown all enhanced systems."""
    from .resource_manager import shutdown_resource_manager
    from .semantic_cache import shutdown_semantic_cache
    from .performance_analytics import shutdown_performance_analytics
    from .reliability import shutdown_reliability_system
    
    # Shutdown all systems
    await shutdown_resource_manager()
    await shutdown_semantic_cache()
    await shutdown_performance_analytics()
    await shutdown_reliability_system()
    
    logging.getLogger('EnhancedSystems').info("Enhanced systems shutdown completed")
