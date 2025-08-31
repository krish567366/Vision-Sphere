"""
Circuit Breaker and Graceful Degradation System
Production-grade reliability patterns for tool and service failures.
"""

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from enum import Enum
from functools import wraps


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking calls due to failures
    HALF_OPEN = "half_open"  # Testing if service recovered


class FallbackStrategy(str, Enum):
    """Types of fallback strategies."""
    CACHED_RESULT = "cached_result"
    SIMPLIFIED_ALGORITHM = "simplified_algorithm"
    DEFAULT_RESPONSE = "default_response"
    ALTERNATIVE_SERVICE = "alternative_service"
    GRACEFUL_SKIP = "graceful_skip"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5       # Failures before opening
    success_threshold: int = 3       # Successes to close from half-open
    timeout_seconds: float = 60.0    # Time before trying half-open
    reset_timeout: float = 300.0     # Time before auto-reset
    
    # Failure detection
    error_rate_threshold: float = 0.5  # 50% error rate
    latency_threshold_ms: float = 10000  # 10 seconds
    
    # Monitoring
    sliding_window_size: int = 20    # Window for error rate calculation


@dataclass
class CircuitMetrics:
    """Circuit breaker metrics."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    state_changes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    
    @property
    def error_rate(self) -> float:
        """Calculate current error rate."""
        if self.total_calls == 0:
            return 0.0
        return self.failed_calls / self.total_calls
    
    @property
    def success_rate(self) -> float:
        """Calculate current success rate."""
        return 1.0 - self.error_rate


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class ToolCircuitBreaker:
    """
    Circuit breaker implementation for tool reliability.
    Prevents cascade failures and provides fallback mechanisms.
    """
    
    def __init__(self, 
                 tool_name: str,
                 config: Optional[CircuitBreakerConfig] = None):
        self.tool_name = tool_name
        self.config = config or CircuitBreakerConfig()
        
        self.state = CircuitState.CLOSED
        self.metrics = CircuitMetrics()
        
        # Sliding window for recent calls
        self.recent_calls: List[Tuple[float, bool]] = []  # (timestamp, success)
        
        # State transition times
        self.state_changed_at = time.time()
        self.last_half_open_attempt = 0.0
        
        self.logger = logging.getLogger(f"CircuitBreaker.{tool_name}")
        self._lock = asyncio.Lock()
    
    async def call_with_protection(self, 
                                  tool_func: Callable,
                                  *args,
                                  fallback_func: Optional[Callable] = None,
                                  **kwargs) -> Any:
        """
        Execute tool with circuit breaker protection.
        
        Args:
            tool_func: Function to execute
            *args: Function arguments
            fallback_func: Optional fallback function
            **kwargs: Function keyword arguments
            
        Returns:
            Function result or fallback result
            
        Raises:
            CircuitBreakerOpenError: When circuit is open and no fallback
        """
        async with self._lock:
            # Check circuit state
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.state_changed_at = time.time()
                    self.last_half_open_attempt = time.time()
                    self.logger.info(f"Circuit breaker for {self.tool_name} moved to HALF_OPEN")
                else:
                    # Circuit is open, try fallback
                    if fallback_func:
                        self.logger.warning(f"Using fallback for {self.tool_name} (circuit open)")
                        return await fallback_func(*args, **kwargs)
                    else:
                        raise CircuitBreakerOpenError(
                            f"Circuit breaker for {self.tool_name} is OPEN"
                        )
        
        # Execute function with monitoring
        start_time = time.time()
        success = False
        
        try:
            # Set timeout for half-open state
            if self.state == CircuitState.HALF_OPEN:
                result = await asyncio.wait_for(
                    tool_func(*args, **kwargs),
                    timeout=self.config.timeout_seconds
                )
            else:
                result = await tool_func(*args, **kwargs)
            
            success = True
            await self._on_success()
            return result
            
        except asyncio.TimeoutError as e:
            await self._on_failure(f"Timeout: {e}")
            
            # Try fallback on timeout
            if fallback_func:
                self.logger.warning(f"Using fallback for {self.tool_name} (timeout)")
                return await fallback_func(*args, **kwargs)
            raise
            
        except Exception as e:
            await self._on_failure(str(e))
            
            # Try fallback on error
            if fallback_func:
                self.logger.warning(f"Using fallback for {self.tool_name} (error: {e})")
                return await fallback_func(*args, **kwargs)
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset from OPEN to HALF_OPEN."""
        time_since_open = time.time() - self.state_changed_at
        return time_since_open >= self.config.timeout_seconds
    
    async def _on_success(self):
        """Handle successful function execution."""
        async with self._lock:
            self.metrics.total_calls += 1
            self.metrics.successful_calls += 1
            self.metrics.last_success_time = time.time()
            
            # Add to recent calls
            self.recent_calls.append((time.time(), True))
            self._cleanup_recent_calls()
            
            # State transitions
            if self.state == CircuitState.HALF_OPEN:
                # Count consecutive successes in half-open
                recent_successes = sum(
                    1 for _, success in self.recent_calls[-self.config.success_threshold:]
                    if success
                )
                
                if recent_successes >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.state_changed_at = time.time()
                    self.metrics.state_changes += 1
                    self.logger.info(f"Circuit breaker for {self.tool_name} CLOSED (recovered)")
    
    async def _on_failure(self, error_message: str):
        """Handle failed function execution."""
        async with self._lock:
            self.metrics.total_calls += 1
            self.metrics.failed_calls += 1
            self.metrics.last_failure_time = time.time()
            
            # Add to recent calls
            self.recent_calls.append((time.time(), False))
            self._cleanup_recent_calls()
            
            self.logger.warning(f"Tool {self.tool_name} failed: {error_message}")
            
            # Check if circuit should open
            await self._check_circuit_conditions()
    
    def _cleanup_recent_calls(self):
        """Remove old calls from sliding window."""
        cutoff_time = time.time() - 300  # Keep 5 minutes of history
        self.recent_calls = [
            (timestamp, success) for timestamp, success in self.recent_calls
            if timestamp > cutoff_time
        ]
    
    async def _check_circuit_conditions(self):
        """Check if circuit breaker should change state."""
        if self.state == CircuitState.OPEN:
            return
        
        # Check failure threshold
        recent_failures = sum(
            1 for _, success in self.recent_calls[-self.config.failure_threshold:]
            if not success
        )
        
        if recent_failures >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            self.state_changed_at = time.time()
            self.metrics.state_changes += 1
            self.logger.error(f"Circuit breaker for {self.tool_name} OPENED (failure threshold reached)")
            return
        
        # Check error rate in sliding window
        if len(self.recent_calls) >= self.config.sliding_window_size:
            window_failures = sum(
                1 for _, success in self.recent_calls[-self.config.sliding_window_size:]
                if not success
            )
            error_rate = window_failures / self.config.sliding_window_size
            
            if error_rate >= self.config.error_rate_threshold:
                self.state = CircuitState.OPEN
                self.state_changed_at = time.time()
                self.metrics.state_changes += 1
                self.logger.error(
                    f"Circuit breaker for {self.tool_name} OPENED (high error rate: {error_rate:.2%})"
                )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        return {
            "tool_name": self.tool_name,
            "state": self.state.value,
            "metrics": {
                "total_calls": self.metrics.total_calls,
                "success_rate": self.metrics.success_rate,
                "error_rate": self.metrics.error_rate,
                "state_changes": self.metrics.state_changes
            },
            "timing": {
                "state_changed_at": self.state_changed_at,
                "time_in_current_state": time.time() - self.state_changed_at,
                "last_failure": self.metrics.last_failure_time,
                "last_success": self.metrics.last_success_time
            }
        }


class GracefulDegradation:
    """
    Graceful degradation system with multiple fallback strategies.
    """
    
    def __init__(self):
        self.fallback_strategies: Dict[str, Dict[FallbackStrategy, Callable]] = {}
        self.cached_fallbacks: Dict[str, Any] = {}
        self.degradation_stats: Dict[str, int] = defaultdict(int)
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def register_fallback(self,
                         tool_name: str,
                         strategy: FallbackStrategy,
                         fallback_func: Callable):
        """Register a fallback strategy for a tool."""
        if tool_name not in self.fallback_strategies:
            self.fallback_strategies[tool_name] = {}
        
        self.fallback_strategies[tool_name][strategy] = fallback_func
        self.logger.debug(f"Registered {strategy.value} fallback for {tool_name}")
    
    async def handle_tool_failure(self,
                                 tool_name: str,
                                 error: Exception,
                                 original_args: tuple = (),
                                 original_kwargs: Dict[str, Any] = None) -> Any:
        """
        Handle tool failure with appropriate fallback strategy.
        
        Args:
            tool_name: Name of the failed tool
            error: Exception that occurred
            original_args: Original function arguments
            original_kwargs: Original function keyword arguments
            
        Returns:
            Fallback result
        """
        if original_kwargs is None:
            original_kwargs = {}
        
        self.degradation_stats[tool_name] += 1
        
        # Get available fallback strategies for this tool
        strategies = self.fallback_strategies.get(tool_name, {})
        
        if not strategies:
            return await self._generic_fallback(tool_name, error)
        
        # Try strategies in order of preference
        strategy_order = [
            FallbackStrategy.CACHED_RESULT,
            FallbackStrategy.ALTERNATIVE_SERVICE,
            FallbackStrategy.SIMPLIFIED_ALGORITHM,
            FallbackStrategy.DEFAULT_RESPONSE,
            FallbackStrategy.GRACEFUL_SKIP
        ]
        
        for strategy in strategy_order:
            if strategy in strategies:
                try:
                    self.logger.info(f"Attempting {strategy.value} fallback for {tool_name}")
                    fallback_func = strategies[strategy]
                    result = await fallback_func(*original_args, **original_kwargs)
                    
                    self.logger.info(f"Successfully used {strategy.value} fallback for {tool_name}")
                    return result
                    
                except Exception as fallback_error:
                    self.logger.warning(
                        f"Fallback {strategy.value} failed for {tool_name}: {fallback_error}"
                    )
                    continue
        
        # All fallbacks failed
        return await self._generic_fallback(tool_name, error)
    
    async def _generic_fallback(self, tool_name: str, error: Exception) -> Dict[str, Any]:
        """Generic fallback when no specific strategies are available."""
        self.logger.error(f"All fallbacks failed for {tool_name}, using generic response")
        
        return {
            "error": True,
            "message": f"Service temporarily unavailable: {tool_name}",
            "fallback_used": True,
            "original_error": str(error),
            "timestamp": time.time()
        }
    
    # Common fallback implementations
    async def cached_result_fallback(self,
                                   cache_key: str,
                                   default_value: Any = None) -> Any:
        """Fallback using cached results."""
        cached_result = self.cached_fallbacks.get(cache_key)
        if cached_result:
            return cached_result
        
        if default_value is not None:
            return default_value
        
        raise Exception("No cached result available")
    
    async def simplified_algorithm_fallback(self,
                                          simple_func: Callable,
                                          *args,
                                          **kwargs) -> Any:
        """Fallback using a simplified algorithm."""
        return await simple_func(*args, **kwargs)
    
    async def default_response_fallback(self, default_response: Any) -> Any:
        """Fallback using a default response."""
        return default_response
    
    def get_degradation_stats(self) -> Dict[str, Any]:
        """Get degradation statistics."""
        total_degradations = sum(self.degradation_stats.values())
        
        return {
            "total_degradations": total_degradations,
            "degradations_by_tool": dict(self.degradation_stats),
            "registered_fallbacks": {
                tool: list(strategies.keys())
                for tool, strategies in self.fallback_strategies.items()
            },
            "cached_fallbacks": len(self.cached_fallbacks)
        }


class ReliabilityManager:
    """
    Central reliability management system combining circuit breakers and graceful degradation.
    """
    
    def __init__(self):
        self.circuit_breakers: Dict[str, ToolCircuitBreaker] = {}
        self.degradation_manager = GracefulDegradation()
        
        # Health monitoring
        self.health_checks: Dict[str, Callable] = {}
        self.service_status: Dict[str, bool] = {}
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self._monitoring_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
    
    async def start(self):
        """Start the reliability manager."""
        self._monitoring_task = asyncio.create_task(self._health_monitoring_loop())
        self.logger.info("Reliability Manager started")
    
    async def stop(self):
        """Stop the reliability manager."""
        self._shutdown_event.set()
        if self._monitoring_task:
            await self._monitoring_task
        self.logger.info("Reliability Manager stopped")
    
    def get_circuit_breaker(self, 
                           tool_name: str,
                           config: Optional[CircuitBreakerConfig] = None) -> ToolCircuitBreaker:
        """Get or create circuit breaker for a tool."""
        if tool_name not in self.circuit_breakers:
            self.circuit_breakers[tool_name] = ToolCircuitBreaker(tool_name, config)
            self.logger.debug(f"Created circuit breaker for {tool_name}")
        
        return self.circuit_breakers[tool_name]
    
    def register_health_check(self, service_name: str, health_check_func: Callable):
        """Register health check for a service."""
        self.health_checks[service_name] = health_check_func
        self.service_status[service_name] = True  # Assume healthy initially
        self.logger.debug(f"Registered health check for {service_name}")
    
    async def execute_with_reliability(self,
                                     tool_name: str,
                                     tool_func: Callable,
                                     *args,
                                     fallback_strategies: Optional[Dict[FallbackStrategy, Callable]] = None,
                                     circuit_config: Optional[CircuitBreakerConfig] = None,
                                     **kwargs) -> Any:
        """
        Execute tool with full reliability protection.
        
        Args:
            tool_name: Name of the tool
            tool_func: Function to execute
            *args: Function arguments
            fallback_strategies: Available fallback strategies
            circuit_config: Circuit breaker configuration
            **kwargs: Function keyword arguments
            
        Returns:
            Function result or fallback result
        """
        # Register fallback strategies
        if fallback_strategies:
            for strategy, fallback_func in fallback_strategies.items():
                self.degradation_manager.register_fallback(tool_name, strategy, fallback_func)
        
        # Get circuit breaker
        circuit_breaker = self.get_circuit_breaker(tool_name, circuit_config)
        
        # Create fallback function that uses degradation manager
        async def integrated_fallback(*f_args, **f_kwargs):
            return await self.degradation_manager.handle_tool_failure(
                tool_name, Exception("Circuit breaker fallback"), f_args, f_kwargs
            )
        
        # Execute with circuit breaker protection
        try:
            return await circuit_breaker.call_with_protection(
                tool_func, *args, fallback_func=integrated_fallback, **kwargs
            )
        except CircuitBreakerOpenError:
            # Circuit is open, use degradation
            return await self.degradation_manager.handle_tool_failure(
                tool_name, CircuitBreakerOpenError("Circuit breaker is open"), args, kwargs
            )
    
    async def _health_monitoring_loop(self):
        """Monitor service health and update circuit breakers."""
        while not self._shutdown_event.is_set():
            try:
                # Run health checks
                for service_name, health_check in self.health_checks.items():
                    try:
                        is_healthy = await health_check()
                        previous_status = self.service_status.get(service_name, True)
                        self.service_status[service_name] = is_healthy
                        
                        # Log status changes
                        if previous_status != is_healthy:
                            status_text = "healthy" if is_healthy else "unhealthy"
                            self.logger.info(f"Service {service_name} is now {status_text}")
                        
                    except Exception as e:
                        self.service_status[service_name] = False
                        self.logger.error(f"Health check failed for {service_name}: {e}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(30)
    
    def get_reliability_status(self) -> Dict[str, Any]:
        """Get comprehensive reliability status."""
        return {
            "circuit_breakers": {
                name: cb.get_status()
                for name, cb in self.circuit_breakers.items()
            },
            "degradation_stats": self.degradation_manager.get_degradation_stats(),
            "service_health": self.service_status,
            "total_protected_tools": len(self.circuit_breakers)
        }


# Global reliability manager
reliability_manager = ReliabilityManager()


# Decorator for automatic reliability protection
def reliable_tool(tool_name: Optional[str] = None,
                 circuit_config: Optional[CircuitBreakerConfig] = None,
                 fallback_strategies: Optional[Dict[FallbackStrategy, Callable]] = None):
    """
    Decorator to add reliability protection to any function.
    
    Args:
        tool_name: Tool name (defaults to function name)
        circuit_config: Circuit breaker configuration
        fallback_strategies: Available fallback strategies
    """
    def decorator(func):
        actual_tool_name = tool_name or func.__name__
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await reliability_manager.execute_with_reliability(
                actual_tool_name,
                func,
                *args,
                fallback_strategies=fallback_strategies,
                circuit_config=circuit_config,
                **kwargs
            )
        
        return wrapper
    return decorator


# Common fallback functions for vision tasks
async def simple_face_detection_fallback(*args, **kwargs):
    """Simplified face detection using basic OpenCV."""
    # Simplified implementation for fallback
    return {
        "faces": [],
        "message": "Using simplified face detection",
        "fallback": True
    }


async def cached_classification_fallback(*args, **kwargs):
    """Use cached classification results."""
    return {
        "predictions": [{"label": "unknown", "confidence": 0.5}],
        "message": "Using cached classification",
        "fallback": True
    }


async def default_object_detection_fallback(*args, **kwargs):
    """Default object detection response."""
    return {
        "objects": [],
        "message": "Object detection temporarily unavailable",
        "fallback": True
    }


async def initialize_reliability_system():
    """Initialize the reliability management system."""
    await reliability_manager.start()
    
    # Register common fallbacks
    degradation = reliability_manager.degradation_manager
    
    degradation.register_fallback(
        "face_detection", 
        FallbackStrategy.SIMPLIFIED_ALGORITHM, 
        simple_face_detection_fallback
    )
    
    degradation.register_fallback(
        "image_classification",
        FallbackStrategy.CACHED_RESULT,
        cached_classification_fallback
    )
    
    degradation.register_fallback(
        "object_detection",
        FallbackStrategy.DEFAULT_RESPONSE,
        default_object_detection_fallback
    )
    
    logging.getLogger('ReliabilitySystem').info("Reliability Management System initialized")


async def shutdown_reliability_system():
    """Shutdown the reliability system."""
    await reliability_manager.stop()
