"""
Enhanced Async-First Base Agent with Performance Optimizations
Inspired by Youtu-agent's architecture for production-scale performance.
"""

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import logging
import cv2
import numpy as np
from ..utils.caching import async_file_cache, cached_tool_call
from ..utils.tracing import traced_operation, SpanType, trace_manager
from ..utils.streaming import publish_agent_started, publish_agent_completed, event_manager, EventType


@dataclass
class AsyncProcessingResult:
    """Enhanced result structure for async agent processing."""
    success: bool
    data: Dict[str, Any]
    confidence: Optional[float] = None
    inference_time_ms: Optional[float] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    trace_id: Optional[str] = None
    cache_hit: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'success': self.success,
            'data': self.data,
            'confidence': self.confidence,
            'inference_time_ms': self.inference_time_ms,
            'error': self.error,
            'metadata': self.metadata or {},
            'trace_id': self.trace_id,
            'cache_hit': self.cache_hit
        }


class AsyncBaseAgent(ABC):
    """
    Enhanced async-first base agent with performance optimizations.
    
    Features:
    - Fully asynchronous processing pipeline
    - Automatic caching with configurable expiration
    - Comprehensive tracing and observability
    - Real-time event streaming
    - Graceful error handling and recovery
    - Resource management with proper cleanup
    - Batch processing support
    - Concurrent request handling
    """
    
    def __init__(self, 
                 agent_name: str,
                 device: Optional[str] = None,
                 model_path: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the enhanced async base agent.
        
        Args:
            agent_name: Unique agent identifier
            device: Target device ('cuda', 'cpu', or None for auto-detection)
            model_path: Path to model files
            config: Agent configuration parameters
        """
        self.agent_name = agent_name
        self.device = self._get_device(device)
        self.model_path = model_path
        self.config = config or {}
        self.logger = logging.getLogger(f"Agent.{agent_name}")
        
        # Performance settings
        self.enable_caching = self.config.get('enable_caching', True)
        self.cache_expire_time = self.config.get('cache_expire_time', 3600)
        self.max_concurrent_requests = self.config.get('max_concurrent_requests', 10)
        self.batch_size = self.config.get('batch_size', 1)
        
        # State management
        self.model = None
        self._is_initialized = False
        self._initialization_lock = asyncio.Lock()
        self._request_semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        self._cleanup_tasks: List[asyncio.Task] = []
        
        # Metrics
        self._request_count = 0
        self._total_inference_time = 0.0
        self._error_count = 0
    
    def _get_device(self, device: Optional[str] = None) -> str:
        """
        Determine the best available device for processing.
        
        Args:
            device: Preferred device or None for auto-detection
            
        Returns:
            Device string ('cuda' or 'cpu')
        """
        if device and device != 'auto':
            return device
        
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                self.logger.info(f"GPU detected: {device_name}")
                return 'cuda'
        except ImportError:
            pass
        
        self.logger.info("Using CPU device")
        return 'cpu'
    
    @abstractmethod
    async def _initialize_model(self) -> bool:
        """
        Initialize the agent's model and resources.
        Must be implemented by subclasses.
        
        Returns:
            True if initialization successful
        """
        pass
    
    @abstractmethod
    async def _process_internal(self, input_data: Any, trace_id: str) -> Dict[str, Any]:
        """
        Internal processing method to be implemented by subclasses.
        
        Args:
            input_data: Input data to process
            trace_id: Trace ID for monitoring
            
        Returns:
            Processing results dictionary
        """
        pass
    
    async def initialize(self) -> bool:
        """
        Initialize the agent with thread-safe lazy loading.
        
        Returns:
            True if initialization successful
        """
        if self._is_initialized:
            return True
        
        async with self._initialization_lock:
            # Double-check after acquiring lock
            if self._is_initialized:
                return True
            
            async with traced_operation(
                f"{self.agent_name}.initialize", 
                SpanType.AGENT
            ) as span:
                span.set_attribute("agent.name", self.agent_name)
                span.set_attribute("agent.device", self.device)
                
                try:
                    success = await self._initialize_model()
                    
                    if success:
                        self._is_initialized = True
                        span.set_attribute("initialization.success", True)
                        self.logger.info(f"Agent {self.agent_name} initialized successfully")
                    else:
                        span.set_attribute("initialization.success", False)
                        self.logger.error(f"Agent {self.agent_name} initialization failed")
                    
                    return success
                
                except Exception as e:
                    span.set_attribute("initialization.error", str(e))
                    self.logger.error(f"Agent {self.agent_name} initialization error: {str(e)}")
                    return False
    
    async def process(self, 
                     input_data: Any,
                     use_cache: bool = True,
                     trace_id: Optional[str] = None) -> AsyncProcessingResult:
        """
        Process input data with full performance optimizations.
        
        Args:
            input_data: Input data to process
            use_cache: Whether to use caching
            trace_id: Optional trace ID for request correlation
            
        Returns:
            AsyncProcessingResult with processing results
        """
        if trace_id is None:
            trace_id = str(uuid.uuid4())
        
        # Ensure initialization
        if not self._is_initialized:
            if not await self.initialize():
                return AsyncProcessingResult(
                    success=False,
                    data={},
                    error="Agent initialization failed",
                    trace_id=trace_id
                )
        
        # Apply concurrency limiting
        async with self._request_semaphore:
            async with traced_operation(
                f"{self.agent_name}.process",
                SpanType.AGENT,
                trace_id=trace_id
            ) as span:
                
                span.set_attribute("agent.name", self.agent_name)
                span.set_attribute("input.type", type(input_data).__name__)
                
                # Publish start event
                await publish_agent_started(self.agent_name, trace_id, {
                    "type": type(input_data).__name__,
                    "config": self.config
                })
                
                try:
                    start_time = time.perf_counter()
                    
                    # Try cache first if enabled
                    cache_hit = False
                    if use_cache and self.enable_caching:
                        cached_result = await self._try_get_cached_result(input_data)
                        if cached_result:
                            cache_hit = True
                            span.add_event("cache_hit")
                            
                            return AsyncProcessingResult(
                                success=True,
                                data=cached_result,
                                trace_id=trace_id,
                                cache_hit=True
                            )
                    
                    if not cache_hit:
                        span.add_event("cache_miss")
                    
                    # Process the input
                    result_data = await self._process_internal(input_data, trace_id)
                    
                    # Calculate timing
                    end_time = time.perf_counter()
                    inference_time = (end_time - start_time) * 1000
                    
                    # Cache result if enabled
                    if use_cache and self.enable_caching:
                        await self._cache_result(input_data, result_data)
                    
                    # Update metrics
                    self._request_count += 1
                    self._total_inference_time += inference_time
                    
                    # Set span attributes
                    span.set_attribute("result.success", True)
                    span.set_attribute("result.inference_time_ms", inference_time)
                    span.set_attribute("result.cache_hit", cache_hit)
                    
                    # Create result
                    result = AsyncProcessingResult(
                        success=True,
                        data=result_data,
                        inference_time_ms=inference_time,
                        trace_id=trace_id,
                        cache_hit=cache_hit,
                        metadata={
                            "agent_name": self.agent_name,
                            "device": self.device,
                            "request_count": self._request_count
                        }
                    )
                    
                    # Publish completion event
                    await publish_agent_completed(self.agent_name, trace_id, {
                        "success": True,
                        "inference_time": inference_time,
                        "summary": result_data
                    })
                    
                    return result
                
                except Exception as e:
                    self._error_count += 1
                    error_msg = str(e)
                    
                    span.set_attribute("result.success", False)
                    span.set_attribute("result.error", error_msg)
                    
                    self.logger.error(f"Processing error in {self.agent_name}: {error_msg}")
                    
                    # Publish error event
                    await event_manager.publish_event(
                        EventType.AGENT_ERROR,
                        {"error": error_msg},
                        agent_name=self.agent_name,
                        trace_id=trace_id
                    )
                    
                    return AsyncProcessingResult(
                        success=False,
                        data={},
                        error=error_msg,
                        trace_id=trace_id
                    )
    
    async def process_batch(self, 
                           input_batch: List[Any],
                           use_cache: bool = True) -> List[AsyncProcessingResult]:
        """
        Process multiple inputs with optimized batching.
        
        Args:
            input_batch: List of input data
            use_cache: Whether to use caching
            
        Returns:
            List of processing results
        """
        if not input_batch:
            return []
        
        # Create semaphore for batch concurrency
        batch_semaphore = asyncio.Semaphore(min(len(input_batch), self.batch_size))
        
        async def process_single(input_data):
            async with batch_semaphore:
                return await self.process(input_data, use_cache)
        
        # Process all inputs concurrently
        tasks = [process_single(input_data) for input_data in input_batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in batch results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(AsyncProcessingResult(
                    success=False,
                    data={},
                    error=str(result),
                    metadata={"batch_index": i}
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    @async_file_cache(expire_time=3600)
    async def _try_get_cached_result(self, input_data: Any) -> Optional[Dict[str, Any]]:
        """
        Try to get cached result for input data.
        
        Args:
            input_data: Input data
            
        Returns:
            Cached result or None
        """
        # This method is decorated with caching, so it will automatically
        # cache and retrieve results based on input_data
        return None  # Cache miss - will be handled by decorator
    
    async def _cache_result(self, input_data: Any, result_data: Dict[str, Any]):
        """
        Cache processing result.
        
        Args:
            input_data: Original input data
            result_data: Processing result to cache
        """
        # The caching is handled by the decorator on _try_get_cached_result
        # This method can be overridden for custom caching logic
        pass
    
    async def _preprocess_image_async(self, image_input: Union[str, np.ndarray, bytes]) -> np.ndarray:
        """
        Asynchronously preprocess image input.
        
        Args:
            image_input: Image as file path, numpy array, or bytes
            
        Returns:
            Image as numpy array in BGR format
        """
        if isinstance(image_input, str):
            # File path or URL
            if image_input.startswith(('http://', 'https://')):
                # Download from URL asynchronously
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get(image_input) as response:
                        image_bytes = await response.read()
                        nparr = np.frombuffer(image_bytes, np.uint8)
                        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                # Load from file
                image = cv2.imread(image_input)
                if image is None:
                    raise ValueError(f"Could not load image from path: {image_input}")
        
        elif isinstance(image_input, bytes):
            # Raw bytes - decode asynchronously
            nparr = np.frombuffer(image_input, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        elif isinstance(image_input, np.ndarray):
            # Already numpy array
            image = image_input
        
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
        
        return image
    
    async def get_metrics(self) -> Dict[str, Any]:
        """
        Get agent performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        avg_inference_time = (
            self._total_inference_time / self._request_count 
            if self._request_count > 0 else 0.0
        )
        
        return {
            'agent_name': self.agent_name,
            'initialized': self._is_initialized,
            'total_requests': self._request_count,
            'total_errors': self._error_count,
            'avg_inference_time_ms': avg_inference_time,
            'error_rate': self._error_count / max(self._request_count, 1),
            'device': self.device,
            'concurrent_limit': self.max_concurrent_requests,
            'caching_enabled': self.enable_caching
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the agent.
        
        Returns:
            Health status dictionary
        """
        status = {
            'healthy': True,
            'agent_name': self.agent_name,
            'initialized': self._is_initialized,
            'device': self.device,
            'last_check': time.time()
        }
        
        # Test basic functionality if initialized
        if self._is_initialized:
            try:
                # Create a small test image
                test_image = np.zeros((100, 100, 3), dtype=np.uint8)
                
                # Test preprocessing
                await self._preprocess_image_async(test_image)
                
                status['preprocessing'] = True
            except Exception as e:
                status['healthy'] = False
                status['preprocessing'] = False
                status['error'] = str(e)
        
        return status
    
    @asynccontextmanager
    async def processing_context(self, trace_id: str):
        """
        Context manager for processing operations with proper resource management.
        
        Args:
            trace_id: Trace ID for monitoring
        """
        async with traced_operation(
            f"{self.agent_name}.processing_context",
            SpanType.AGENT,
            trace_id=trace_id
        ) as span:
            try:
                yield span
            finally:
                # Cleanup any temporary resources
                await self._cleanup_processing_resources()
    
    async def _cleanup_processing_resources(self):
        """Clean up any temporary resources created during processing."""
        # Override in subclasses for specific cleanup
        pass
    
    async def stream_process(self, 
                           input_data: Any,
                           trace_id: Optional[str] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream processing results in real-time.
        
        Args:
            input_data: Input data to process
            trace_id: Optional trace ID
            
        Yields:
            Processing updates and final results
        """
        if trace_id is None:
            trace_id = str(uuid.uuid4())
        
        # Ensure initialization
        if not self._is_initialized:
            yield {"type": "status", "message": "Initializing agent..."}
            if not await self.initialize():
                yield {"type": "error", "message": "Agent initialization failed"}
                return
        
        yield {"type": "status", "message": "Processing started"}
        
        try:
            # Process with streaming updates
            async with self.processing_context(trace_id) as span:
                result = await self.process(input_data, trace_id=trace_id)
                
                # Yield final result
                yield {
                    "type": "result",
                    "data": result.to_dict()
                }
        
        except Exception as e:
            yield {"type": "error", "message": str(e)}
    
    async def shutdown(self):
        """
        Gracefully shutdown the agent and cleanup resources.
        """
        self.logger.info(f"Shutting down agent: {self.agent_name}")
        
        # Cancel background tasks
        for task in self._cleanup_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._cleanup_tasks:
            await asyncio.gather(*self._cleanup_tasks, return_exceptions=True)
        
        # Agent-specific cleanup
        await self._cleanup_processing_resources()
        
        self._is_initialized = False
        self.logger.info(f"Agent {self.agent_name} shutdown complete")
    
    def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.shutdown()


# Agent registry for dynamic loading
class AgentRegistry:
    """Registry for managing agent instances with lifecycle management."""
    
    def __init__(self):
        self.agents: Dict[str, AsyncBaseAgent] = {}
        self.logger = logging.getLogger('AgentRegistry')
        self._lock = asyncio.Lock()
    
    async def register_agent(self, agent: AsyncBaseAgent) -> bool:
        """
        Register an agent instance.
        
        Args:
            agent: Agent instance to register
            
        Returns:
            True if registration successful
        """
        async with self._lock:
            if agent.agent_name in self.agents:
                self.logger.warning(f"Agent {agent.agent_name} already registered, replacing")
                await self.agents[agent.agent_name].shutdown()
            
            self.agents[agent.agent_name] = agent
            self.logger.info(f"Registered agent: {agent.agent_name}")
            return True
    
    async def get_agent(self, agent_name: str) -> Optional[AsyncBaseAgent]:
        """Get agent by name."""
        return self.agents.get(agent_name)
    
    async def shutdown_all(self):
        """Shutdown all registered agents."""
        self.logger.info("Shutting down all agents...")
        
        shutdown_tasks = [
            agent.shutdown() for agent in self.agents.values()
        ]
        
        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        
        self.agents.clear()
        self.logger.info("All agents shutdown complete")
    
    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Perform health check on all agents."""
        health_results = {}
        
        for agent_name, agent in self.agents.items():
            try:
                health_results[agent_name] = await agent.health_check()
            except Exception as e:
                health_results[agent_name] = {
                    'healthy': False,
                    'error': str(e)
                }
        
        return health_results
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary metrics for all agents."""
        total_requests = sum(agent._request_count for agent in self.agents.values())
        total_errors = sum(agent._error_count for agent in self.agents.values())
        
        return {
            'total_agents': len(self.agents),
            'active_agents': sum(1 for agent in self.agents.values() if agent._is_initialized),
            'total_requests': total_requests,
            'total_errors': total_errors,
            'overall_error_rate': total_errors / max(total_requests, 1),
            'agents': {
                name: {
                    'requests': agent._request_count,
                    'errors': agent._error_count,
                    'initialized': agent._is_initialized
                }
                for name, agent in self.agents.items()
            }
        }


# Global agent registry
agent_registry = AgentRegistry()
