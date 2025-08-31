"""
Adaptive Resource Management System
Dynamic concurrency control and resource allocation based on system load and task complexity.
"""

import asyncio
import psutil
import time
import logging
from dataclasses import dataclass
from typing import Dict, Optional, List, Any
from contextlib import asynccontextmanager
from enum import Enum
from functools import wraps
from functools import wraps


class TaskComplexity(str, Enum):
    """Task complexity levels for resource allocation."""
    SIMPLE = "simple"      # Basic image classification
    MEDIUM = "medium"      # Face detection, object detection
    COMPLEX = "complex"    # Video analysis, multi-modal processing
    INTENSIVE = "intensive" # Batch processing, large model inference


@dataclass
class SystemMetrics:
    """Current system resource metrics."""
    cpu_percent: float
    memory_percent: float
    disk_io_percent: float
    gpu_memory_percent: Optional[float] = None
    active_tasks: int = 0
    queue_length: int = 0
    
    @property
    def overall_load(self) -> float:
        """Calculate overall system load score (0-1)."""
        weights = {
            'cpu': 0.4,
            'memory': 0.3,
            'disk': 0.2,
            'tasks': 0.1
        }
        
        task_load = min(self.active_tasks / 20, 1.0)  # Normalize to 20 max tasks
        
        return (
            weights['cpu'] * (self.cpu_percent / 100) +
            weights['memory'] * (self.memory_percent / 100) +
            weights['disk'] * (self.disk_io_percent / 100) +
            weights['tasks'] * task_load
        )


@dataclass
class ResourceProfile:
    """Resource requirements for different task types."""
    cpu_weight: float      # CPU importance (0-1)
    memory_weight: float   # Memory importance (0-1)
    io_weight: float       # I/O importance (0-1)
    expected_duration: float  # Expected duration in seconds
    parallelizable: bool   # Can run in parallel
    
    def calculate_cost(self, metrics: SystemMetrics) -> float:
        """Calculate resource cost for this task given current metrics."""
        return (
            self.cpu_weight * (metrics.cpu_percent / 100) +
            self.memory_weight * (metrics.memory_percent / 100) +
            self.io_weight * (metrics.disk_io_percent / 100)
        )


class AdaptiveResourceManager:
    """
    Intelligent resource manager that adapts concurrency based on:
    - Current system load (CPU, memory, I/O)
    - Task complexity and resource requirements
    - Historical performance data
    - Queue length and backpressure
    """
    
    def __init__(self, 
                 max_concurrency: int = 10,
                 min_concurrency: int = 1,
                 monitoring_interval: float = 5.0):
        self.max_concurrency = max_concurrency
        self.min_concurrency = min_concurrency
        self.monitoring_interval = monitoring_interval
        
        # Current state
        self.current_concurrency = min_concurrency
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.metrics_history: List[SystemMetrics] = []
        
        # Resource profiles for different task types
        self.resource_profiles = {
            TaskComplexity.SIMPLE: ResourceProfile(
                cpu_weight=0.2, memory_weight=0.1, io_weight=0.1,
                expected_duration=0.5, parallelizable=True
            ),
            TaskComplexity.MEDIUM: ResourceProfile(
                cpu_weight=0.5, memory_weight=0.3, io_weight=0.2,
                expected_duration=2.0, parallelizable=True
            ),
            TaskComplexity.COMPLEX: ResourceProfile(
                cpu_weight=0.7, memory_weight=0.6, io_weight=0.4,
                expected_duration=10.0, parallelizable=False
            ),
            TaskComplexity.INTENSIVE: ResourceProfile(
                cpu_weight=0.9, memory_weight=0.8, io_weight=0.6,
                expected_duration=30.0, parallelizable=False
            )
        }
        
        # Performance tracking
        self.performance_history: Dict[TaskComplexity, List[float]] = {
            complexity: [] for complexity in TaskComplexity
        }
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self._monitoring_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
    
    async def start(self):
        """Start the resource manager and monitoring."""
        self._monitoring_task = asyncio.create_task(self._monitor_resources())
        self.logger.info("Adaptive Resource Manager started")
    
    async def stop(self):
        """Stop the resource manager."""
        self._shutdown_event.set()
        if self._monitoring_task:
            await self._monitoring_task
        
        # Cancel all active tasks
        for task in self.active_tasks.values():
            task.cancel()
        
        await asyncio.gather(*self.active_tasks.values(), return_exceptions=True)
        self.logger.info("Adaptive Resource Manager stopped")
    
    def get_current_metrics(self) -> SystemMetrics:
        """Get current system resource metrics."""
        # Get basic system metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()
        
        # Calculate disk I/O percentage (simplified)
        disk_io_percent = min(
            (disk_io.read_bytes + disk_io.write_bytes) / (1024 * 1024 * 100), 
            100
        ) if disk_io else 0
        
        # Try to get GPU metrics if available
        gpu_memory_percent = None
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_memory_percent = gpus[0].memoryUtil * 100
        except ImportError:
            pass
        
        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_io_percent=disk_io_percent,
            gpu_memory_percent=gpu_memory_percent,
            active_tasks=len(self.active_tasks),
            queue_length=self.task_queue.qsize()
        )
    
    def calculate_optimal_concurrency(self, 
                                    current_metrics: SystemMetrics,
                                    task_complexity: TaskComplexity) -> int:
        """
        Calculate optimal concurrency based on current system state.
        
        Args:
            current_metrics: Current system metrics
            task_complexity: Complexity of incoming tasks
            
        Returns:
            Optimal concurrency level
        """
        profile = self.resource_profiles[task_complexity]
        
        # Base calculation on overall system load
        load_factor = 1.0 - current_metrics.overall_load
        base_concurrency = int(self.max_concurrency * load_factor)
        
        # Adjust for task complexity
        if task_complexity == TaskComplexity.INTENSIVE:
            # Intensive tasks should run with lower concurrency
            base_concurrency = max(1, base_concurrency // 3)
        elif task_complexity == TaskComplexity.COMPLEX:
            base_concurrency = max(2, base_concurrency // 2)
        elif task_complexity == TaskComplexity.SIMPLE:
            # Simple tasks can use higher concurrency
            base_concurrency = min(self.max_concurrency, base_concurrency * 2)
        
        # Apply queue backpressure
        if current_metrics.queue_length > 10:
            base_concurrency = max(self.min_concurrency, base_concurrency // 2)
        
        # Historical performance adjustment
        avg_performance = self._get_average_performance(task_complexity)
        if avg_performance and avg_performance > profile.expected_duration * 2:
            # Tasks taking too long, reduce concurrency
            base_concurrency = max(self.min_concurrency, base_concurrency - 1)
        
        return max(self.min_concurrency, min(self.max_concurrency, base_concurrency))
    
    def _get_average_performance(self, complexity: TaskComplexity) -> Optional[float]:
        """Get average performance for task complexity."""
        history = self.performance_history[complexity]
        return sum(history) / len(history) if history else None
    
    @asynccontextmanager
    async def acquire_slot(self, 
                          task_id: str,
                          complexity: TaskComplexity,
                          timeout: float = 60.0):
        """
        Acquire a processing slot with adaptive concurrency control.
        
        Args:
            task_id: Unique task identifier
            complexity: Task complexity level
            timeout: Maximum wait time for slot acquisition
        """
        start_time = time.time()
        
        while True:
            current_metrics = self.get_current_metrics()
            optimal_concurrency = self.calculate_optimal_concurrency(
                current_metrics, complexity
            )
            
            # Check if we can acquire a slot
            if len(self.active_tasks) < optimal_concurrency:
                # Create placeholder task
                placeholder_task = asyncio.create_task(asyncio.sleep(0))
                self.active_tasks[task_id] = placeholder_task
                
                self.logger.debug(
                    f"Acquired slot for {task_id} "
                    f"(complexity: {complexity.value}, "
                    f"concurrency: {len(self.active_tasks)}/{optimal_concurrency})"
                )
                
                try:
                    task_start = time.time()
                    yield
                    
                    # Record performance
                    duration = time.time() - task_start
                    self.performance_history[complexity].append(duration)
                    
                    # Keep only recent history
                    if len(self.performance_history[complexity]) > 100:
                        self.performance_history[complexity] = \
                            self.performance_history[complexity][-50:]
                    
                finally:
                    # Release slot
                    self.active_tasks.pop(task_id, None)
                    placeholder_task.cancel()
                
                break
            
            # Check timeout
            if time.time() - start_time > timeout:
                raise asyncio.TimeoutError(f"Could not acquire slot for {task_id} within {timeout}s")
            
            # Wait before retrying
            await asyncio.sleep(0.1)
    
    async def _monitor_resources(self):
        """Background task to monitor system resources."""
        while not self._shutdown_event.is_set():
            try:
                metrics = self.get_current_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only recent history
                if len(self.metrics_history) > 100:
                    self.metrics_history = self.metrics_history[-50:]
                
                # Log resource usage if high
                if metrics.overall_load > 0.8:
                    self.logger.warning(
                        f"High system load detected: {metrics.overall_load:.2f} "
                        f"(CPU: {metrics.cpu_percent:.1f}%, "
                        f"Memory: {metrics.memory_percent:.1f}%, "
                        f"Active tasks: {metrics.active_tasks})"
                    )
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get resource management statistics."""
        current_metrics = self.get_current_metrics()
        
        return {
            "current_concurrency": len(self.active_tasks),
            "optimal_concurrency": {
                complexity.value: self.calculate_optimal_concurrency(current_metrics, complexity)
                for complexity in TaskComplexity
            },
            "system_metrics": {
                "cpu_percent": current_metrics.cpu_percent,
                "memory_percent": current_metrics.memory_percent,
                "overall_load": current_metrics.overall_load,
                "active_tasks": current_metrics.active_tasks,
                "queue_length": current_metrics.queue_length
            },
            "performance_averages": {
                complexity.value: self._get_average_performance(complexity)
                for complexity in TaskComplexity
            }
        }


# Global resource manager instance
resource_manager = AdaptiveResourceManager()


# Decorator for automatic resource management
def adaptive_resource_control(complexity: TaskComplexity):
    """Decorator to automatically manage resources for agent methods."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            task_id = f"{func.__name__}_{int(time.time() * 1000000)}"
            
            async with resource_manager.acquire_slot(task_id, complexity):
                return await func(*args, **kwargs)
        
        return wrapper
    return decorator


async def initialize_resource_manager():
    """Initialize the global resource manager."""
    await resource_manager.start()
    logging.getLogger('ResourceManager').info("Adaptive Resource Manager initialized")


async def shutdown_resource_manager():
    """Shutdown the global resource manager."""
    await resource_manager.stop()
