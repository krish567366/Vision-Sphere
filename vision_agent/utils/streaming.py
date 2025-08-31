"""
Streaming Event System for Real-Time Agent Monitoring
Provides real-time updates for agent processing with WebSocket support.
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any, Dict, List, Optional, Callable, AsyncGenerator
import logging


class EventType(str, Enum):
    """Types of streaming events."""
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"
    AGENT_ERROR = "agent_error"
    PROCESSING_UPDATE = "processing_update"
    MODEL_INFERENCE = "model_inference"
    TOOL_EXECUTION = "tool_execution"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    SYSTEM_STATUS = "system_status"


@dataclass
class StreamEvent:
    """Individual streaming event."""
    event_id: str
    event_type: EventType
    timestamp: float
    trace_id: Optional[str]
    agent_name: Optional[str]
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class StreamEventManager:
    """
    Central event management system for real-time streaming.
    
    Features:
    - Event subscription and publishing
    - Filtering by event type and agent
    - Buffer management for slow consumers
    - Automatic cleanup of old events
    """
    
    def __init__(self, max_buffer_size: int = 1000, cleanup_interval: int = 300):
        """
        Initialize event manager.
        
        Args:
            max_buffer_size: Maximum events to buffer per subscriber
            cleanup_interval: Cleanup interval in seconds
        """
        self.max_buffer_size = max_buffer_size
        self.cleanup_interval = cleanup_interval
        
        # Event storage and subscribers
        self.event_history: List[StreamEvent] = []
        self.subscribers: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger('StreamEventManager')
        
        # Background cleanup task (initialized lazily)
        self._cleanup_task: Optional[asyncio.Task] = None
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize the event manager."""
        if not self.is_initialized:
            self._cleanup_task = asyncio.create_task(self._background_cleanup())
            self.is_initialized = True
            self.logger.info("Stream event manager initialized")
    
    async def shutdown(self):
        """Shutdown the event manager."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
        self.is_initialized = False
    
    async def publish_event(self, 
                           event_type: EventType,
                           data: Dict[str, Any],
                           agent_name: Optional[str] = None,
                           trace_id: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> StreamEvent:
        """
        Publish an event to all subscribers.
        
        Args:
            event_type: Type of event
            data: Event data
            agent_name: Name of the agent generating the event
            trace_id: Associated trace ID
            metadata: Additional metadata
            
        Returns:
            Created stream event
        """
        event = StreamEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=time.time(),
            trace_id=trace_id,
            agent_name=agent_name,
            data=data,
            metadata=metadata
        )
        
        # Add to history
        self.event_history.append(event)
        
        # Trim history if too long
        if len(self.event_history) > self.max_buffer_size * 2:
            self.event_history = self.event_history[-self.max_buffer_size:]
        
        # Notify subscribers
        await self._notify_subscribers(event)
        
        return event
    
    async def subscribe(self, 
                       subscriber_id: str,
                       event_types: Optional[List[EventType]] = None,
                       agent_filter: Optional[str] = None,
                       buffer_size: Optional[int] = None) -> AsyncGenerator[StreamEvent, None]:
        """
        Subscribe to streaming events.
        
        Args:
            subscriber_id: Unique subscriber identifier
            event_types: Filter by event types (None for all)
            agent_filter: Filter by agent name (None for all)
            buffer_size: Custom buffer size for this subscriber
            
        Yields:
            Stream events matching filters
        """
        # Create subscriber entry
        subscriber_queue = asyncio.Queue(maxsize=buffer_size or self.max_buffer_size)
        
        self.subscribers[subscriber_id] = {
            'queue': subscriber_queue,
            'event_types': event_types,
            'agent_filter': agent_filter,
            'created_at': time.time(),
            'last_activity': time.time()
        }
        
        self.logger.info(f"New subscriber: {subscriber_id}")
        
        try:
            while True:
                # Wait for events
                event = await subscriber_queue.get()
                
                # Update activity timestamp
                self.subscribers[subscriber_id]['last_activity'] = time.time()
                
                yield event
        
        except asyncio.CancelledError:
            # Cleanup on cancellation
            self.logger.info(f"Subscriber {subscriber_id} cancelled")
        
        finally:
            # Remove subscriber
            if subscriber_id in self.subscribers:
                del self.subscribers[subscriber_id]
                self.logger.info(f"Removed subscriber: {subscriber_id}")
    
    async def _notify_subscribers(self, event: StreamEvent):
        """Notify all matching subscribers of an event."""
        for subscriber_id, subscriber_info in list(self.subscribers.items()):
            try:
                # Apply filters
                if not self._event_matches_filters(event, subscriber_info):
                    continue
                
                queue = subscriber_info['queue']
                
                # Non-blocking put with overflow handling
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    # Remove oldest event and add new one
                    try:
                        queue.get_nowait()
                        queue.put_nowait(event)
                        self.logger.warning(f"Buffer overflow for subscriber {subscriber_id}")
                    except asyncio.QueueEmpty:
                        pass
            
            except Exception as e:
                self.logger.error(f"Error notifying subscriber {subscriber_id}: {str(e)}")
                # Remove problematic subscriber
                if subscriber_id in self.subscribers:
                    del self.subscribers[subscriber_id]
    
    def _event_matches_filters(self, event: StreamEvent, subscriber_info: Dict[str, Any]) -> bool:
        """Check if event matches subscriber filters."""
        # Event type filter
        event_types = subscriber_info['event_types']
        if event_types and event.event_type not in event_types:
            return False
        
        # Agent filter
        agent_filter = subscriber_info['agent_filter']
        if agent_filter and event.agent_name != agent_filter:
            return False
        
        return True
    
    async def _background_cleanup(self):
        """Background task for cleaning up inactive subscribers."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                current_time = time.time()
                inactive_threshold = 300  # 5 minutes
                
                # Find inactive subscribers
                inactive_subscribers = [
                    subscriber_id for subscriber_id, info in self.subscribers.items()
                    if current_time - info['last_activity'] > inactive_threshold
                ]
                
                # Remove inactive subscribers
                for subscriber_id in inactive_subscribers:
                    del self.subscribers[subscriber_id]
                    self.logger.info(f"Removed inactive subscriber: {subscriber_id}")
                
                # Cleanup old events
                if len(self.event_history) > self.max_buffer_size:
                    self.event_history = self.event_history[-self.max_buffer_size // 2:]
            
            except Exception as e:
                self.logger.error(f"Cleanup error: {str(e)}")
    
    def get_subscriber_count(self) -> int:
        """Get current number of active subscribers."""
        return len(self.subscribers)
    
    def get_event_history(self, 
                         limit: int = 100,
                         event_types: Optional[List[EventType]] = None,
                         agent_filter: Optional[str] = None) -> List[StreamEvent]:
        """
        Get recent event history with filters.
        
        Args:
            limit: Maximum number of events to return
            event_types: Filter by event types
            agent_filter: Filter by agent name
            
        Returns:
            List of matching events
        """
        filtered_events = []
        
        for event in reversed(self.event_history):
            # Apply filters
            if event_types and event.event_type not in event_types:
                continue
            
            if agent_filter and event.agent_name != agent_filter:
                continue
            
            filtered_events.append(event)
            
            if len(filtered_events) >= limit:
                break
        
        return list(reversed(filtered_events))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event manager statistics."""
        return {
            'active_subscribers': len(self.subscribers),
            'total_events': len(self.event_history),
            'initialized': self.is_initialized
        }


# Global event manager instance
event_manager = StreamEventManager()


# Convenience functions for publishing events
async def publish_agent_started(agent_name: str, trace_id: str, input_info: Dict[str, Any]):
    """Publish agent started event."""
    await event_manager.publish_event(
        EventType.AGENT_STARTED,
        {
            "input_type": input_info.get("type"),
            "input_size": input_info.get("size"),
            "config": input_info.get("config", {})
        },
        agent_name=agent_name,
        trace_id=trace_id
    )


async def publish_agent_completed(agent_name: str, trace_id: str, result_info: Dict[str, Any]):
    """Publish agent completed event."""
    await event_manager.publish_event(
        EventType.AGENT_COMPLETED,
        {
            "success": result_info.get("success"),
            "inference_time_ms": result_info.get("inference_time"),
            "result_summary": result_info.get("summary", {})
        },
        agent_name=agent_name,
        trace_id=trace_id
    )


async def publish_processing_update(agent_name: str, trace_id: str, progress_info: Dict[str, Any]):
    """Publish processing progress update."""
    await event_manager.publish_event(
        EventType.PROCESSING_UPDATE,
        progress_info,
        agent_name=agent_name,
        trace_id=trace_id
    )


async def publish_model_inference(model_name: str, trace_id: str, inference_info: Dict[str, Any]):
    """Publish model inference event."""
    await event_manager.publish_event(
        EventType.MODEL_INFERENCE,
        {
            "model_name": model_name,
            "inference_time_ms": inference_info.get("inference_time"),
            "device": inference_info.get("device"),
            "batch_size": inference_info.get("batch_size", 1)
        },
        trace_id=trace_id
    )


async def publish_cache_event(cache_type: str, cache_key: str, hit: bool):
    """Publish cache hit/miss event."""
    event_type = EventType.CACHE_HIT if hit else EventType.CACHE_MISS
    
    await event_manager.publish_event(
        event_type,
        {
            "cache_type": cache_type,
            "cache_key": cache_key
        }
    )


async def publish_detection_event(detection_type: str, trace_id: str, detection_info: Dict[str, Any]):
    """Publish detection event."""
    await event_manager.publish_event(
        EventType.AGENT_COMPLETED,  # Using existing event type
        {
            "detection_type": detection_type,
            **detection_info
        },
        trace_id=trace_id
    )


async def subscribe_to_events(event_types: Optional[List[EventType]] = None,
                            agent_filter: Optional[str] = None) -> AsyncGenerator[StreamEvent, None]:
    """
    Subscribe to events stream.
    
    Args:
        event_types: Optional event type filters
        agent_filter: Optional agent name filter
        
    Yields:
        Stream events
    """
    subscriber_id = str(uuid.uuid4())
    
    try:
        async for event in event_manager.subscribe(subscriber_id, event_types, agent_filter):
            yield event
    finally:
        # Cleanup is handled by the event manager
        pass


# Decorator for automatic event publishing
def stream_events(agent_name: str):
    """
    Decorator for automatic event streaming from agent methods.
    
    Args:
        agent_name: Name of the agent
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            trace_id = str(uuid.uuid4())
            
            # Publish start event
            await publish_agent_started(agent_name, trace_id, {
                "type": type(args[0]).__name__ if args else "unknown",
                "config": getattr(self, 'config', {})
            })
            
            try:
                # Execute function
                result = await func(self, *args, **kwargs)
                
                # Publish completion event
                await publish_agent_completed(agent_name, trace_id, {
                    "success": getattr(result, 'success', True),
                    "inference_time": getattr(result, 'inference_time', None),
                    "summary": getattr(result, 'data', {})
                })
                
                return result
            
            except Exception as e:
                # Publish error event
                await event_manager.publish_event(
                    EventType.AGENT_ERROR,
                    {"error": str(e)},
                    agent_name=agent_name,
                    trace_id=trace_id
                )
                raise
        
        return wrapper
    return decorator
