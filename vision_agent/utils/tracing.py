"""
Advanced Tracing and Observability System
OpenTelemetry-based tracing with custom span processing for agent workflows.
"""

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from datetime import datetime
from functools import wraps
from typing import Any, Dict, List, Optional, AsyncGenerator, Callable
import logging
from enum import Enum


class SpanType(str, Enum):
    """Types of spans for categorization."""
    AGENT = "agent"
    TOOL = "tool"
    MODEL = "model"
    PROCESSING = "processing"
    NETWORK = "network"
    DATABASE = "database"


@dataclass
class SpanEvent:
    """Individual event within a span."""
    timestamp: float
    name: str
    attributes: Dict[str, Any]


@dataclass 
class Span:
    """Custom span implementation for detailed tracing."""
    span_id: str
    parent_span_id: Optional[str]
    trace_id: str
    name: str
    span_type: SpanType
    start_time: float
    end_time: Optional[float] = None
    status: str = "RUNNING"  # RUNNING, SUCCESS, ERROR
    attributes: Dict[str, Any] = None
    events: List[SpanEvent] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}
        if self.events is None:
            self.events = []
    
    @property
    def duration_ms(self) -> Optional[float]:
        """Get span duration in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add an event to the span."""
        event = SpanEvent(
            timestamp=time.time(),
            name=name,
            attributes=attributes or {}
        )
        self.events.append(event)
    
    def set_attribute(self, key: str, value: Any):
        """Set span attribute."""
        self.attributes[key] = value
    
    def set_status(self, status: str, error: Optional[str] = None):
        """Set span status."""
        self.status = status
        if error:
            self.error = error
            self.set_attribute("error.message", error)
    
    def finish(self):
        """Mark span as finished."""
        self.end_time = time.time()
        if self.status == "RUNNING":
            self.status = "SUCCESS"


class TracingProcessor:
    """Base class for trace processing."""
    
    async def on_span_start(self, span: Span) -> None:
        """Called when a span starts."""
        pass
    
    async def on_span_end(self, span: Span) -> None:
        """Called when a span ends."""
        pass
    
    async def on_span_event(self, span: Span, event: SpanEvent) -> None:
        """Called when an event is added to a span."""
        pass


class ConsoleTracingProcessor(TracingProcessor):
    """Console-based tracing processor for development."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.logger = logging.getLogger('TracingProcessor')
    
    async def on_span_start(self, span: Span) -> None:
        """Log span start."""
        if self.verbose:
            self.logger.info(f"🟢 [{span.span_type.value}] {span.name} started")
    
    async def on_span_end(self, span: Span) -> None:
        """Log span completion."""
        status_emoji = "✅" if span.status == "SUCCESS" else "❌" if span.status == "ERROR" else "⏸️"
        duration = f"{span.duration_ms:.2f}ms" if span.duration_ms else "N/A"
        
        self.logger.info(f"{status_emoji} [{span.span_type.value}] {span.name} ({duration})")
        
        if span.error:
            self.logger.error(f"  Error: {span.error}")


class FileTracingProcessor(TracingProcessor):
    """File-based tracing processor for persistent logging."""
    
    def __init__(self, log_file: str = "./logs/traces.jsonl"):
        self.log_file = log_file
        self.logger = logging.getLogger('FileTracingProcessor')
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    async def on_span_end(self, span: Span) -> None:
        """Write span to JSON lines file."""
        try:
            span_data = asdict(span)
            span_data['timestamp'] = datetime.utcnow().isoformat()
            
            # Append to log file
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(span_data) + '\n')
        
        except Exception as e:
            self.logger.error(f"Failed to write trace: {str(e)}")


class StreamingTracingProcessor(TracingProcessor):
    """Streaming tracing processor for real-time monitoring."""
    
    def __init__(self):
        self.subscribers: List[Callable[[Span], None]] = []
        self.event_queue: asyncio.Queue = asyncio.Queue()
    
    def subscribe(self, callback: Callable[[Span], None]):
        """Subscribe to span events."""
        self.subscribers.append(callback)
    
    async def on_span_end(self, span: Span) -> None:
        """Notify all subscribers."""
        for callback in self.subscribers:
            try:
                await callback(span)
            except Exception as e:
                logging.getLogger('StreamingTracingProcessor').error(
                    f"Subscriber error: {str(e)}"
                )


class TraceManager:
    """Central trace management system."""
    
    def __init__(self):
        self.processors: List[TracingProcessor] = []
        self.active_spans: Dict[str, Span] = {}
        self.traces: Dict[str, List[Span]] = {}
        self._lock = asyncio.Lock()
    
    def add_processor(self, processor: TracingProcessor):
        """Add a tracing processor."""
        self.processors.append(processor)
    
    async def start_span(self,
                        name: str,
                        span_type: SpanType,
                        parent_span_id: Optional[str] = None,
                        trace_id: Optional[str] = None,
                        attributes: Optional[Dict[str, Any]] = None) -> Span:
        """
        Start a new span.
        
        Args:
            name: Span name
            span_type: Type of span
            parent_span_id: Parent span ID
            trace_id: Trace ID (generated if None)
            attributes: Initial attributes
            
        Returns:
            Started span
        """
        if trace_id is None:
            trace_id = str(uuid.uuid4())
        
        span = Span(
            span_id=str(uuid.uuid4()),
            parent_span_id=parent_span_id,
            trace_id=trace_id,
            name=name,
            span_type=span_type,
            start_time=time.time(),
            attributes=attributes or {}
        )
        
        async with self._lock:
            self.active_spans[span.span_id] = span
            
            if trace_id not in self.traces:
                self.traces[trace_id] = []
            self.traces[trace_id].append(span)
        
        # Notify processors
        for processor in self.processors:
            await processor.on_span_start(span)
        
        return span
    
    async def end_span(self, span_id: str, status: str = "SUCCESS", error: Optional[str] = None):
        """
        End an active span.
        
        Args:
            span_id: Span ID to end
            status: Final status
            error: Error message if any
        """
        async with self._lock:
            span = self.active_spans.get(span_id)
            if not span:
                return
            
            span.finish()
            span.set_status(status, error)
            
            del self.active_spans[span_id]
        
        # Notify processors
        for processor in self.processors:
            await processor.on_span_end(span)
    
    async def add_span_event(self, span_id: str, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add an event to an active span."""
        span = self.active_spans.get(span_id)
        if span:
            span.add_event(name, attributes)
            
            # Notify processors
            for processor in self.processors:
                await processor.on_span_event(span, span.events[-1])
    
    def get_trace(self, trace_id: str) -> List[Span]:
        """Get all spans for a trace ID."""
        return self.traces.get(trace_id, []).copy()
    
    def get_active_spans(self) -> List[Span]:
        """Get all currently active spans."""
        return list(self.active_spans.values())


# Global trace manager instance
trace_manager = TraceManager()


@asynccontextmanager
async def traced_operation(name: str, 
                          span_type: SpanType = SpanType.PROCESSING,
                          attributes: Optional[Dict[str, Any]] = None,
                          trace_id: Optional[str] = None) -> AsyncGenerator[Span, None]:
    """
    Context manager for tracing operations.
    
    Args:
        name: Operation name
        span_type: Type of span
        attributes: Initial attributes
        trace_id: Trace ID
        
    Yields:
        Active span
    """
    span = await trace_manager.start_span(name, span_type, attributes=attributes, trace_id=trace_id)
    
    try:
        yield span
        await trace_manager.end_span(span.span_id, "SUCCESS")
    except Exception as e:
        await trace_manager.end_span(span.span_id, "ERROR", str(e))
        raise


def traced_method(span_type: SpanType = SpanType.PROCESSING):
    """
    Decorator for automatically tracing method calls.
    
    Args:
        span_type: Type of span to create
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            method_name = f"{self.__class__.__name__}.{func.__name__}"
            
            async with traced_operation(method_name, span_type) as span:
                # Add method attributes
                span.set_attribute("method.class", self.__class__.__name__)
                span.set_attribute("method.name", func.__name__)
                span.set_attribute("method.args_count", len(args))
                span.set_attribute("method.kwargs_count", len(kwargs))
                
                # Execute function
                result = await func(self, *args, **kwargs)
                
                # Add result attributes
                if hasattr(result, 'success'):
                    span.set_attribute("result.success", result.success)
                if hasattr(result, 'inference_time'):
                    span.set_attribute("result.inference_time_ms", result.inference_time)
                
                return result
        
        return wrapper
    return decorator


# Setup default processors
def setup_tracing(verbose: bool = False, 
                 log_file: Optional[str] = None,
                 enable_streaming: bool = False):
    """
    Setup tracing with default processors.
    
    Args:
        verbose: Enable verbose console logging
        log_file: File path for trace logging
        enable_streaming: Enable streaming processor
    """
    # Console processor
    console_processor = ConsoleTracingProcessor(verbose=verbose)
    trace_manager.add_processor(console_processor)
    
    # File processor
    if log_file:
        file_processor = FileTracingProcessor(log_file)
        trace_manager.add_processor(file_processor)
    
    # Streaming processor
    if enable_streaming:
        streaming_processor = StreamingTracingProcessor()
        trace_manager.add_processor(streaming_processor)
    
    logging.getLogger('TraceManager').info("Tracing system initialized")


# Utility functions for common tracing patterns
async def trace_agent_processing(agent_name: str, input_data: Any, trace_id: Optional[str] = None) -> str:
    """Start tracing for agent processing."""
    span = await trace_manager.start_span(
        f"{agent_name}.process",
        SpanType.AGENT,
        trace_id=trace_id,
        attributes={
            "agent.name": agent_name,
            "input.type": type(input_data).__name__,
            "input.size": len(str(input_data)) if hasattr(input_data, '__len__') else 0
        }
    )
    return span.span_id


async def trace_model_inference(model_name: str, trace_id: str) -> str:
    """Start tracing for model inference."""
    span = await trace_manager.start_span(
        f"model.inference.{model_name}",
        SpanType.MODEL,
        trace_id=trace_id,
        attributes={
            "model.name": model_name,
            "model.type": "vision"
        }
    )
    return span.span_id


async def trace_tool_execution(tool_name: str, trace_id: str) -> str:
    """Start tracing for tool execution."""
    span = await trace_manager.start_span(
        f"tool.{tool_name}",
        SpanType.TOOL,
        trace_id=trace_id,
        attributes={
            "tool.name": tool_name
        }
    )
    return span.span_id
