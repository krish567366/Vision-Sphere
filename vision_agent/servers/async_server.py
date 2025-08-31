"""
Enhanced Async FastAPI Server with Performance Optimizations
Production-ready API server with advanced caching, tracing, and streaming.
"""

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Union, Any
import logging
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# Import agents
from agents import (
    AsyncFaceAgent, AsyncObjectAgent, AsyncVideoAgent, AsyncClassificationAgent,
    agent_registry, AsyncProcessingResult
)

# Import utilities
from ..utils.caching import cache_manager
from ..utils.tracing import trace_manager, traced_operation, SpanType
from utils.streaming import event_manager, EventType, subscribe_to_events
from utils.config_advanced import get_hierarchical_config


# Pydantic models for API
class ProcessingRequest(BaseModel):
    """Base processing request model."""
    use_cache: bool = Field(default=True, description="Enable caching for this request")
    trace_id: Optional[str] = Field(default=None, description="Optional trace ID for request correlation")
    config_overrides: Optional[Dict[str, Any]] = Field(default=None, description="Runtime configuration overrides")


class FaceDetectionRequest(ProcessingRequest):
    """Face detection request model."""
    enable_recognition: bool = Field(default=True, description="Enable face recognition")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Detection confidence threshold")


class ObjectDetectionRequest(ProcessingRequest):
    """Object detection request model."""
    allowed_classes: Optional[List[str]] = Field(default=None, description="Filter detections to these classes")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Detection confidence threshold")
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0, description="IoU threshold for NMS")
    enable_tracking: bool = Field(default=True, description="Enable object tracking")


class ClassificationRequest(ProcessingRequest):
    """Image classification request model."""
    top_k: int = Field(default=5, ge=1, le=20, description="Number of top predictions to return")
    extract_features: bool = Field(default=False, description="Extract feature embeddings")


class VideoAnalysisRequest(ProcessingRequest):
    """Video analysis request model."""
    frame_skip: int = Field(default=1, ge=0, description="Number of frames to skip between analyses")
    max_frames: Optional[int] = Field(default=None, description="Maximum frames to process")
    enable_face_analytics: bool = Field(default=True, description="Enable face analytics")
    enable_object_analytics: bool = Field(default=True, description="Enable object analytics")
    enable_scene_detection: bool = Field(default=True, description="Enable scene change detection")


class WebSocketManager:
    """WebSocket connection manager for real-time streaming."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_subscriptions: Dict[str, List[str]] = {}
        self.logger = logging.getLogger('WebSocketManager')
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.connection_subscriptions[client_id] = []
        self.logger.info(f"WebSocket client {client_id} connected")
    
    def disconnect(self, client_id: str):
        """Disconnect WebSocket client."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.connection_subscriptions:
            del self.connection_subscriptions[client_id]
        self.logger.info(f"WebSocket client {client_id} disconnected")
    
    async def send_to_client(self, client_id: str, message: Dict[str, Any]):
        """Send message to specific client."""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json(message)
            except Exception as e:
                self.logger.error(f"Error sending to client {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast(self, message: Dict[str, Any], event_types: Optional[List[str]] = None):
        """Broadcast message to all connected clients."""
        if not self.active_connections:
            return
        
        # Filter clients by subscriptions
        target_clients = []
        
        if event_types:
            for client_id, subscriptions in self.connection_subscriptions.items():
                if any(et in subscriptions for et in event_types) or not subscriptions:
                    target_clients.append(client_id)
        else:
            target_clients = list(self.active_connections.keys())
        
        # Send to target clients
        tasks = [
            self.send_to_client(client_id, message)
            for client_id in target_clients
        ]
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


# Global instances
websocket_manager = WebSocketManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger = logging.getLogger('FastAPI')
    logger.info("Starting VisionAgent API server...")
    
    # Load configuration
    config = get_hierarchical_config()
    
    # Initialize agents
    face_agent = AsyncFaceAgent(config=config.get('face_agent', {}))
    object_agent = AsyncObjectAgent(config=config.get('object_agent', {}))
    video_agent = AsyncVideoAgent(config=config.get('video_agent', {}))
    classification_agent = AsyncClassificationAgent(config=config.get('classification_agent', {}))
    
    # Register agents
    await agent_registry.register_agent(face_agent)
    await agent_registry.register_agent(object_agent)
    await agent_registry.register_agent(video_agent)
    await agent_registry.register_agent(classification_agent)
    
    # Initialize tracing
    await trace_manager.initialize()
    
    # Setup event streaming
    await event_manager.initialize()
    
    logger.info("VisionAgent API server started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down VisionAgent API server...")
    
    # Cleanup agents
    await agent_registry.shutdown_all()
    
    # Cleanup tracing
    await trace_manager.shutdown()
    
    # Cleanup event manager
    await event_manager.shutdown()
    
    # Cleanup cache
    await cache_manager.cleanup()
    
    logger.info("VisionAgent API server shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="VisionAgent API",
    description="Production-ready multi-modal AI agent framework for image, video, and face analytics",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def save_uploaded_file(upload_file: UploadFile) -> str:
    """Save uploaded file and return path."""
    # Create temp directory if it doesn't exist
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    
    # Generate unique filename
    file_path = temp_dir / f"{uuid.uuid4()}_{upload_file.filename}"
    
    # Save file
    with open(file_path, "wb") as f:
        content = await upload_file.read()
        f.write(content)
    
    return str(file_path)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "VisionAgent API - Multi-modal AI Agent Framework",
        "version": "1.0.0",
        "endpoints": {
            "face_detection": "/face",
            "object_detection": "/object", 
            "video_analysis": "/video",
            "image_classification": "/classify",
            "health_check": "/health",
            "metrics": "/metrics",
            "streaming": "ws://localhost:8000/ws/{client_id}"
        },
        "documentation": "/docs"
    }


@app.get("/health")
async def health_check():
    """Comprehensive health check for all agents."""
    async with traced_operation("api.health_check", SpanType.API) as span:
        try:
            # Check all agents
            health_results = await agent_registry.health_check_all()
            
            # Check system components
            system_health = {
                'cache_manager': cache_manager.is_healthy(),
                'trace_manager': trace_manager.is_initialized,
                'event_manager': event_manager.is_initialized,
                'websocket_connections': len(websocket_manager.active_connections)
            }
            
            # Determine overall health
            agent_healthy = all(result.get('healthy', False) for result in health_results.values())
            system_healthy = all(system_health.values())
            overall_healthy = agent_healthy and system_healthy
            
            span.set_attribute("health.overall", overall_healthy)
            span.set_attribute("health.agents", agent_healthy)
            span.set_attribute("health.system", system_healthy)
            
            return {
                "healthy": overall_healthy,
                "timestamp": time.time(),
                "agents": health_results,
                "system": system_health
            }
        
        except Exception as e:
            span.set_attribute("health.error", str(e))
            raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.get("/metrics")
async def get_metrics():
    """Get comprehensive system metrics."""
    async with traced_operation("api.metrics", SpanType.API) as span:
        try:
            # Get agent metrics
            agent_metrics = agent_registry.get_metrics_summary()
            
            # Get system metrics
            system_metrics = {
                'cache_stats': await cache_manager.get_stats(),
                'trace_stats': trace_manager.get_stats(),
                'event_stats': event_manager.get_stats(),
                'websocket_stats': {
                    'active_connections': len(websocket_manager.active_connections),
                    'total_subscriptions': sum(len(subs) for subs in websocket_manager.connection_subscriptions.values())
                }
            }
            
            span.set_attribute("metrics.collected", True)
            
            return {
                "timestamp": time.time(),
                "agents": agent_metrics,
                "system": system_metrics
            }
        
        except Exception as e:
            span.set_attribute("metrics.error", str(e))
            raise HTTPException(status_code=500, detail=f"Metrics collection failed: {str(e)}")


@app.post("/face")
async def detect_faces(
    file: UploadFile = File(...),
    request: FaceDetectionRequest = FaceDetectionRequest(),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Enhanced face detection endpoint with performance optimizations."""
    trace_id = request.trace_id or str(uuid.uuid4())
    
    async with traced_operation("api.face_detection", SpanType.API, trace_id=trace_id) as span:
        try:
            # Save uploaded file
            file_path = await save_uploaded_file(file)
            background_tasks.add_task(lambda: Path(file_path).unlink())
            
            span.set_attribute("file.name", file.filename)
            span.set_attribute("file.size", file.size)
            
            # Get face agent
            face_agent = await agent_registry.get_agent("AsyncFaceAgent")
            if not face_agent:
                raise HTTPException(status_code=500, detail="Face agent not available")
            
            # Apply config overrides
            if request.config_overrides:
                original_config = face_agent.config.copy()
                face_agent.config.update(request.config_overrides)
            
            # Process image
            result = await face_agent.process(
                file_path, 
                use_cache=request.use_cache,
                trace_id=trace_id
            )
            
            # Restore original config
            if request.config_overrides:
                face_agent.config = original_config
            
            span.set_attribute("result.success", result.success)
            span.set_attribute("result.face_count", result.data.get('face_count', 0) if result.success else 0)
            
            if result.success:
                return {
                    "success": True,
                    "data": result.data,
                    "metadata": {
                        "trace_id": trace_id,
                        "inference_time_ms": result.inference_time_ms,
                        "cache_hit": result.cache_hit,
                        "file_processed": file.filename
                    }
                }
            else:
                raise HTTPException(status_code=500, detail=result.error)
        
        except Exception as e:
            span.set_attribute("error", str(e))
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/object") 
async def detect_objects(
    file: UploadFile = File(...),
    request: ObjectDetectionRequest = ObjectDetectionRequest(),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Enhanced object detection endpoint with tracking and filtering."""
    trace_id = request.trace_id or str(uuid.uuid4())
    
    async with traced_operation("api.object_detection", SpanType.API, trace_id=trace_id) as span:
        try:
            # Save uploaded file
            file_path = await save_uploaded_file(file)
            background_tasks.add_task(lambda: Path(file_path).unlink())
            
            span.set_attribute("file.name", file.filename)
            span.set_attribute("file.size", file.size)
            
            # Get object agent
            object_agent = await agent_registry.get_agent("AsyncObjectAgent")
            if not object_agent:
                raise HTTPException(status_code=500, detail="Object agent not available")
            
            # Apply config overrides
            original_config = {}
            if request.config_overrides:
                original_config = object_agent.config.copy()
                object_agent.config.update(request.config_overrides)
            
            # Apply request parameters
            if request.allowed_classes:
                original_config['allowed_classes'] = object_agent.config.get('allowed_classes')
                object_agent.config['allowed_classes'] = request.allowed_classes
            
            # Process image
            result = await object_agent.process(
                file_path,
                use_cache=request.use_cache,
                trace_id=trace_id
            )
            
            # Restore original config
            if original_config:
                object_agent.config.update(original_config)
            
            span.set_attribute("result.success", result.success)
            span.set_attribute("result.detection_count", result.data.get('detection_count', 0) if result.success else 0)
            
            if result.success:
                return {
                    "success": True,
                    "data": result.data,
                    "metadata": {
                        "trace_id": trace_id,
                        "inference_time_ms": result.inference_time_ms,
                        "cache_hit": result.cache_hit,
                        "file_processed": file.filename
                    }
                }
            else:
                raise HTTPException(status_code=500, detail=result.error)
        
        except Exception as e:
            span.set_attribute("error", str(e))
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify")
async def classify_image(
    file: UploadFile = File(...),
    request: ClassificationRequest = ClassificationRequest(),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Enhanced image classification endpoint with feature extraction."""
    trace_id = request.trace_id or str(uuid.uuid4())
    
    async with traced_operation("api.image_classification", SpanType.API, trace_id=trace_id) as span:
        try:
            # Save uploaded file
            file_path = await save_uploaded_file(file)
            background_tasks.add_task(lambda: Path(file_path).unlink())
            
            span.set_attribute("file.name", file.filename)
            span.set_attribute("file.size", file.size)
            
            # Get classification agent
            classification_agent = await agent_registry.get_agent("AsyncClassificationAgent")
            if not classification_agent:
                raise HTTPException(status_code=500, detail="Classification agent not available")
            
            # Apply config overrides
            original_config = {}
            if request.config_overrides:
                original_config = classification_agent.config.copy()
                classification_agent.config.update(request.config_overrides)
            
            # Apply request parameters
            original_config['top_k'] = classification_agent.config.get('top_k')
            original_config['enable_features'] = classification_agent.config.get('enable_features')
            
            classification_agent.config['top_k'] = request.top_k
            classification_agent.config['enable_features'] = request.extract_features
            
            # Process image
            result = await classification_agent.process(
                file_path,
                use_cache=request.use_cache,
                trace_id=trace_id
            )
            
            # Restore original config
            classification_agent.config.update(original_config)
            
            span.set_attribute("result.success", result.success)
            span.set_attribute("result.prediction_count", result.data.get('prediction_count', 0) if result.success else 0)
            
            if result.success:
                return {
                    "success": True,
                    "data": result.data,
                    "metadata": {
                        "trace_id": trace_id,
                        "inference_time_ms": result.inference_time_ms,
                        "cache_hit": result.cache_hit,
                        "file_processed": file.filename
                    }
                }
            else:
                raise HTTPException(status_code=500, detail=result.error)
        
        except Exception as e:
            span.set_attribute("error", str(e))
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/video")
async def analyze_video(
    file: UploadFile = File(...),
    request: VideoAnalysisRequest = VideoAnalysisRequest(),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Enhanced video analysis endpoint with comprehensive analytics."""
    trace_id = request.trace_id or str(uuid.uuid4())
    
    async with traced_operation("api.video_analysis", SpanType.API, trace_id=trace_id) as span:
        try:
            # Save uploaded file
            file_path = await save_uploaded_file(file)
            background_tasks.add_task(lambda: Path(file_path).unlink())
            
            span.set_attribute("file.name", file.filename)
            span.set_attribute("file.size", file.size)
            
            # Get video agent
            video_agent = await agent_registry.get_agent("AsyncVideoAgent")
            if not video_agent:
                raise HTTPException(status_code=500, detail="Video agent not available")
            
            # Apply config overrides
            original_config = {}
            if request.config_overrides:
                original_config = video_agent.config.copy()
                video_agent.config.update(request.config_overrides)
            
            # Apply request parameters
            video_agent.config.update({
                'frame_skip': request.frame_skip,
                'max_frames': request.max_frames,
                'enable_face_analytics': request.enable_face_analytics,
                'enable_object_analytics': request.enable_object_analytics,
                'enable_scene_detection': request.enable_scene_detection
            })
            
            # Process video
            result = await video_agent.process(
                file_path,
                use_cache=request.use_cache,
                trace_id=trace_id
            )
            
            # Restore original config
            if original_config:
                video_agent.config.update(original_config)
            
            span.set_attribute("result.success", result.success)
            
            if result.success:
                return {
                    "success": True,
                    "data": result.data,
                    "metadata": {
                        "trace_id": trace_id,
                        "inference_time_ms": result.inference_time_ms,
                        "cache_hit": result.cache_hit,
                        "file_processed": file.filename
                    }
                }
            else:
                raise HTTPException(status_code=500, detail=result.error)
        
        except Exception as e:
            span.set_attribute("error", str(e))
            raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time streaming."""
    await websocket_manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive client messages
            data = await websocket.receive_json()
            message_type = data.get('type')
            
            if message_type == 'subscribe':
                # Subscribe to event types
                event_types = data.get('event_types', [])
                websocket_manager.connection_subscriptions[client_id] = event_types
                
                await websocket.send_json({
                    "type": "subscription_confirmed",
                    "event_types": event_types
                })
            
            elif message_type == 'process_image':
                # Real-time image processing
                image_data = data.get('image_data')
                agent_type = data.get('agent_type', 'classification')
                
                if image_data:
                    # Process based on agent type
                    agent = await agent_registry.get_agent(f"Async{agent_type.title()}Agent")
                    if agent:
                        # Stream processing results
                        async for update in agent.stream_process(image_data):
                            await websocket.send_json({
                                "type": "processing_update",
                                "data": update
                            })
            
            elif message_type == 'ping':
                await websocket.send_json({"type": "pong"})
    
    except WebSocketDisconnect:
        websocket_manager.disconnect(client_id)
    except Exception as e:
        logging.error(f"WebSocket error for client {client_id}: {e}")
        websocket_manager.disconnect(client_id)


@app.get("/stream/video/{agent_type}")
async def stream_video_analysis(agent_type: str, video_url: str):
    """Stream video analysis results."""
    # Validate agent type
    valid_agents = ['face', 'object', 'classification']
    if agent_type not in valid_agents:
        raise HTTPException(status_code=400, detail=f"Invalid agent type. Choose from: {valid_agents}")
    
    # Get agent
    agent = await agent_registry.get_agent(f"Async{agent_type.title()}Agent")
    if not agent:
        raise HTTPException(status_code=500, detail=f"{agent_type} agent not available")
    
    async def generate_stream():
        """Generate streaming response."""
        try:
            if hasattr(agent, 'stream_classification'):
                # Classification agent
                async for update in agent.stream_classification(video_url):
                    yield f"data: {update}\n\n"
            
            elif hasattr(agent, 'detect_and_recognize_streaming'):
                # Face agent  
                result = await agent.detect_and_recognize_streaming(video_url)
                yield f"data: {result.to_dict()}\n\n"
            
            elif hasattr(agent, 'track_objects_in_video'):
                # Object agent
                result = await agent.track_objects_in_video(video_url)
                yield f"data: {result.to_dict()}\n\n"
            
            else:
                yield f"data: {{'error': 'Streaming not supported for {agent_type}'}}\n\n"
        
        except Exception as e:
            yield f"data: {{'error': '{str(e)}'}}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.post("/batch/{agent_type}")
async def process_batch(
    agent_type: str,
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Batch processing endpoint for multiple files."""
    trace_id = str(uuid.uuid4())
    
    async with traced_operation("api.batch_processing", SpanType.API, trace_id=trace_id) as span:
        try:
            # Validate agent type
            valid_agents = ['face', 'object', 'classification']
            if agent_type not in valid_agents:
                raise HTTPException(status_code=400, detail=f"Invalid agent type. Choose from: {valid_agents}")
            
            # Get agent
            agent = await agent_registry.get_agent(f"Async{agent_type.title()}Agent")
            if not agent:
                raise HTTPException(status_code=500, detail=f"{agent_type} agent not available")
            
            # Save all files
            file_paths = []
            for file in files:
                file_path = await save_uploaded_file(file)
                file_paths.append(file_path)
                background_tasks.add_task(lambda p=file_path: Path(p).unlink())
            
            span.set_attribute("batch.file_count", len(files))
            
            # Process batch
            if hasattr(agent, 'process_batch'):
                results = await agent.process_batch(file_paths)
            else:
                # Fallback to individual processing
                tasks = [agent.process(path, trace_id=f"{trace_id}_file_{i}") for i, path in enumerate(file_paths)]
                results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Format results
            batch_response = {
                "success": True,
                "batch_size": len(files),
                "results": [],
                "metadata": {
                    "trace_id": trace_id,
                    "agent_type": agent_type
                }
            }
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    batch_response["results"].append({
                        "file_index": i,
                        "filename": files[i].filename,
                        "success": False,
                        "error": str(result)
                    })
                else:
                    batch_response["results"].append({
                        "file_index": i,
                        "filename": files[i].filename,
                        "success": result.success,
                        "data": result.data if result.success else None,
                        "error": result.error,
                        "inference_time_ms": result.inference_time_ms,
                        "cache_hit": result.cache_hit
                    })
            
            span.set_attribute("batch.processed", len(results))
            successful_results = sum(1 for r in results if hasattr(r, 'success') and r.success)
            span.set_attribute("batch.successful", successful_results)
            
            return batch_response
        
        except Exception as e:
            span.set_attribute("error", str(e))
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/agents/stats")
async def get_agent_statistics():
    """Get detailed statistics for all agents."""
    async with traced_operation("api.agent_statistics", SpanType.API) as span:
        try:
            stats = {}
            
            for agent_name in ['AsyncFaceAgent', 'AsyncObjectAgent', 'AsyncVideoAgent', 'AsyncClassificationAgent']:
                agent = await agent_registry.get_agent(agent_name)
                if agent:
                    if hasattr(agent, 'get_face_statistics'):
                        stats[agent_name] = await agent.get_face_statistics()
                    elif hasattr(agent, 'get_class_statistics'):
                        stats[agent_name] = await agent.get_class_statistics()
                    elif hasattr(agent, 'get_video_statistics'):
                        stats[agent_name] = await agent.get_video_statistics()
                    elif hasattr(agent, 'get_classification_statistics'):
                        stats[agent_name] = await agent.get_classification_statistics()
                    else:
                        stats[agent_name] = await agent.get_metrics()
            
            span.set_attribute("stats.agents_collected", len(stats))
            
            return {
                "timestamp": time.time(),
                "agent_statistics": stats
            }
        
        except Exception as e:
            span.set_attribute("error", str(e))
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/cache/clear")
async def clear_cache():
    """Clear all caches."""
    async with traced_operation("api.cache_clear", SpanType.API) as span:
        try:
            await cache_manager.clear_all_caches()
            span.set_attribute("cache.cleared", True)
            
            return {
                "success": True,
                "message": "All caches cleared",
                "timestamp": time.time()
            }
        
        except Exception as e:
            span.set_attribute("error", str(e))
            raise HTTPException(status_code=500, detail=str(e))


# Event streaming setup
async def setup_event_streaming():
    """Setup event streaming to WebSocket clients."""
    async def stream_events():
        """Stream events to WebSocket clients."""
        async for event in subscribe_to_events():
            await websocket_manager.broadcast(
                {
                    "type": "event",
                    "event_type": event.event_type.value,
                    "data": event.data,
                    "timestamp": event.timestamp,
                    "trace_id": event.trace_id
                },
                event_types=[event.event_type.value]
            )
    
    # Start event streaming task
    asyncio.create_task(stream_events())


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler with tracing."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": time.time(),
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler with tracing."""
    logging.error(f"Unhandled exception: {str(exc)}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": time.time(),
            "status_code": 500
        }
    )


if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "async_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
