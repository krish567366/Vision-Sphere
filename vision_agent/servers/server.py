"""
FastAPI Server for VisionAgent Framework
Production-ready API server with async endpoints and WebSocket support.
"""

import asyncio
import io
import os
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import cv2
import numpy as np

from agents import FaceAgent, ObjectAgent, VideoAgent, ClassificationAgent
from config import get_config, VisionAgentConfig
from utils import setup_logging, validate_image_input, load_image_safe, get_device_info


# Pydantic models for API requests/responses
class ProcessingRequest(BaseModel):
    """Base request model for processing operations."""
    config_override: Optional[Dict[str, Any]] = Field(None, description="Override default agent configuration")


class ImageProcessingRequest(ProcessingRequest):
    """Request model for image processing."""
    image_url: Optional[str] = Field(None, description="URL to image file")
    return_annotated: bool = Field(False, description="Return annotated image with bounding boxes")


class VideoProcessingRequest(ProcessingRequest):
    """Request model for video processing."""
    video_url: Optional[str] = Field(None, description="URL to video file")
    frame_skip: int = Field(1, description="Number of frames to skip between analysis")
    max_frames: int = Field(100, description="Maximum number of frames to process")
    output_format: str = Field("summary", description="Output format: 'summary' or 'detailed'")


class ProcessingResponse(BaseModel):
    """Standard response model for all processing operations."""
    success: bool
    data: Dict[str, Any]
    inference_time_ms: Optional[float] = None
    agent_info: Optional[Dict[str, Any]] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    agents: Dict[str, bool]
    system_info: Dict[str, Any]


class WebSocketMessage(BaseModel):
    """WebSocket message model."""
    type: str  # 'frame_result', 'error', 'status'
    data: Dict[str, Any]
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# Global agent instances
agents: Dict[str, Any] = {}
app_config: VisionAgentConfig = None
logger = None


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    global app_config, logger, agents
    
    # Load configuration
    app_config = get_config()
    
    # Setup logging
    logger = setup_logging(app_config.logging.__dict__)
    
    # Create FastAPI app
    app = FastAPI(
        title="VisionAgent API",
        description="Professional Multi-Modal AI Agent Framework for image, video, and face analytics",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    if app_config.server.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=app_config.server.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Initialize agents
    initialize_agents()
    
    return app


def initialize_agents():
    """Initialize all AI agents."""
    global agents, app_config, logger
    
    logger.info("Initializing AI agents...")
    
    try:
        # Face Agent
        if app_config.face_agent.enabled:
            face_config = app_config.face_agent.model.custom_params or {}
            agents['face'] = FaceAgent(
                device=app_config.default_device,
                config=face_config
            )
            if agents['face'].initialize():
                logger.info("Face Agent initialized successfully")
            else:
                logger.error("Face Agent initialization failed")
        
        # Object Agent  
        if app_config.object_agent.enabled:
            object_config = app_config.object_agent.model.custom_params or {}
            object_config['confidence_threshold'] = app_config.object_agent.model.confidence_threshold
            agents['object'] = ObjectAgent(
                device=app_config.default_device,
                model_path=app_config.object_agent.model.path,
                config=object_config
            )
            if agents['object'].initialize():
                logger.info("Object Agent initialized successfully")
            else:
                logger.error("Object Agent initialization failed")
        
        # Video Agent
        if app_config.video_agent.enabled:
            video_config = app_config.video_agent.processing_params or {}
            video_config['face_config'] = app_config.face_agent.model.custom_params or {}
            video_config['object_config'] = app_config.object_agent.model.custom_params or {}
            agents['video'] = VideoAgent(
                device=app_config.default_device,
                config=video_config
            )
            if agents['video'].initialize():
                logger.info("Video Agent initialized successfully")
            else:
                logger.error("Video Agent initialization failed")
        
        # Classification Agent
        if app_config.classification_agent.enabled:
            classification_config = app_config.classification_agent.model.custom_params or {}
            agents['classification'] = ClassificationAgent(
                device=app_config.default_device,
                model_path=app_config.classification_agent.model.path,
                config=classification_config
            )
            if agents['classification'].initialize():
                logger.info("Classification Agent initialized successfully")
            else:
                logger.error("Classification Agent initialization failed")
        
        logger.info(f"Initialized {len(agents)} agents")
        
    except Exception as e:
        logger.error(f"Failed to initialize agents: {str(e)}")


# Create app instance
app = create_app()


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "VisionAgent API",
        "version": "1.0.0",
        "description": "Multi-Modal AI Agent Framework",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    agent_status = {}
    for agent_name, agent in agents.items():
        agent_status[agent_name] = agent._is_initialized if agent else False
    
    return HealthResponse(
        status="healthy" if all(agent_status.values()) else "degraded",
        timestamp=datetime.utcnow().isoformat(),
        agents=agent_status,
        system_info=get_device_info()
    )


@app.post("/face", response_model=ProcessingResponse)
async def process_face(
    request: ImageProcessingRequest = None,
    file: UploadFile = File(None)
):
    """
    Face detection and recognition endpoint.
    
    Accepts either form data with file upload or JSON with image URL.
    """
    if 'face' not in agents:
        raise HTTPException(status_code=503, detail="Face agent not available")
    
    try:
        # Get image data
        if file:
            image_data = await file.read()
        elif request and request.image_url:
            image_data = request.image_url
        else:
            raise HTTPException(status_code=400, detail="No image provided")
        
        # Process with face agent
        result = agents['face'].process(image_data)
        
        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)
        
        return ProcessingResponse(
            success=result.success,
            data=result.data,
            inference_time_ms=result.inference_time,
            agent_info=agents['face'].get_info()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Face processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/object", response_model=ProcessingResponse)
async def process_object(
    request: ImageProcessingRequest = None,
    file: UploadFile = File(None)
):
    """
    Object detection endpoint.
    
    Accepts either form data with file upload or JSON with image URL.
    """
    if 'object' not in agents:
        raise HTTPException(status_code=503, detail="Object agent not available")
    
    try:
        # Get image data
        if file:
            image_data = await file.read()
        elif request and request.image_url:
            image_data = request.image_url
        else:
            raise HTTPException(status_code=400, detail="No image provided")
        
        # Process with object agent
        result = agents['object'].process(image_data)
        
        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)
        
        return ProcessingResponse(
            success=result.success,
            data=result.data,
            inference_time_ms=result.inference_time,
            agent_info=agents['object'].get_info()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Object processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/video", response_model=ProcessingResponse)
async def process_video(
    request: VideoProcessingRequest = None,
    file: UploadFile = File(None)
):
    """
    Video analysis endpoint.
    
    Accepts either form data with file upload or JSON with video URL.
    """
    if 'video' not in agents:
        raise HTTPException(status_code=503, detail="Video agent not available")
    
    try:
        # Get video data
        if file:
            # Save uploaded file temporarily
            temp_path = f"./temp/{uuid.uuid4()}.mp4"
            os.makedirs('./temp', exist_ok=True)
            
            with open(temp_path, 'wb') as f:
                content = await file.read()
                f.write(content)
            
            video_source = temp_path
        elif request and request.video_url:
            video_source = request.video_url
        else:
            raise HTTPException(status_code=400, detail="No video provided")
        
        # Update agent configuration if provided
        if request and request.config_override:
            # Temporarily update agent config
            original_config = agents['video'].config.copy()
            agents['video'].config.update(request.config_override)
        
        # Process with video agent
        result = agents['video'].process(video_source)
        
        # Restore original config
        if request and request.config_override:
            agents['video'].config = original_config
        
        # Cleanup temp file
        if file and os.path.exists(video_source):
            os.remove(video_source)
        
        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)
        
        return ProcessingResponse(
            success=result.success,
            data=result.data,
            inference_time_ms=result.inference_time,
            agent_info=agents['video'].get_info()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Video processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify", response_model=ProcessingResponse)
async def classify_image(
    request: ImageProcessingRequest = None,
    file: UploadFile = File(None)
):
    """
    Image classification endpoint.
    
    Accepts either form data with file upload or JSON with image URL.
    """
    if 'classification' not in agents:
        raise HTTPException(status_code=503, detail="Classification agent not available")
    
    try:
        # Get image data
        if file:
            image_data = await file.read()
        elif request and request.image_url:
            image_data = request.image_url
        else:
            raise HTTPException(status_code=400, detail="No image provided")
        
        # Process with classification agent
        result = agents['classification'].process(image_data)
        
        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)
        
        return ProcessingResponse(
            success=result.success,
            data=result.data,
            inference_time_ms=result.inference_time,
            agent_info=agents['classification'].get_info()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agents", response_model=Dict[str, Any])
async def list_agents():
    """List all available agents and their status."""
    agent_info = {}
    
    for agent_name, agent in agents.items():
        if agent:
            agent_info[agent_name] = agent.get_info()
        else:
            agent_info[agent_name] = {"initialized": False, "error": "Agent not loaded"}
    
    return {
        "agents": agent_info,
        "total_agents": len(agents),
        "active_agents": sum(1 for agent in agents.values() if agent and agent._is_initialized)
    }


@app.websocket("/ws/video")
async def video_stream_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time video processing.
    
    Client sends video frames, server returns analysis results.
    """
    await websocket.accept()
    
    if 'video' not in agents:
        await websocket.send_json({
            "type": "error",
            "data": {"message": "Video agent not available"}
        })
        await websocket.close()
        return
    
    try:
        while True:
            # Receive frame data
            data = await websocket.receive_bytes()
            
            # Decode frame
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": "Invalid frame data"}
                })
                continue
            
            # Process frame
            result = agents['video']._process_single_frame(frame, 0)
            
            # Send result
            response = WebSocketMessage(
                type="frame_result",
                data=result
            )
            
            await websocket.send_json(response.dict())
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.send_json({
            "type": "error", 
            "data": {"message": str(e)}
        })


@app.post("/batch/classify", response_model=List[ProcessingResponse])
async def batch_classify(files: List[UploadFile] = File(...)):
    """
    Batch image classification endpoint.
    
    Accepts multiple image files for batch processing.
    """
    if 'classification' not in agents:
        raise HTTPException(status_code=503, detail="Classification agent not available")
    
    try:
        # Read all files
        image_data_list = []
        for file in files:
            data = await file.read()
            image_data_list.append(data)
        
        # Process batch
        results = agents['classification'].process_batch(image_data_list)
        
        # Convert to response format
        responses = []
        for i, result in enumerate(results):
            response = ProcessingResponse(
                success=result.success,
                data=result.data,
                inference_time_ms=result.inference_time,
                agent_info=agents['classification'].get_info()
            )
            responses.append(response)
        
        return responses
        
    except Exception as e:
        logger.error(f"Batch classification error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/face/add_known")
async def add_known_face(
    name: str,
    file: UploadFile = File(...)
):
    """
    Add a known face for recognition.
    
    Args:
        name: Name to associate with the face
        file: Image file containing the face
    """
    if 'face' not in agents:
        raise HTTPException(status_code=503, detail="Face agent not available")
    
    try:
        image_data = await file.read()
        success = agents['face'].add_known_face(image_data, name)
        
        if success:
            return {"success": True, "message": f"Added known face: {name}"}
        else:
            raise HTTPException(status_code=400, detail="Failed to add known face")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Add known face error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/face/known")
async def list_known_faces():
    """List all known faces."""
    if 'face' not in agents:
        raise HTTPException(status_code=503, detail="Face agent not available")
    
    try:
        known_faces = agents['face'].get_known_faces()
        return {
            "known_faces": known_faces,
            "count": len(known_faces)
        }
        
    except Exception as e:
        logger.error(f"List known faces error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/face/known/{name}")
async def remove_known_face(name: str):
    """Remove a known face."""
    if 'face' not in agents:
        raise HTTPException(status_code=503, detail="Face agent not available")
    
    try:
        success = agents['face'].remove_known_face(name)
        
        if success:
            return {"success": True, "message": f"Removed known face: {name}"}
        else:
            raise HTTPException(status_code=404, detail=f"Known face not found: {name}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Remove known face error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/object/classes")
async def list_object_classes():
    """List all available object classes."""
    if 'object' not in agents:
        raise HTTPException(status_code=503, detail="Object agent not available")
    
    try:
        classes = agents['object'].get_available_classes()
        return {
            "classes": classes,
            "count": len(classes)
        }
        
    except Exception as e:
        logger.error(f"List object classes error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/classification/classes")
async def list_classification_classes():
    """List all available classification classes."""
    if 'classification' not in agents:
        raise HTTPException(status_code=503, detail="Classification agent not available")
    
    try:
        classes = agents['classification'].get_class_names()
        return {
            "classes": classes,
            "count": len(classes)
        }
        
    except Exception as e:
        logger.error(f"List classification classes error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config")
async def get_current_config():
    """Get current server configuration."""
    return {
        "server": app_config.server.__dict__,
        "agents": {
            "face": app_config.face_agent.__dict__ if app_config.face_agent else None,
            "object": app_config.object_agent.__dict__ if app_config.object_agent else None,
            "video": app_config.video_agent.__dict__ if app_config.video_agent else None,
            "classification": app_config.classification_agent.__dict__ if app_config.classification_agent else None
        },
        "logging": app_config.logging.__dict__
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


if __name__ == "__main__":
    """Run the server directly."""
    config = get_config()
    
    uvicorn.run(
        "server:app",
        host=config.server.host,
        port=config.server.port,
        workers=config.server.workers,
        reload=False,
        access_log=True
    )
