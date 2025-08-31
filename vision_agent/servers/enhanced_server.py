"""
Enhanced Production Server with Enterprise-Grade Performance Patterns
Integration of all advanced systems for world-class autonomous agent platform.
"""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import numpy as np
import cv2

# Import all enhanced systems
from ..utils.enhanced_base_agent import (
    initialize_enhanced_systems, shutdown_enhanced_systems
)
from ..agents.enhanced_face_agent import create_enhanced_face_agent
from ..agents.async_object_agent import AsyncObjectAgent
from ..agents.async_video_agent import AsyncVideoAgent  
from ..agents.async_classification_agent import AsyncClassificationAgent

# Import performance systems
from ..utils.resource_manager import resource_manager
from ..utils.semantic_cache import semantic_cache_manager
from ..utils.speculative_execution import speculative_runner, cost_optimizer
from ..utils.performance_analytics import performance_analytics
from ..utils.reliability import reliability_manager
from ..utils.streaming import event_manager
from ..utils.tracing import trace_manager


# Enhanced request/response models
class EnhancedProcessingRequest(BaseModel):
    """Enhanced processing request with performance options."""
    # Performance options
    enable_speculation: bool = True
    enable_caching: bool = True
    priority: int = Field(default=5, ge=1, le=10)
    max_wait_time: float = Field(default=30.0, gt=0)
    
    # Quality options
    quality_threshold: float = Field(default=0.5, ge=0, le=1)
    include_quality_metrics: bool = True
    include_performance_data: bool = False
    
    # Tracing
    trace_id: Optional[str] = None


class EnhancedFaceRequest(EnhancedProcessingRequest):
    """Enhanced face detection request."""
    confidence_threshold: float = Field(default=0.7, ge=0, le=1)
    include_encodings: bool = False
    include_demographics: bool = False
    include_emotions: bool = False
    max_faces: int = Field(default=50, ge=1, le=100)


class EnhancedProcessingResponse(BaseModel):
    """Enhanced processing response with comprehensive metadata."""
    # Core result
    result: Any
    success: bool
    confidence: float
    
    # Performance metadata
    processing_time_ms: float
    cache_hit: bool
    fallback_used: bool
    model_used: Optional[str] = None
    estimated_cost: Optional[float] = None
    
    # Quality metrics
    quality_metrics: Optional[Dict[str, Any]] = None
    
    # Tracing
    trace_id: str
    span_id: str
    
    # System metrics (optional)
    system_metrics: Optional[Dict[str, Any]] = None


class SystemHealthResponse(BaseModel):
    """System health status response."""
    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: float
    
    # Component health
    agents: Dict[str, str]
    services: Dict[str, bool]
    
    # Performance metrics
    overall_performance: Dict[str, float]
    resource_usage: Dict[str, float]
    
    # Reliability status
    circuit_breakers: Dict[str, Dict[str, Any]]
    active_alerts: List[Dict[str, Any]]


class WebSocketManager:
    """Enhanced WebSocket manager with streaming analytics."""
    
    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, List[str]] = {}  # client_id -> event_types
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Connect client to WebSocket."""
        await websocket.accept()
        self.connections[client_id] = websocket
        self.subscriptions[client_id] = []
        self.logger.info(f"Client {client_id} connected")
    
    async def disconnect(self, client_id: str):
        """Disconnect client."""
        self.connections.pop(client_id, None)
        self.subscriptions.pop(client_id, None)
        self.logger.info(f"Client {client_id} disconnected")
    
    async def subscribe(self, client_id: str, event_types: List[str]):
        """Subscribe client to event types."""
        if client_id in self.subscriptions:
            self.subscriptions[client_id] = event_types
            self.logger.debug(f"Client {client_id} subscribed to: {event_types}")
    
    async def broadcast_performance_update(self, data: Dict[str, Any]):
        """Broadcast performance updates to subscribed clients."""
        message = {
            "type": "performance_update",
            "data": data,
            "timestamp": time.time()
        }
        
        await self._broadcast_to_subscribers(["performance"], message)
    
    async def broadcast_alert(self, alert_data: Dict[str, Any]):
        """Broadcast alerts to subscribed clients."""
        message = {
            "type": "alert",
            "data": alert_data,
            "timestamp": time.time()
        }
        
        await self._broadcast_to_subscribers(["alerts"], message)
    
    async def _broadcast_to_subscribers(self, event_types: List[str], message: Dict[str, Any]):
        """Broadcast message to subscribed clients."""
        for client_id, subscriptions in self.subscriptions.items():
            if any(event_type in subscriptions for event_type in event_types):
                websocket = self.connections.get(client_id)
                if websocket:
                    try:
                        await websocket.send_json(message)
                    except Exception as e:
                        self.logger.warning(f"Failed to send to {client_id}: {e}")
                        await self.disconnect(client_id)


# Global instances
websocket_manager = WebSocketManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced application lifespan with all systems."""
    # Startup
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("EnhancedVisionServer")
    
    try:
        logger.info("🚀 Starting Enhanced VisionAgent Server...")
        
        # Initialize all enhanced systems
        await initialize_enhanced_systems()
        
        # Create enhanced face agent (primary enterprise agent)
        app.state.face_agent = await create_enhanced_face_agent()
        logger.info("✅ Enhanced Face Agent initialized successfully")
        
        # Initialize other agents with robust error handling
        try:
            object_config = {
                "model_name": "yolov8s.pt", 
                "confidence_threshold": 0.5,
                "enable_caching": True,
                "cache_expire_time": 3600
            }
            app.state.object_agent = AsyncObjectAgent(device="cpu", config=object_config)
            await app.state.object_agent.initialize()
            logger.info("✅ Object Agent initialized successfully")
        except Exception as e:
            logger.error(f"❌ Object Agent initialization failed: {e}")
            app.state.object_agent = None
        
        try:
            video_config = {
                "frame_skip": 5, 
                "max_frames": 100,
                "enable_caching": True,
                "cache_expire_time": 3600
            }
            app.state.video_agent = AsyncVideoAgent(device="cpu", config=video_config)
            await app.state.video_agent.initialize()
            logger.info("✅ Video Agent initialized successfully")
        except Exception as e:
            logger.error(f"❌ Video Agent initialization failed: {e}")
            app.state.video_agent = None
        
        try:
            classification_config = {
                "top_k": 5, 
                "confidence_threshold": 0.3,
                "enable_caching": True,
                "cache_expire_time": 3600
            }
            app.state.classification_agent = AsyncClassificationAgent(device="cpu", config=classification_config)
            await app.state.classification_agent.initialize()
            logger.info("✅ Classification Agent initialized successfully")
        except Exception as e:
            logger.error(f"❌ Classification Agent initialization failed: {e}")
            app.state.classification_agent = None
        
        # Setup performance monitoring callbacks
        try:
            performance_analytics.subscribe_to_alerts(websocket_manager.broadcast_alert)
        except Exception as e:
            logger.warning(f"Could not setup alert callbacks: {e}")
            # Continue without alert callbacks
        
        # Start periodic performance broadcasting
        async def broadcast_performance():
            while True:
                try:
                    dashboard_data = performance_analytics.get_real_time_dashboard_data()
                    await websocket_manager.broadcast_performance_update(dashboard_data)
                    await asyncio.sleep(10)  # Broadcast every 10 seconds
                except Exception as e:
                    logger.error(f"Performance broadcast error: {e}")
                    await asyncio.sleep(10)
        
        asyncio.create_task(broadcast_performance())
        
        logger.info("✅ Enhanced VisionAgent Server started successfully!")
        
        yield
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise
    
    finally:
        # Shutdown
        logger.info("🛑 Shutting down Enhanced VisionAgent Server...")
        await shutdown_enhanced_systems()
        logger.info("✅ Enhanced VisionAgent Server shutdown completed")


# Create enhanced FastAPI application
app = FastAPI(
    title="Enhanced VisionAgent API",
    description="Enterprise-grade multi-modal AI agent platform with advanced performance patterns",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Enhanced CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Enhanced endpoints with comprehensive analytics

@app.post("/api/v2/face/detect", response_model=EnhancedProcessingResponse)
async def enhanced_face_detection(
    request: EnhancedFaceRequest,
    file: UploadFile = File(...)
):
    """Enhanced face detection with enterprise performance patterns."""
    start_time = time.time()
    
    try:
        # Read and validate image
        image_data = await file.read()
        if len(image_data) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Convert to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Process with enhanced face agent
        result = await app.state.face_agent.process(
            image,
            trace_id=request.trace_id,
            confidence_threshold=request.confidence_threshold,
            include_encodings=request.include_encodings,
            include_demographics=request.include_demographics,
            include_emotions=request.include_emotions
        )
        
        # Build enhanced response
        response = EnhancedProcessingResponse(
            result=result.primary_result.__dict__ if hasattr(result.primary_result, '__dict__') else result.primary_result,
            success=True,
            confidence=result.confidence,
            processing_time_ms=result.processing_time_ms,
            cache_hit=result.cache_hit,
            fallback_used=result.fallback_used,
            model_used=result.model_used,
            estimated_cost=result.estimated_cost,
            quality_metrics=result.primary_result.quality_metrics if hasattr(result.primary_result, 'quality_metrics') else None,
            trace_id=result.trace_id,
            span_id=result.span_id
        )
        
        # Include system metrics if requested
        if request.include_performance_data:
            response.system_metrics = await _get_system_metrics()
        
        return response
        
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        
        return EnhancedProcessingResponse(
            result={"error": str(e)},
            success=False,
            confidence=0.0,
            processing_time_ms=processing_time,
            cache_hit=False,
            fallback_used=True,
            trace_id=request.trace_id or "error",
            span_id="error"
        )


@app.post("/api/v2/face/compare")
async def enhanced_face_comparison(
    request: EnhancedProcessingRequest,
    file1: UploadFile = File(...),
    file2: UploadFile = File(...)
):
    """Enhanced face comparison with similarity analysis."""
    try:
        # Read both images
        image_data1 = await file1.read()
        image_data2 = await file2.read()
        
        nparr1 = np.frombuffer(image_data1, np.uint8)
        nparr2 = np.frombuffer(image_data2, np.uint8)
        
        image1 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)
        image2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)
        
        if image1 is None or image2 is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Compare faces using enhanced agent
        comparison_result = await app.state.face_agent.compare_faces(image1, image2)
        
        return {
            "success": True,
            "comparison": comparison_result,
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/health", response_model=SystemHealthResponse)
async def enhanced_system_health():
    """Comprehensive system health check with enterprise metrics."""
    try:
        # Get component health
        agents_health = {}
        for agent_name in ["face_agent", "object_agent", "video_agent", "classification_agent"]:
            agent = getattr(app.state, agent_name, None)
            if agent is None:
                agents_health[agent_name] = "not_initialized"
            elif hasattr(agent, '_health_check'):
                try:
                    is_healthy = await agent._health_check()
                    agents_health[agent_name] = "healthy" if is_healthy else "unhealthy"
                except Exception as e:
                    agents_health[agent_name] = f"unhealthy: {str(e)}"
            elif hasattr(agent, 'logger'):
                # Agent has logger, assume healthy if initialized
                agents_health[agent_name] = "healthy"
            else:
                agents_health[agent_name] = "unknown"
        
        # Get service health from reliability manager
        reliability_status = reliability_manager.get_reliability_status()
        service_health = reliability_status.get("service_health", {})
        
        # Get performance metrics
        dashboard_data = performance_analytics.get_real_time_dashboard_data()
        current_metrics = dashboard_data.get("current_metrics", {})
        
        # Get resource usage
        resource_stats = resource_manager.get_resource_stats()
        system_metrics = resource_stats.get("system_metrics", {})
        
        # Determine overall status
        overall_status = "healthy"
        if any(status == "unhealthy" for status in agents_health.values()):
            overall_status = "degraded"
        
        if system_metrics.get("overall_load", 0) > 0.9:
            overall_status = "degraded"
        
        if current_metrics.get("error_rate", 0) > 0.1:
            overall_status = "unhealthy"
        
        return SystemHealthResponse(
            status=overall_status,
            timestamp=time.time(),
            agents=agents_health,
            services=service_health,
            overall_performance={
                "avg_latency_ms": current_metrics.get("avg_latency_ms", 0),
                "error_rate": current_metrics.get("error_rate", 0),
                "throughput_per_minute": current_metrics.get("throughput_per_minute", 0)
            },
            resource_usage={
                "cpu_percent": system_metrics.get("cpu_percent", 0),
                "memory_percent": system_metrics.get("memory_percent", 0),
                "overall_load": system_metrics.get("overall_load", 0)
            },
            circuit_breakers=reliability_status.get("circuit_breakers", {}),
            active_alerts=dashboard_data.get("active_alerts", [])
        )
        
    except Exception as e:
        logging.getLogger("HealthCheck").error(f"Health check error: {e}")
        return SystemHealthResponse(
            status="unhealthy",
            timestamp=time.time(),
            agents={},
            services={},
            overall_performance={},
            resource_usage={},
            circuit_breakers={},
            active_alerts=[{"level": "error", "message": f"Health check failed: {e}"}]
        )


@app.get("/api/v2/analytics/dashboard")
async def analytics_dashboard():
    """Real-time analytics dashboard data."""
    try:
        dashboard_data = performance_analytics.get_real_time_dashboard_data()
        
        # Add cache statistics
        cache_stats = semantic_cache_manager.get_stats()
        dashboard_data["cache_performance"] = cache_stats
        
        # Add resource statistics
        resource_stats = resource_manager.get_resource_stats()
        dashboard_data["resource_management"] = resource_stats
        
        # Add speculation statistics
        speculation_stats = speculative_runner.get_stats()
        dashboard_data["speculative_execution"] = speculation_stats
        
        # Add reliability statistics
        reliability_stats = reliability_manager.get_reliability_status()
        dashboard_data["reliability"] = reliability_stats
        
        return dashboard_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics error: {e}")


@app.get("/api/v2/analytics/agent/{agent_name}")
async def agent_analytics(agent_name: str):
    """Detailed analytics for specific agent."""
    try:
        agent = getattr(app.state, f"{agent_name}_agent", None)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        if hasattr(agent, 'get_agent_stats'):
            stats = agent.get_agent_stats()
        else:
            stats = performance_analytics.get_agent_analytics(agent_name)
        
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v2/optimization/trigger")
async def trigger_system_optimization():
    """Trigger system-wide performance optimization."""
    try:
        optimization_tasks = []
        
        # Optimize each agent
        for agent_name in ["face_agent", "object_agent", "video_agent", "classification_agent"]:
            agent = getattr(app.state, agent_name, None)
            if agent and hasattr(agent, 'optimize_performance'):
                optimization_tasks.append(agent.optimize_performance())
        
        # Run optimizations concurrently
        await asyncio.gather(*optimization_tasks, return_exceptions=True)
        
        # Clear low-relevance caches
        await semantic_cache_manager._cleanup_low_relevance_entries(0)
        
        return {
            "success": True,
            "message": "System optimization completed",
            "optimized_components": len(optimization_tasks),
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization error: {e}")


@app.websocket("/ws/analytics")
async def analytics_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time analytics streaming."""
    client_id = f"client_{int(time.time() * 1000)}"
    
    try:
        await websocket_manager.connect(websocket, client_id)
        
        # Send initial dashboard data
        dashboard_data = performance_analytics.get_real_time_dashboard_data()
        await websocket.send_json({
            "type": "initial_data",
            "data": dashboard_data
        })
        
        # Handle client messages
        while True:
            try:
                message = await websocket.receive_json()
                
                if message.get("type") == "subscribe":
                    event_types = message.get("event_types", ["performance", "alerts"])
                    await websocket_manager.subscribe(client_id, event_types)
                    
                    await websocket.send_json({
                        "type": "subscribed",
                        "event_types": event_types
                    })
                
                elif message.get("type") == "get_stats":
                    stats = {
                        "resource_stats": resource_manager.get_resource_stats(),
                        "cache_stats": semantic_cache_manager.get_stats(),
                        "speculation_stats": speculative_runner.get_stats(),
                        "reliability_stats": reliability_manager.get_reliability_status()
                    }
                    
                    await websocket.send_json({
                        "type": "stats_response",
                        "data": stats
                    })
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logging.getLogger("WebSocket").error(f"WebSocket error: {e}")
                break
    
    finally:
        await websocket_manager.disconnect(client_id)


@app.post("/api/v2/batch/process")
async def enhanced_batch_processing(
    request: EnhancedProcessingRequest,
    files: List[UploadFile] = File(...)
):
    """Enhanced batch processing with intelligent load balancing."""
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Too many files (max 50)")
    
    try:
        # Determine batch processing strategy
        strategy = await _determine_batch_strategy(len(files), request)
        
        # Process based on strategy
        if strategy == "parallel":
            results = await _process_batch_parallel(files, request)
        elif strategy == "sequential":
            results = await _process_batch_sequential(files, request)
        else:  # adaptive
            results = await _process_batch_adaptive(files, request)
        
        # Calculate batch statistics
        batch_stats = _calculate_batch_stats(results)
        
        return {
            "success": True,
            "results": results,
            "batch_stats": batch_stats,
            "processing_strategy": strategy,
            "total_files": len(files)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing error: {e}")


@app.get("/api/v2/cost/estimate")
async def cost_estimation(
    task_type: str,
    complexity: str = "medium",
    estimated_tokens: int = 1000,
    batch_size: int = 1
):
    """Estimate processing costs for different scenarios."""
    try:
        complexity_score = {
            "simple": 0.2,
            "medium": 0.5,
            "complex": 0.8,
            "intensive": 1.0
        }.get(complexity, 0.5)
        
        # Get optimal model for this complexity
        optimal_model = cost_optimizer.route_request(
            complexity_score=complexity_score,
            budget_per_request=1.0,  # High budget for estimation
            estimated_tokens=estimated_tokens
        )
        
        # Calculate costs
        base_cost_per_request = 0.01 * complexity_score
        model_multiplier = {"gpt-4o-mini": 1, "gpt-4o": 3, "o1-preview": 10}.get(optimal_model, 1)
        
        estimated_cost_per_request = base_cost_per_request * model_multiplier
        total_estimated_cost = estimated_cost_per_request * batch_size
        
        return {
            "task_type": task_type,
            "complexity": complexity,
            "optimal_model": optimal_model,
            "estimated_cost_per_request": estimated_cost_per_request,
            "total_estimated_cost": total_estimated_cost,
            "batch_size": batch_size,
            "currency": "USD"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Enhanced utility functions

async def _get_system_metrics() -> Dict[str, Any]:
    """Get comprehensive system metrics."""
    return {
        "resource_usage": resource_manager.get_resource_stats(),
        "cache_performance": semantic_cache_manager.get_stats(),
        "speculation_stats": speculative_runner.get_stats(),
        "reliability_status": reliability_manager.get_reliability_status()
    }


async def _determine_batch_strategy(batch_size: int, request: EnhancedProcessingRequest) -> str:
    """Determine optimal batch processing strategy."""
    current_metrics = resource_manager.get_current_metrics()
    
    if batch_size <= 3:
        return "parallel"
    elif current_metrics.overall_load > 0.7:
        return "sequential"
    else:
        return "adaptive"


async def _process_batch_parallel(files: List[UploadFile], request: EnhancedProcessingRequest) -> List[Dict[str, Any]]:
    """Process batch in parallel."""
    tasks = []
    
    for file in files:
        # Create individual request
        individual_request = EnhancedFaceRequest(
            **request.dict(),
            confidence_threshold=0.7
        )
        
        # Create processing task
        task = asyncio.create_task(
            enhanced_face_detection(individual_request, file)
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Convert results to serializable format
    serializable_results = []
    for result in results:
        if isinstance(result, Exception):
            serializable_results.append({
                "success": False,
                "error": str(result)
            })
        else:
            serializable_results.append(result.dict() if hasattr(result, 'dict') else result)
    
    return serializable_results


async def _process_batch_sequential(files: List[UploadFile], request: EnhancedProcessingRequest) -> List[Dict[str, Any]]:
    """Process batch sequentially."""
    results = []
    
    for file in files:
        try:
            individual_request = EnhancedFaceRequest(
                **request.dict(),
                confidence_threshold=0.7
            )
            
            result = await enhanced_face_detection(individual_request, file)
            results.append(result.dict() if hasattr(result, 'dict') else result)
            
        except Exception as e:
            results.append({
                "success": False,
                "error": str(e)
            })
    
    return results


async def _process_batch_adaptive(files: List[UploadFile], request: EnhancedProcessingRequest) -> List[Dict[str, Any]]:
    """Process batch with adaptive strategy."""
    # Start with smaller parallel chunks
    chunk_size = min(5, len(files))
    results = []
    
    for i in range(0, len(files), chunk_size):
        chunk = files[i:i + chunk_size]
        chunk_results = await _process_batch_parallel(chunk, request)
        results.extend(chunk_results)
        
        # Adaptive adjustment based on performance
        current_metrics = resource_manager.get_current_metrics()
        if current_metrics.overall_load > 0.8:
            chunk_size = max(1, chunk_size - 1)
        elif current_metrics.overall_load < 0.3:
            chunk_size = min(10, chunk_size + 1)
    
    return results


def _calculate_batch_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate statistics for batch processing results."""
    total_results = len(results)
    successful_results = sum(1 for r in results if r.get("success", False))
    
    processing_times = [
        r.get("processing_time_ms", 0) 
        for r in results 
        if r.get("success", False)
    ]
    
    cache_hits = sum(1 for r in results if r.get("cache_hit", False))
    fallback_uses = sum(1 for r in results if r.get("fallback_used", False))
    
    return {
        "total_files": total_results,
        "successful_processing": successful_results,
        "success_rate": successful_results / total_results if total_results > 0 else 0,
        "cache_hit_rate": cache_hits / total_results if total_results > 0 else 0,
        "fallback_usage_rate": fallback_uses / total_results if total_results > 0 else 0,
        "avg_processing_time_ms": np.mean(processing_times) if processing_times else 0,
        "total_processing_time_ms": sum(processing_times)
    }


# Development server with enhanced configuration
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced VisionAgent Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--log-level", default="info", help="Log level")
    parser.add_argument("--enable-reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    print("🚀 Starting Enhanced VisionAgent Server...")
    print(f"   Host: {args.host}:{args.port}")
    print(f"   Workers: {args.workers}")
    print(f"   Log level: {args.log_level}")
    print("   Enterprise features: Enabled")
    print("   Performance patterns: Active")
    
    uvicorn.run(
        "enhanced_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
        reload=args.enable_reload
    )
