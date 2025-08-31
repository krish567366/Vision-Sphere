"""
VisionAgent - Multi-modal AI Agent Framework
Provides specialized agents for computer vision tasks.
"""

from .base_agent import BaseAgent
from .face_agent import FaceAgent
from .object_agent import ObjectAgent
from .video_agent import VideoAgent
from .classification_agent import ClassificationAgent

# Enhanced async agents with performance optimizations
from .async_base_agent import AsyncBaseAgent, AsyncProcessingResult, agent_registry
from .async_face_agent import AsyncFaceAgent
from .async_object_agent import AsyncObjectAgent
from .async_video_agent import AsyncVideoAgent, VideoAnalysisResult
from .async_classification_agent import AsyncClassificationAgent

__all__ = [
    # Synchronous agents
    'BaseAgent',
    'FaceAgent', 
    'ObjectAgent',
    'VideoAgent',
    'ClassificationAgent',
    
    # Asynchronous agents
    'AsyncBaseAgent',
    'AsyncProcessingResult',
    'AsyncFaceAgent',
    'AsyncObjectAgent', 
    'AsyncVideoAgent',
    'AsyncClassificationAgent',
    'VideoAnalysisResult',
    'agent_registry'
]
