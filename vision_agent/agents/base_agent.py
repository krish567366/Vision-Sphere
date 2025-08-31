"""
Base Agent Abstract Class
Provides the foundation for all AI agents in the VisionAgent framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import time
import logging
from dataclasses import dataclass
import torch
import cv2
import numpy as np


@dataclass
class ProcessingResult:
    """Standard result structure for all agent processing operations."""
    success: bool
    data: Dict[str, Any]
    confidence: Optional[float] = None
    inference_time: Optional[float] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseAgent(ABC):
    """
    Abstract base class for all VisionAgent AI agents.
    
    All agents must implement the process method and follow the standard
    interface for consistent behavior across the framework.
    """
    
    def __init__(self, 
                 device: Optional[str] = None,
                 model_path: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base agent.
        
        Args:
            device: Target device ('cuda', 'cpu', or None for auto-detection)
            model_path: Path to the model files
            config: Additional configuration parameters
        """
        self.device = self._get_device(device)
        self.model_path = model_path
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None
        self._is_initialized = False
        
    def _get_device(self, device: Optional[str] = None) -> str:
        """
        Determine the best available device for processing.
        
        Args:
            device: Preferred device or None for auto-detection
            
        Returns:
            Device string ('cuda' or 'cpu')
        """
        if device:
            return device
            
        if torch.cuda.is_available():
            self.logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
            return 'cuda'
        else:
            self.logger.info("No GPU available, using CPU")
            return 'cpu'
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the agent's model and resources.
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def process(self, input_data: Any) -> ProcessingResult:
        """
        Process input data and return structured results.
        
        Args:
            input_data: Input data to process (image, video, path, etc.)
            
        Returns:
            ProcessingResult with success status, data, confidence, and timing
        """
        pass
    
    def _preprocess_image(self, image_input: Union[str, np.ndarray, bytes]) -> np.ndarray:
        """
        Standardize image input to numpy array format.
        
        Args:
            image_input: Image as file path, numpy array, or bytes
            
        Returns:
            Image as numpy array in BGR format
        """
        if isinstance(image_input, str):
            # File path
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Could not load image from path: {image_input}")
        elif isinstance(image_input, bytes):
            # Raw bytes
            nparr = np.frombuffer(image_input, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(image_input, np.ndarray):
            # Already numpy array
            image = image_input
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
            
        return image
    
    def _measure_inference_time(self, func, *args, **kwargs) -> tuple:
        """
        Measure execution time of a function.
        
        Args:
            func: Function to measure
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Tuple of (result, inference_time_ms)
        """
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        return result, inference_time
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get agent information and status.
        
        Returns:
            Dictionary with agent details
        """
        return {
            'agent_type': self.__class__.__name__,
            'device': self.device,
            'model_path': self.model_path,
            'initialized': self._is_initialized,
            'config': self.config
        }
