"""
Object Detection Agent
Implements object detection using YOLOv8 and other state-of-the-art models.
"""

import cv2
import numpy as np
from typing import Any, Dict, List, Optional, Union
import torch
from ultralytics import YOLO
from .base_agent import BaseAgent, ProcessingResult


class ObjectAgent(BaseAgent):
    """
    Agent for object detection and classification.
    
    Features:
    - YOLOv8 object detection
    - Real-time inference
    - Batch processing support
    - Custom class filtering
    - Confidence thresholding
    """
    
    def __init__(self, 
                 device: Optional[str] = None,
                 model_path: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize Object Detection Agent.
        
        Args:
            device: Target device (CPU/GPU)
            model_path: Path to YOLO model weights
            config: Configuration parameters including:
                - model_size: 'nano', 'small', 'medium', 'large', 'xlarge'
                - confidence_threshold: Minimum confidence for detections
                - iou_threshold: IoU threshold for NMS
                - max_detections: Maximum number of detections per image
                - target_classes: List of class names to detect (None for all)
        """
        super().__init__(device, model_path, config)
        
        # Configuration defaults
        self.model_size = self.config.get('model_size', 'small')
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        self.iou_threshold = self.config.get('iou_threshold', 0.45)
        self.max_detections = self.config.get('max_detections', 100)
        self.target_classes = self.config.get('target_classes', None)
        
        # Model will be loaded in initialize()
        self.model: Optional[YOLO] = None
        self.class_names: List[str] = []
    
    def initialize(self) -> bool:
        """
        Initialize YOLO model for object detection.
        
        Returns:
            True if initialization successful
        """
        try:
            self.logger.info("Initializing Object Detection Agent...")
            
            # Determine model path
            if self.model_path:
                model_path = self.model_path
            else:
                # Use pre-trained YOLOv8 model
                model_map = {
                    'nano': 'yolov8n.pt',
                    'small': 'yolov8s.pt', 
                    'medium': 'yolov8m.pt',
                    'large': 'yolov8l.pt',
                    'xlarge': 'yolov8x.pt'
                }
                model_path = model_map.get(self.model_size, 'yolov8s.pt')
            
            # Load YOLO model
            self.model = YOLO(model_path)
            
            # Move to appropriate device
            if self.device == 'cuda' and torch.cuda.is_available():
                self.model.to('cuda')
            
            # Get class names
            self.class_names = list(self.model.names.values())
            
            self._is_initialized = True
            self.logger.info(f"Object Agent initialized with {model_path} on {self.device}")
            self.logger.info(f"Available classes: {len(self.class_names)} total")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Object Agent: {str(e)}")
            return False
    
    def process(self, input_data: Any) -> ProcessingResult:
        """
        Process image for object detection.
        
        Args:
            input_data: Image data (path, numpy array, or bytes)
            
        Returns:
            ProcessingResult with object detection results
        """
        if not self._is_initialized:
            if not self.initialize():
                return ProcessingResult(
                    success=False,
                    data={},
                    error="Agent not initialized"
                )
        
        try:
            # Preprocess image
            image = self._preprocess_image(input_data)
            
            # Perform object detection
            result, inference_time = self._measure_inference_time(
                self._detect_objects, image
            )
            
            return ProcessingResult(
                success=True,
                data=result,
                inference_time=inference_time,
                metadata={
                    'image_shape': image.shape,
                    'model_size': self.model_size,
                    'confidence_threshold': self.confidence_threshold
                }
            )
            
        except Exception as e:
            self.logger.error(f"Object detection error: {str(e)}")
            return ProcessingResult(
                success=False,
                data={},
                error=str(e)
            )
    
    def _detect_objects(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect objects in the image using YOLO.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary with detection results
        """
        # Run YOLO inference
        results = self.model(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            max_det=self.max_detections,
            verbose=False
        )
        
        detections = []
        
        # Process detection results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract box data
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    
                    class_name = self.class_names[cls_id]
                    
                    # Filter by target classes if specified
                    if self.target_classes and class_name not in self.target_classes:
                        continue
                    
                    detection = {
                        'class_id': cls_id,
                        'class_name': class_name,
                        'confidence': conf,
                        'bounding_box': {
                            'x1': int(xyxy[0]),
                            'y1': int(xyxy[1]),
                            'x2': int(xyxy[2]),
                            'y2': int(xyxy[3])
                        },
                        'center': {
                            'x': int((xyxy[0] + xyxy[2]) / 2),
                            'y': int((xyxy[1] + xyxy[3]) / 2)
                        },
                        'area': int((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]))
                    }
                    
                    detections.append(detection)
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'detections': detections,
            'detection_count': len(detections),
            'image_dimensions': {
                'height': image.shape[0],
                'width': image.shape[1],
                'channels': image.shape[2]
            },
            'class_summary': self._get_class_summary(detections)
        }
    
    def _get_class_summary(self, detections: List[Dict]) -> Dict[str, int]:
        """
        Get summary of detected classes and their counts.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Dictionary mapping class names to counts
        """
        class_counts = {}
        for detection in detections:
            class_name = detection['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        return class_counts
    
    def process_batch(self, image_batch: List[Any]) -> List[ProcessingResult]:
        """
        Process multiple images in batch for better performance.
        
        Args:
            image_batch: List of image inputs
            
        Returns:
            List of ProcessingResult objects
        """
        if not self._is_initialized:
            if not self.initialize():
                return [ProcessingResult(
                    success=False,
                    data={},
                    error="Agent not initialized"
                ) for _ in image_batch]
        
        results = []
        for image_input in image_batch:
            result = self.process(image_input)
            results.append(result)
        
        return results
    
    def set_target_classes(self, class_names: Optional[List[str]]) -> None:
        """
        Set target classes to filter detections.
        
        Args:
            class_names: List of class names to detect, or None for all classes
        """
        self.target_classes = class_names
        if class_names:
            self.logger.info(f"Target classes set to: {class_names}")
        else:
            self.logger.info("Target classes cleared - detecting all classes")
    
    def get_available_classes(self) -> List[str]:
        """
        Get list of all available class names.
        
        Returns:
            List of class names the model can detect
        """
        return self.class_names.copy()
