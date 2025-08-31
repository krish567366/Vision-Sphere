"""
Face Detection and Recognition Agent
Implements face detection using OpenCV's DNN face detector and basic face analysis.
"""

import cv2
import numpy as np
from typing import Any, Dict, List, Optional, Union
import os
from urllib.request import urlretrieve
from .base_agent import BaseAgent, ProcessingResult


class FaceAgent(BaseAgent):
    """
    Agent for face detection and basic analysis using OpenCV.
    
    Features:
    - Face detection using OpenCV DNN
    - Basic face analysis and feature extraction
    - Face comparison using feature vectors
    - Bounding box detection
    - Age/gender estimation (with additional models)
    """
    
    def __init__(self, 
                 device: Optional[str] = None,
                 model_path: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize Face Agent.
        
        Args:
            device: Target device (CPU/GPU)
            model_path: Path to face detection models
            config: Configuration parameters including:
                - confidence_threshold: Minimum confidence for face detection
                - nms_threshold: Non-maximum suppression threshold
                - input_size: Input size for face detection model
        """
        super().__init__(device, model_path, config)
        
        # Configuration defaults
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.nms_threshold = self.config.get('nms_threshold', 0.4)
        self.input_size = self.config.get('input_size', (300, 300))
        
        # Model paths - using Haar cascades instead of DNN
        self.model_dir = self.model_path or './models/face'
        self.cascade_path = None
        
        # Face cascade classifier
        self.face_cascade = None
        
        # Known face features for recognition
        self.known_face_features: List[np.ndarray] = []
        self.known_face_names: List[str] = []
    
    def initialize(self) -> bool:
        """
        Initialize face detection model.
        
        Returns:
            True if initialization successful
        """
        try:
            self.logger.info("Initializing Face Agent...")
            
            # Ensure model directory exists
            os.makedirs(self.model_dir, exist_ok=True)
            
            # Download models if not present
            if not self._download_models():
                return False
            
            # Load DNN model
            self.net = cv2.dnn.readNetFromTensorflow(
                self.model_weights_path, 
                self.proto_path
            )
            
            # Set backend and target
            if self.device == 'cuda' and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                self.logger.info("Using CUDA backend for face detection")
            else:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.logger.info("Using CPU backend for face detection")
            
            self._is_initialized = True
            self.logger.info("Face Agent initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Face Agent: {str(e)}")
            return False
    
    def _download_models(self) -> bool:
        """
        Setup face detection using OpenCV's built-in Haar cascades.
        
        Returns:
            True if models are available
        """
        try:
            # Use OpenCV's built-in Haar cascade for face detection
            self.cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            
            if not os.path.exists(self.cascade_path):
                self.logger.error("OpenCV Haar cascade not found")
                return False
            
            self.logger.info("Using OpenCV Haar cascade for face detection")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup face detection models: {str(e)}")
            return False
    
    def process(self, input_data: Any) -> ProcessingResult:
        """
        Process image for face detection and analysis.
        
        Args:
            input_data: Image data (path, numpy array, or bytes)
            
        Returns:
            ProcessingResult with face detection results
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
            
            # Perform face detection
            result, inference_time = self._measure_inference_time(
                self._detect_faces, image
            )
            
            return ProcessingResult(
                success=True,
                data=result,
                inference_time=inference_time,
                metadata={
                    'image_shape': image.shape,
                    'confidence_threshold': self.confidence_threshold
                }
            )
            
        except Exception as e:
            self.logger.error(f"Face processing error: {str(e)}")
            return ProcessingResult(
                success=False,
                data={},
                error=str(e)
            )
    
    def _detect_faces(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect faces in the image using OpenCV DNN.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary with face detection results
        """
        h, w = image.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(
            image, 
            1.0, 
            self.input_size, 
            [104, 117, 123]
        )
        
        # Set input to the model
        self.net.setInput(blob)
        
        # Run forward pass
        detections = self.net.forward()
        
        faces = []
        
        # Process detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.confidence_threshold:
                # Get bounding box coordinates
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                
                # Extract face features
                face_roi = image[y1:y2, x1:x2]
                features = self._extract_face_features(face_roi)
                
                # Face recognition against known faces
                name, recognition_confidence = self._recognize_face(features)
                
                face_data = {
                    'face_id': i,
                    'bounding_box': {
                        'left': x1,
                        'top': y1,
                        'right': x2,
                        'bottom': y2
                    },
                    'center': {
                        'x': (x1 + x2) // 2,
                        'y': (y1 + y2) // 2
                    },
                    'confidence': float(confidence),
                    'recognition': {
                        'name': name,
                        'confidence': float(recognition_confidence),
                        'is_known': name != "Unknown"
                    },
                    'area': (x2 - x1) * (y2 - y1),
                    'features': features.tolist() if features is not None else None
                }
                
                faces.append(face_data)
        
        return {
            'faces': faces,
            'face_count': len(faces),
            'image_dimensions': {
                'height': h,
                'width': w,
                'channels': image.shape[2] if len(image.shape) > 2 else 1
            }
        }
    
    def _extract_face_features(self, face_roi: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract features from face region.
        
        Args:
            face_roi: Face region of interest
            
        Returns:
            Feature vector as numpy array
        """
        if face_roi.size == 0:
            return None
        
        try:
            # Resize face to standard size
            face_resized = cv2.resize(face_roi, (128, 128))
            
            # Convert to grayscale for feature extraction
            if len(face_resized.shape) == 3:
                face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            else:
                face_gray = face_resized
            
            # Extract HOG features as a simple feature representation
            # This is a basic implementation - can be enhanced with deep learning features
            features = cv2.HOGDescriptor((128, 128), (16, 16), (8, 8), (8, 8), 9).compute(face_gray)
            
            return features.flatten()
            
        except Exception as e:
            self.logger.warning(f"Failed to extract features: {str(e)}")
            return None
    
    def _recognize_face(self, features: Optional[np.ndarray]) -> tuple[str, float]:
        """
        Recognize face using stored features.
        
        Args:
            features: Face feature vector
            
        Returns:
            Tuple of (name, confidence)
        """
        if features is None or len(self.known_face_features) == 0:
            return "Unknown", 0.0
        
        try:
            # Calculate distances to all known faces
            distances = []
            for known_features in self.known_face_features:
                # Cosine similarity
                similarity = np.dot(features, known_features) / (
                    np.linalg.norm(features) * np.linalg.norm(known_features)
                )
                distance = 1.0 - similarity
                distances.append(distance)
            
            # Find best match
            min_distance_idx = np.argmin(distances)
            min_distance = distances[min_distance_idx]
            
            # Convert distance to confidence (threshold-based)
            if min_distance < self.tolerance:
                confidence = 1.0 - min_distance
                name = self.known_face_names[min_distance_idx]
                return name, confidence
            
            return "Unknown", 0.0
            
        except Exception as e:
            self.logger.warning(f"Face recognition error: {str(e)}")
            return "Unknown", 0.0
    
    def add_known_face(self, face_image: Union[str, np.ndarray], name: str) -> bool:
        """
        Add a known face for recognition.
        
        Args:
            face_image: Image containing the face
            name: Name to associate with the face
            
        Returns:
            True if face was successfully added
        """
        try:
            image = self._preprocess_image(face_image)
            
            # Detect faces in the image
            result = self._detect_faces(image)
            
            if result['face_count'] == 0:
                self.logger.warning(f"No face found in image for {name}")
                return False
            
            if result['face_count'] > 1:
                self.logger.warning(f"Multiple faces found for {name}, using first one")
            
            # Extract features from first face
            first_face = result['faces'][0]
            features = np.array(first_face['features']) if first_face['features'] else None
            
            if features is None:
                self.logger.warning(f"Could not extract features for {name}")
                return False
            
            self.known_face_features.append(features)
            self.known_face_names.append(name)
            
            self.logger.info(f"Added known face: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add known face {name}: {str(e)}")
            return False
    
    def remove_known_face(self, name: str) -> bool:
        """
        Remove a known face from recognition database.
        
        Args:
            name: Name of the face to remove
            
        Returns:
            True if face was found and removed
        """
        try:
            indices_to_remove = [i for i, n in enumerate(self.known_face_names) if n == name]
            
            if not indices_to_remove:
                self.logger.warning(f"Face {name} not found in known faces")
                return False
            
            # Remove in reverse order to maintain indices
            for index in reversed(indices_to_remove):
                del self.known_face_features[index]
                del self.known_face_names[index]
            
            self.logger.info(f"Removed {len(indices_to_remove)} encodings for {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to remove known face {name}: {str(e)}")
            return False
    
    def get_known_faces(self) -> List[str]:
        """
        Get list of all known face names.
        
        Returns:
            List of known face names
        """
        return self.known_face_names.copy()
    
    def clear_known_faces(self) -> None:
        """Clear all known faces from the recognition database."""
        self.known_face_features.clear()
        self.known_face_names.clear()
        self.logger.info("Cleared all known faces")
