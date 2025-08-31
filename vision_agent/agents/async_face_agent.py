"""
Enhanced Async Face Agent with OpenCV DNN and Performance Optimizations
Production-ready face detection and recognition with advanced caching and tracing.
"""

import asyncio
import time
import hashlib
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import cv2
import numpy as np

from .async_base_agent import AsyncBaseAgent, AsyncProcessingResult
from ..utils.caching import cached_tool_call
from ..utils.tracing import traced_operation, SpanType
from ..utils.streaming import publish_detection_event, EventType


class AsyncFaceAgent(AsyncBaseAgent):
    """
    Enhanced async face detection and recognition agent.
    
    Features:
    - OpenCV DNN face detection with Haar cascade fallback
    - Async face encoding and recognition
    - Face landmark detection (optional)
    - Real-time streaming support
    - Comprehensive caching and tracing
    - Batch processing optimization
    """
    
    def __init__(self, 
                 device: Optional[str] = None,
                 model_path: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the async face agent.
        
        Args:
            device: Target device for processing
            model_path: Path to face detection models
            config: Agent configuration
        """
        super().__init__("AsyncFaceAgent", device, model_path, config)
        
        # Model paths
        self.dnn_model_path = config.get('dnn_model_path', 'models/face/opencv_face_detector.pbtxt')
        self.dnn_weights_path = config.get('dnn_weights_path', 'models/face/opencv_face_detector_uint8.pb')
        self.haar_cascade_path = config.get('haar_cascade_path', cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detection parameters
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.nms_threshold = config.get('nms_threshold', 0.4)
        self.enable_landmarks = config.get('enable_landmarks', False)
        self.enable_recognition = config.get('enable_recognition', True)
        
        # Models
        self.dnn_net = None
        self.haar_cascade = None
        self.landmark_predictor = None
        
        # Known faces database
        self.known_faces: Dict[str, np.ndarray] = {}
        self.face_recognition_threshold = config.get('face_recognition_threshold', 0.6)
        
        self.logger = logging.getLogger('AsyncFaceAgent')
    
    async def _initialize_model(self) -> bool:
        """
        Initialize face detection models.
        
        Returns:
            True if initialization successful
        """
        try:
            # Load DNN model first (more accurate)
            try:
                self.dnn_net = cv2.dnn.readNetFromTensorflow(
                    self.dnn_weights_path, 
                    self.dnn_model_path
                )
                self.logger.info("DNN face detector loaded successfully")
            except Exception as e:
                self.logger.warning(f"Failed to load DNN model: {e}")
            
            # Load Haar cascade as fallback
            try:
                self.haar_cascade = cv2.CascadeClassifier(self.haar_cascade_path)
                if self.haar_cascade.empty():
                    raise ValueError("Failed to load Haar cascade")
                self.logger.info("Haar cascade face detector loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load Haar cascade: {e}")
                if self.dnn_net is None:
                    return False
            
            # Initialize landmark predictor if enabled
            if self.enable_landmarks:
                try:
                    # This would use dlib's shape predictor
                    # For now, we'll skip landmarks to avoid dlib dependency
                    self.logger.info("Landmark detection disabled (dlib not available)")
                    self.enable_landmarks = False
                except Exception as e:
                    self.logger.warning(f"Landmark predictor not available: {e}")
                    self.enable_landmarks = False
            
            return True
        
        except Exception as e:
            self.logger.error(f"Model initialization failed: {e}")
            return False
    
    @cached_tool_call(expire_time=1800)  # 30 minutes cache
    async def _detect_faces_dnn(self, 
                               image: np.ndarray, 
                               trace_id: str) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect faces using OpenCV DNN with caching.
        
        Args:
            image: Input image as numpy array
            trace_id: Trace ID for monitoring
            
        Returns:
            List of face bounding boxes with confidence scores
        """
        async with traced_operation(
            "face_detection.dnn",
            SpanType.INFERENCE,
            trace_id=trace_id
        ) as span:
            
            h, w = image.shape[:2]
            span.set_attribute("image.height", h)
            span.set_attribute("image.width", w)
            
            # Create blob from image
            blob = cv2.dnn.blobFromImage(
                image, 
                1.0, 
                (300, 300), 
                [104, 117, 123],
                swapRB=False,
                crop=False
            )
            
            # Set input to the network
            self.dnn_net.setInput(blob)
            
            # Run inference
            detections = self.dnn_net.forward()
            
            faces = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                if confidence > self.confidence_threshold:
                    x1 = int(detections[0, 0, i, 3] * w)
                    y1 = int(detections[0, 0, i, 4] * h)
                    x2 = int(detections[0, 0, i, 5] * w)
                    y2 = int(detections[0, 0, i, 6] * h)
                    
                    faces.append((x1, y1, x2, y2, confidence))
            
            span.set_attribute("detections.count", len(faces))
            return faces
    
    async def _detect_faces_haar(self, 
                                image: np.ndarray, 
                                trace_id: str) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect faces using Haar cascade fallback.
        
        Args:
            image: Input image as numpy array
            trace_id: Trace ID for monitoring
            
        Returns:
            List of face bounding boxes with confidence scores
        """
        async with traced_operation(
            "face_detection.haar",
            SpanType.INFERENCE,
            trace_id=trace_id
        ) as span:
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.haar_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Convert to consistent format (x1, y1, x2, y2, confidence)
            face_list = []
            for (x, y, w, h) in faces:
                face_list.append((x, y, x + w, y + h, 0.8))  # Fixed confidence for Haar
            
            span.set_attribute("detections.count", len(face_list))
            return face_list
    
    async def _extract_face_encoding(self, 
                                   face_image: np.ndarray,
                                   trace_id: str) -> np.ndarray:
        """
        Extract face encoding using OpenCV features.
        
        Args:
            face_image: Cropped face image
            trace_id: Trace ID for monitoring
            
        Returns:
            Face encoding vector
        """
        async with traced_operation(
            "face_encoding.extraction",
            SpanType.INFERENCE,
            trace_id=trace_id
        ) as span:
            
            # Convert to grayscale
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
            # Resize to standard size
            resized = cv2.resize(gray, (128, 128))
            
            # Extract HOG features
            hog = cv2.HOGDescriptor()
            features = hog.compute(resized)
            
            if features is not None:
                encoding = features.flatten()
                span.set_attribute("encoding.dimensions", len(encoding))
                return encoding
            else:
                # Fallback: use raw pixel values
                encoding = resized.flatten().astype(np.float32) / 255.0
                span.set_attribute("encoding.dimensions", len(encoding))
                span.add_event("fallback_to_raw_pixels")
                return encoding
    
    async def _recognize_face(self, 
                            encoding: np.ndarray,
                            trace_id: str) -> Optional[Tuple[str, float]]:
        """
        Recognize face from encoding.
        
        Args:
            encoding: Face encoding vector
            trace_id: Trace ID for monitoring
            
        Returns:
            (name, similarity_score) or None if no match
        """
        if not self.known_faces:
            return None
        
        async with traced_operation(
            "face_recognition.matching",
            SpanType.INFERENCE,
            trace_id=trace_id
        ) as span:
            
            best_match = None
            best_similarity = 0.0
            
            for name, known_encoding in self.known_faces.items():
                # Calculate cosine similarity
                similarity = np.dot(encoding, known_encoding) / (
                    np.linalg.norm(encoding) * np.linalg.norm(known_encoding)
                )
                
                if similarity > best_similarity and similarity > self.face_recognition_threshold:
                    best_similarity = similarity
                    best_match = name
            
            span.set_attribute("recognition.candidates", len(self.known_faces))
            span.set_attribute("recognition.best_similarity", best_similarity)
            
            if best_match:
                span.set_attribute("recognition.matched_name", best_match)
                return best_match, best_similarity
            
            return None
    
    async def _process_internal(self, input_data: Any, trace_id: str) -> Dict[str, Any]:
        """
        Internal face processing implementation.
        
        Args:
            input_data: Image input data
            trace_id: Trace ID for monitoring
            
        Returns:
            Processing results dictionary
        """
        # Preprocess image
        image = await self._preprocess_image_async(input_data)
        
        # Detect faces
        faces = []
        
        # Try DNN first
        if self.dnn_net is not None:
            try:
                faces = await self._detect_faces_dnn(image, trace_id)
            except Exception as e:
                self.logger.warning(f"DNN detection failed: {e}")
        
        # Fallback to Haar cascade
        if not faces and self.haar_cascade is not None:
            faces = await self._detect_faces_haar(image, trace_id)
        
        # Process each detected face
        processed_faces = []
        for i, (x1, y1, x2, y2, confidence) in enumerate(faces):
            face_info = {
                'face_id': i,
                'bounding_box': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                'confidence': confidence,
                'recognition': None,
                'landmarks': None
            }
            
            # Extract face for recognition
            if self.enable_recognition:
                try:
                    face_crop = image[y1:y2, x1:x2]
                    if face_crop.size > 0:
                        encoding = await self._extract_face_encoding(face_crop, trace_id)
                        recognition_result = await self._recognize_face(encoding, trace_id)
                        
                        if recognition_result:
                            face_info['recognition'] = {
                                'name': recognition_result[0],
                                'similarity': recognition_result[1]
                            }
                except Exception as e:
                    self.logger.warning(f"Face recognition failed for face {i}: {e}")
            
            processed_faces.append(face_info)
            
            # Publish detection event
            await publish_detection_event(
                'face',
                trace_id,
                {'face_id': i, 'confidence': confidence}
            )
        
        return {
            'faces': processed_faces,
            'face_count': len(processed_faces),
            'image_dimensions': {'height': image.shape[0], 'width': image.shape[1]},
            'detection_method': 'dnn' if self.dnn_net and faces else 'haar'
        }
    
    async def add_known_face(self, 
                           name: str, 
                           face_image: Union[str, np.ndarray, bytes]) -> bool:
        """
        Add a known face to the recognition database.
        
        Args:
            name: Person's name
            face_image: Face image input
            
        Returns:
            True if face added successfully
        """
        try:
            image = await self._preprocess_image_async(face_image)
            
            # Detect face in the image
            trace_id = f"add_known_face_{name}"
            
            if self.dnn_net:
                faces = await self._detect_faces_dnn(image, trace_id)
            else:
                faces = await self._detect_faces_haar(image, trace_id)
            
            if not faces:
                self.logger.error(f"No face detected in image for {name}")
                return False
            
            # Use the largest face
            largest_face = max(faces, key=lambda f: (f[2] - f[0]) * (f[3] - f[1]))
            x1, y1, x2, y2, _ = largest_face
            
            # Extract face crop
            face_crop = image[y1:y2, x1:x2]
            
            # Extract encoding
            encoding = await self._extract_face_encoding(face_crop, trace_id)
            
            # Store in known faces
            self.known_faces[name] = encoding
            
            self.logger.info(f"Added known face: {name}")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to add known face {name}: {e}")
            return False
    
    async def remove_known_face(self, name: str) -> bool:
        """
        Remove a known face from the database.
        
        Args:
            name: Person's name to remove
            
        Returns:
            True if removed successfully
        """
        if name in self.known_faces:
            del self.known_faces[name]
            self.logger.info(f"Removed known face: {name}")
            return True
        
        return False
    
    async def list_known_faces(self) -> List[str]:
        """
        Get list of all known face names.
        
        Returns:
            List of known face names
        """
        return list(self.known_faces.keys())
    
    async def detect_and_recognize_streaming(self, 
                                           video_source: Union[str, int],
                                           callback_func: Optional[callable] = None) -> AsyncProcessingResult:
        """
        Perform streaming face detection and recognition.
        
        Args:
            video_source: Video file path or camera index
            callback_func: Optional callback for each frame result
            
        Returns:
            Streaming processing result
        """
        trace_id = f"streaming_face_{time.time()}"
        
        async with traced_operation(
            "face_detection.streaming",
            SpanType.STREAMING,
            trace_id=trace_id
        ) as span:
            
            try:
                cap = cv2.VideoCapture(video_source)
                if not cap.isOpened():
                    raise ValueError(f"Could not open video source: {video_source}")
                
                frame_count = 0
                total_faces = 0
                processing_times = []
                
                span.set_attribute("video.source", str(video_source))
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_start = time.perf_counter()
                    
                    # Process frame
                    result = await self.process(frame, trace_id=f"{trace_id}_frame_{frame_count}")
                    
                    frame_end = time.perf_counter()
                    frame_time = (frame_end - frame_start) * 1000
                    processing_times.append(frame_time)
                    
                    if result.success:
                        face_count = result.data.get('face_count', 0)
                        total_faces += face_count
                        
                        # Call callback if provided
                        if callback_func:
                            await callback_func(frame_count, frame, result)
                        
                        # Publish streaming event
                        await publish_detection_event(
                            'streaming_face',
                            trace_id,
                            {
                                'frame': frame_count,
                                'faces': face_count,
                                'processing_time': frame_time
                            }
                        )
                    
                    frame_count += 1
                    
                    # Allow other tasks to run
                    await asyncio.sleep(0.001)
                
                cap.release()
                
                # Calculate summary metrics
                avg_processing_time = np.mean(processing_times) if processing_times else 0
                
                span.set_attribute("streaming.total_frames", frame_count)
                span.set_attribute("streaming.total_faces", total_faces)
                span.set_attribute("streaming.avg_processing_time", avg_processing_time)
                
                return AsyncProcessingResult(
                    success=True,
                    data={
                        'total_frames': frame_count,
                        'total_faces': total_faces,
                        'avg_processing_time_ms': avg_processing_time,
                        'fps': frame_count / (sum(processing_times) / 1000) if processing_times else 0
                    },
                    trace_id=trace_id,
                    metadata={'streaming': True}
                )
            
            except Exception as e:
                error_msg = f"Streaming face detection failed: {str(e)}"
                span.set_attribute("error", error_msg)
                return AsyncProcessingResult(
                    success=False,
                    data={},
                    error=error_msg,
                    trace_id=trace_id
                )
    
    async def batch_face_recognition(self, 
                                   face_images: List[Union[str, np.ndarray, bytes]],
                                   batch_size: Optional[int] = None) -> List[AsyncProcessingResult]:
        """
        Process multiple face images in optimized batches.
        
        Args:
            face_images: List of face images to process
            batch_size: Optional batch size override
            
        Returns:
            List of processing results
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        # Process in batches
        results = []
        for i in range(0, len(face_images), batch_size):
            batch = face_images[i:i + batch_size]
            batch_results = await self.process_batch(batch)
            results.extend(batch_results)
        
        return results
    
    async def _cleanup_processing_resources(self):
        """Clean up any temporary face processing resources."""
        # Clean up any temporary face crops or encodings
        pass
    
    async def get_face_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive face detection statistics.
        
        Returns:
            Statistics dictionary
        """
        base_metrics = await self.get_metrics()
        
        face_stats = {
            'known_faces_count': len(self.known_faces),
            'known_faces': list(self.known_faces.keys()),
            'detection_method': 'dnn' if self.dnn_net else 'haar',
            'recognition_enabled': self.enable_recognition,
            'landmarks_enabled': self.enable_landmarks,
            'confidence_threshold': self.confidence_threshold,
            'recognition_threshold': self.face_recognition_threshold
        }
        
        return {**base_metrics, **face_stats}


# Utility functions for face analysis
async def calculate_face_quality(face_image: np.ndarray) -> float:
    """
    Calculate face image quality score.
    
    Args:
        face_image: Cropped face image
        
    Returns:
        Quality score (0.0 to 1.0)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    
    # Calculate sharpness using Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Normalize sharpness score
    sharpness_score = min(laplacian_var / 500.0, 1.0)
    
    # Check brightness
    brightness = np.mean(gray)
    brightness_score = 1.0 - abs(brightness - 128) / 128.0
    
    # Calculate overall quality
    quality = (sharpness_score * 0.7) + (brightness_score * 0.3)
    
    return min(max(quality, 0.0), 1.0)


async def compare_faces(encoding1: np.ndarray, encoding2: np.ndarray) -> float:
    """
    Compare two face encodings and return similarity score.
    
    Args:
        encoding1: First face encoding
        encoding2: Second face encoding
        
    Returns:
        Similarity score (0.0 to 1.0)
    """
    # Normalize encodings
    norm1 = encoding1 / np.linalg.norm(encoding1)
    norm2 = encoding2 / np.linalg.norm(encoding2)
    
    # Calculate cosine similarity
    similarity = np.dot(norm1, norm2)
    
    # Convert to 0-1 range
    return (similarity + 1.0) / 2.0
