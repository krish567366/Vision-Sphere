"""
Enhanced Async Object Detection Agent with YOLOv8 and Performance Optimizations
Production-ready object detection with advanced tracking and batch processing.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import cv2
import numpy as np

from .async_base_agent import AsyncBaseAgent, AsyncProcessingResult
from ..utils.caching import cached_tool_call
from ..utils.tracing import traced_operation, SpanType
from ..utils.streaming import publish_detection_event, EventType


class AsyncObjectAgent(AsyncBaseAgent):
    """
    Enhanced async object detection agent using YOLOv8.
    
    Features:
    - YOLOv8 object detection with multiple model sizes
    - Real-time object tracking
    - Class filtering and confidence thresholding
    - Batch processing optimization
    - Advanced caching and tracing
    - Stream processing support
    """
    
    def __init__(self, 
                 device: Optional[str] = None,
                 model_path: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the async object detection agent.
        
        Args:
            device: Target device for processing
            model_path: Path to YOLO model
            config: Agent configuration
        """
        super().__init__("AsyncObjectAgent", device, model_path, config)
        
        # Ensure config is not None
        config = config or {}
        
        # Model configuration
        self.model_name = config.get('model_name', 'yolov8s.pt')
        self.model_size = config.get('model_size', 640)
        
        # Detection parameters
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.iou_threshold = config.get('iou_threshold', 0.45)
        self.max_detections = config.get('max_detections', 100)
        
        # Class filtering
        self.allowed_classes = config.get('allowed_classes', None)  # None = all classes
        self.ignored_classes = config.get('ignored_classes', [])
        
        # Tracking parameters
        self.enable_tracking = config.get('enable_tracking', True)
        self.max_tracking_age = config.get('max_tracking_age', 30)
        
        # YOLO model
        self.model = None
        self.class_names = []
        
        # Tracking state
        self.trackers: Dict[int, Dict[str, Any]] = {}
        self.next_track_id = 0
        
        # Logger is already set by parent AsyncBaseAgent.__init__()
    
    async def _initialize_model(self) -> bool:
        """
        Initialize YOLO model for object detection.
        
        Returns:
            True if initialization successful
        """
        try:
            # Import YOLO
            try:
                from ultralytics import YOLO
            except ImportError:
                self.logger.error("ultralytics not installed. Install with: pip install ultralytics")
                return False
            
            # Load model
            model_path = self.model_path or self.model_name
            self.model = YOLO(model_path)
            
            # Set device
            if self.device == 'cuda':
                self.model.to('cuda')
            
            # Get class names
            self.class_names = list(self.model.names.values())
            
            self.logger.info(f"YOLO model loaded: {model_path}")
            self.logger.info(f"Available classes: {len(self.class_names)}")
            self.logger.info(f"Device: {self.device}")
            
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to initialize YOLO model: {e}")
            return False
    
    @cached_tool_call(expire_time=900)  # 15 minutes cache
    async def _detect_objects_yolo(self, 
                                  image: np.ndarray,
                                  trace_id: str) -> List[Dict[str, Any]]:
        """
        Detect objects using YOLO with caching.
        
        Args:
            image: Input image as numpy array
            trace_id: Trace ID for monitoring
            
        Returns:
            List of detected objects
        """
        async with traced_operation(
            "object_detection.yolo",
            SpanType.INFERENCE,
            trace_id=trace_id
        ) as span:
            
            h, w = image.shape[:2]
            span.set_attribute("image.height", h)
            span.set_attribute("image.width", w)
            span.set_attribute("model.confidence_threshold", self.confidence_threshold)
            span.set_attribute("model.iou_threshold", self.iou_threshold)
            
            # Run inference
            results = self.model.predict(
                image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                verbose=False
            )
            
            detections = []
            
            if results and len(results) > 0:
                result = results[0]  # First result
                
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                        class_name = self.class_names[cls_id]
                        
                        # Apply class filtering
                        if self.allowed_classes and class_name not in self.allowed_classes:
                            continue
                        if class_name in self.ignored_classes:
                            continue
                        
                        x1, y1, x2, y2 = box.astype(int)
                        
                        detection = {
                            'detection_id': i,
                            'class_name': class_name,
                            'class_id': int(cls_id),
                            'confidence': float(conf),
                            'bounding_box': {
                                'x1': int(x1), 'y1': int(y1),
                                'x2': int(x2), 'y2': int(y2)
                            },
                            'area': int((x2 - x1) * (y2 - y1)),
                            'center': {
                                'x': int((x1 + x2) / 2),
                                'y': int((y1 + y2) / 2)
                            }
                        }
                        
                        detections.append(detection)
            
            span.set_attribute("detections.count", len(detections))
            return detections
    
    async def _update_tracking(self, 
                              detections: List[Dict[str, Any]],
                              trace_id: str) -> List[Dict[str, Any]]:
        """
        Update object tracking for detections.
        
        Args:
            detections: List of current frame detections
            trace_id: Trace ID for monitoring
            
        Returns:
            Detections with tracking IDs
        """
        if not self.enable_tracking:
            return detections
        
        async with traced_operation(
            "object_tracking.update",
            SpanType.TRACKING,
            trace_id=trace_id
        ) as span:
            
            # Simple tracking based on IoU overlap
            tracked_detections = []
            
            for detection in detections:
                bbox = detection['bounding_box']
                center = detection['center']
                class_name = detection['class_name']
                
                # Find best matching tracker
                best_tracker_id = None
                best_iou = 0.0
                
                for tracker_id, tracker in self.trackers.items():
                    if tracker['class_name'] != class_name:
                        continue
                    
                    # Calculate IoU with last known position
                    iou = self._calculate_iou(bbox, tracker['last_bbox'])
                    
                    if iou > best_iou and iou > 0.3:  # Minimum IoU threshold
                        best_iou = iou
                        best_tracker_id = tracker_id
                
                if best_tracker_id is not None:
                    # Update existing tracker
                    self.trackers[best_tracker_id].update({
                        'last_bbox': bbox,
                        'last_center': center,
                        'last_seen': time.time(),
                        'age': self.trackers[best_tracker_id]['age'] + 1
                    })
                    
                    detection['track_id'] = best_tracker_id
                    detection['track_age'] = self.trackers[best_tracker_id]['age']
                
                else:
                    # Create new tracker
                    track_id = self.next_track_id
                    self.next_track_id += 1
                    
                    self.trackers[track_id] = {
                        'class_name': class_name,
                        'last_bbox': bbox,
                        'last_center': center,
                        'last_seen': time.time(),
                        'age': 1,
                        'created_at': time.time()
                    }
                    
                    detection['track_id'] = track_id
                    detection['track_age'] = 1
                
                tracked_detections.append(detection)
            
            # Remove old trackers
            current_time = time.time()
            expired_trackers = [
                tid for tid, tracker in self.trackers.items()
                if current_time - tracker['last_seen'] > self.max_tracking_age
            ]
            
            for tid in expired_trackers:
                del self.trackers[tid]
            
            span.set_attribute("tracking.active_trackers", len(self.trackers))
            span.set_attribute("tracking.expired_trackers", len(expired_trackers))
            
            return tracked_detections
    
    def _calculate_iou(self, bbox1: Dict[str, int], bbox2: Dict[str, int]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            bbox1: First bounding box
            bbox2: Second bounding box
            
        Returns:
            IoU score
        """
        # Calculate intersection
        x1 = max(bbox1['x1'], bbox2['x1'])
        y1 = max(bbox1['y1'], bbox2['y1'])
        x2 = min(bbox1['x2'], bbox2['x2'])
        y2 = min(bbox1['y2'], bbox2['y2'])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = (bbox1['x2'] - bbox1['x1']) * (bbox1['y2'] - bbox1['y1'])
        area2 = (bbox2['x2'] - bbox2['x1']) * (bbox2['y2'] - bbox2['y1'])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    async def _process_internal(self, input_data: Any, trace_id: str) -> Dict[str, Any]:
        """
        Internal object detection processing implementation.
        
        Args:
            input_data: Image input data
            trace_id: Trace ID for monitoring
            
        Returns:
            Processing results dictionary
        """
        # Preprocess image
        image = await self._preprocess_image_async(input_data)
        
        # Detect objects
        detections = await self._detect_objects_yolo(image, trace_id)
        
        # Update tracking if enabled
        if self.enable_tracking:
            detections = await self._update_tracking(detections, trace_id)
        
        # Publish detection events
        for detection in detections:
            await publish_detection_event(
                'object',
                trace_id,
                {
                    'class': detection['class_name'],
                    'confidence': detection['confidence'],
                    'track_id': detection.get('track_id')
                }
            )
        
        # Calculate summary statistics
        class_counts = {}
        total_area = 0
        avg_confidence = 0.0
        
        for detection in detections:
            class_name = detection['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            total_area += detection['area']
            avg_confidence += detection['confidence']
        
        avg_confidence = avg_confidence / len(detections) if detections else 0.0
        
        return {
            'detections': detections,
            'detection_count': len(detections),
            'class_counts': class_counts,
            'avg_confidence': avg_confidence,
            'total_detection_area': total_area,
            'image_dimensions': {'height': image.shape[0], 'width': image.shape[1]},
            'tracking_enabled': self.enable_tracking,
            'active_tracks': len(self.trackers) if self.enable_tracking else 0
        }
    
    async def detect_objects_in_region(self, 
                                     image_input: Union[str, np.ndarray, bytes],
                                     region: Dict[str, int],
                                     trace_id: Optional[str] = None) -> AsyncProcessingResult:
        """
        Detect objects within a specific region of the image.
        
        Args:
            image_input: Input image
            region: Region of interest {'x1', 'y1', 'x2', 'y2'}
            trace_id: Optional trace ID
            
        Returns:
            Processing result for the region
        """
        if trace_id is None:
            trace_id = f"region_detection_{time.time()}"
        
        # Preprocess full image
        image = await self._preprocess_image_async(image_input)
        
        # Extract region
        x1, y1, x2, y2 = region['x1'], region['y1'], region['x2'], region['y2']
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return AsyncProcessingResult(
                success=False,
                data={},
                error="Invalid region specified",
                trace_id=trace_id
            )
        
        # Process region
        result = await self.process(roi, trace_id=trace_id)
        
        if result.success:
            # Adjust coordinates back to full image space
            for detection in result.data.get('detections', []):
                bbox = detection['bounding_box']
                bbox['x1'] += x1
                bbox['y1'] += y1
                bbox['x2'] += x1
                bbox['y2'] += y1
                
                detection['center']['x'] += x1
                detection['center']['y'] += y1
        
        return result
    
    async def track_objects_in_video(self, 
                                   video_source: Union[str, int],
                                   callback_func: Optional[callable] = None) -> AsyncProcessingResult:
        """
        Track objects throughout a video with temporal consistency.
        
        Args:
            video_source: Video file path or camera index
            callback_func: Optional callback for each frame result
            
        Returns:
            Video processing result with tracking summary
        """
        trace_id = f"video_tracking_{time.time()}"
        
        async with traced_operation(
            "object_tracking.video",
            SpanType.STREAMING,
            trace_id=trace_id
        ) as span:
            
            try:
                cap = cv2.VideoCapture(video_source)
                if not cap.isOpened():
                    raise ValueError(f"Could not open video source: {video_source}")
                
                frame_count = 0
                total_detections = 0
                processing_times = []
                track_histories: Dict[int, List[Dict[str, Any]]] = {}
                
                span.set_attribute("video.source", str(video_source))
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_start = time.perf_counter()
                    
                    # Process frame with tracking
                    frame_trace_id = f"{trace_id}_frame_{frame_count}"
                    result = await self.process(frame, trace_id=frame_trace_id)
                    
                    frame_end = time.perf_counter()
                    frame_time = (frame_end - frame_start) * 1000
                    processing_times.append(frame_time)
                    
                    if result.success:
                        detections = result.data.get('detections', [])
                        total_detections += len(detections)
                        
                        # Update tracking histories
                        for detection in detections:
                            track_id = detection.get('track_id')
                            if track_id is not None:
                                if track_id not in track_histories:
                                    track_histories[track_id] = []
                                
                                track_histories[track_id].append({
                                    'frame': frame_count,
                                    'timestamp': time.time(),
                                    'bbox': detection['bounding_box'],
                                    'confidence': detection['confidence'],
                                    'class_name': detection['class_name']
                                })
                        
                        # Call callback if provided
                        if callback_func:
                            await callback_func(frame_count, frame, result)
                        
                        # Publish streaming event
                        await publish_detection_event(
                            'video_tracking',
                            trace_id,
                            {
                                'frame': frame_count,
                                'detections': len(detections),
                                'active_tracks': len(self.trackers),
                                'processing_time': frame_time
                            }
                        )
                    
                    frame_count += 1
                    
                    # Allow other tasks to run
                    await asyncio.sleep(0.001)
                
                cap.release()
                
                # Generate tracking summary
                track_summary = self._generate_track_summary(track_histories)
                
                avg_processing_time = np.mean(processing_times) if processing_times else 0
                fps = frame_count / (sum(processing_times) / 1000) if processing_times else 0
                
                span.set_attribute("video.total_frames", frame_count)
                span.set_attribute("video.total_detections", total_detections)
                span.set_attribute("video.unique_tracks", len(track_histories))
                span.set_attribute("video.avg_processing_time", avg_processing_time)
                span.set_attribute("video.fps", fps)
                
                return AsyncProcessingResult(
                    success=True,
                    data={
                        'video_summary': {
                            'total_frames': frame_count,
                            'total_detections': total_detections,
                            'unique_tracks': len(track_histories),
                            'avg_processing_time_ms': avg_processing_time,
                            'fps': fps
                        },
                        'track_summary': track_summary,
                        'track_histories': track_histories
                    },
                    trace_id=trace_id,
                    metadata={'video_processing': True}
                )
            
            except Exception as e:
                error_msg = f"Video tracking failed: {str(e)}"
                span.set_attribute("error", error_msg)
                return AsyncProcessingResult(
                    success=False,
                    data={},
                    error=error_msg,
                    trace_id=trace_id
                )
    
    def _generate_track_summary(self, track_histories: Dict[int, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Generate summary statistics for object tracks.
        
        Args:
            track_histories: Dictionary of track histories
            
        Returns:
            Track summary statistics
        """
        summary = {
            'total_tracks': len(track_histories),
            'track_durations': [],
            'track_classes': {},
            'long_lived_tracks': 0,
            'avg_track_confidence': 0.0
        }
        
        total_confidence = 0.0
        confidence_count = 0
        
        for track_id, history in track_histories.items():
            if not history:
                continue
            
            # Track duration
            duration = len(history)
            summary['track_durations'].append(duration)
            
            # Track class
            class_name = history[0]['class_name']
            summary['track_classes'][class_name] = summary['track_classes'].get(class_name, 0) + 1
            
            # Long-lived tracks (more than 10 frames)
            if duration > 10:
                summary['long_lived_tracks'] += 1
            
            # Average confidence
            track_confidences = [entry['confidence'] for entry in history]
            total_confidence += sum(track_confidences)
            confidence_count += len(track_confidences)
        
        # Calculate averages
        if summary['track_durations']:
            summary['avg_track_duration'] = np.mean(summary['track_durations'])
            summary['max_track_duration'] = max(summary['track_durations'])
        
        if confidence_count > 0:
            summary['avg_track_confidence'] = total_confidence / confidence_count
        
        return summary
    
    async def get_class_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about detected object classes.
        
        Returns:
            Class detection statistics
        """
        base_metrics = await self.get_metrics()
        
        object_stats = {
            'available_classes': self.class_names,
            'total_classes': len(self.class_names),
            'allowed_classes': self.allowed_classes,
            'ignored_classes': self.ignored_classes,
            'model_name': self.model_name,
            'model_size': self.model_size,
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'tracking_enabled': self.enable_tracking,
            'active_trackers': len(self.trackers)
        }
        
        return {**base_metrics, **object_stats}
    
    async def clear_tracking_state(self):
        """Clear all tracking state."""
        self.trackers.clear()
        self.next_track_id = 0
        self.logger.info("Tracking state cleared")
    
    async def filter_detections_by_area(self, 
                                      detections: List[Dict[str, Any]],
                                      min_area: int = 100,
                                      max_area: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Filter detections by bounding box area.
        
        Args:
            detections: List of detections
            min_area: Minimum area threshold
            max_area: Maximum area threshold (None for no limit)
            
        Returns:
            Filtered detections
        """
        filtered = []
        
        for detection in detections:
            area = detection['area']
            
            if area >= min_area:
                if max_area is None or area <= max_area:
                    filtered.append(detection)
        
        return filtered
    
    async def get_detection_heatmap(self, 
                                  video_source: Union[str, int],
                                  output_size: Tuple[int, int] = (640, 480)) -> np.ndarray:
        """
        Generate a heatmap of object detections across video frames.
        
        Args:
            video_source: Video file path or camera index
            output_size: Output heatmap dimensions
            
        Returns:
            Heatmap as numpy array
        """
        heatmap = np.zeros(output_size[::-1], dtype=np.float32)  # (height, width)
        
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {video_source}")
        
        frame_count = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                result = await self.process(frame)
                
                if result.success:
                    detections = result.data.get('detections', [])
                    
                    # Add detections to heatmap
                    for detection in detections:
                        bbox = detection['bounding_box']
                        confidence = detection['confidence']
                        
                        # Scale coordinates to heatmap size
                        h_scale = output_size[1] / frame.shape[0]
                        w_scale = output_size[0] / frame.shape[1]
                        
                        x1 = int(bbox['x1'] * w_scale)
                        y1 = int(bbox['y1'] * h_scale)
                        x2 = int(bbox['x2'] * w_scale)
                        y2 = int(bbox['y2'] * h_scale)
                        
                        # Add weighted detection to heatmap
                        heatmap[y1:y2, x1:x2] += confidence
                
                frame_count += 1
                
                # Limit processing for performance
                if frame_count % 10 == 0:
                    await asyncio.sleep(0.001)
        
        finally:
            cap.release()
        
        # Normalize heatmap
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return heatmap
    
    async def _cleanup_processing_resources(self):
        """Clean up object detection resources."""
        # Clear old tracking data
        current_time = time.time()
        expired_trackers = [
            tid for tid, tracker in self.trackers.items()
            if current_time - tracker['last_seen'] > self.max_tracking_age * 2
        ]
        
        for tid in expired_trackers:
            del self.trackers[tid]
