"""
Enhanced Async Video Processing Agent with Advanced Analytics
Production-ready video analysis with multi-modal integration and streaming support.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple, Union, AsyncGenerator
import logging
import cv2
import numpy as np
from dataclasses import dataclass

from .async_base_agent import AsyncBaseAgent, AsyncProcessingResult
from .async_face_agent import AsyncFaceAgent
from .async_object_agent import AsyncObjectAgent
from ..utils.caching import cached_tool_call
from ..utils.tracing import traced_operation, SpanType
from ..utils.streaming import publish_detection_event, event_manager, EventType


@dataclass
class VideoAnalysisResult:
    """Comprehensive video analysis result."""
    frame_count: int
    duration_seconds: float
    fps: float
    resolution: Tuple[int, int]
    face_analytics: Dict[str, Any]
    object_analytics: Dict[str, Any]
    temporal_analytics: Dict[str, Any]
    scene_changes: List[int]
    processing_time_ms: float
    trace_id: str


class AsyncVideoAgent(AsyncBaseAgent):
    """
    Enhanced async video processing agent with multi-modal analytics.
    
    Features:
    - Integrated face and object detection
    - Scene change detection
    - Temporal analysis and tracking
    - Real-time streaming processing
    - Advanced video analytics
    - Batch frame processing
    """
    
    def __init__(self, 
                 device: Optional[str] = None,
                 model_path: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the async video processing agent.
        
        Args:
            device: Target device for processing
            model_path: Path to model files
            config: Agent configuration
        """
        super().__init__("AsyncVideoAgent", device, model_path, config)
        
        # Ensure config is not None
        config = config or {}
        
        # Sub-agent configuration
        face_config = config.get('face_config', {})
        object_config = config.get('object_config', {})
        
        # Initialize sub-agents
        self.face_agent = AsyncFaceAgent(device, model_path, face_config)
        self.object_agent = AsyncObjectAgent(device, model_path, object_config)
        
        # Video processing parameters
        self.frame_skip = config.get('frame_skip', 1)
        self.max_frames = config.get('max_frames', None)
        self.scene_change_threshold = config.get('scene_change_threshold', 0.3)
        self.enable_scene_detection = config.get('enable_scene_detection', True)
        
        # Analytics configuration
        self.enable_face_analytics = config.get('enable_face_analytics', True)
        self.enable_object_analytics = config.get('enable_object_analytics', True)
        self.enable_temporal_analysis = config.get('enable_temporal_analysis', True)
        
        # Processing state
        self.previous_frame = None
        self.frame_features_history = []
        
        self.logger = logging.getLogger('AsyncVideoAgent')
    
    async def _initialize_model(self) -> bool:
        """
        Initialize video processing models and sub-agents.
        
        Returns:
            True if initialization successful
        """
        try:
            # Initialize sub-agents
            face_init_success = await self.face_agent.initialize()
            object_init_success = await self.object_agent.initialize()
            
            if not face_init_success:
                self.logger.warning("Face agent initialization failed")
                self.enable_face_analytics = False
            
            if not object_init_success:
                self.logger.warning("Object agent initialization failed")
                self.enable_object_analytics = False
            
            if not face_init_success and not object_init_success:
                self.logger.error("Both sub-agents failed to initialize")
                return False
            
            self.logger.info(f"Video agent initialized - Face: {face_init_success}, Object: {object_init_success}")
            return True
        
        except Exception as e:
            self.logger.error(f"Video agent initialization failed: {e}")
            return False
    
    async def _extract_frame_features(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract features from frame for scene change detection.
        
        Args:
            frame: Video frame
            
        Returns:
            Feature vector
        """
        # Convert to grayscale and resize for speed
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64))
        
        # Calculate histogram
        hist = cv2.calcHist([resized], [0], None, [256], [0, 256])
        
        # Normalize and flatten
        features = hist.flatten() / hist.sum()
        
        return features
    
    async def _detect_scene_change(self, current_frame: np.ndarray) -> Tuple[bool, float]:
        """
        Detect scene change between current and previous frame.
        
        Args:
            current_frame: Current video frame
            
        Returns:
            (is_scene_change, similarity_score)
        """
        if self.previous_frame is None:
            self.previous_frame = current_frame
            return False, 1.0
        
        # Extract features
        current_features = await self._extract_frame_features(current_frame)
        previous_features = await self._extract_frame_features(self.previous_frame)
        
        # Calculate similarity
        similarity = cv2.compareHist(
            current_features.astype(np.float32),
            previous_features.astype(np.float32),
            cv2.HISTCMP_CORREL
        )
        
        # Detect scene change
        is_scene_change = similarity < (1.0 - self.scene_change_threshold)
        
        self.previous_frame = current_frame
        
        return is_scene_change, similarity
    
    @cached_tool_call(expire_time=1800)  # 30 minutes cache
    async def _process_video_frame(self, 
                                  frame: np.ndarray,
                                  frame_number: int,
                                  trace_id: str) -> Dict[str, Any]:
        """
        Process a single video frame with multi-modal analysis.
        
        Args:
            frame: Video frame
            frame_number: Frame index
            trace_id: Trace ID for monitoring
            
        Returns:
            Frame analysis results
        """
        async with traced_operation(
            "video_frame.analysis",
            SpanType.INFERENCE,
            trace_id=trace_id
        ) as span:
            
            span.set_attribute("frame.number", frame_number)
            span.set_attribute("frame.height", frame.shape[0])
            span.set_attribute("frame.width", frame.shape[1])
            
            frame_result = {
                'frame_number': frame_number,
                'timestamp': time.time(),
                'scene_change': False,
                'similarity_score': 1.0,
                'face_results': None,
                'object_results': None
            }
            
            # Scene change detection
            if self.enable_scene_detection:
                scene_change, similarity = await self._detect_scene_change(frame)
                frame_result['scene_change'] = scene_change
                frame_result['similarity_score'] = similarity
                
                span.set_attribute("scene.change_detected", scene_change)
                span.set_attribute("scene.similarity_score", similarity)
            
            # Face analysis
            if self.enable_face_analytics:
                try:
                    face_result = await self.face_agent.process(
                        frame, 
                        trace_id=f"{trace_id}_face"
                    )
                    frame_result['face_results'] = face_result.to_dict()
                    span.set_attribute("face.detection_count", 
                                     face_result.data.get('face_count', 0))
                except Exception as e:
                    self.logger.warning(f"Face analysis failed for frame {frame_number}: {e}")
            
            # Object analysis
            if self.enable_object_analytics:
                try:
                    object_result = await self.object_agent.process(
                        frame,
                        trace_id=f"{trace_id}_object"
                    )
                    frame_result['object_results'] = object_result.to_dict()
                    span.set_attribute("object.detection_count",
                                     object_result.data.get('detection_count', 0))
                except Exception as e:
                    self.logger.warning(f"Object analysis failed for frame {frame_number}: {e}")
            
            return frame_result
    
    async def _process_internal(self, input_data: Any, trace_id: str) -> Dict[str, Any]:
        """
        Internal video processing implementation.
        
        Args:
            input_data: Video file path or numpy array
            trace_id: Trace ID for monitoring
            
        Returns:
            Comprehensive video analysis results
        """
        if isinstance(input_data, str):
            # Video file path
            return await self._process_video_file(input_data, trace_id)
        
        elif isinstance(input_data, np.ndarray):
            # Single frame
            frame_result = await self._process_video_frame(input_data, 0, trace_id)
            
            return {
                'frame_analysis': frame_result,
                'video_type': 'single_frame',
                'total_frames': 1
            }
        
        else:
            raise ValueError(f"Unsupported video input type: {type(input_data)}")
    
    async def _process_video_file(self, video_path: str, trace_id: str) -> Dict[str, Any]:
        """
        Process complete video file with comprehensive analytics.
        
        Args:
            video_path: Path to video file
            trace_id: Trace ID for monitoring
            
        Returns:
            Complete video analysis results
        """
        async with traced_operation(
            "video_file.complete_analysis",
            SpanType.VIDEO_PROCESSING,
            trace_id=trace_id
        ) as span:
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            try:
                # Get video properties
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = total_frames / fps if fps > 0 else 0
                
                span.set_attribute("video.path", video_path)
                span.set_attribute("video.total_frames", total_frames)
                span.set_attribute("video.fps", fps)
                span.set_attribute("video.duration", duration)
                span.set_attribute("video.resolution", f"{width}x{height}")
                
                # Limit frames if specified
                frames_to_process = min(total_frames, self.max_frames) if self.max_frames else total_frames
                
                # Process frames
                frame_results = []
                scene_changes = []
                processing_start = time.perf_counter()
                
                frame_number = 0
                processed_frames = 0
                
                # Create semaphore for concurrent frame processing
                frame_semaphore = asyncio.Semaphore(min(5, self.max_concurrent_requests))
                
                async def process_frame_async(frame, frame_idx):
                    async with frame_semaphore:
                        frame_trace_id = f"{trace_id}_frame_{frame_idx}"
                        return await self._process_video_frame(frame, frame_idx, frame_trace_id)
                
                # Process frames in batches
                batch_size = 10
                frame_batch = []
                frame_indices = []
                
                while cap.isOpened() and processed_frames < frames_to_process:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Skip frames if configured
                    if frame_number % (self.frame_skip + 1) != 0:
                        frame_number += 1
                        continue
                    
                    frame_batch.append(frame.copy())
                    frame_indices.append(frame_number)
                    
                    # Process batch when full
                    if len(frame_batch) >= batch_size:
                        tasks = [
                            process_frame_async(f, idx) 
                            for f, idx in zip(frame_batch, frame_indices)
                        ]
                        
                        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                        
                        for i, result in enumerate(batch_results):
                            if isinstance(result, Exception):
                                self.logger.warning(f"Frame {frame_indices[i]} processing failed: {result}")
                                continue
                            
                            frame_results.append(result)
                            
                            # Track scene changes
                            if result.get('scene_change', False):
                                scene_changes.append(frame_indices[i])
                        
                        # Clear batch
                        frame_batch = []
                        frame_indices = []
                        processed_frames += len(batch_results)
                        
                        # Publish progress
                        await event_manager.publish_event(
                            EventType.PROCESSING_PROGRESS,
                            {
                                'progress': processed_frames / frames_to_process,
                                'frames_processed': processed_frames,
                                'total_frames': frames_to_process
                            },
                            agent_name=self.agent_name,
                            trace_id=trace_id
                        )
                    
                    frame_number += 1
                
                # Process remaining frames
                if frame_batch:
                    tasks = [
                        process_frame_async(f, idx) 
                        for f, idx in zip(frame_batch, frame_indices)
                    ]
                    
                    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for i, result in enumerate(batch_results):
                        if not isinstance(result, Exception):
                            frame_results.append(result)
                            if result.get('scene_change', False):
                                scene_changes.append(frame_indices[i])
                
                processing_end = time.perf_counter()
                total_processing_time = (processing_end - processing_start) * 1000
                
                # Generate comprehensive analytics
                analytics = await self._generate_video_analytics(
                    frame_results, 
                    scene_changes,
                    trace_id
                )
                
                span.set_attribute("processing.total_time_ms", total_processing_time)
                span.set_attribute("processing.frames_analyzed", len(frame_results))
                span.set_attribute("analytics.scene_changes", len(scene_changes))
                
                return {
                    'video_info': {
                        'path': video_path,
                        'total_frames': total_frames,
                        'fps': fps,
                        'duration_seconds': duration,
                        'resolution': {'width': width, 'height': height},
                        'frames_analyzed': len(frame_results)
                    },
                    'analytics': analytics,
                    'scene_changes': scene_changes,
                    'frame_results': frame_results,
                    'processing_time_ms': total_processing_time,
                    'performance_metrics': {
                        'avg_frame_time': total_processing_time / len(frame_results) if frame_results else 0,
                        'processing_fps': len(frame_results) / (total_processing_time / 1000) if total_processing_time > 0 else 0
                    }
                }
            
            finally:
                cap.release()
    
    async def _generate_video_analytics(self, 
                                       frame_results: List[Dict[str, Any]],
                                       scene_changes: List[int],
                                       trace_id: str) -> Dict[str, Any]:
        """
        Generate comprehensive video analytics from frame results.
        
        Args:
            frame_results: Results from all processed frames
            scene_changes: List of frame numbers with scene changes
            trace_id: Trace ID for monitoring
            
        Returns:
            Comprehensive analytics dictionary
        """
        async with traced_operation(
            "video_analytics.generation",
            SpanType.ANALYTICS,
            trace_id=trace_id
        ) as span:
            
            analytics = {
                'face_analytics': {},
                'object_analytics': {},
                'temporal_analytics': {},
                'scene_analytics': {}
            }
            
            if not frame_results:
                return analytics
            
            # Face analytics
            if self.enable_face_analytics:
                analytics['face_analytics'] = await self._analyze_face_patterns(frame_results)
            
            # Object analytics
            if self.enable_object_analytics:
                analytics['object_analytics'] = await self._analyze_object_patterns(frame_results)
            
            # Temporal analytics
            if self.enable_temporal_analysis:
                analytics['temporal_analytics'] = await self._analyze_temporal_patterns(frame_results)
            
            # Scene analytics
            analytics['scene_analytics'] = {
                'total_scenes': len(scene_changes) + 1,
                'scene_changes': scene_changes,
                'avg_scene_duration': len(frame_results) / (len(scene_changes) + 1) if scene_changes else len(frame_results),
                'scene_change_frequency': len(scene_changes) / len(frame_results) if frame_results else 0
            }
            
            span.set_attribute("analytics.scenes_detected", len(scene_changes))
            span.set_attribute("analytics.frames_analyzed", len(frame_results))
            
            return analytics
    
    async def _analyze_face_patterns(self, frame_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze face detection patterns across video.
        
        Args:
            frame_results: Frame analysis results
            
        Returns:
            Face analytics
        """
        total_faces = 0
        face_frames = 0
        recognized_faces = {}
        face_confidences = []
        
        for frame_result in frame_results:
            face_data = frame_result.get('face_results', {})
            if isinstance(face_data, dict) and 'data' in face_data:
                faces = face_data['data'].get('faces', [])
                
                if faces:
                    face_frames += 1
                    total_faces += len(faces)
                    
                    for face in faces:
                        # Track confidences
                        face_confidences.append(face.get('confidence', 0))
                        
                        # Track recognized faces
                        recognition = face.get('recognition')
                        if recognition:
                            name = recognition['name']
                            recognized_faces[name] = recognized_faces.get(name, 0) + 1
        
        return {
            'total_faces_detected': total_faces,
            'frames_with_faces': face_frames,
            'face_presence_ratio': face_frames / len(frame_results) if frame_results else 0,
            'avg_faces_per_frame': total_faces / face_frames if face_frames > 0 else 0,
            'avg_face_confidence': np.mean(face_confidences) if face_confidences else 0,
            'recognized_faces': recognized_faces,
            'unique_recognized_people': len(recognized_faces)
        }
    
    async def _analyze_object_patterns(self, frame_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze object detection patterns across video.
        
        Args:
            frame_results: Frame analysis results
            
        Returns:
            Object analytics
        """
        total_objects = 0
        object_frames = 0
        class_counts = {}
        object_confidences = []
        track_durations = {}
        
        for frame_result in frame_results:
            object_data = frame_result.get('object_results', {})
            if isinstance(object_data, dict) and 'data' in object_data:
                detections = object_data['data'].get('detections', [])
                
                if detections:
                    object_frames += 1
                    total_objects += len(detections)
                    
                    for detection in detections:
                        # Track class counts
                        class_name = detection['class_name']
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
                        
                        # Track confidences
                        object_confidences.append(detection.get('confidence', 0))
                        
                        # Track durations
                        track_id = detection.get('track_id')
                        if track_id is not None:
                            track_durations[track_id] = detection.get('track_age', 1)
        
        return {
            'total_objects_detected': total_objects,
            'frames_with_objects': object_frames,
            'object_presence_ratio': object_frames / len(frame_results) if frame_results else 0,
            'avg_objects_per_frame': total_objects / object_frames if object_frames > 0 else 0,
            'avg_object_confidence': np.mean(object_confidences) if object_confidences else 0,
            'class_distribution': class_counts,
            'unique_classes': len(class_counts),
            'total_tracks': len(track_durations),
            'avg_track_duration': np.mean(list(track_durations.values())) if track_durations else 0,
            'max_track_duration': max(track_durations.values()) if track_durations else 0
        }
    
    async def _analyze_temporal_patterns(self, frame_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze temporal patterns in the video.
        
        Args:
            frame_results: Frame analysis results
            
        Returns:
            Temporal analytics
        """
        # Activity levels over time
        activity_scores = []
        face_counts = []
        object_counts = []
        
        for frame_result in frame_results:
            # Calculate activity score
            face_count = 0
            object_count = 0
            
            # Extract counts
            face_data = frame_result.get('face_results', {})
            if isinstance(face_data, dict) and 'data' in face_data:
                face_count = face_data['data'].get('face_count', 0)
            
            object_data = frame_result.get('object_results', {})
            if isinstance(object_data, dict) and 'data' in object_data:
                object_count = object_data['data'].get('detection_count', 0)
            
            face_counts.append(face_count)
            object_counts.append(object_count)
            
            # Activity score combines face and object presence
            activity_score = (face_count * 0.3) + (object_count * 0.7)
            activity_scores.append(activity_score)
        
        return {
            'activity_timeline': activity_scores,
            'face_count_timeline': face_counts,
            'object_count_timeline': object_counts,
            'avg_activity_score': np.mean(activity_scores) if activity_scores else 0,
            'peak_activity_frame': int(np.argmax(activity_scores)) if activity_scores else 0,
            'activity_variance': np.var(activity_scores) if activity_scores else 0
        }
    
    async def stream_video_analysis(self, 
                                  video_source: Union[str, int],
                                  callback_func: Optional[callable] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream video analysis results in real-time.
        
        Args:
            video_source: Video file path or camera index
            callback_func: Optional callback for each frame
            
        Yields:
            Real-time analysis updates
        """
        trace_id = f"stream_video_{time.time()}"
        
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            yield {"type": "error", "message": f"Could not open video source: {video_source}"}
            return
        
        try:
            frame_number = 0
            
            # Get video info
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            yield {
                "type": "video_info",
                "data": {
                    "fps": fps,
                    "resolution": {"width": width, "height": height}
                }
            }
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames if configured
                if frame_number % (self.frame_skip + 1) != 0:
                    frame_number += 1
                    continue
                
                # Process frame
                frame_start = time.perf_counter()
                frame_result = await self._process_video_frame(
                    frame, 
                    frame_number, 
                    f"{trace_id}_frame_{frame_number}"
                )
                frame_end = time.perf_counter()
                
                frame_time = (frame_end - frame_start) * 1000
                
                # Yield frame result
                yield {
                    "type": "frame_result",
                    "data": {
                        "frame_number": frame_number,
                        "processing_time_ms": frame_time,
                        "analysis": frame_result
                    }
                }
                
                # Call callback if provided
                if callback_func:
                    await callback_func(frame_number, frame, frame_result)
                
                frame_number += 1
                
                # Allow other tasks to run
                await asyncio.sleep(0.001)
        
        finally:
            cap.release()
            yield {"type": "stream_complete", "data": {"total_frames": frame_number}}
    
    async def extract_video_summary(self, 
                                  video_path: str,
                                  summary_frames: int = 10) -> AsyncProcessingResult:
        """
        Extract key frames and generate video summary.
        
        Args:
            video_path: Path to video file
            summary_frames: Number of key frames to extract
            
        Returns:
            Video summary with key frames
        """
        trace_id = f"video_summary_{time.time()}"
        
        async with traced_operation(
            "video_summary.extraction",
            SpanType.ANALYTICS,
            trace_id=trace_id
        ) as span:
            
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise ValueError(f"Could not open video file: {video_path}")
                
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Calculate frame intervals for key frame extraction
                frame_interval = max(1, total_frames // summary_frames)
                key_frames = []
                key_frame_analyses = []
                
                frame_number = 0
                
                while cap.isOpened() and len(key_frames) < summary_frames:
                    # Seek to next key frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                    ret, frame = cap.read()
                    
                    if not ret:
                        break
                    
                    # Analyze key frame
                    frame_analysis = await self._process_video_frame(
                        frame,
                        frame_number,
                        f"{trace_id}_keyframe_{len(key_frames)}"
                    )
                    
                    key_frames.append({
                        'frame_number': frame_number,
                        'timestamp': frame_number / cap.get(cv2.CAP_PROP_FPS),
                        'analysis': frame_analysis
                    })
                    key_frame_analyses.append(frame_analysis)
                    
                    frame_number += frame_interval
                
                cap.release()
                
                # Generate summary analytics
                summary_analytics = await self._generate_video_analytics(
                    key_frame_analyses,
                    [],  # No scene changes for summary
                    trace_id
                )
                
                span.set_attribute("summary.key_frames", len(key_frames))
                
                return AsyncProcessingResult(
                    success=True,
                    data={
                        'key_frames': key_frames,
                        'summary_analytics': summary_analytics,
                        'video_summary': {
                            'dominant_classes': list(summary_analytics['object_analytics'].get('class_distribution', {}).keys())[:5],
                            'avg_activity': summary_analytics['temporal_analytics'].get('avg_activity_score', 0),
                            'face_presence': summary_analytics['face_analytics'].get('face_presence_ratio', 0)
                        }
                    },
                    trace_id=trace_id,
                    metadata={'summary_extraction': True}
                )
            
            except Exception as e:
                error_msg = f"Video summary extraction failed: {str(e)}"
                span.set_attribute("error", error_msg)
                return AsyncProcessingResult(
                    success=False,
                    data={},
                    error=error_msg,
                    trace_id=trace_id
                )
    
    async def compare_videos(self, 
                           video1_path: str,
                           video2_path: str) -> AsyncProcessingResult:
        """
        Compare two videos for similarity analysis.
        
        Args:
            video1_path: Path to first video
            video2_path: Path to second video
            
        Returns:
            Video comparison results
        """
        trace_id = f"video_comparison_{time.time()}"
        
        async with traced_operation(
            "video_comparison.analysis",
            SpanType.ANALYTICS,
            trace_id=trace_id
        ) as span:
            
            try:
                # Process both videos concurrently
                video1_task = self.process(video1_path, trace_id=f"{trace_id}_video1")
                video2_task = self.process(video2_path, trace_id=f"{trace_id}_video2")
                
                video1_result, video2_result = await asyncio.gather(video1_task, video2_task)
                
                if not (video1_result.success and video2_result.success):
                    return AsyncProcessingResult(
                        success=False,
                        data={},
                        error="Failed to process one or both videos",
                        trace_id=trace_id
                    )
                
                # Compare analytics
                comparison = await self._compare_video_analytics(
                    video1_result.data['analytics'],
                    video2_result.data['analytics']
                )
                
                span.set_attribute("comparison.completed", True)
                
                return AsyncProcessingResult(
                    success=True,
                    data={
                        'video1_analysis': video1_result.data,
                        'video2_analysis': video2_result.data,
                        'comparison': comparison
                    },
                    trace_id=trace_id,
                    metadata={'video_comparison': True}
                )
            
            except Exception as e:
                error_msg = f"Video comparison failed: {str(e)}"
                span.set_attribute("error", error_msg)
                return AsyncProcessingResult(
                    success=False,
                    data={},
                    error=error_msg,
                    trace_id=trace_id
                )
    
    async def _compare_video_analytics(self, 
                                     analytics1: Dict[str, Any],
                                     analytics2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare analytics between two videos.
        
        Args:
            analytics1: Analytics from first video
            analytics2: Analytics from second video
            
        Returns:
            Comparison results
        """
        comparison = {
            'similarity_scores': {},
            'differences': {},
            'overall_similarity': 0.0
        }
        
        # Compare face analytics
        face1 = analytics1.get('face_analytics', {})
        face2 = analytics2.get('face_analytics', {})
        
        face_similarity = self._calculate_feature_similarity(
            face1.get('face_presence_ratio', 0),
            face2.get('face_presence_ratio', 0)
        )
        comparison['similarity_scores']['face_presence'] = face_similarity
        
        # Compare object analytics
        obj1 = analytics1.get('object_analytics', {})
        obj2 = analytics2.get('object_analytics', {})
        
        obj_similarity = self._calculate_feature_similarity(
            obj1.get('object_presence_ratio', 0),
            obj2.get('object_presence_ratio', 0)
        )
        comparison['similarity_scores']['object_presence'] = obj_similarity
        
        # Compare class distributions
        classes1 = set(obj1.get('class_distribution', {}).keys())
        classes2 = set(obj2.get('class_distribution', {}).keys())
        
        class_overlap = len(classes1.intersection(classes2)) / len(classes1.union(classes2)) if (classes1 or classes2) else 1.0
        comparison['similarity_scores']['class_overlap'] = class_overlap
        
        # Calculate overall similarity
        similarities = list(comparison['similarity_scores'].values())
        comparison['overall_similarity'] = np.mean(similarities) if similarities else 0.0
        
        return comparison
    
    def _calculate_feature_similarity(self, value1: float, value2: float) -> float:
        """Calculate similarity between two feature values."""
        if value1 == 0 and value2 == 0:
            return 1.0
        
        max_val = max(value1, value2)
        min_val = min(value1, value2)
        
        return min_val / max_val if max_val > 0 else 0.0
    
    async def _cleanup_processing_resources(self):
        """Clean up video processing resources."""
        # Reset frame state
        self.previous_frame = None
        self.frame_features_history.clear()
        
        # Cleanup sub-agents
        if hasattr(self.face_agent, '_cleanup_processing_resources'):
            await self.face_agent._cleanup_processing_resources()
        
        if hasattr(self.object_agent, '_cleanup_processing_resources'):
            await self.object_agent._cleanup_processing_resources()
    
    async def get_video_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive video processing statistics.
        
        Returns:
            Statistics dictionary
        """
        base_metrics = await self.get_metrics()
        
        video_stats = {
            'frame_skip': self.frame_skip,
            'max_frames': self.max_frames,
            'scene_detection_enabled': self.enable_scene_detection,
            'scene_change_threshold': self.scene_change_threshold,
            'face_analytics_enabled': self.enable_face_analytics,
            'object_analytics_enabled': self.enable_object_analytics,
            'temporal_analysis_enabled': self.enable_temporal_analysis
        }
        
        # Get sub-agent stats
        if self.enable_face_analytics:
            face_stats = await self.face_agent.get_metrics()
            video_stats['face_agent_stats'] = face_stats
        
        if self.enable_object_analytics:
            object_stats = await self.object_agent.get_metrics()
            video_stats['object_agent_stats'] = object_stats
        
        return {**base_metrics, **video_stats}
