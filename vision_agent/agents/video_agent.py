"""
Video Analysis Agent
Implements video processing, tracking, and temporal analysis.
"""

import cv2
import numpy as np
from typing import Any, Dict, List, Optional, Union, Generator
import threading
import queue
from .base_agent import BaseAgent, ProcessingResult
from .face_agent import FaceAgent
from .object_agent import ObjectAgent


class VideoAgent(BaseAgent):
    """
    Agent for video analysis and processing.
    
    Features:
    - Frame-by-frame analysis
    - Object tracking across frames
    - Temporal pattern detection
    - Real-time streaming support
    - Video summarization
    - Integration with Face and Object agents
    """
    
    def __init__(self, 
                 device: Optional[str] = None,
                 model_path: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize Video Analysis Agent.
        
        Args:
            device: Target device (CPU/GPU)
            model_path: Path to video analysis models
            config: Configuration parameters including:
                - frame_skip: Number of frames to skip between analysis
                - max_frames: Maximum frames to process
                - track_objects: Enable object tracking
                - track_faces: Enable face tracking
                - output_format: 'summary' or 'detailed'
        """
        super().__init__(device, model_path, config)
        
        # Configuration defaults
        self.frame_skip = self.config.get('frame_skip', 1)
        self.max_frames = self.config.get('max_frames', 1000)
        self.track_objects = self.config.get('track_objects', True)
        self.track_faces = self.config.get('track_faces', True)
        self.output_format = self.config.get('output_format', 'summary')
        
        # Initialize sub-agents
        self.face_agent: Optional[FaceAgent] = None
        self.object_agent: Optional[ObjectAgent] = None
        
        # Tracking data
        self.trackers = {}
        self.track_id_counter = 0
    
    def initialize(self) -> bool:
        """
        Initialize video processing capabilities and sub-agents.
        
        Returns:
            True if initialization successful
        """
        try:
            self.logger.info("Initializing Video Agent...")
            
            # Initialize sub-agents if tracking is enabled
            if self.track_faces:
                self.face_agent = FaceAgent(device=self.device, config=self.config.get('face_config', {}))
                if not self.face_agent.initialize():
                    self.logger.warning("Face agent initialization failed")
                    self.track_faces = False
            
            if self.track_objects:
                self.object_agent = ObjectAgent(device=self.device, config=self.config.get('object_config', {}))
                if not self.object_agent.initialize():
                    self.logger.warning("Object agent initialization failed")
                    self.track_objects = False
            
            self._is_initialized = True
            self.logger.info(f"Video Agent initialized on {self.device}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Video Agent: {str(e)}")
            return False
    
    def process(self, input_data: Any) -> ProcessingResult:
        """
        Process video for analysis and tracking.
        
        Args:
            input_data: Video file path, URL, or camera index
            
        Returns:
            ProcessingResult with video analysis results
        """
        if not self._is_initialized:
            if not self.initialize():
                return ProcessingResult(
                    success=False,
                    data={},
                    error="Agent not initialized"
                )
        
        try:
            # Open video source
            cap = cv2.VideoCapture(input_data)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video source: {input_data}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Process video
            result, inference_time = self._measure_inference_time(
                self._analyze_video, cap
            )
            
            cap.release()
            
            # Add video metadata
            result['video_info'] = {
                'fps': fps,
                'total_frames': frame_count,
                'width': width,
                'height': height,
                'duration_seconds': frame_count / fps if fps > 0 else 0
            }
            
            return ProcessingResult(
                success=True,
                data=result,
                inference_time=inference_time,
                metadata={
                    'frames_processed': result.get('frames_analyzed', 0),
                    'output_format': self.output_format
                }
            )
            
        except Exception as e:
            self.logger.error(f"Video processing error: {str(e)}")
            return ProcessingResult(
                success=False,
                data={},
                error=str(e)
            )
    
    def _analyze_video(self, cap: cv2.VideoCapture) -> Dict[str, Any]:
        """
        Analyze video frame by frame.
        
        Args:
            cap: OpenCV VideoCapture object
            
        Returns:
            Dictionary with video analysis results
        """
        frame_results = []
        frame_number = 0
        frames_analyzed = 0
        
        # Tracking data
        object_tracks = {}
        face_tracks = {}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames if configured
            if frame_number % (self.frame_skip + 1) != 0:
                frame_number += 1
                continue
            
            # Stop if max frames reached
            if frames_analyzed >= self.max_frames:
                break
            
            frame_data = {
                'frame_number': frame_number,
                'timestamp': frame_number / cap.get(cv2.CAP_PROP_FPS)
            }
            
            # Object detection
            if self.track_objects and self.object_agent:
                obj_result = self.object_agent.process(frame)
                if obj_result.success:
                    frame_data['objects'] = obj_result.data['detections']
                    # Update object tracks
                    self._update_object_tracks(obj_result.data['detections'], frame_number)
            
            # Face detection
            if self.track_faces and self.face_agent:
                face_result = self.face_agent.process(frame)
                if face_result.success:
                    frame_data['faces'] = face_result.data['faces']
                    # Update face tracks
                    self._update_face_tracks(face_result.data['faces'], frame_number)
            
            frame_results.append(frame_data)
            frames_analyzed += 1
            frame_number += 1
        
        # Compile final results
        if self.output_format == 'summary':
            return self._generate_video_summary(frame_results, object_tracks, face_tracks)
        else:
            return {
                'frames': frame_results,
                'frames_analyzed': frames_analyzed,
                'object_tracks': object_tracks,
                'face_tracks': face_tracks
            }
    
    def _update_object_tracks(self, detections: List[Dict], frame_number: int) -> None:
        """
        Update object tracking across frames.
        
        Args:
            detections: Object detections for current frame
            frame_number: Current frame number
        """
        # Simple centroid-based tracking (can be enhanced with more sophisticated methods)
        for detection in detections:
            center = detection['center']
            class_name = detection['class_name']
            
            # Find closest existing track
            best_track_id = None
            min_distance = float('inf')
            
            for track_id, track_data in self.trackers.items():
                if track_data['class_name'] == class_name:
                    last_center = track_data['last_center']
                    distance = np.sqrt((center['x'] - last_center['x'])**2 + 
                                     (center['y'] - last_center['y'])**2)
                    
                    if distance < min_distance and distance < 100:  # Distance threshold
                        min_distance = distance
                        best_track_id = track_id
            
            if best_track_id:
                # Update existing track
                self.trackers[best_track_id]['last_center'] = center
                self.trackers[best_track_id]['last_frame'] = frame_number
                self.trackers[best_track_id]['frame_count'] += 1
                detection['track_id'] = best_track_id
            else:
                # Create new track
                track_id = self.track_id_counter
                self.trackers[track_id] = {
                    'class_name': class_name,
                    'first_frame': frame_number,
                    'last_frame': frame_number,
                    'last_center': center,
                    'frame_count': 1
                }
                detection['track_id'] = track_id
                self.track_id_counter += 1
    
    def _update_face_tracks(self, faces: List[Dict], frame_number: int) -> None:
        """
        Update face tracking across frames.
        
        Args:
            faces: Face detections for current frame
            frame_number: Current frame number
        """
        # Similar tracking logic for faces
        for face in faces:
            bbox = face['bounding_box']
            center = {
                'x': (bbox['left'] + bbox['right']) // 2,
                'y': (bbox['top'] + bbox['bottom']) // 2
            }
            
            # Find closest face track (could use face encodings for better matching)
            best_track_id = None
            min_distance = float('inf')
            
            for track_id, track_data in self.trackers.items():
                if track_data.get('type') == 'face':
                    last_center = track_data['last_center']
                    distance = np.sqrt((center['x'] - last_center['x'])**2 + 
                                     (center['y'] - last_center['y'])**2)
                    
                    if distance < min_distance and distance < 80:
                        min_distance = distance
                        best_track_id = track_id
            
            if best_track_id:
                self.trackers[best_track_id]['last_center'] = center
                self.trackers[best_track_id]['last_frame'] = frame_number
                self.trackers[best_track_id]['frame_count'] += 1
                face['track_id'] = best_track_id
            else:
                track_id = self.track_id_counter
                self.trackers[track_id] = {
                    'type': 'face',
                    'name': face['recognition']['name'],
                    'first_frame': frame_number,
                    'last_frame': frame_number,
                    'last_center': center,
                    'frame_count': 1
                }
                face['track_id'] = track_id
                self.track_id_counter += 1
    
    def _generate_video_summary(self, frame_results: List[Dict], 
                               object_tracks: Dict, face_tracks: Dict) -> Dict[str, Any]:
        """
        Generate a summary of the entire video analysis.
        
        Args:
            frame_results: Results from all analyzed frames
            object_tracks: Object tracking data
            face_tracks: Face tracking data
            
        Returns:
            Video summary dictionary
        """
        # Aggregate object statistics
        all_objects = []
        for frame in frame_results:
            if 'objects' in frame:
                all_objects.extend(frame['objects'])
        
        object_summary = {}
        for obj in all_objects:
            class_name = obj['class_name']
            if class_name not in object_summary:
                object_summary[class_name] = {
                    'count': 0,
                    'avg_confidence': 0.0,
                    'confidences': []
                }
            object_summary[class_name]['count'] += 1
            object_summary[class_name]['confidences'].append(obj['confidence'])
        
        # Calculate average confidences
        for class_name in object_summary:
            confidences = object_summary[class_name]['confidences']
            object_summary[class_name]['avg_confidence'] = sum(confidences) / len(confidences)
            del object_summary[class_name]['confidences']  # Remove raw data
        
        # Aggregate face statistics
        all_faces = []
        for frame in frame_results:
            if 'faces' in frame:
                all_faces.extend(frame['faces'])
        
        face_summary = {}
        for face in all_faces:
            name = face['recognition']['name']
            if name not in face_summary:
                face_summary[name] = {
                    'appearances': 0,
                    'avg_confidence': 0.0,
                    'confidences': []
                }
            face_summary[name]['appearances'] += 1
            face_summary[name]['confidences'].append(face['recognition']['confidence'])
        
        # Calculate average confidences for faces
        for name in face_summary:
            confidences = face_summary[name]['confidences']
            face_summary[name]['avg_confidence'] = sum(confidences) / len(confidences)
            del face_summary[name]['confidences']
        
        return {
            'summary': {
                'frames_analyzed': len(frame_results),
                'unique_objects': len(object_summary),
                'unique_faces': len(face_summary),
                'total_object_detections': len(all_objects),
                'total_face_detections': len(all_faces)
            },
            'object_summary': object_summary,
            'face_summary': face_summary,
            'tracking_summary': {
                'object_tracks': len([t for t in self.trackers.values() if t.get('type') != 'face']),
                'face_tracks': len([t for t in self.trackers.values() if t.get('type') == 'face'])
            },
            'frames_analyzed': len(frame_results)
        }
    
    def process_stream(self, video_source: Any) -> Generator[ProcessingResult, None, None]:
        """
        Process video stream in real-time with generator pattern.
        
        Args:
            video_source: Video source (file path, URL, or camera index)
            
        Yields:
            ProcessingResult for each processed frame
        """
        if not self._is_initialized:
            if not self.initialize():
                yield ProcessingResult(
                    success=False,
                    data={},
                    error="Agent not initialized"
                )
                return
        
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            yield ProcessingResult(
                success=False,
                data={},
                error=f"Could not open video source: {video_source}"
            )
            return
        
        frame_number = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames if configured
                if frame_number % (self.frame_skip + 1) != 0:
                    frame_number += 1
                    continue
                
                # Process single frame
                frame_result = self._process_single_frame(frame, frame_number)
                
                yield ProcessingResult(
                    success=True,
                    data=frame_result,
                    metadata={
                        'frame_number': frame_number,
                        'stream_source': str(video_source)
                    }
                )
                
                frame_number += 1
                
        except Exception as e:
            yield ProcessingResult(
                success=False,
                data={},
                error=str(e)
            )
        finally:
            cap.release()
    
    def _process_single_frame(self, frame: np.ndarray, frame_number: int) -> Dict[str, Any]:
        """
        Process a single video frame.
        
        Args:
            frame: Video frame as numpy array
            frame_number: Frame number in sequence
            
        Returns:
            Dictionary with frame analysis results
        """
        frame_data = {
            'frame_number': frame_number,
            'frame_shape': frame.shape
        }
        
        # Object detection
        if self.track_objects and self.object_agent:
            obj_result = self.object_agent.process(frame)
            if obj_result.success:
                frame_data['objects'] = obj_result.data['detections']
                frame_data['object_count'] = obj_result.data['detection_count']
        
        # Face detection
        if self.track_faces and self.face_agent:
            face_result = self.face_agent.process(frame)
            if face_result.success:
                frame_data['faces'] = face_result.data['faces']
                frame_data['face_count'] = face_result.data['face_count']
        
        return frame_data
    
    def extract_frames(self, video_path: str, 
                      output_dir: str,
                      interval_seconds: float = 1.0) -> ProcessingResult:
        """
        Extract frames from video at specified intervals.
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save extracted frames
            interval_seconds: Time interval between extracted frames
            
        Returns:
            ProcessingResult with extraction results
        """
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(fps * interval_seconds)
            
            frame_number = 0
            extracted_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_number % frame_interval == 0:
                    frame_filename = f"frame_{frame_number:06d}.jpg"
                    frame_path = os.path.join(output_dir, frame_filename)
                    cv2.imwrite(frame_path, frame)
                    extracted_count += 1
                
                frame_number += 1
            
            cap.release()
            
            return ProcessingResult(
                success=True,
                data={
                    'extracted_frames': extracted_count,
                    'total_frames': frame_number,
                    'output_directory': output_dir,
                    'interval_seconds': interval_seconds
                }
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                data={},
                error=str(e)
            )
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Get detailed information about a video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video properties
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {'error': f'Could not open video: {video_path}'}
        
        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'codec': int(cap.get(cv2.CAP_PROP_FOURCC)),
            'duration_seconds': 0
        }
        
        if info['fps'] > 0:
            info['duration_seconds'] = info['frame_count'] / info['fps']
        
        cap.release()
        return info
