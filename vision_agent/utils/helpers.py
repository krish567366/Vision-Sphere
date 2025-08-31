"""
Utility Functions and Helpers for VisionAgent Framework
Common functionality shared across agents and components.
"""

import os
import cv2
import numpy as np
import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import requests
from urllib.parse import urlparse
import tempfile
import shutil


def setup_logging(config: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """
    Set up logging configuration for the application.
    
    Args:
        config: Logging configuration dictionary
        
    Returns:
        Configured logger instance
    """
    # Default configuration
    log_config = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_path': None,
        'max_file_size_mb': 10,
        'backup_count': 5
    }
    
    if config:
        log_config.update(config)
    
    # Set up logging
    log_level = getattr(logging, log_config['level'].upper())
    logging.basicConfig(
        level=log_level,
        format=log_config['format']
    )
    
    logger = logging.getLogger('VisionAgent')
    
    # Add file handler if specified
    if log_config['file_path']:
        from logging.handlers import RotatingFileHandler
        
        # Ensure log directory exists
        log_dir = os.path.dirname(log_config['file_path'])
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_config['file_path'],
            maxBytes=log_config['max_file_size_mb'] * 1024 * 1024,
            backupCount=log_config['backup_count']
        )
        file_handler.setFormatter(logging.Formatter(log_config['format']))
        logger.addHandler(file_handler)
    
    return logger


def validate_image_input(input_data: Any) -> bool:
    """
    Validate if input data is a valid image format.
    
    Args:
        input_data: Input to validate
        
    Returns:
        True if valid image input
    """
    try:
        if isinstance(input_data, str):
            # File path or URL
            if os.path.isfile(input_data):
                return True
            elif urlparse(input_data).scheme in ['http', 'https']:
                return True
        elif isinstance(input_data, (np.ndarray, bytes)):
            return True
        elif hasattr(input_data, 'read'):
            # File-like object
            return True
    except:
        pass
    
    return False


def download_file(url: str, dest_path: str, chunk_size: int = 8192) -> bool:
    """
    Download a file from URL to local path.
    
    Args:
        url: URL to download from
        dest_path: Local destination path
        chunk_size: Download chunk size in bytes
        
    Returns:
        True if download successful
    """
    try:
        # Create destination directory if needed
        dest_dir = os.path.dirname(dest_path)
        if dest_dir:
            os.makedirs(dest_dir, exist_ok=True)
        
        # Download file
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
        
        return True
        
    except Exception as e:
        logging.getLogger('VisionAgent').error(f"Failed to download {url}: {str(e)}")
        return False


def get_file_hash(file_path: str, algorithm: str = 'md5') -> str:
    """
    Calculate hash of a file.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
        
    Returns:
        Hex digest of file hash
    """
    hash_func = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def load_image_from_url(url: str) -> Optional[np.ndarray]:
    """
    Load image from URL.
    
    Args:
        url: Image URL
        
    Returns:
        Image as numpy array or None if failed
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Convert to numpy array
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        return image
        
    except Exception as e:
        logging.getLogger('VisionAgent').error(f"Failed to load image from {url}: {str(e)}")
        return None


def resize_image(image: np.ndarray, 
                target_size: Union[int, Tuple[int, int]], 
                maintain_aspect: bool = True) -> np.ndarray:
    """
    Resize image to target size.
    
    Args:
        image: Input image
        target_size: Target size as (width, height) or single dimension
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized image
    """
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    
    if maintain_aspect:
        # Calculate scaling factor
        h, w = image.shape[:2]
        scale = min(target_size[0] / w, target_size[1] / h)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize and pad
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create padded image
        padded = np.zeros((target_size[1], target_size[0], image.shape[2]), dtype=image.dtype)
        
        # Center the resized image
        y_offset = (target_size[1] - new_h) // 2
        x_offset = (target_size[0] - new_w) // 2
        
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return padded
    else:
        return cv2.resize(image, target_size)


def create_thumbnail(image: np.ndarray, size: int = 128) -> np.ndarray:
    """
    Create thumbnail version of image.
    
    Args:
        image: Input image
        size: Thumbnail size (square)
        
    Returns:
        Thumbnail image
    """
    return resize_image(image, (size, size), maintain_aspect=True)


def draw_bounding_box(image: np.ndarray, 
                     bbox: Dict[str, int], 
                     label: str = "", 
                     confidence: float = 0.0,
                     color: Tuple[int, int, int] = (0, 255, 0),
                     thickness: int = 2) -> np.ndarray:
    """
    Draw bounding box with label on image.
    
    Args:
        image: Input image
        bbox: Bounding box coordinates
        label: Label text
        confidence: Confidence score
        color: Box color (BGR)
        thickness: Line thickness
        
    Returns:
        Image with bounding box drawn
    """
    img_copy = image.copy()
    
    # Extract coordinates
    if 'x1' in bbox and 'y1' in bbox:
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
    elif 'left' in bbox and 'top' in bbox:
        x1, y1 = bbox['left'], bbox['top']
        x2, y2 = bbox['right'], bbox['bottom']
    else:
        raise ValueError("Invalid bounding box format")
    
    # Draw rectangle
    cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
    
    # Draw label
    if label or confidence > 0:
        label_text = label
        if confidence > 0:
            label_text += f" ({confidence:.2f})"
        
        # Get text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        text_thickness = 1
        
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, font, font_scale, text_thickness
        )
        
        # Draw background rectangle for text
        cv2.rectangle(
            img_copy,
            (x1, y1 - text_height - 10),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            img_copy,
            label_text,
            (x1, y1 - 5),
            font,
            font_scale,
            (255, 255, 255),
            text_thickness
        )
    
    return img_copy


def save_results_to_json(results: Dict[str, Any], output_path: str) -> bool:
    """
    Save analysis results to JSON file.
    
    Args:
        results: Results dictionary
        output_path: Output file path
        
    Returns:
        True if saved successfully
    """
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy(results), f, indent=2, ensure_ascii=False)
        
        return True
        
    except Exception as e:
        logging.getLogger('VisionAgent').error(f"Failed to save results to {output_path}: {str(e)}")
        return False


def load_image_safe(image_input: Union[str, np.ndarray, bytes]) -> Optional[np.ndarray]:
    """
    Safely load image with error handling.
    
    Args:
        image_input: Image input (path, array, or bytes)
        
    Returns:
        Image as numpy array or None if failed
    """
    try:
        if isinstance(image_input, str):
            if urlparse(image_input).scheme in ['http', 'https']:
                return load_image_from_url(image_input)
            else:
                image = cv2.imread(image_input)
                return image if image is not None else None
        elif isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(image_input, np.ndarray):
            return image_input
    except Exception as e:
        logging.getLogger('VisionAgent').error(f"Failed to load image: {str(e)}")
    
    return None


def get_device_info() -> Dict[str, Any]:
    """
    Get information about available compute devices.
    
    Returns:
        Dictionary with device information
    """
    device_info = {
        'cpu': {
            'available': True,
            'cores': os.cpu_count()
        },
        'cuda': {
            'available': False,
            'devices': []
        }
    }
    
    try:
        import torch
        if torch.cuda.is_available():
            device_info['cuda']['available'] = True
            device_info['cuda']['device_count'] = torch.cuda.device_count()
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                device_info['cuda']['devices'].append({
                    'id': i,
                    'name': props.name,
                    'memory_gb': props.total_memory / (1024**3),
                    'compute_capability': f"{props.major}.{props.minor}"
                })
    except ImportError:
        pass
    
    return device_info


def create_temp_file(suffix: str = '.tmp', prefix: str = 'visionagent_') -> str:
    """
    Create a temporary file.
    
    Args:
        suffix: File suffix
        prefix: File prefix
        
    Returns:
        Path to temporary file
    """
    fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
    os.close(fd)  # Close the file descriptor
    return temp_path


def cleanup_temp_files(temp_dir: str, max_age_hours: int = 24) -> int:
    """
    Clean up old temporary files.
    
    Args:
        temp_dir: Temporary directory path
        max_age_hours: Maximum age of files to keep in hours
        
    Returns:
        Number of files cleaned up
    """
    if not os.path.exists(temp_dir):
        return 0
    
    import time
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    cleaned_count = 0
    
    try:
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getmtime(file_path)
                
                if file_age > max_age_seconds:
                    os.remove(file_path)
                    cleaned_count += 1
    except Exception as e:
        logging.getLogger('VisionAgent').error(f"Error cleaning temp files: {str(e)}")
    
    return cleaned_count


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def calculate_iou(box1: Dict[str, int], box2: Dict[str, int]) -> float:
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        box1: First bounding box
        box2: Second bounding box
        
    Returns:
        IoU value between 0 and 1
    """
    # Normalize box format
    def normalize_box(box):
        if 'x1' in box:
            return box['x1'], box['y1'], box['x2'], box['y2']
        else:
            return box['left'], box['top'], box['right'], box['bottom']
    
    x1_1, y1_1, x2_1, y2_1 = normalize_box(box1)
    x1_2, y1_2, x2_2, y2_2 = normalize_box(box2)
    
    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0


def non_max_suppression(detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
    """
    Apply Non-Maximum Suppression to filter overlapping detections.
    
    Args:
        detections: List of detection dictionaries
        iou_threshold: IoU threshold for suppression
        
    Returns:
        Filtered list of detections
    """
    if not detections:
        return []
    
    # Sort by confidence
    sorted_detections = sorted(detections, key=lambda x: x.get('confidence', 0), reverse=True)
    
    filtered_detections = []
    
    for detection in sorted_detections:
        # Check if this detection overlaps with any already selected detection
        suppress = False
        
        for selected_detection in filtered_detections:
            # Only suppress if same class
            if detection.get('class_name') == selected_detection.get('class_name'):
                iou = calculate_iou(detection['bounding_box'], selected_detection['bounding_box'])
                
                if iou > iou_threshold:
                    suppress = True
                    break
        
        if not suppress:
            filtered_detections.append(detection)
    
    return filtered_detections


def create_video_thumbnail(video_path: str, 
                          timestamp_seconds: float = 1.0,
                          size: Tuple[int, int] = (320, 240)) -> Optional[np.ndarray]:
    """
    Create thumbnail from video at specified timestamp.
    
    Args:
        video_path: Path to video file
        timestamp_seconds: Timestamp to extract frame from
        size: Thumbnail size
        
    Returns:
        Thumbnail image or None if failed
    """
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None
        
        # Seek to timestamp
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(timestamp_seconds * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        # Read frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None
        
        # Resize to thumbnail size
        thumbnail = cv2.resize(frame, size)
        return thumbnail
        
    except Exception as e:
        logging.getLogger('VisionAgent').error(f"Failed to create video thumbnail: {str(e)}")
        return None


def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration to validate
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check required sections
    required_sections = ['server', 'logging']
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: {section}")
    
    # Validate server config
    if 'server' in config:
        server_config = config['server']
        
        if 'port' in server_config:
            port = server_config['port']
            if not isinstance(port, int) or port < 1 or port > 65535:
                errors.append(f"Invalid port number: {port}")
        
        if 'host' in server_config:
            host = server_config['host']
            if not isinstance(host, str) or not host.strip():
                errors.append(f"Invalid host: {host}")
    
    # Validate logging config
    if 'logging' in config:
        logging_config = config['logging']
        
        if 'level' in logging_config:
            level = logging_config['level'].upper()
            valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            if level not in valid_levels:
                errors.append(f"Invalid log level: {level}")
    
    return len(errors) == 0, errors


class ModelCache:
    """Cache for downloaded models and their metadata."""
    
    def __init__(self, cache_dir: str = './models'):
        """
        Initialize model cache.
        
        Args:
            cache_dir: Directory to store cached models
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.metadata_file = self.cache_dir / 'cache_metadata.json'
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
    
    def _save_metadata(self) -> None:
        """Save cache metadata to disk."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logging.getLogger('VisionAgent').error(f"Failed to save cache metadata: {str(e)}")
    
    def get_model_path(self, model_name: str) -> Optional[str]:
        """
        Get local path for cached model.
        
        Args:
            model_name: Model identifier
            
        Returns:
            Local path if cached, None otherwise
        """
        if model_name in self.metadata:
            model_path = self.cache_dir / self.metadata[model_name]['filename']
            if model_path.exists():
                return str(model_path)
        
        return None
    
    def cache_model(self, model_name: str, model_url: str) -> Optional[str]:
        """
        Download and cache a model.
        
        Args:
            model_name: Model identifier
            model_url: URL to download model from
            
        Returns:
            Local path to cached model or None if failed
        """
        try:
            # Generate filename
            filename = f"{model_name}.pt"
            model_path = self.cache_dir / filename
            
            # Download model
            if download_file(model_url, str(model_path)):
                # Update metadata
                self.metadata[model_name] = {
                    'filename': filename,
                    'url': model_url,
                    'size_bytes': model_path.stat().st_size,
                    'hash': get_file_hash(str(model_path)),
                    'cached_at': str(int(time.time()))
                }
                
                self._save_metadata()
                return str(model_path)
        
        except Exception as e:
            logging.getLogger('VisionAgent').error(f"Failed to cache model {model_name}: {str(e)}")
        
        return None
    
    def clear_cache(self) -> int:
        """
        Clear all cached models.
        
        Returns:
            Number of files removed
        """
        removed_count = 0
        
        try:
            for file_path in self.cache_dir.iterdir():
                if file_path.is_file() and file_path.name != 'cache_metadata.json':
                    file_path.unlink()
                    removed_count += 1
            
            # Clear metadata
            self.metadata.clear()
            self._save_metadata()
            
        except Exception as e:
            logging.getLogger('VisionAgent').error(f"Failed to clear cache: {str(e)}")
        
        return removed_count


# Global model cache instance
model_cache = ModelCache()
