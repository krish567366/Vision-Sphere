"""
Enhanced Async Classification Agent with HuggingFace Integration
Production-ready image classification with advanced model management and caching.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple, Union, AsyncGenerator
import logging
import numpy as np
from PIL import Image
import cv2

from .async_base_agent import AsyncBaseAgent, AsyncProcessingResult
from ..utils.caching import cached_tool_call
from ..utils.tracing import traced_operation, SpanType
from ..utils.streaming import publish_detection_event, EventType


class AsyncClassificationAgent(AsyncBaseAgent):
    """
    Enhanced async image classification agent using HuggingFace models.
    
    Features:
    - Multiple pre-trained model support
    - Custom model loading
    - Feature extraction capabilities
    - Batch classification optimization
    - Advanced caching and model management
    - Real-time classification streaming
    """
    
    def __init__(self, 
                 device: Optional[str] = None,
                 model_path: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the async classification agent.
        
        Args:
            device: Target device for processing
            model_path: Path to custom model or HuggingFace model name
            config: Agent configuration
        """
        super().__init__("AsyncClassificationAgent", device, model_path, config)
        
        # Ensure config is not None
        config = config or {}
        
        # Model configuration
        self.model_name = model_path or config.get('model_name', 'microsoft/resnet-50')
        self.top_k = config.get('top_k', 5)
        self.confidence_threshold = config.get('confidence_threshold', 0.1)
        
        # Feature extraction
        self.enable_features = config.get('enable_features', True)
        self.feature_layer = config.get('feature_layer', 'pooler')
        
        # Models and processors
        self.model = None
        self.processor = None
        self.feature_extractor = None
        
        # Model metadata
        self.model_info = {}
        self.class_labels = []
        
        # Logger is already set by parent AsyncBaseAgent.__init__()
    
    async def _initialize_model(self) -> bool:
        """
        Initialize classification model and processor.
        
        Returns:
            True if initialization successful
        """
        try:
            # Import HuggingFace transformers
            try:
                from transformers import AutoImageProcessor, AutoModelForImageClassification
                import torch
            except ImportError:
                self.logger.error("transformers or torch not installed. Install with: pip install transformers torch")
                return False
            
            # Load processor and model
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForImageClassification.from_pretrained(self.model_name)
            
            # Move to device
            if self.device == 'cuda' and torch.cuda.is_available():
                self.model = self.model.to('cuda')
                self.logger.info(f"Model moved to GPU: {torch.cuda.get_device_name()}")
            
            # Set to evaluation mode
            self.model.eval()
            
            # Get model info
            self.model_info = {
                'model_name': self.model_name,
                'num_labels': self.model.num_labels if hasattr(self.model, 'num_labels') else 'unknown',
                'architecture': self.model.config.architectures[0] if hasattr(self.model.config, 'architectures') else 'unknown'
            }
            
            # Get class labels if available
            if hasattr(self.model.config, 'id2label'):
                self.class_labels = [
                    self.model.config.id2label[i] 
                    for i in range(len(self.model.config.id2label))
                ]
            
            # Initialize feature extractor for embeddings
            if self.enable_features:
                self.feature_extractor = self.model
            
            self.logger.info(f"Classification model loaded: {self.model_name}")
            self.logger.info(f"Number of classes: {len(self.class_labels)}")
            self.logger.info(f"Device: {self.device}")
            
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to initialize classification model: {e}")
            return False
    
    @cached_tool_call(expire_time=1800)  # 30 minutes cache
    async def _classify_image_internal(self, 
                                     image: np.ndarray,
                                     trace_id: str) -> Dict[str, Any]:
        """
        Internal image classification with caching.
        
        Args:
            image: Input image as numpy array
            trace_id: Trace ID for monitoring
            
        Returns:
            Classification results
        """
        async with traced_operation(
            "classification.inference",
            SpanType.INFERENCE,
            trace_id=trace_id
        ) as span:
            
            import torch
            
            # Convert OpenCV BGR to RGB PIL Image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            span.set_attribute("image.height", image.shape[0])
            span.set_attribute("image.width", image.shape[1])
            span.set_attribute("model.name", self.model_name)
            
            # Process image
            inputs = self.processor(pil_image, return_tensors="pt")
            
            # Move inputs to device
            if self.device == 'cuda':
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Get probabilities
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                
                # Get top-k predictions
                top_probs, top_indices = torch.topk(probabilities, self.top_k)
                
                # Move back to CPU for processing
                top_probs = top_probs.cpu().numpy()[0]
                top_indices = top_indices.cpu().numpy()[0]
            
            # Format predictions
            predictions = []
            for prob, idx in zip(top_probs, top_indices):
                if prob >= self.confidence_threshold:
                    label = self.class_labels[idx] if idx < len(self.class_labels) else f"class_{idx}"
                    
                    predictions.append({
                        'class_name': label,
                        'class_id': int(idx),
                        'confidence': float(prob),
                        'probability': float(prob)
                    })
            
            # Extract features if enabled
            features = None
            if self.enable_features:
                try:
                    # Extract features from the model's hidden states
                    with torch.no_grad():
                        feature_outputs = self.model(**inputs, output_hidden_states=True)
                        
                        # Use pooled output or last hidden state
                        if hasattr(feature_outputs, 'pooler_output') and feature_outputs.pooler_output is not None:
                            features = feature_outputs.pooler_output.cpu().numpy()[0]
                        else:
                            # Use mean of last hidden state
                            last_hidden = feature_outputs.hidden_states[-1]
                            features = torch.mean(last_hidden, dim=1).cpu().numpy()[0]
                        
                except Exception as e:
                    self.logger.warning(f"Feature extraction failed: {e}")
            
            span.set_attribute("predictions.count", len(predictions))
            span.set_attribute("predictions.top_confidence", float(top_probs[0]) if len(top_probs) > 0 else 0)
            
            result = {
                'predictions': predictions,
                'top_prediction': predictions[0] if predictions else None,
                'features': features.tolist() if features is not None else None,
                'feature_dimensions': len(features) if features is not None else 0,
                'model_info': self.model_info
            }
            
            return result
    
    async def _process_internal(self, input_data: Any, trace_id: str) -> Dict[str, Any]:
        """
        Internal classification processing implementation.
        
        Args:
            input_data: Image input data
            trace_id: Trace ID for monitoring
            
        Returns:
            Classification results dictionary
        """
        # Preprocess image
        image = await self._preprocess_image_async(input_data)
        
        # Classify image
        classification_result = await self._classify_image_internal(image, trace_id)
        
        # Publish classification event
        if classification_result['predictions']:
            top_prediction = classification_result['top_prediction']
            await publish_detection_event(
                'classification',
                trace_id,
                {
                    'class': top_prediction['class_name'],
                    'confidence': top_prediction['confidence']
                }
            )
        
        # Add image metadata
        classification_result.update({
            'image_dimensions': {'height': image.shape[0], 'width': image.shape[1]},
            'prediction_count': len(classification_result['predictions']),
            'classification_confidence': classification_result['top_prediction']['confidence'] if classification_result['top_prediction'] else 0.0
        })
        
        return classification_result
    
    async def classify_batch(self, 
                           images: List[Union[str, np.ndarray, bytes]],
                           batch_size: Optional[int] = None) -> List[AsyncProcessingResult]:
        """
        Classify multiple images in optimized batches.
        
        Args:
            images: List of images to classify
            batch_size: Optional batch size override
            
        Returns:
            List of classification results
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        # Process in batches for optimal GPU utilization
        results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            # Process batch concurrently
            batch_tasks = [
                self.process(image, trace_id=f"batch_{i}_{j}")
                for j, image in enumerate(batch)
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Handle exceptions in batch
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    results.append(AsyncProcessingResult(
                        success=False,
                        data={},
                        error=str(result),
                        metadata={'batch_index': i + j}
                    ))
                else:
                    results.append(result)
        
        return results
    
    async def extract_features_batch(self, 
                                   images: List[Union[str, np.ndarray, bytes]]) -> List[Optional[np.ndarray]]:
        """
        Extract feature vectors from multiple images.
        
        Args:
            images: List of images
            
        Returns:
            List of feature vectors
        """
        if not self.enable_features:
            self.logger.warning("Feature extraction is disabled")
            return [None] * len(images)
        
        # Process images to get features
        results = await self.classify_batch(images)
        
        features = []
        for result in results:
            if result.success and result.data.get('features'):
                features.append(np.array(result.data['features']))
            else:
                features.append(None)
        
        return features
    
    async def find_similar_images(self, 
                                query_image: Union[str, np.ndarray, bytes],
                                candidate_images: List[Union[str, np.ndarray, bytes]],
                                similarity_threshold: float = 0.8) -> List[Tuple[int, float]]:
        """
        Find images similar to a query image using feature similarity.
        
        Args:
            query_image: Query image to match against
            candidate_images: List of candidate images
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of (image_index, similarity_score) for similar images
        """
        trace_id = f"similarity_search_{time.time()}"
        
        async with traced_operation(
            "similarity_search.features",
            SpanType.ANALYTICS,
            trace_id=trace_id
        ) as span:
            
            # Extract features for query image
            query_result = await self.process(query_image, trace_id=f"{trace_id}_query")
            
            if not query_result.success or not query_result.data.get('features'):
                raise ValueError("Could not extract features from query image")
            
            query_features = np.array(query_result.data['features'])
            
            # Extract features for all candidate images
            candidate_features = await self.extract_features_batch(candidate_images)
            
            # Calculate similarities
            similarities = []
            
            for i, features in enumerate(candidate_features):
                if features is None:
                    continue
                
                # Calculate cosine similarity
                similarity = np.dot(query_features, features) / (
                    np.linalg.norm(query_features) * np.linalg.norm(features)
                )
                
                if similarity >= similarity_threshold:
                    similarities.append((i, float(similarity)))
            
            # Sort by similarity score (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            span.set_attribute("similarity.query_processed", True)
            span.set_attribute("similarity.candidates_processed", len(candidate_features))
            span.set_attribute("similarity.matches_found", len(similarities))
            
            return similarities
    
    async def classify_with_custom_labels(self, 
                                        image_input: Union[str, np.ndarray, bytes],
                                        custom_labels: List[str]) -> AsyncProcessingResult:
        """
        Classify image with custom label mapping.
        
        Args:
            image_input: Input image
            custom_labels: Custom labels to map to
            
        Returns:
            Classification result with custom labels
        """
        trace_id = f"custom_classification_{time.time()}"
        
        # Get standard classification
        result = await self.process(image_input, trace_id=trace_id)
        
        if not result.success:
            return result
        
        # Map predictions to custom labels using semantic similarity
        predictions = result.data.get('predictions', [])
        mapped_predictions = []
        
        for pred in predictions:
            # Find best matching custom label
            best_match = await self._find_best_label_match(
                pred['class_name'], 
                custom_labels
            )
            
            mapped_pred = pred.copy()
            mapped_pred['original_class'] = pred['class_name']
            mapped_pred['custom_class'] = best_match
            mapped_predictions.append(mapped_pred)
        
        # Update result
        updated_data = result.data.copy()
        updated_data['custom_predictions'] = mapped_predictions
        updated_data['custom_labels'] = custom_labels
        
        return AsyncProcessingResult(
            success=True,
            data=updated_data,
            confidence=result.confidence,
            inference_time_ms=result.inference_time_ms,
            trace_id=trace_id,
            metadata={**(result.metadata or {}), 'custom_labels': True}
        )
    async def _find_best_label_match(self,
                                   original_label: str,
                                   custom_labels: List[str]) -> str:
        """
        Find best matching custom label for original prediction.
        
        Args:
            original_label: Original model prediction
            custom_labels: List of custom labels
            
        Returns:
            Best matching custom label
        """
        # Simple keyword matching (can be enhanced with semantic similarity)
        original_lower = original_label.lower()
        
        best_match = custom_labels[0] if custom_labels else original_label
        best_score = 0.0
        
        for custom_label in custom_labels:
            custom_lower = custom_label.lower()
            
            # Calculate simple similarity
            if original_lower == custom_lower:
                return custom_label
            
            # Check for substring matches
            if original_lower in custom_lower or custom_lower in original_lower:
                score = len(set(original_lower.split()) & set(custom_lower.split())) / len(set(original_lower.split()) | set(custom_lower.split()))
                if score > best_score:
                    best_score = score
                    best_match = custom_label
        
        return best_match
    
    async def analyze_image_composition(self, 
                                      image_input: Union[str, np.ndarray, bytes]) -> AsyncProcessingResult:
        """
        Analyze image composition and visual elements.
        
        Args:
            image_input: Input image
            
        Returns:
            Composition analysis results
        """
        trace_id = f"composition_analysis_{time.time()}"
        
        async with traced_operation(
            "composition.analysis",
            SpanType.ANALYTICS,
            trace_id=trace_id
        ) as span:
            
            try:
                # Preprocess image
                image = await self._preprocess_image_async(image_input)
                
                # Get classification
                classification_result = await self.process(image, trace_id=trace_id)
                
                if not classification_result.success:
                    return classification_result
                
                # Analyze composition
                composition_analysis = await self._analyze_visual_composition(image)
                
                # Combine results
                combined_data = {
                    **classification_result.data,
                    'composition_analysis': composition_analysis
                }
                
                span.set_attribute("composition.analyzed", True)
                
                return AsyncProcessingResult(
                    success=True,
                    data=combined_data,
                    confidence=classification_result.confidence,
                    inference_time_ms=classification_result.inference_time_ms,
                    trace_id=trace_id,
                    metadata={'composition_analysis': True}
                )

            except Exception as e:
                error_msg = f"Composition analysis failed: {str(e)}"
                span.set_attribute("error", error_msg)
                return AsyncProcessingResult(
                    success=False,
                    data={},
                    error=error_msg,
                    trace_id=trace_id
                )
    
    async def _analyze_visual_composition(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze visual composition of image.
        
        Args:
            image: Input image
            
        Returns:
            Composition analysis
        """
        # Color analysis
        color_analysis = await self._analyze_colors(image)
        
        # Texture analysis
        texture_analysis = await self._analyze_texture(image)
        
        # Brightness and contrast
        brightness_contrast = await self._analyze_brightness_contrast(image)
        
        return {
            'color_analysis': color_analysis,
            'texture_analysis': texture_analysis,
            'brightness_contrast': brightness_contrast
        }
    
    async def _analyze_colors(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze color distribution in image."""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate color histograms
        h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
        
        # Dominant colors
        dominant_hue = np.argmax(h_hist)
        dominant_saturation = np.argmax(s_hist)
        dominant_value = np.argmax(v_hist)
        
        # Color statistics
        avg_hue = np.average(range(180), weights=h_hist.flatten())
        avg_saturation = np.average(range(256), weights=s_hist.flatten())
        avg_value = np.average(range(256), weights=v_hist.flatten())
        
        return {
            'dominant_hue': int(dominant_hue),
            'dominant_saturation': int(dominant_saturation),
            'dominant_value': int(dominant_value),
            'avg_hue': float(avg_hue),
            'avg_saturation': float(avg_saturation),
            'avg_brightness': float(avg_value),
            'color_diversity': float(np.std(h_hist))
        }
    
    async def _analyze_texture(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze texture characteristics of image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate texture measures
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Sobel gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        avg_gradient = np.mean(gradient_magnitude)
        
        return {
            'sharpness': float(laplacian_var),
            'avg_gradient': float(avg_gradient),
            'texture_complexity': float(np.std(gray))
        }
    
    async def _analyze_brightness_contrast(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze brightness and contrast characteristics."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Basic statistics
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        # Contrast (RMS contrast)
        contrast = np.sqrt(np.mean((gray - mean_brightness) ** 2))
        
        # Dynamic range
        dynamic_range = np.max(gray) - np.min(gray)
        
        return {
            'mean_brightness': float(mean_brightness),
            'brightness_std': float(std_brightness),
            'contrast': float(contrast),
            'dynamic_range': float(dynamic_range),
            'brightness_category': self._categorize_brightness(mean_brightness)
        }
    
    def _categorize_brightness(self, brightness: float) -> str:
        """Categorize brightness level."""
        if brightness < 50:
            return 'dark'
        elif brightness < 100:
            return 'dim'
        elif brightness < 150:
            return 'normal'
        elif brightness < 200:
            return 'bright'
        else:
            return 'very_bright'
    
    async def stream_classification(self, 
                                  video_source: Union[str, int],
                                  callback_func: Optional[callable] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream classification results from video in real-time.
        
        Args:
            video_source: Video file path or camera index
            callback_func: Optional callback for each frame
            
        Yields:
            Real-time classification updates
        """
        trace_id = f"stream_classification_{time.time()}"
        
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
                
                # Classify frame
                frame_start = time.perf_counter()
                result = await self.process(frame, trace_id=f"{trace_id}_frame_{frame_number}")
                frame_end = time.perf_counter()
                
                processing_time = (frame_end - frame_start) * 1000
                
                # Yield classification result
                yield {
                    "type": "classification_result",
                    "data": {
                        "frame_number": frame_number,
                        "processing_time_ms": processing_time,
                        "classification": result.to_dict()
                    }
                }
                
                # Call callback if provided
                if callback_func:
                    await callback_func(frame_number, frame, result)
                
                frame_number += 1
                
                # Allow other tasks to run
                await asyncio.sleep(0.001)
        
        finally:
            cap.release()
            yield {"type": "stream_complete", "data": {"total_frames": frame_number}}
    
    async def get_model_capabilities(self) -> Dict[str, Any]:
        """
        Get detailed information about model capabilities.
        
        Returns:
            Model capabilities and metadata
        """
        base_metrics = await self.get_metrics()
        
        capabilities = {
            'model_info': self.model_info,
            'supported_classes': self.class_labels,
            'total_classes': len(self.class_labels),
            'top_k': self.top_k,
            'confidence_threshold': self.confidence_threshold,
            'feature_extraction_enabled': self.enable_features,
            'feature_dimensions': None,
            'supported_formats': ['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            'batch_processing': True,
            'streaming_support': True
        }
        
        # Get feature dimensions if available
        if self.enable_features and self.model:
            try:
                # Create dummy input to get feature dimensions
                dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
                dummy_result = await self.process(dummy_image)
                
                if dummy_result.success and dummy_result.data.get('features'):
                    capabilities['feature_dimensions'] = dummy_result.data['feature_dimensions']
            except Exception:
                pass
        
        return {**base_metrics, **capabilities}
    
    async def benchmark_performance(self, 
                                  test_images: List[Union[str, np.ndarray, bytes]],
                                  iterations: int = 3) -> Dict[str, Any]:
        """
        Benchmark classification performance.
        
        Args:
            test_images: Test images for benchmarking
            iterations: Number of benchmark iterations
            
        Returns:
            Performance benchmark results
        """
        trace_id = f"benchmark_{time.time()}"
        
        async with traced_operation(
            "classification.benchmark",
            SpanType.ANALYTICS,
            trace_id=trace_id
        ) as span:
            
            benchmark_results = {
                'test_images': len(test_images),
                'iterations': iterations,
                'timings': [],
                'throughput': [],
                'memory_usage': []
            }
            
            for iteration in range(iterations):
                iteration_start = time.perf_counter()
                
                # Process all test images
                results = await self.classify_batch(test_images)
                
                iteration_end = time.perf_counter()
                iteration_time = (iteration_end - iteration_start) * 1000
                
                # Calculate throughput
                successful_results = sum(1 for r in results if r.success)
                throughput = successful_results / (iteration_time / 1000)
                
                benchmark_results['timings'].append(iteration_time)
                benchmark_results['throughput'].append(throughput)
                
                # Allow system to cool down
                await asyncio.sleep(0.1)
            
            # Calculate statistics
            benchmark_results['avg_time_ms'] = np.mean(benchmark_results['timings'])
            benchmark_results['std_time_ms'] = np.std(benchmark_results['timings'])
            benchmark_results['avg_throughput_fps'] = np.mean(benchmark_results['throughput'])
            benchmark_results['min_time_ms'] = np.min(benchmark_results['timings'])
            benchmark_results['max_time_ms'] = np.max(benchmark_results['timings'])
            
            span.set_attribute("benchmark.completed", True)
            span.set_attribute("benchmark.avg_time_ms", benchmark_results['avg_time_ms'])
            span.set_attribute("benchmark.avg_throughput", benchmark_results['avg_throughput_fps'])
            
            return benchmark_results
    
    async def _cleanup_processing_resources(self):
        """Clean up classification processing resources."""
        # Clear any cached features or temporary data
        if hasattr(self, '_feature_cache'):
            self._feature_cache.clear()
    
    async def get_classification_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive classification statistics.
        
        Returns:
            Statistics dictionary
        """
        base_metrics = await self.get_metrics()
        
        classification_stats = {
            'model_name': self.model_name,
            'model_info': self.model_info,
            'available_classes': len(self.class_labels),
            'top_k_predictions': self.top_k,
            'confidence_threshold': self.confidence_threshold,
            'feature_extraction': self.enable_features,
            'batch_processing': True,
            'streaming_capable': True
        }
        
        return {**base_metrics, **classification_stats}
