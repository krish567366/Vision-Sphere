"""
Classification Agent
Implements image classification using HuggingFace Transformers and custom models.
"""

import cv2
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch.nn.functional as F
from .base_agent import BaseAgent, ProcessingResult


class ClassificationAgent(BaseAgent):
    """
    Agent for image classification and feature extraction.
    
    Features:
    - HuggingFace Transformers integration
    - Custom model support
    - Multi-class and multi-label classification
    - Feature vector extraction
    - Batch processing
    - Top-k predictions
    """
    
    def __init__(self, 
                 device: Optional[str] = None,
                 model_path: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize Classification Agent.
        
        Args:
            device: Target device (CPU/GPU)
            model_path: Path to model or HuggingFace model name
            config: Configuration parameters including:
                - model_name: HuggingFace model name
                - top_k: Number of top predictions to return
                - threshold: Minimum confidence threshold
                - image_size: Target image size for processing
                - return_features: Whether to return feature vectors
        """
        super().__init__(device, model_path, config)
        
        # Configuration defaults
        self.model_name = self.config.get('model_name', 'microsoft/resnet-50')
        self.top_k = self.config.get('top_k', 5)
        self.threshold = self.config.get('threshold', 0.1)
        self.image_size = self.config.get('image_size', 224)
        self.return_features = self.config.get('return_features', False)
        
        # Model components
        self.model: Optional[AutoModelForImageClassification] = None
        self.processor: Optional[AutoImageProcessor] = None
        self.class_names: List[str] = []
    
    def initialize(self) -> bool:
        """
        Initialize classification model and processor.
        
        Returns:
            True if initialization successful
        """
        try:
            self.logger.info("Initializing Classification Agent...")
            
            # Load model and processor
            model_name_or_path = self.model_path or self.model_name
            
            self.processor = AutoImageProcessor.from_pretrained(model_name_or_path)
            self.model = AutoModelForImageClassification.from_pretrained(model_name_or_path)
            
            # Move to appropriate device
            if self.device == 'cuda' and torch.cuda.is_available():
                self.model = self.model.to('cuda')
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Get class names
            if hasattr(self.model.config, 'id2label'):
                self.class_names = list(self.model.config.id2label.values())
            else:
                self.class_names = [f"class_{i}" for i in range(self.model.config.num_labels)]
            
            self._is_initialized = True
            self.logger.info(f"Classification Agent initialized with {model_name_or_path} on {self.device}")
            self.logger.info(f"Available classes: {len(self.class_names)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Classification Agent: {str(e)}")
            return False
    
    def process(self, input_data: Any) -> ProcessingResult:
        """
        Process image for classification.
        
        Args:
            input_data: Image data (path, numpy array, or bytes)
            
        Returns:
            ProcessingResult with classification results
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
            pil_image = self._convert_to_pil(image)
            
            # Perform classification
            result, inference_time = self._measure_inference_time(
                self._classify_image, pil_image
            )
            
            return ProcessingResult(
                success=True,
                data=result,
                inference_time=inference_time,
                metadata={
                    'model_name': self.model_name,
                    'top_k': self.top_k,
                    'image_size': self.image_size
                }
            )
            
        except Exception as e:
            self.logger.error(f"Classification error: {str(e)}")
            return ProcessingResult(
                success=False,
                data={},
                error=str(e)
            )
    
    def _convert_to_pil(self, cv_image: np.ndarray) -> Image.Image:
        """
        Convert OpenCV image to PIL Image.
        
        Args:
            cv_image: Image in BGR format
            
        Returns:
            PIL Image in RGB format
        """
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_image)
    
    def _classify_image(self, pil_image: Image.Image) -> Dict[str, Any]:
        """
        Classify image using the loaded model.
        
        Args:
            pil_image: PIL Image to classify
            
        Returns:
            Dictionary with classification results
        """
        # Process image
        inputs = self.processor(pil_image, return_tensors="pt")
        
        # Move inputs to device
        if self.device == 'cuda':
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Apply softmax to get probabilities
            probabilities = F.softmax(logits, dim=-1)
            
            # Get top-k predictions
            top_k_probs, top_k_indices = torch.topk(probabilities, self.top_k, dim=-1)
            
            # Extract feature vector if requested
            features = None
            if self.return_features:
                # Get features from the layer before classification
                if hasattr(self.model, 'classifier'):
                    # For models with a classifier layer
                    features = self.model.classifier.in_features
                features = logits.cpu().numpy().flatten().tolist()
        
        # Format results
        predictions = []
        for prob, idx in zip(top_k_probs[0], top_k_indices[0]):
            prob_value = float(prob.cpu().numpy())
            if prob_value >= self.threshold:
                predictions.append({
                    'class_id': int(idx.cpu().numpy()),
                    'class_name': self.class_names[int(idx.cpu().numpy())],
                    'confidence': prob_value,
                    'probability': prob_value
                })
        
        result = {
            'predictions': predictions,
            'top_prediction': predictions[0] if predictions else None,
            'all_probabilities': probabilities[0].cpu().numpy().tolist(),
            'prediction_count': len(predictions)
        }
        
        if features is not None:
            result['features'] = features
        
        return result
    
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
        
        try:
            # Convert all images to PIL format
            pil_images = []
            for image_input in image_batch:
                cv_image = self._preprocess_image(image_input)
                pil_image = self._convert_to_pil(cv_image)
                pil_images.append(pil_image)
            
            # Process batch
            inputs = self.processor(pil_images, return_tensors="pt", padding=True)
            
            # Move to device
            if self.device == 'cuda':
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=-1)
            
            # Process results for each image
            results = []
            for i in range(len(image_batch)):
                image_probs = probabilities[i]
                top_k_probs, top_k_indices = torch.topk(image_probs, self.top_k)
                
                predictions = []
                for prob, idx in zip(top_k_probs, top_k_indices):
                    prob_value = float(prob.cpu().numpy())
                    if prob_value >= self.threshold:
                        predictions.append({
                            'class_id': int(idx.cpu().numpy()),
                            'class_name': self.class_names[int(idx.cpu().numpy())],
                            'confidence': prob_value,
                            'probability': prob_value
                        })
                
                result_data = {
                    'predictions': predictions,
                    'top_prediction': predictions[0] if predictions else None,
                    'prediction_count': len(predictions)
                }
                
                results.append(ProcessingResult(
                    success=True,
                    data=result_data,
                    metadata={'batch_index': i}
                ))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch classification error: {str(e)}")
            return [ProcessingResult(
                success=False,
                data={},
                error=str(e)
            ) for _ in image_batch]
    
    def extract_features(self, input_data: Any) -> ProcessingResult:
        """
        Extract feature vectors from an image.
        
        Args:
            input_data: Image data
            
        Returns:
            ProcessingResult with feature vectors
        """
        # Temporarily enable feature extraction
        original_return_features = self.return_features
        self.return_features = True
        
        result = self.process(input_data)
        
        # Restore original setting
        self.return_features = original_return_features
        
        return result
    
    def get_class_names(self) -> List[str]:
        """
        Get list of all available class names.
        
        Returns:
            List of class names the model can predict
        """
        return self.class_names.copy()
    
    def set_threshold(self, threshold: float) -> None:
        """
        Set confidence threshold for predictions.
        
        Args:
            threshold: Minimum confidence threshold (0.0 to 1.0)
        """
        self.threshold = max(0.0, min(1.0, threshold))
        self.logger.info(f"Classification threshold set to {self.threshold}")
