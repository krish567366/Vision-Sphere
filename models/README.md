# VisionAgent Models Directory

## This directory stores downloaded AI models and weights

## Model Organization

### Face Recognition Models

- `face_recognition` library models (automatically managed)
- Custom face recognition models
- Facial landmark detection models

### Object Detection Models

- YOLOv8 variants (nano, small, medium, large, xlarge)
- Custom trained YOLO models
- Other object detection frameworks

### Classification Models

- HuggingFace Transformers models
- Custom classification models
- Feature extraction models

### Video Analysis Models

- Tracking models
- Temporal analysis models
- Action recognition models

## Model Caching

Models are automatically downloaded and cached when first used:

- YOLO models from Ultralytics
- HuggingFace models from transformers library
- Custom models from configured URLs

## Adding Custom Models

Place your custom model files in this directory and update the configuration:

```yaml
object_agent:
  model:
    path: "./models/custom_yolo.pt"
```

## Model Versioning

The framework supports model versioning through the cache metadata system.
See `cache_metadata.json` for tracked model information.
