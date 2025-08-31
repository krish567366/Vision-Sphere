# VisionAgent - Professional Multi-Modal AI Agent Framework

A cutting-edge, production-ready AI agent platform for image, video, and face analytics built with modern Python and state-of-the-art AI models.

## 🚀 Features

### Core Capabilities

- **Face Detection & Recognition** - Advanced face detection, encoding, and recognition with facial landmarks
- **Object Detection** - YOLOv8-powered object detection with real-time inference
- **Video Analysis** - Frame-by-frame video processing with object/face tracking
- **Image Classification** - HuggingFace Transformers integration for image classification
- **Real-time Processing** - WebSocket streaming for live video analytics

### Technical Excellence

- **Modular Architecture** - Easily extendable agent framework
- **GPU Acceleration** - Automatic CUDA detection with CPU fallback
- **Async Processing** - FastAPI with async endpoints for high performance
- **Production Ready** - Docker support, logging, metrics, and error handling
- **Type Safety** - Full type hints and Pydantic models
- **Scalable** - Batch processing and concurrent request handling

## 🏗️ Architecture

```txt
vision-sphere/
├── agents/                 # AI Agent implementations
│   ├── base_agent.py      # Abstract base class
│   ├── face_agent.py      # Face detection & recognition
│   ├── object_agent.py    # Object detection (YOLOv8)
│   ├── video_agent.py     # Video analysis & tracking
│   └── classification_agent.py  # Image classification
├── models/                # Downloaded/trained models
├── utils/                 # Common utilities
│   └── helpers.py         # Helper functions
├── server.py              # FastAPI application
├── config.py              # Configuration management
├── cli.py                 # Command-line interface
├── requirements.txt       # Python dependencies
└── Dockerfile            # Container deployment
```

## 🛠️ Installation

### Prerequisites

- Python 3.11+
- CUDA 11.8+ (optional, for GPU acceleration)
- 8GB+ RAM (16GB+ recommended for video processing)

### Quick Start

1. **Clone and Setup**

   ```bash
   git clone <repository-url>
   cd vision-sphere
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Run API Server**

   ```bash
   python server.py
   ```

3. **Access API Documentation**

   - Open [http://localhost:8000/docs](http://localhost:8000/docs) for interactive API docs
   - Or [http://localhost:8000/redoc](http://localhost:8000/redoc) for alternative documentation

### Docker Deployment

```bash
# Build image
docker build -t visionagent .

# Run with GPU support
docker run --gpus all -p 8000:8000 visionagent

# Run CPU-only
docker run -p 8000:8000 visionagent
```

## 📖 Usage

### Command Line Interface

```bash
# Face detection
python cli.py face image.jpg --output results.json

# Object detection  
python cli.py object image.jpg --confidence 0.7 --verbose

# Video analysis
python cli.py video video.mp4 --max-frames 500 --format detailed

# Image classification
python cli.py classify image.jpg --top-k 10 --confidence 0.1

# System information
python cli.py info

# Start server
python cli.py server --host 0.0.0.0 --port 8000
```

### API Endpoints

#### Face Detection

```bash
# Upload file
curl -X POST "http://localhost:8000/face" \
     -F "file=@image.jpg"

# Or use image URL
curl -X POST "http://localhost:8000/face" \
     -H "Content-Type: application/json" \
     -d '{"image_url": "https://example.com/image.jpg"}'
```

#### Object Detection

```bash
curl -X POST "http://localhost:8000/object" \
     -F "file=@image.jpg"
```

#### Video Analysis

```bash
curl -X POST "http://localhost:8000/video" \
     -F "file=@video.mp4"
```

#### Image Classification

```bash
curl -X POST "http://localhost:8000/classify" \
     -F "file=@image.jpg"
```

#### Batch Processing

```bash
curl -X POST "http://localhost:8000/batch/classify" \
     -F "files=@image1.jpg" \
     -F "files=@image2.jpg" \
     -F "files=@image3.jpg"
```

### WebSocket Streaming

```javascript
// Real-time video processing
const ws = new WebSocket('ws://localhost:8000/ws/video');

ws.onopen = function() {
    // Send video frames as binary data
    ws.send(frameData);
};

ws.onmessage = function(event) {
    const result = JSON.parse(event.data);
    console.log('Analysis result:', result);
};
```

## ⚙️ Configuration

Create a `config.yaml` file to customize the framework:

```yaml
# Global settings
default_device: "auto"  # auto, cpu, cuda
model_cache_dir: "./models"
temp_dir: "./temp"

# Face Agent
face_agent:
  enabled: true
  model:
    name: "face_recognition"
    confidence_threshold: 0.6
    custom_params:
      face_detection_model: "hog"  # hog, cnn
      num_jitters: 1
      tolerance: 0.6

# Object Agent
object_agent:
  enabled: true
  model:
    name: "yolov8s.pt"
    confidence_threshold: 0.5
    custom_params:
      iou_threshold: 0.45
      max_detections: 100

# Video Agent
video_agent:
  enabled: true
  processing_params:
    frame_skip: 1
    max_frames: 1000
    track_objects: true
    track_faces: true

# Classification Agent
classification_agent:
  enabled: true
  model:
    name: "microsoft/resnet-50"
    custom_params:
      top_k: 5
      threshold: 0.1
      return_features: false

# Server Configuration
server:
  host: "0.0.0.0"
  port: 8000
  workers: 1
  max_file_size_mb: 100
  enable_websocket: true
  rate_limit_per_minute: 60

# Logging
logging:
  level: "INFO"
  file_path: "./logs/visionagent.log"
  max_file_size_mb: 10
  backup_count: 5
```

### Environment Variables

```bash
# Override configuration with environment variables
export VISIONAGENT_CONFIG=/path/to/config.yaml
export VISIONAGENT_DEVICE=cuda
export VISIONAGENT_HOST=0.0.0.0
export VISIONAGENT_PORT=8000
export VISIONAGENT_LOG_LEVEL=DEBUG
export VISIONAGENT_MODEL_CACHE_DIR=/app/models
```

## 🧩 Extending the Framework

### Creating Custom Agents

```python
from agents.base_agent import BaseAgent, ProcessingResult

class CustomAgent(BaseAgent):
    def initialize(self) -> bool:
        # Initialize your model here
        self._is_initialized = True
        return True
    
    def process(self, input_data: Any) -> ProcessingResult:
        # Implement your processing logic
        try:
            # Your processing code here
            result_data = {"custom_analysis": "results"}
            
            return ProcessingResult(
                success=True,
                data=result_data,
                confidence=0.95,
                inference_time=50.0
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                data={},
                error=str(e)
            )
```

## 📊 API Response Format

All endpoints return standardized responses:

```json
{
  "success": true,
  "data": {
    "detections": [...],
    "detection_count": 5,
    "class_summary": {...}
  },
  "inference_time_ms": 45.2,
  "agent_info": {
    "agent_type": "ObjectAgent",
    "device": "cuda",
    "initialized": true
  },
  "timestamp": "2025-08-31T12:00:00.000Z",
  "request_id": "uuid-string"
}
```

## 🔧 Development

### Setup Development Environment

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-asyncio black flake8 mypy

# Run tests
pytest

# Format code
black .

# Lint code
flake8 .
mypy .
```

### Project Structure Guidelines

- **agents/** - All AI agent implementations inherit from `BaseAgent`
- **models/** - Downloaded model files and weights
- **utils/** - Shared utilities and helper functions
- **server.py** - FastAPI application with all endpoints
- **config.py** - Centralized configuration management
- **cli.py** - Command-line interface for all agents

## 🚀 Production Deployment

sDocker Deployment

```bash
# Build production image
docker build -t visionagent:latest .

# Run with GPU support
docker run --gpus all \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  -e VISIONAGENT_LOG_LEVEL=INFO \
  visionagent:latest

# Docker Compose (recommended)
docker-compose up -d
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: visionagent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: visionagent
  template:
    metadata:
      labels:
        app: visionagent
    spec:
      containers:
      - name: visionagent
        image: visionagent:latest
        ports:
        - containerPort: 8000
        env:
        - name: VISIONAGENT_DEVICE
          value: "cuda"
        resources:
          requests:
            nvidia.com/gpu: 1
          limits:
            nvidia.com/gpu: 1
```

## 📈 Performance Optimization

### GPU Acceleration

- Automatic CUDA detection and device selection
- Batch processing for multiple images
- Memory-efficient model loading

### Scalability Features

- Async FastAPI endpoints
- WebSocket streaming for real-time processing
- Configurable worker processes
- Model caching and lazy loading

## 🔒 Security Considerations

- File size limits for uploads
- Input validation and sanitization
- Non-root container execution
- Rate limiting support
- CORS configuration

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_agents.py
pytest tests/test_api.py
pytest tests/test_utils.py

# Run with coverage
pytest --cov=agents --cov=utils --cov-report=html
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🆘 Support

For issues and questions:

- Check the [API documentation](http://localhost:8000/docs)
- Review the [configuration guide]([def]: #configuration)
- Check system requirements and GPU setup
- Enable debug logging for detailed error information

## 🎯 Roadmap

- [ ] ONNX model support for cross-platform deployment
- [ ] Advanced video tracking algorithms
- [ ] Real-time face recognition optimization
- [ ] Model quantization for edge deployment
- [ ] Multi-camera support
- [ ] Advanced analytics and reporting
- [ ] Model fine-tuning utilities
- [ ] REST API rate limiting and authentication
