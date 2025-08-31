"""
Command Line Interface for VisionAgent Framework
Provides CLI access to all agent functionalities.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from vision_agent import (
    FaceAgent, ObjectAgent, VideoAgent, ClassificationAgent,
    VisionAgent, setup_logging
)
from config import get_config

# Utility functions
def save_results_to_json(data, filepath):
    """Save results to JSON file."""
    import json
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def get_device_info():
    """Get device information."""
    import torch
    info = {
        'cpu_count': 8,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    if torch.cuda.is_available():
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory
    return info


def create_agent(agent_type: str, config_override: Optional[Dict[str, Any]] = None):
    """
    Create and initialize an agent.
    
    Args:
        agent_type: Type of agent to create
        config_override: Configuration overrides
        
    Returns:
        Initialized agent instance
    """
    config = get_config()
    device = config.default_device
    
    if agent_type == 'face':
        agent_config = config.face_agent.model.custom_params or {}
        if config_override:
            agent_config.update(config_override)
        agent = FaceAgent(device=device, config=agent_config)
    elif agent_type == 'object':
        agent_config = config.object_agent.model.custom_params or {}
        agent_config['confidence_threshold'] = config.object_agent.model.confidence_threshold
        if config_override:
            agent_config.update(config_override)
        agent = ObjectAgent(
            device=device,
            model_path=config.object_agent.model.path,
            config=agent_config
        )
    elif agent_type == 'video':
        agent_config = config.video_agent.processing_params or {}
        if config_override:
            agent_config.update(config_override)
        agent = VideoAgent(device=device, config=agent_config)
    elif agent_type == 'classification':
        agent_config = config.classification_agent.model.custom_params or {}
        if config_override:
            agent_config.update(config_override)
        agent = ClassificationAgent(
            device=device,
            model_path=config.classification_agent.model.path,
            config=agent_config
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    if not agent.initialize():
        raise RuntimeError(f"Failed to initialize {agent_type} agent")
    
    return agent


def process_face_command(args):
    """Handle face detection command."""
    print(f"Processing face detection on: {args.input}")
    
    config_override = {}
    if args.confidence:
        config_override['tolerance'] = 1.0 - args.confidence
    
    agent = create_agent('face', config_override)
    result = agent.process(args.input)
    
    if result.success:
        print(f"✅ Found {result.data['face_count']} face(s)")
        print(f"⏱️  Inference time: {result.inference_time:.2f}ms")
        
        for i, face in enumerate(result.data['faces']):
            print(f"  Face {i+1}:")
            print(f"    Recognition: {face['recognition']['name']} ({face['recognition']['confidence']:.3f})")
            bbox = face['bounding_box']
            print(f"    Location: ({bbox['left']}, {bbox['top']}) - ({bbox['right']}, {bbox['bottom']})")
        
        if args.output:
            save_results_to_json(result.data, args.output)
            print(f"💾 Results saved to: {args.output}")
    else:
        print(f"❌ Error: {result.error}")
        return 1
    
    return 0


def process_object_command(args):
    """Handle object detection command."""
    print(f"Processing object detection on: {args.input}")
    
    config_override = {}
    if args.confidence:
        config_override['confidence_threshold'] = args.confidence
    
    agent = create_agent('object', config_override)
    result = agent.process(args.input)
    
    if result.success:
        print(f"✅ Found {result.data['detection_count']} object(s)")
        print(f"⏱️  Inference time: {result.inference_time:.2f}ms")
        
        # Show class summary
        class_summary = result.data['class_summary']
        print("📊 Object Summary:")
        for class_name, count in class_summary.items():
            print(f"  {class_name}: {count}")
        
        if args.verbose:
            print("\n🔍 Detailed Detections:")
            for i, detection in enumerate(result.data['detections']):
                print(f"  {i+1}. {detection['class_name']} ({detection['confidence']:.3f})")
                bbox = detection['bounding_box']
                print(f"     Location: ({bbox['x1']}, {bbox['y1']}) - ({bbox['x2']}, {bbox['y2']})")
        
        if args.output:
            save_results_to_json(result.data, args.output)
            print(f"💾 Results saved to: {args.output}")
    else:
        print(f"❌ Error: {result.error}")
        return 1
    
    return 0


def process_video_command(args):
    """Handle video analysis command."""
    print(f"Processing video analysis on: {args.input}")
    
    config_override = {
        'frame_skip': args.frame_skip,
        'max_frames': args.max_frames,
        'output_format': args.format
    }
    
    agent = create_agent('video', config_override)
    result = agent.process(args.input)
    
    if result.success:
        print(f"✅ Analyzed {result.data['frames_analyzed']} frames")
        print(f"⏱️  Inference time: {result.inference_time:.2f}ms")
        
        if 'summary' in result.data:
            summary = result.data['summary']
            print(f"📊 Summary:")
            print(f"  Objects detected: {summary['total_object_detections']}")
            print(f"  Faces detected: {summary['total_face_detections']}")
            print(f"  Unique objects: {summary['unique_objects']}")
            print(f"  Unique faces: {summary['unique_faces']}")
        
        if args.output:
            save_results_to_json(result.data, args.output)
            print(f"💾 Results saved to: {args.output}")
    else:
        print(f"❌ Error: {result.error}")
        return 1
    
    return 0


def process_classify_command(args):
    """Handle image classification command."""
    print(f"Processing image classification on: {args.input}")
    
    config_override = {}
    if args.confidence:
        config_override['threshold'] = args.confidence
    if args.top_k:
        config_override['top_k'] = args.top_k
    
    agent = create_agent('classification', config_override)
    result = agent.process(args.input)
    
    if result.success:
        predictions = result.data['predictions']
        print(f"✅ Classification complete")
        print(f"⏱️  Inference time: {result.inference_time:.2f}ms")
        print(f"🏆 Top prediction: {result.data['top_prediction']['class_name']} ({result.data['top_prediction']['confidence']:.3f})")
        
        if args.verbose and len(predictions) > 1:
            print(f"\n🔍 All predictions:")
            for i, pred in enumerate(predictions):
                print(f"  {i+1}. {pred['class_name']}: {pred['confidence']:.3f}")
        
        if args.output:
            save_results_to_json(result.data, args.output)
            print(f"💾 Results saved to: {args.output}")
    else:
        print(f"❌ Error: {result.error}")
        return 1
    
    return 0


def info_command(args):
    """Handle info command."""
    print("🖥️  VisionAgent Framework Information")
    print("=" * 50)
    
    # System info
    device_info = get_device_info()
    print(f"CPU cores: {device_info['cpu']['cores']}")
    
    if device_info['cuda']['available']:
        print(f"CUDA available: ✅")
        for device in device_info['cuda']['devices']:
            print(f"  GPU {device['id']}: {device['name']} ({device['memory_gb']:.1f} GB)")
    else:
        print(f"CUDA available: ❌")
    
    # Configuration info
    config = get_config()
    print(f"\nConfiguration:")
    print(f"  Default device: {config.default_device}")
    print(f"  Model cache: {config.model_cache_dir}")
    print(f"  Temp directory: {config.temp_dir}")
    
    # Agent status
    print(f"\nAgent Configuration:")
    agents_config = {
        'face': config.face_agent,
        'object': config.object_agent,
        'video': config.video_agent,
        'classification': config.classification_agent
    }
    
    for name, agent_config in agents_config.items():
        status = "✅" if agent_config.enabled else "❌"
        print(f"  {name.capitalize()} Agent: {status}")
        if agent_config.model:
            print(f"    Model: {agent_config.model.name}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="VisionAgent - Multi-Modal AI Agent Framework CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Global arguments
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], help='Processing device')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Face detection command
    face_parser = subparsers.add_parser('face', help='Face detection and recognition')
    face_parser.add_argument('input', help='Input image path or URL')
    face_parser.add_argument('--confidence', type=float, help='Recognition confidence threshold')
    face_parser.add_argument('--output', help='Output JSON file path')
    
    # Object detection command
    object_parser = subparsers.add_parser('object', help='Object detection')
    object_parser.add_argument('input', help='Input image path or URL')
    object_parser.add_argument('--confidence', type=float, help='Detection confidence threshold')
    object_parser.add_argument('--output', help='Output JSON file path')
    
    # Video analysis command
    video_parser = subparsers.add_parser('video', help='Video analysis')
    video_parser.add_argument('input', help='Input video path or URL')
    video_parser.add_argument('--frame-skip', type=int, default=1, help='Frames to skip between analysis')
    video_parser.add_argument('--max-frames', type=int, default=100, help='Maximum frames to process')
    video_parser.add_argument('--format', choices=['summary', 'detailed'], default='summary', help='Output format')
    video_parser.add_argument('--output', help='Output JSON file path')
    
    # Classification command
    classify_parser = subparsers.add_parser('classify', help='Image classification')
    classify_parser.add_argument('input', help='Input image path or URL')
    classify_parser.add_argument('--confidence', type=float, help='Classification confidence threshold')
    classify_parser.add_argument('--top-k', type=int, help='Number of top predictions to return')
    classify_parser.add_argument('--output', help='Output JSON file path')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show system and configuration information')
    
    # Server command
    server_parser = subparsers.add_parser('server', help='Start API server')
    server_parser.add_argument('--host', help='Server host')
    server_parser.add_argument('--port', type=int, help='Server port')
    server_parser.add_argument('--workers', type=int, help='Number of workers')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logging({'level': log_level})
    
    # Handle commands
    try:
        if args.command == 'face':
            return process_face_command(args)
        elif args.command == 'object':
            return process_object_command(args)
        elif args.command == 'video':
            return process_video_command(args)
        elif args.command == 'classify':
            return process_classify_command(args)
        elif args.command == 'info':
            info_command(args)
            return 0
        elif args.command == 'server':
            # Start server
            import uvicorn
            config = get_config()
            
            host = args.host or config.server.host
            port = args.port or config.server.port
            workers = args.workers or config.server.workers
            
            print(f"🚀 Starting VisionAgent API server on {host}:{port}")
            uvicorn.run(
                "server:app",
                host=host,
                port=port,
                workers=workers,
                reload=False
            )
            return 0
        else:
            print(f"Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
        return 0
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
