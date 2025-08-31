"""
VisionAgent Main Entry Point
Launch the enhanced server with all advanced features.

Usage:
    python -m vision_agent.main
    python -m vision_agent.main --port 8080 --host 0.0.0.0
"""

import argparse
import asyncio
import sys
from pathlib import Path

def main():
    """Main entry point for VisionAgent server."""
    parser = argparse.ArgumentParser(description="VisionAgent Enhanced Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    # Print banner
    from . import print_banner
    print_banner()
    
    # Import and run server
    try:
        # Import the app directly
        from .servers.enhanced_server import app
        import uvicorn
        
        # Configure logging
        log_level = "debug" if args.debug else "info"
        
        print(f"\n🚀 Starting VisionAgent Enhanced Server...")
        print(f"📡 Server: http://{args.host}:{args.port}")
        print(f"📊 Analytics: http://{args.host}:{args.port}/analytics")
        print(f"🔍 Health: http://{args.host}:{args.port}/health")
        print(f"📚 Docs: http://{args.host}:{args.port}/docs")
        
        # Run server
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level=log_level,
            workers=args.workers,
            access_log=args.debug
        )
        
    except KeyboardInterrupt:
        print("\n👋 VisionAgent server stopped.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
