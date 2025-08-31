"""
VisionAgent Server Entry Point
Simple wrapper to start the enhanced server.
"""

import uvicorn
import sys
import argparse
from .servers.enhanced_server import app

def start_server():
    """Start the enhanced VisionAgent server."""
    parser = argparse.ArgumentParser(description='Start VisionAgent Enhanced Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--workers', type=int, default=1, help='Number of workers')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    
    args = parser.parse_args()
    
    print(f"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                     🚀 VisionAgent Server 🚀                     ║
    ║              World-Class Multi-Modal AI Framework                ║
    ║                                                                  ║
    ║  🌐 Server: http://{args.host}:{args.port}                                 ║
    ║  📊 Dashboard: http://{args.host}:{args.port}/dashboard                    ║
    ║  📖 Docs: http://{args.host}:{args.port}/docs                             ║
    ║                                                                  ║
    ║  Authors: Krishna Bajpai & Vedanshi Gupta                       ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload
    )

if __name__ == "__main__":
    start_server()
