"""
VisionAgent Enhanced Server Entry Point
Starts the production-grade enhanced server with all enterprise features.
"""

import uvicorn
import sys
import argparse

def start_enhanced_server():
    """Start the enhanced VisionAgent server with all enterprise features."""
    parser = argparse.ArgumentParser(description='Start VisionAgent Enhanced Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--workers', type=int, default=1, help='Number of workers')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    
    args = parser.parse_args()
    
    print(f"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                🚀 VisionAgent Enhanced Server 🚀                 ║
    ║              Enterprise-Grade Multi-Modal AI Platform            ║
    ║                                                                  ║
    ║  🌐 Server: http://{args.host}:{args.port}                                 ║
    ║  📊 Dashboard: http://{args.host}:{args.port}/dashboard                    ║
    ║  📈 Analytics: http://{args.host}:{args.port}/analytics                   ║
    ║  🎨 Canvas: http://{args.host}:{args.port}/canvas                         ║
    ║  📖 Docs: http://{args.host}:{args.port}/docs                             ║
    ║                                                                  ║
    ║  ⚡ Features: Token Recycling, Predictive Scaling, Cost Optimization ║
    ║  🛡️ Security: AI Safety Sandbox, Circuit Breakers, Privacy       ║
    ║                                                                  ║
    ║  Authors: Krishna Bajpai & Vedanshi Gupta                       ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "vision_agent.servers.enhanced_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload
    )

if __name__ == "__main__":
    start_enhanced_server()
