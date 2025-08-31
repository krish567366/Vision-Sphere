"""
VisionAgent Servers Module
=========================

Production-grade servers with enterprise features:
- Enhanced Server: Full enterprise features
- Async Server: High-performance async processing
- Simple Enhanced Server: Lightweight version
"""

# Note: Import the app objects, not modules to avoid circular imports
try:
    from .enhanced_server import app as enhanced_app
except ImportError:
    enhanced_app = None

try:
    from .async_server import app as async_app  
except ImportError:
    async_app = None

try:
    from .simple_enhanced_server import app as simple_app
except ImportError:
    simple_app = None

__all__ = [
    'enhanced_app',
    'async_app', 
    'simple_app',
]
