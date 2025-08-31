#!/usr/bin/env python3
"""
VisionAgent Launcher - Quick development server
"""

import sys
import subprocess
from pathlib import Path

if __name__ == "__main__":
    # Add current directory to Python path
    vision_sphere_path = Path(__file__).parent.absolute()
    sys.path.insert(0, str(vision_sphere_path))
    
    # Launch the enhanced server
    try:
        from vision_agent.main import main
        main()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Try: pip install -e .")
        sys.exit(1)
