"""
VisionAgent - World-Class Multi-Modal AI Agent Framework
========================================================

A cutting-edge AI agent framework for image, video, and face analytics with revolutionary features:

🚀 **Performance Breakthroughs:**
- Token Recycling Engine: 2x speed improvements through intelligent token prediction
- Byte-Level Processing: 50% FLOP reduction with adaptive patching
- Predictive Resource Scaling: Proactive ML-based resource management

🧠 **Advanced Intelligence:**
- Self-Auditing Cost Predictor: Models that predict cost/risk before execution
- Canvas-Based Tool Exploration: Revolutionary 2D interface for workflow design
- Adaptive Metacognitive Orchestration: Agents that learn and self-improve

🔬 **Research-Inspired Features:**
- Hierarchical Tool Discovery: Dynamic tool creation from natural language
- Emergent Behavior Detection: Monitor for unexpected capabilities  
- Universal Agent Protocol: Interoperability with other frameworks
- Multi-Modal Tool Integration: Unified vision, audio, and language reasoning

🛡️ **Enterprise-Grade Security:**
- AI Safety Sandbox: Advanced isolation for risky operations
- Differential Privacy: Privacy-preserving training data protection
- Circuit Breaker Patterns: Reliability and fault tolerance

Authors: Krishna Bajpai & Vedanshi Gupta

Installation:
    pip install vision-agent-framework

Quick Start:
    ```python
    from vision_agent import VisionAgent, EnhancedServer
    
    # Create agent with all advanced features
    agent = VisionAgent.create_enhanced()
    
    # Start production server
    server = EnhancedServer()
    server.start()
    ```

For complete documentation and examples, visit: https://github.com/krishna-bajpai/vision-agent
"""

from setuptools import setup, find_packages
import os
import re

# Read version from __init__.py
def get_version():
    init_file = os.path.join(os.path.dirname(__file__), 'vision_agent', '__init__.py')
    with open(init_file, 'r', encoding='utf-8') as f:
        content = f.read()
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", content, re.M)
        if version_match:
            return version_match.group(1)
    return '0.1.0'

# Read README for long description
def get_long_description():
    readme_file = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_file):
        with open(readme_file, 'r', encoding='utf-8') as f:
            return f.read()
    return __doc__

# Core dependencies - minimum required
CORE_REQUIREMENTS = [
    'fastapi>=0.104.0',
    'uvicorn[standard]>=0.24.0',
    'numpy>=1.24.0',
    'opencv-python>=4.8.0',
    'Pillow>=10.0.0',
    'pydantic>=2.0.0',
    'python-multipart>=0.0.6',
    'aiofiles>=23.0.0',
    'psutil>=5.9.0',
]

# AI/ML dependencies
AI_REQUIREMENTS = [
    'torch>=2.0.0',
    'torchvision>=0.15.0',
    'transformers>=4.35.0',
    'ultralytics>=8.0.0',
    'scikit-learn>=1.3.0',
    'sentence-transformers>=2.2.0',
]

# Advanced features dependencies
ADVANCED_REQUIREMENTS = [
    'redis>=5.0.0',
    'sqlalchemy>=2.0.0',
    'alembic>=1.12.0',
    'celery>=5.3.0',
    'prometheus-client>=0.19.0',
    'elastic-apm>=6.15.0',
]

# Development dependencies
DEV_REQUIREMENTS = [
    'pytest>=7.4.0',
    'pytest-asyncio>=0.21.0',
    'pytest-cov>=4.1.0',
    'black>=23.0.0',
    'isort>=5.12.0',
    'flake8>=6.0.0',
    'mypy>=1.5.0',
    'pre-commit>=3.4.0',
]

# Optional dependency groups
EXTRAS_REQUIRE = {
    'ai': AI_REQUIREMENTS,
    'advanced': ADVANCED_REQUIREMENTS,
    'dev': DEV_REQUIREMENTS,
    'full': AI_REQUIREMENTS + ADVANCED_REQUIREMENTS,
    'enterprise': AI_REQUIREMENTS + ADVANCED_REQUIREMENTS + [
        'kubernetes>=24.0.0',
        'docker>=6.1.0',
        'boto3>=1.29.0',
        'azure-storage-blob>=12.19.0',
        'google-cloud-storage>=2.10.0',
    ]
}

setup(
    name='vision-agent-framework',
    version=get_version(),
    author='Krishna Bajpai, Vedanshi Gupta',
    author_email='krishna@krishnabajpai.me, vedanshigupta158@gmail.com',
    description='World-Class Multi-Modal AI Agent Framework with Revolutionary Performance Features',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/krishna-bajpai/vision-agent',
    project_urls={
        'Documentation': 'https://vision-agent.readthedocs.io',
        'Bug Reports': 'https://github.com/krishna-bajpai/vision-agent/issues',
        'Source': 'https://github.com/krishna-bajpai/vision-agent',
        'Changelog': 'https://github.com/krishna-bajpai/vision-agent/blob/main/CHANGELOG.md',
    },
    packages=find_packages(exclude=['tests', 'tests.*', 'examples', 'docs']),
    classifiers=[
        # Development Status
        'Development Status :: 4 - Beta',
        
        # Intended Audience
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Information Technology',
        
        # Topic
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Multimedia :: Video',
        'Topic :: Internet :: WWW/HTTP :: HTTP Servers',
        
        # License
        'License :: OSI Approved :: MIT License',
        
        # Programming Language
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        
        # Operating System
        'Operating System :: OS Independent',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS',
        
        # Framework
        'Framework :: AsyncIO',
        'Framework :: FastAPI',
        
        # Environment
        'Environment :: Web Environment',
        'Environment :: GPU :: NVIDIA CUDA',
        
        # Natural Language
        'Natural Language :: English',
    ],
    keywords=[
        'ai', 'machine-learning', 'computer-vision', 'agent-framework',
        'multi-modal', 'face-detection', 'object-detection', 'video-processing',
        'fastapi', 'async', 'performance-optimization', 'enterprise',
        'token-recycling', 'predictive-scaling', 'cost-prediction',
        'canvas-interface', 'workflow-automation', 'differential-privacy'
    ],
    python_requires='>=3.9',
    install_requires=CORE_REQUIREMENTS,
    extras_require=EXTRAS_REQUIRE,
    
    # Entry points for CLI
    entry_points={
        'console_scripts': [
            'vision-agent=vision_agent.cli:main',
            'vision-server=vision_agent.server:start_server',
            'vision-enhanced=vision_agent.enhanced_server:start_enhanced_server',
        ],
    },
    
    # Package data
    include_package_data=True,
    package_data={
        'vision_agent': [
            'models/**/*',
            'configs/**/*',
            'templates/**/*',
            'static/**/*',
        ],
    },
    
    # Zip safety
    zip_safe=False,
    
    # Platform specific
    platforms=['any'],
    
    # Additional metadata
    maintainer='Krishna Bajpai, Vedanshi Gupta',
    maintainer_email='krishna.bajpai@example.com',
    download_url='https://github.com/krishna-bajpai/vision-agent/archive/main.zip',
    
    # Testing
    test_suite='tests',
    tests_require=DEV_REQUIREMENTS,
    
    # Command line interface
    scripts=[],
)
