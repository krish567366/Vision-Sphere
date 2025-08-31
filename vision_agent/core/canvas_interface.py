"""
Canvas-Based Tool Exploration - 2D spatial navigation for tool workflows
Revolutionary interface for exploring and organizing tool execution flows.
"""

import asyncio
import json
import math
import time
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import logging

logger = logging.getLogger(__name__)

class NodeType(Enum):
    ACTION = "action"
    DATA = "data"
    RESULT = "result"
    CONDITION = "condition"
    LOOP = "loop"

class ConnectionType(Enum):
    SEQUENCE = "sequence"  # A -> B in sequence
    DEPENDENCY = "dependency"  # B depends on A
    CONDITION = "condition"  # A if condition
    PARALLEL = "parallel"  # A || B
    FEEDBACK = "feedback"  # Result feeds back to input

@dataclass
class Position:
    """2D position in canvas space."""
    x: float
    y: float
    
    def distance_to(self, other: 'Position') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

@dataclass
class BoundingBox:
    """Rectangular region in canvas space."""
    top_left: Position
    bottom_right: Position
    
    def contains(self, point: Position) -> bool:
        return (self.top_left.x <= point.x <= self.bottom_right.x and
                self.top_left.y <= point.y <= self.bottom_right.y)
    
    def area(self) -> float:
        width = self.bottom_right.x - self.top_left.x
        height = self.bottom_right.y - self.top_left.y
        return width * height

@dataclass
class CanvasNode:
    """A node in the canvas representing an action or data."""
    id: str
    label: str
    node_type: NodeType
    position: Position
    tool_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    connections: List[str] = field(default_factory=list)  # IDs of connected nodes
    execution_status: str = "pending"  # pending, running, completed, failed
    result_preview: Optional[str] = None
    cost_estimate: float = 0.0
    risk_level: str = "low"
    priority: float = 1.0
    created_at: float = field(default_factory=time.time)

@dataclass
class CanvasConnection:
    """Connection between canvas nodes."""
    id: str
    from_node_id: str
    to_node_id: str
    connection_type: ConnectionType
    weight: float = 1.0
    condition: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class SemanticLayoutEngine:
    """Generate semantic layouts for tool nodes."""
    
    def __init__(self, canvas_width: float = 2000, canvas_height: float = 1500):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.layout_algorithms = {
            'semantic_clustering': self._semantic_clustering_layout,
            'workflow_flow': self._workflow_flow_layout,
            'priority_radial': self._priority_radial_layout,
            'temporal_timeline': self._temporal_timeline_layout
        }
        
    async def calculate_semantic_position(self, action: Dict[str, Any], 
                                        existing_nodes: List[CanvasNode] = None) -> Position:
        """Calculate optimal position based on semantic similarity."""
        if existing_nodes is None:
            existing_nodes = []
        
        # Extract semantic features
        features = await self._extract_semantic_features(action)
        
        if not existing_nodes:
            # First node - place in center
            return Position(self.canvas_width / 2, self.canvas_height / 2)
        
        # Find most similar existing nodes
        similarities = []
        for node in existing_nodes:
            node_features = await self._extract_semantic_features({
                'tool_name': node.tool_name,
                'parameters': node.parameters
            })
            similarity = self._calculate_similarity(features, node_features)
            similarities.append((node, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Position near similar nodes but avoid overlap
        if similarities and similarities[0][1] > 0.7:  # High similarity
            similar_node = similarities[0][0]
            # Place nearby but offset
            offset_angle = np.random.uniform(0, 2 * np.pi)
            offset_distance = 150 + np.random.uniform(0, 50)
            
            new_x = similar_node.position.x + offset_distance * np.cos(offset_angle)
            new_y = similar_node.position.y + offset_distance * np.sin(offset_angle)
            
            # Ensure within canvas bounds
            new_x = max(50, min(self.canvas_width - 50, new_x))
            new_y = max(50, min(self.canvas_height - 50, new_y))
            
            return Position(new_x, new_y)
        
        # Place in open area
        return await self._find_open_position(existing_nodes)
    
    async def _extract_semantic_features(self, action: Dict[str, Any]) -> np.ndarray:
        """Extract semantic features from action for positioning."""
        tool_name = action.get('tool_name', '')
        parameters = action.get('parameters', {})
        
        # Create feature vector based on tool characteristics
        features = np.zeros(20)
        
        # Tool category features
        if 'face' in tool_name.lower():
            features[0] = 1.0
        if 'object' in tool_name.lower():
            features[1] = 1.0
        if 'classify' in tool_name.lower():
            features[2] = 1.0
        if 'video' in tool_name.lower():
            features[3] = 1.0
        if 'image' in tool_name.lower():
            features[4] = 1.0
        
        # Input type features
        if any(key in parameters for key in ['image', 'image_path']):
            features[5] = 1.0
        if any(key in parameters for key in ['video', 'video_path']):
            features[6] = 1.0
        if any(key in parameters for key in ['text', 'query']):
            features[7] = 1.0
        
        # Complexity features
        param_count = len(parameters)
        features[8] = min(1.0, param_count / 10)  # Parameter complexity
        
        # Processing type features
        if any(term in tool_name.lower() for term in ['detect', 'find', 'locate']):
            features[9] = 1.0  # Detection
        if any(term in tool_name.lower() for term in ['analyze', 'process']):
            features[10] = 1.0  # Analysis
        if any(term in tool_name.lower() for term in ['enhance', 'improve']):
            features[11] = 1.0  # Enhancement
        
        return features
    
    def _calculate_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate cosine similarity between feature vectors."""
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(features1, features2) / (norm1 * norm2)
    
    async def _find_open_position(self, existing_nodes: List[CanvasNode]) -> Position:
        """Find an open position that doesn't overlap with existing nodes."""
        min_distance = 200  # Minimum distance between nodes
        max_attempts = 50
        
        for _ in range(max_attempts):
            # Random position
            x = np.random.uniform(100, self.canvas_width - 100)
            y = np.random.uniform(100, self.canvas_height - 100)
            candidate = Position(x, y)
            
            # Check if it's far enough from existing nodes
            too_close = any(
                candidate.distance_to(node.position) < min_distance
                for node in existing_nodes
            )
            
            if not too_close:
                return candidate
        
        # Fallback to grid placement
        grid_x = len(existing_nodes) % 8
        grid_y = len(existing_nodes) // 8
        
        return Position(
            100 + grid_x * 250,
            100 + grid_y * 200
        )
    
    async def apply_layout_algorithm(self, nodes: List[CanvasNode], 
                                   algorithm: str = 'semantic_clustering') -> List[CanvasNode]:
        """Apply specific layout algorithm to nodes."""
        if algorithm in self.layout_algorithms:
            return await self.layout_algorithms[algorithm](nodes)
        else:
            logger.warning(f"Unknown layout algorithm: {algorithm}")
            return nodes
    
    async def _semantic_clustering_layout(self, nodes: List[CanvasNode]) -> List[CanvasNode]:
        """Cluster nodes by semantic similarity."""
        if len(nodes) <= 1:
            return nodes
        
        # Extract features for all nodes
        node_features = []
        for node in nodes:
            features = await self._extract_semantic_features({
                'tool_name': node.tool_name,
                'parameters': node.parameters
            })
            node_features.append(features)
        
        # Simple clustering (K-means would be better)
        clusters = self._simple_clustering(nodes, node_features, num_clusters=min(4, len(nodes)))
        
        # Position clusters
        cluster_centers = [
            Position(
                self.canvas_width * (0.25 + 0.5 * (i % 2)),
                self.canvas_height * (0.25 + 0.5 * (i // 2))
            )
            for i in range(len(clusters))
        ]
        
        # Position nodes within clusters
        updated_nodes = []
        for cluster_idx, cluster_nodes in enumerate(clusters):
            center = cluster_centers[cluster_idx]
            
            for i, node in enumerate(cluster_nodes):
                angle = 2 * np.pi * i / len(cluster_nodes)
                radius = 50 + 20 * (len(cluster_nodes) / 4)
                
                node.position = Position(
                    center.x + radius * np.cos(angle),
                    center.y + radius * np.sin(angle)
                )
                updated_nodes.append(node)
        
        return updated_nodes
    
    def _simple_clustering(self, nodes: List[CanvasNode], features: List[np.ndarray], 
                          num_clusters: int) -> List[List[CanvasNode]]:
        """Simple clustering algorithm."""
        if num_clusters >= len(nodes):
            return [[node] for node in nodes]
        
        # Initialize clusters with random nodes
        clusters = [[] for _ in range(num_clusters)]
        
        # Assign each node to closest cluster (simplified)
        for i, node in enumerate(nodes):
            cluster_idx = i % num_clusters
            clusters[cluster_idx].append(node)
        
        return [cluster for cluster in clusters if cluster]

class InteractiveCanvas:
    """Main interactive canvas for tool exploration."""
    
    def __init__(self, width: float = 2000, height: float = 1500):
        self.width = width
        self.height = height
        self.nodes: Dict[str, CanvasNode] = {}
        self.connections: Dict[str, CanvasConnection] = {}
        self.layout_engine = SemanticLayoutEngine(width, height)
        
        # View state
        self.viewport = BoundingBox(
            Position(0, 0),
            Position(width, height)
        )
        self.zoom_level = 1.0
        self.selected_nodes: Set[str] = set()
        
        # Interaction history
        self.interaction_history: List[Dict[str, Any]] = []
        
    async def add_node(self, action: Dict[str, Any], 
                      position: Optional[Position] = None) -> str:
        """Add a new node to the canvas."""
        node_id = f"node_{len(self.nodes)}_{int(time.time() * 1000)}"
        
        # Calculate position if not provided
        if position is None:
            position = await self.layout_engine.calculate_semantic_position(
                action, list(self.nodes.values())
            )
        
        # Determine node type
        node_type = self._classify_node_type(action)
        
        # Create node
        node = CanvasNode(
            id=node_id,
            label=self._generate_node_label(action),
            node_type=node_type,
            position=position,
            tool_name=action.get('tool_name', 'unknown'),
            parameters=action.get('parameters', {}),
            metadata=action.get('metadata', {}),
            cost_estimate=action.get('cost_estimate', 0.0),
            risk_level=action.get('risk_level', 'low'),
            priority=action.get('priority', 1.0)
        )
        
        self.nodes[node_id] = node
        
        # Log interaction
        self.interaction_history.append({
            'action': 'add_node',
            'node_id': node_id,
            'timestamp': time.time(),
            'position': {'x': position.x, 'y': position.y}
        })
        
        logger.info(f"Added node {node_id}: {node.label} at ({position.x:.0f}, {position.y:.0f})")
        return node_id
    
    async def add_nodes(self, actions: List[Dict[str, Any]]) -> List[str]:
        """Add multiple nodes with automatic layout."""
        node_ids = []
        
        for action in actions:
            node_id = await self.add_node(action)
            node_ids.append(node_id)
        
        # Apply layout algorithm to organize nodes
        await self.apply_layout('semantic_clustering')
        
        return node_ids
    
    def add_connection(self, from_node_id: str, to_node_id: str, 
                      connection_type: ConnectionType = ConnectionType.SEQUENCE,
                      condition: Optional[str] = None) -> str:
        """Add a connection between nodes."""
        if from_node_id not in self.nodes or to_node_id not in self.nodes:
            raise ValueError("Both nodes must exist before creating connection")
        
        connection_id = f"conn_{from_node_id}_{to_node_id}"
        
        connection = CanvasConnection(
            id=connection_id,
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            connection_type=connection_type,
            condition=condition
        )
        
        self.connections[connection_id] = connection
        
        # Update node connections
        if connection_id not in self.nodes[from_node_id].connections:
            self.nodes[from_node_id].connections.append(connection_id)
        if connection_id not in self.nodes[to_node_id].connections:
            self.nodes[to_node_id].connections.append(connection_id)
        
        logger.info(f"Connected {from_node_id} -> {to_node_id} ({connection_type.value})")
        return connection_id
    
    def _classify_node_type(self, action: Dict[str, Any]) -> NodeType:
        """Classify the type of node based on action."""
        tool_name = action.get('tool_name', '').lower()
        
        if any(term in tool_name for term in ['if', 'condition', 'check']):
            return NodeType.CONDITION
        elif any(term in tool_name for term in ['loop', 'repeat', 'iterate']):
            return NodeType.LOOP
        elif any(term in tool_name for term in ['data', 'input', 'load']):
            return NodeType.DATA
        elif any(term in tool_name for term in ['result', 'output', 'save']):
            return NodeType.RESULT
        else:
            return NodeType.ACTION
    
    def _generate_node_label(self, action: Dict[str, Any]) -> str:
        """Generate a descriptive label for the node."""
        tool_name = action.get('tool_name', 'Unknown')
        parameters = action.get('parameters', {})
        
        # Create concise but descriptive label
        if tool_name == 'face_detection':
            return f"Detect Faces"
        elif tool_name == 'object_detection':
            return f"Detect Objects"
        elif tool_name == 'image_classification':
            return f"Classify Image"
        elif tool_name == 'video_processing':
            return f"Process Video"
        elif 'image' in parameters:
            return f"{tool_name.title()} (Image)"
        elif 'video' in parameters:
            return f"{tool_name.title()} (Video)"
        else:
            return tool_name.replace('_', ' ').title()
    
    async def apply_layout(self, algorithm: str = 'semantic_clustering'):
        """Apply layout algorithm to all nodes."""
        nodes_list = list(self.nodes.values())
        updated_nodes = await self.layout_engine.apply_layout_algorithm(nodes_list, algorithm)
        
        # Update node positions
        for updated_node in updated_nodes:
            if updated_node.id in self.nodes:
                self.nodes[updated_node.id].position = updated_node.position
    
    def pan_to_region(self, region: BoundingBox):
        """Pan viewport to show specific region."""
        self.viewport = region
        
        # Log interaction
        self.interaction_history.append({
            'action': 'pan',
            'region': {
                'x': region.top_left.x,
                'y': region.top_left.y,
                'width': region.bottom_right.x - region.top_left.x,
                'height': region.bottom_right.y - region.top_left.y
            },
            'timestamp': time.time()
        })
    
    def zoom_to_level(self, zoom_level: float, center: Optional[Position] = None):
        """Zoom to specific level around center point."""
        if center is None:
            center = Position(
                (self.viewport.top_left.x + self.viewport.bottom_right.x) / 2,
                (self.viewport.top_left.y + self.viewport.bottom_right.y) / 2
            )
        
        self.zoom_level = max(0.1, min(5.0, zoom_level))
        
        # Adjust viewport based on zoom
        view_width = self.width / self.zoom_level
        view_height = self.height / self.zoom_level
        
        self.viewport = BoundingBox(
            Position(
                max(0, center.x - view_width / 2),
                max(0, center.y - view_height / 2)
            ),
            Position(
                min(self.width, center.x + view_width / 2),
                min(self.height, center.y + view_height / 2)
            )
        )
        
        # Log interaction
        self.interaction_history.append({
            'action': 'zoom',
            'zoom_level': zoom_level,
            'center': {'x': center.x, 'y': center.y},
            'timestamp': time.time()
        })
    
    def filter_nodes(self, filter_criteria: Dict[str, Any]) -> List[str]:
        """Filter nodes based on criteria."""
        filtered_node_ids = []
        
        for node_id, node in self.nodes.items():
            matches = True
            
            # Filter by node type
            if 'node_type' in filter_criteria:
                if node.node_type.value != filter_criteria['node_type']:
                    matches = False
            
            # Filter by tool name
            if 'tool_name' in filter_criteria:
                if filter_criteria['tool_name'].lower() not in node.tool_name.lower():
                    matches = False
            
            # Filter by execution status
            if 'status' in filter_criteria:
                if node.execution_status != filter_criteria['status']:
                    matches = False
            
            # Filter by cost range
            if 'max_cost' in filter_criteria:
                if node.cost_estimate > filter_criteria['max_cost']:
                    matches = False
            
            # Filter by risk level
            if 'risk_level' in filter_criteria:
                if node.risk_level != filter_criteria['risk_level']:
                    matches = False
            
            if matches:
                filtered_node_ids.append(node_id)
        
        return filtered_node_ids
    
    def get_nodes_in_region(self, region: BoundingBox) -> List[str]:
        """Get all nodes within a specific region."""
        nodes_in_region = []
        
        for node_id, node in self.nodes.items():
            if region.contains(node.position):
                nodes_in_region.append(node_id)
        
        return nodes_in_region
    
    def render(self) -> Dict[str, Any]:
        """Render the current canvas state."""
        return {
            'canvas': {
                'width': self.width,
                'height': self.height,
                'viewport': {
                    'x': self.viewport.top_left.x,
                    'y': self.viewport.top_left.y,
                    'width': self.viewport.bottom_right.x - self.viewport.top_left.x,
                    'height': self.viewport.bottom_right.y - self.viewport.top_left.y
                },
                'zoom_level': self.zoom_level
            },
            'nodes': [
                {
                    'id': node.id,
                    'label': node.label,
                    'type': node.node_type.value,
                    'position': {'x': node.position.x, 'y': node.position.y},
                    'tool_name': node.tool_name,
                    'status': node.execution_status,
                    'cost': node.cost_estimate,
                    'risk': node.risk_level,
                    'priority': node.priority,
                    'selected': node.id in self.selected_nodes
                }
                for node in self.nodes.values()
            ],
            'connections': [
                {
                    'id': conn.id,
                    'from': conn.from_node_id,
                    'to': conn.to_node_id,
                    'type': conn.connection_type.value,
                    'condition': conn.condition
                }
                for conn in self.connections.values()
            ],
            'stats': {
                'total_nodes': len(self.nodes),
                'total_connections': len(self.connections),
                'completed_nodes': len([n for n in self.nodes.values() if n.execution_status == 'completed']),
                'failed_nodes': len([n for n in self.nodes.values() if n.execution_status == 'failed']),
                'total_cost': sum(n.cost_estimate for n in self.nodes.values())
            }
        }

class WorkflowPlanner:
    """Generate optimal tool workflows from user queries."""
    
    def __init__(self):
        self.tool_templates = {
            'face_analysis': [
                {'tool_name': 'face_detection', 'priority': 1},
                {'tool_name': 'face_recognition', 'priority': 2, 'depends_on': ['face_detection']},
                {'tool_name': 'emotion_analysis', 'priority': 2, 'depends_on': ['face_detection']}
            ],
            'object_analysis': [
                {'tool_name': 'object_detection', 'priority': 1},
                {'tool_name': 'object_classification', 'priority': 2, 'depends_on': ['object_detection']},
                {'tool_name': 'object_tracking', 'priority': 3, 'depends_on': ['object_detection']}
            ],
            'video_analysis': [
                {'tool_name': 'video_preprocessing', 'priority': 1},
                {'tool_name': 'frame_extraction', 'priority': 2, 'depends_on': ['video_preprocessing']},
                {'tool_name': 'face_detection', 'priority': 3, 'depends_on': ['frame_extraction']},
                {'tool_name': 'object_detection', 'priority': 3, 'depends_on': ['frame_extraction']},
                {'tool_name': 'results_aggregation', 'priority': 4, 'depends_on': ['face_detection', 'object_detection']}
            ]
        }
    
    async def generate_actions(self, user_query: str) -> List[Dict[str, Any]]:
        """Generate candidate actions from user query."""
        query_lower = user_query.lower()
        actions = []
        
        # Detect required workflows
        if any(term in query_lower for term in ['face', 'person', 'people']):
            actions.extend(self._instantiate_template('face_analysis', user_query))
        
        if any(term in query_lower for term in ['object', 'thing', 'item', 'detect']):
            actions.extend(self._instantiate_template('object_analysis', user_query))
        
        if any(term in query_lower for term in ['video', 'movie', 'clip']):
            actions.extend(self._instantiate_template('video_analysis', user_query))
        
        # Add general classification if requested
        if any(term in query_lower for term in ['classify', 'categorize', 'identify']):
            actions.append({
                'tool_name': 'image_classification',
                'parameters': self._extract_parameters_from_query(user_query),
                'priority': 1,
                'metadata': {'source': 'user_query'}
            })
        
        # If no specific actions, suggest exploration actions
        if not actions:
            actions = await self._generate_exploration_actions(user_query)
        
        return actions
    
    def _instantiate_template(self, template_name: str, user_query: str) -> List[Dict[str, Any]]:
        """Instantiate a workflow template with user-specific parameters."""
        if template_name not in self.tool_templates:
            return []
        
        template = self.tool_templates[template_name]
        actions = []
        
        # Extract parameters from user query
        base_parameters = self._extract_parameters_from_query(user_query)
        
        for tool_spec in template:
            action = {
                'tool_name': tool_spec['tool_name'],
                'parameters': base_parameters.copy(),
                'priority': tool_spec['priority'],
                'metadata': {
                    'template': template_name,
                    'depends_on': tool_spec.get('depends_on', [])
                }
            }
            actions.append(action)
        
        return actions
    
    def _extract_parameters_from_query(self, user_query: str) -> Dict[str, Any]:
        """Extract parameters from user query."""
        parameters = {}
        
        # Look for file references
        if 'image' in user_query.lower():
            parameters['image_type'] = 'user_provided'
        if 'video' in user_query.lower():
            parameters['video_type'] = 'user_provided'
        
        # Look for quality/performance preferences
        if any(term in user_query.lower() for term in ['fast', 'quick', 'speed']):
            parameters['performance_mode'] = 'fast'
        elif any(term in user_query.lower() for term in ['accurate', 'precise', 'detailed']):
            parameters['performance_mode'] = 'accurate'
        
        # Look for output preferences
        if any(term in user_query.lower() for term in ['json', 'structured']):
            parameters['output_format'] = 'json'
        elif any(term in user_query.lower() for term in ['summary', 'brief']):
            parameters['output_format'] = 'summary'
        
        return parameters
    
    async def _generate_exploration_actions(self, user_query: str) -> List[Dict[str, Any]]:
        """Generate exploratory actions when intent is unclear."""
        return [
            {
                'tool_name': 'query_analysis',
                'parameters': {'query': user_query},
                'priority': 1,
                'metadata': {'type': 'exploration'}
            },
            {
                'tool_name': 'suggest_tools',
                'parameters': {'context': user_query},
                'priority': 2,
                'metadata': {'type': 'suggestion'}
            },
            {
                'tool_name': 'capability_overview',
                'parameters': {},
                'priority': 3,
                'metadata': {'type': 'information'}
            }
        ]
    
    async def augment_actions(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate additional actions based on canvas context."""
        region_nodes = context.get('nodes_in_region', [])
        existing_tools = {node.tool_name for node in region_nodes}
        
        augmented_actions = []
        
        # Suggest complementary tools
        for tool_name in existing_tools:
            complements = self._get_complementary_tools(tool_name)
            for complement in complements:
                if complement not in existing_tools:
                    augmented_actions.append({
                        'tool_name': complement,
                        'parameters': {},
                        'priority': 2,
                        'metadata': {'type': 'complement', 'complements': tool_name}
                    })
        
        # Suggest result analysis tools
        completed_nodes = [node for node in region_nodes if node.execution_status == 'completed']
        if completed_nodes:
            augmented_actions.extend([
                {
                    'tool_name': 'result_comparison',
                    'parameters': {'nodes': [node.id for node in completed_nodes]},
                    'priority': 3,
                    'metadata': {'type': 'analysis'}
                },
                {
                    'tool_name': 'result_visualization',
                    'parameters': {'nodes': [node.id for node in completed_nodes]},
                    'priority': 3,
                    'metadata': {'type': 'visualization'}
                }
            ])
        
        return augmented_actions
    
    def _get_complementary_tools(self, tool_name: str) -> List[str]:
        """Get tools that complement the given tool."""
        complements = {
            'face_detection': ['emotion_analysis', 'age_estimation', 'face_recognition'],
            'object_detection': ['object_classification', 'object_tracking', 'scene_analysis'],
            'image_classification': ['feature_extraction', 'similarity_search', 'confidence_analysis'],
            'video_processing': ['motion_analysis', 'scene_detection', 'highlight_extraction']
        }
        
        return complements.get(tool_name, [])

class CanvasAgentInterface:
    """Main interface for canvas-based tool exploration."""
    
    def __init__(self):
        self.canvas = InteractiveCanvas()
        self.planner = WorkflowPlanner()
        self.tool_nodes: Dict[str, CanvasNode] = {}
        self.execution_queue: List[str] = []
        
    async def generate_tool_graph(self, user_query: str) -> Dict[str, Any]:
        """Generate initial tool graph from user query."""
        # Generate candidate actions
        actions = await self.planner.generate_actions(user_query)
        
        # Add actions to canvas
        node_ids = await self.canvas.add_nodes(actions)
        
        # Create connections based on dependencies
        await self._create_workflow_connections(actions, node_ids)
        
        # Return rendered canvas
        canvas_state = self.canvas.render()
        
        logger.info(f"Generated tool graph with {len(node_ids)} nodes for query: {user_query}")
        
        return {
            'canvas_state': canvas_state,
            'node_ids': node_ids,
            'suggested_actions': [
                'Click nodes to see details',
                'Drag to rearrange layout',
                'Right-click for context menu',
                'Select region to augment with related tools'
            ],
            'query_analysis': {
                'detected_workflows': self._detect_workflows_in_query(user_query),
                'complexity_score': len(actions) / 10,
                'estimated_total_cost': sum(action.get('cost_estimate', 0) for action in actions)
            }
        }
    
    async def augment_region(self, canvas_region: BoundingBox) -> Dict[str, Any]:
        """Augment a selected region with additional tools."""
        # Get nodes in the region
        nodes_in_region_ids = self.canvas.get_nodes_in_region(canvas_region)
        nodes_in_region = [self.canvas.nodes[node_id] for node_id in nodes_in_region_ids]
        
        # Extract context from region
        region_context = {
            'nodes_in_region': nodes_in_region,
            'tool_types': list(set(node.tool_name for node in nodes_in_region)),
            'average_priority': np.mean([node.priority for node in nodes_in_region]) if nodes_in_region else 1.0,
            'region_area': canvas_region.area()
        }
        
        # Generate augmented actions
        new_actions = await self.planner.augment_actions(region_context)
        
        # Add new nodes to canvas
        new_node_ids = []
        for action in new_actions:
            # Position near the region center
            region_center = Position(
                (canvas_region.top_left.x + canvas_region.bottom_right.x) / 2,
                (canvas_region.top_left.y + canvas_region.bottom_right.y) / 2
            )
            
            # Add slight offset for each new node
            offset_angle = np.random.uniform(0, 2 * np.pi)
            offset_distance = 100 + np.random.uniform(0, 50)
            
            position = Position(
                region_center.x + offset_distance * np.cos(offset_angle),
                region_center.y + offset_distance * np.sin(offset_angle)
            )
            
            node_id = await self.canvas.add_node(action, position)
            new_node_ids.append(node_id)
        
        # Create connections to existing nodes if appropriate
        await self._create_augmentation_connections(nodes_in_region_ids, new_node_ids)
        
        logger.info(f"Augmented region with {len(new_node_ids)} new tools")
        
        return {
            'new_nodes': new_node_ids,
            'canvas_state': self.canvas.render(),
            'augmentation_summary': {
                'region_tools': len(nodes_in_region),
                'new_tools': len(new_node_ids),
                'total_tools': len(self.canvas.nodes)
            }
        }
    
    async def _create_workflow_connections(self, actions: List[Dict[str, Any]], 
                                         node_ids: List[str]):
        """Create connections between workflow nodes based on dependencies."""
        node_id_map = {}
        
        # Map actions to node IDs
        for i, action in enumerate(actions):
            if i < len(node_ids):
                node_id_map[action.get('tool_name', f'action_{i}')] = node_ids[i]
        
        # Create connections based on dependencies
        for i, action in enumerate(actions):
            if i < len(node_ids):
                depends_on = action.get('metadata', {}).get('depends_on', [])
                
                for dependency in depends_on:
                    if dependency in node_id_map:
                        self.canvas.add_connection(
                            node_id_map[dependency],
                            node_ids[i],
                            ConnectionType.DEPENDENCY
                        )
    
    async def _create_augmentation_connections(self, existing_node_ids: List[str], 
                                            new_node_ids: List[str]):
        """Create intelligent connections between existing and new nodes."""
        # Connect complementary tools
        for existing_id in existing_node_ids:
            existing_node = self.canvas.nodes[existing_id]
            
            for new_id in new_node_ids:
                new_node = self.canvas.nodes[new_id]
                
                # Check if tools are complementary
                if self._are_tools_complementary(existing_node.tool_name, new_node.tool_name):
                    self.canvas.add_connection(
                        existing_id, new_id, ConnectionType.SEQUENCE
                    )
    
    def _are_tools_complementary(self, tool1: str, tool2: str) -> bool:
        """Check if two tools are complementary."""
        complementary_pairs = [
            ('face_detection', 'emotion_analysis'),
            ('object_detection', 'object_classification'),
            ('face_detection', 'face_recognition'),
            ('video_processing', 'motion_analysis')
        ]
        
        return (tool1, tool2) in complementary_pairs or (tool2, tool1) in complementary_pairs
    
    def _detect_workflows_in_query(self, query: str) -> List[str]:
        """Detect workflow types in user query."""
        workflows = []
        query_lower = query.lower()
        
        if any(term in query_lower for term in ['face', 'person', 'people']):
            workflows.append('face_analysis')
        if any(term in query_lower for term in ['object', 'thing', 'detect']):
            workflows.append('object_analysis')
        if any(term in query_lower for term in ['video', 'clip', 'movie']):
            workflows.append('video_analysis')
        if any(term in query_lower for term in ['classify', 'categorize']):
            workflows.append('classification')
        
        return workflows if workflows else ['general_analysis']
    
    async def execute_workflow(self, node_ids: List[str]) -> Dict[str, Any]:
        """Execute a workflow represented by connected nodes."""
        execution_plan = self._create_execution_plan(node_ids)
        results = {}
        
        for phase in execution_plan:
            phase_results = await self._execute_phase(phase)
            results[f'phase_{len(results)}'] = phase_results
        
        return {
            'workflow_results': results,
            'execution_summary': {
                'total_phases': len(execution_plan),
                'total_nodes': len(node_ids),
                'success_rate': self._calculate_success_rate(results)
            }
        }
    
    def _create_execution_plan(self, node_ids: List[str]) -> List[List[str]]:
        """Create phased execution plan based on dependencies."""
        # Simple topological sort for execution phases
        remaining_nodes = set(node_ids)
        execution_phases = []
        
        while remaining_nodes:
            # Find nodes with no unmet dependencies
            ready_nodes = []
            for node_id in remaining_nodes:
                node = self.canvas.nodes[node_id]
                dependencies = self._get_node_dependencies(node_id)
                
                if all(dep_id not in remaining_nodes for dep_id in dependencies):
                    ready_nodes.append(node_id)
            
            if not ready_nodes:
                # Handle circular dependencies by taking highest priority node
                ready_nodes = [min(remaining_nodes, key=lambda nid: self.canvas.nodes[nid].priority)]
            
            execution_phases.append(ready_nodes)
            remaining_nodes -= set(ready_nodes)
        
        return execution_phases
    
    def _get_node_dependencies(self, node_id: str) -> List[str]:
        """Get dependencies for a node."""
        dependencies = []
        
        for conn in self.canvas.connections.values():
            if (conn.to_node_id == node_id and 
                conn.connection_type in [ConnectionType.DEPENDENCY, ConnectionType.SEQUENCE]):
                dependencies.append(conn.from_node_id)
        
        return dependencies
    
    async def _execute_phase(self, node_ids: List[str]) -> Dict[str, Any]:
        """Execute a phase of nodes in parallel."""
        tasks = []
        
        for node_id in node_ids:
            task = asyncio.create_task(self._execute_node(node_id))
            tasks.append((node_id, task))
        
        results = {}
        for node_id, task in tasks:
            try:
                result = await task
                results[node_id] = result
            except Exception as e:
                results[node_id] = {'error': str(e), 'success': False}
        
        return results
    
    async def _execute_node(self, node_id: str) -> Dict[str, Any]:
        """Execute a single node."""
        node = self.canvas.nodes[node_id]
        
        # Update status
        node.execution_status = 'running'
        
        try:
            # Simulate tool execution
            await asyncio.sleep(np.random.uniform(0.1, 0.5))  # Simulate work
            
            # Generate simulated result
            result = {
                'tool_name': node.tool_name,
                'success': True,
                'output': f"Simulated result from {node.tool_name}",
                'execution_time': np.random.uniform(0.1, 2.0),
                'cost': node.cost_estimate
            }
            
            node.execution_status = 'completed'
            node.result_preview = result['output'][:50]
            
            return result
            
        except Exception as e:
            node.execution_status = 'failed'
            return {'success': False, 'error': str(e)}
    
    def _calculate_success_rate(self, results: Dict[str, Any]) -> float:
        """Calculate success rate for workflow execution."""
        total_executions = 0
        successful_executions = 0
        
        for phase_results in results.values():
            for node_result in phase_results.values():
                total_executions += 1
                if node_result.get('success', False):
                    successful_executions += 1
        
        return successful_executions / max(total_executions, 1)
    
    def export_workflow(self, node_ids: List[str]) -> Dict[str, Any]:
        """Export workflow as reusable template."""
        workflow_nodes = [self.canvas.nodes[node_id] for node_id in node_ids]
        workflow_connections = [
            conn for conn in self.canvas.connections.values()
            if conn.from_node_id in node_ids and conn.to_node_id in node_ids
        ]
        
        return {
            'workflow_template': {
                'name': f"workflow_{int(time.time())}",
                'nodes': [
                    {
                        'tool_name': node.tool_name,
                        'parameters': node.parameters,
                        'position': {'x': node.position.x, 'y': node.position.y},
                        'priority': node.priority
                    }
                    for node in workflow_nodes
                ],
                'connections': [
                    {
                        'from': conn.from_node_id,
                        'to': conn.to_node_id,
                        'type': conn.connection_type.value
                    }
                    for conn in workflow_connections
                ],
                'metadata': {
                    'created_at': time.time(),
                    'node_count': len(workflow_nodes),
                    'estimated_cost': sum(node.cost_estimate for node in workflow_nodes)
                }
            }
        }
