"""
Self-Auditing Cost Predictor - Models that predict cost/risk before tool execution
Provides intelligent cost-benefit analysis and safety checks.
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import hashlib

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class CostType(Enum):
    COMPUTE = "compute"
    MEMORY = "memory"
    NETWORK = "network"
    STORAGE = "storage"
    API_CALLS = "api_calls"
    TIME = "time"

@dataclass
class CostEstimate:
    """Estimated cost for a tool execution."""
    compute_cost: float = 0.0
    memory_cost: float = 0.0
    network_cost: float = 0.0
    storage_cost: float = 0.0
    api_cost: float = 0.0
    time_cost: float = 0.0
    total_cost: float = 0.0
    confidence: float = 0.0
    
    def __post_init__(self):
        self.total_cost = (
            self.compute_cost + self.memory_cost + self.network_cost +
            self.storage_cost + self.api_cost + self.time_cost
        )

@dataclass
class RiskAssessment:
    """Risk assessment for a tool execution."""
    risk_level: RiskLevel
    risk_factors: List[str]
    mitigation_strategies: List[str]
    confidence: float
    requires_confirmation: bool = False

@dataclass
class ToolExecutionHistory:
    """Historical data for a tool execution."""
    tool_name: str
    parameters: Dict[str, Any]
    actual_cost: CostEstimate
    execution_time: float
    success: bool
    timestamp: float
    resource_usage: Dict[str, float] = field(default_factory=dict)

class CostModelingEngine:
    """Advanced cost modeling for different tool types."""
    
    def __init__(self):
        self.tool_cost_models: Dict[str, Dict[str, float]] = {
            # Vision processing tools
            'face_detection': {
                'base_cost': 0.001,
                'per_pixel': 0.000001,
                'per_face': 0.0005,
                'complexity_multiplier': 1.2
            },
            'object_detection': {
                'base_cost': 0.002,
                'per_pixel': 0.000002,
                'per_object': 0.001,
                'complexity_multiplier': 1.5
            },
            'image_classification': {
                'base_cost': 0.0015,
                'per_pixel': 0.0000015,
                'per_class': 0.0001,
                'complexity_multiplier': 1.1
            },
            'video_processing': {
                'base_cost': 0.01,
                'per_frame': 0.005,
                'per_second': 0.02,
                'complexity_multiplier': 2.0
            },
            
            # General tools
            'file_operations': {
                'base_cost': 0.0001,
                'per_mb': 0.00001,
                'complexity_multiplier': 1.0
            },
            'network_requests': {
                'base_cost': 0.0005,
                'per_request': 0.0001,
                'per_mb': 0.00005,
                'complexity_multiplier': 1.3
            },
            'database_operations': {
                'base_cost': 0.001,
                'per_query': 0.0005,
                'per_row': 0.00001,
                'complexity_multiplier': 1.4
            }
        }
        
        # Historical execution data for learning
        self.execution_history: deque = deque(maxlen=1000)
        self.tool_performance_stats: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {'avg_time': 0, 'success_rate': 1.0, 'avg_cost': 0}
        )
    
    async def estimate_tool_cost(self, tool_name: str, parameters: Dict[str, Any]) -> CostEstimate:
        """Estimate the cost of executing a tool with given parameters."""
        try:
            # Get base cost model for tool type
            tool_type = self._classify_tool_type(tool_name)
            base_model = self.tool_cost_models.get(tool_type, self.tool_cost_models['file_operations'])
            
            # Calculate different cost components
            compute_cost = await self._estimate_compute_cost(tool_name, parameters, base_model)
            memory_cost = await self._estimate_memory_cost(tool_name, parameters, base_model)
            network_cost = await self._estimate_network_cost(tool_name, parameters, base_model)
            storage_cost = await self._estimate_storage_cost(tool_name, parameters, base_model)
            api_cost = await self._estimate_api_cost(tool_name, parameters, base_model)
            time_cost = await self._estimate_time_cost(tool_name, parameters, base_model)
            
            # Calculate confidence based on historical data
            confidence = self._calculate_cost_confidence(tool_name, parameters)
            
            return CostEstimate(
                compute_cost=compute_cost,
                memory_cost=memory_cost,
                network_cost=network_cost,
                storage_cost=storage_cost,
                api_cost=api_cost,
                time_cost=time_cost,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Cost estimation failed for {tool_name}: {e}")
            # Return conservative high estimate
            return CostEstimate(
                compute_cost=0.01,
                memory_cost=0.005,
                network_cost=0.002,
                storage_cost=0.001,
                api_cost=0.003,
                time_cost=0.01,
                confidence=0.3
            )
    
    def _classify_tool_type(self, tool_name: str) -> str:
        """Classify tool into cost model category."""
        tool_lower = tool_name.lower()
        
        if any(term in tool_lower for term in ['face', 'detect_faces']):
            return 'face_detection'
        elif any(term in tool_lower for term in ['object', 'detect_objects', 'yolo']):
            return 'object_detection'
        elif any(term in tool_lower for term in ['classify', 'classification']):
            return 'image_classification'
        elif any(term in tool_lower for term in ['video', 'process_video']):
            return 'video_processing'
        elif any(term in tool_lower for term in ['file', 'read', 'write']):
            return 'file_operations'
        elif any(term in tool_lower for term in ['http', 'request', 'fetch']):
            return 'network_requests'
        elif any(term in tool_lower for term in ['query', 'database', 'sql']):
            return 'database_operations'
        else:
            return 'file_operations'  # Default category
    
    async def _estimate_compute_cost(self, tool_name: str, parameters: Dict[str, Any], 
                                   base_model: Dict[str, float]) -> float:
        """Estimate computational cost."""
        base_cost = base_model['base_cost']
        complexity = self._analyze_parameter_complexity(parameters)
        multiplier = base_model['complexity_multiplier']
        
        # Adjust for input size
        input_size_factor = 1.0
        if 'image' in parameters:
            input_size_factor = self._estimate_image_size_factor(parameters['image'])
        elif 'video' in parameters:
            input_size_factor = self._estimate_video_size_factor(parameters['video'])
        elif 'text' in parameters:
            input_size_factor = len(str(parameters['text'])) / 1000  # Per 1K characters
        
        return base_cost * complexity * multiplier * input_size_factor
    
    async def _estimate_memory_cost(self, tool_name: str, parameters: Dict[str, Any], 
                                  base_model: Dict[str, float]) -> float:
        """Estimate memory usage cost."""
        base_memory = 0.001  # Base memory cost
        
        # Memory scales with input size
        if 'image' in parameters:
            # Estimate based on image resolution
            size_factor = self._estimate_image_size_factor(parameters['image'])
            return base_memory * size_factor * 2  # Images require more memory
        elif 'video' in parameters:
            size_factor = self._estimate_video_size_factor(parameters['video'])
            return base_memory * size_factor * 5  # Video requires much more memory
        
        return base_memory
    
    async def _estimate_network_cost(self, tool_name: str, parameters: Dict[str, Any], 
                                   base_model: Dict[str, float]) -> float:
        """Estimate network usage cost."""
        if any(term in tool_name.lower() for term in ['fetch', 'download', 'upload', 'api']):
            return base_model.get('per_request', 0.0001)
        return 0.0
    
    async def _estimate_storage_cost(self, tool_name: str, parameters: Dict[str, Any], 
                                   base_model: Dict[str, float]) -> float:
        """Estimate storage cost."""
        if any(term in tool_name.lower() for term in ['save', 'cache', 'store']):
            return 0.0001  # Small storage cost
        return 0.0
    
    async def _estimate_api_cost(self, tool_name: str, parameters: Dict[str, Any], 
                               base_model: Dict[str, float]) -> float:
        """Estimate external API costs."""
        # Check if tool makes external API calls
        if any(term in tool_name.lower() for term in ['api', 'external', 'cloud']):
            return 0.005  # API call cost
        return 0.0
    
    async def _estimate_time_cost(self, tool_name: str, parameters: Dict[str, Any], 
                                base_model: Dict[str, float]) -> float:
        """Estimate time-based cost (opportunity cost)."""
        # Use historical data if available
        if tool_name in self.tool_performance_stats:
            avg_time = self.tool_performance_stats[tool_name]['avg_time']
            return avg_time * 0.001  # $0.001 per second
        
        # Estimate based on tool type
        tool_type = self._classify_tool_type(tool_name)
        time_estimates = {
            'face_detection': 0.5,
            'object_detection': 1.0,
            'image_classification': 0.3,
            'video_processing': 5.0,
            'file_operations': 0.1,
            'network_requests': 2.0,
            'database_operations': 0.2
        }
        
        estimated_time = time_estimates.get(tool_type, 1.0)
        return estimated_time * 0.001
    
    def _analyze_parameter_complexity(self, parameters: Dict[str, Any]) -> float:
        """Analyze complexity of parameters."""
        complexity_score = 1.0
        
        # Count number of parameters
        param_count_factor = min(2.0, 1.0 + len(parameters) / 10)
        complexity_score *= param_count_factor
        
        # Analyze parameter values
        for key, value in parameters.items():
            if isinstance(value, (list, tuple)) and len(value) > 10:
                complexity_score *= 1.2  # Lists increase complexity
            elif isinstance(value, dict) and len(value) > 5:
                complexity_score *= 1.3  # Nested structures increase complexity
            elif isinstance(value, str) and len(value) > 1000:
                complexity_score *= 1.1  # Large strings increase complexity
        
        return min(complexity_score, 3.0)  # Cap at 3x complexity
    
    def _estimate_image_size_factor(self, image_param: Any) -> float:
        """Estimate size factor for image parameters."""
        # Default medium size
        size_factor = 1.0
        
        if isinstance(image_param, str):
            # Estimate based on string length (could be path or base64)
            if len(image_param) > 100000:  # Likely base64
                size_factor = len(image_param) / 100000
            else:
                size_factor = 1.0  # File path
        elif isinstance(image_param, dict) and 'width' in image_param and 'height' in image_param:
            # Explicit dimensions
            pixels = image_param['width'] * image_param['height']
            size_factor = pixels / (640 * 480)  # Normalize to VGA
        
        return min(size_factor, 10.0)  # Cap at 10x
    
    def _estimate_video_size_factor(self, video_param: Any) -> float:
        """Estimate size factor for video parameters."""
        size_factor = 2.0  # Base video factor
        
        if isinstance(video_param, dict):
            if 'duration' in video_param:
                duration = float(video_param['duration'])
                size_factor *= min(duration / 30, 10)  # Scale with duration, cap at 10x
            if 'resolution' in video_param:
                # Estimate based on resolution
                if '4K' in str(video_param['resolution']) or '2160' in str(video_param['resolution']):
                    size_factor *= 4
                elif 'HD' in str(video_param['resolution']) or '1080' in str(video_param['resolution']):
                    size_factor *= 2
        
        return min(size_factor, 20.0)  # Cap at 20x
    
    def _calculate_cost_confidence(self, tool_name: str, parameters: Dict[str, Any]) -> float:
        """Calculate confidence in cost estimate."""
        if tool_name not in self.tool_performance_stats:
            return 0.5  # Medium confidence for new tools
        
        stats = self.tool_performance_stats[tool_name]
        
        # Confidence based on historical data and success rate
        data_confidence = min(1.0, len(self.execution_history) / 100)
        success_confidence = stats['success_rate']
        
        return (data_confidence + success_confidence) / 2
    
    def add_execution_result(self, tool_name: str, parameters: Dict[str, Any],
                           actual_cost: CostEstimate, execution_time: float, 
                           success: bool, resource_usage: Dict[str, float]):
        """Add actual execution result for learning."""
        history_entry = ToolExecutionHistory(
            tool_name=tool_name,
            parameters=parameters,
            actual_cost=actual_cost,
            execution_time=execution_time,
            success=success,
            timestamp=time.time(),
            resource_usage=resource_usage
        )
        
        self.execution_history.append(history_entry)
        
        # Update performance stats
        stats = self.tool_performance_stats[tool_name]
        
        # Update averages with exponential moving average
        alpha = 0.1  # Learning rate
        stats['avg_time'] = stats['avg_time'] * (1 - alpha) + execution_time * alpha
        stats['avg_cost'] = stats['avg_cost'] * (1 - alpha) + actual_cost.total_cost * alpha
        
        # Update success rate
        total_executions = sum(1 for h in self.execution_history if h.tool_name == tool_name)
        successful_executions = sum(1 for h in self.execution_history 
                                  if h.tool_name == tool_name and h.success)
        stats['success_rate'] = successful_executions / max(total_executions, 1)

class RiskAssessmentEngine:
    """Advanced risk assessment for tool executions."""
    
    def __init__(self):
        self.risk_patterns = {
            # File system risks
            'file_deletion': {'level': RiskLevel.HIGH, 'keywords': ['delete', 'remove', 'rm']},
            'system_modification': {'level': RiskLevel.CRITICAL, 'keywords': ['system', 'registry', 'config']},
            'large_file_operations': {'level': RiskLevel.MEDIUM, 'keywords': ['large', 'big', 'huge']},
            
            # Network risks
            'external_api': {'level': RiskLevel.MEDIUM, 'keywords': ['api', 'external', 'remote']},
            'data_upload': {'level': RiskLevel.HIGH, 'keywords': ['upload', 'send', 'transmit']},
            'download': {'level': RiskLevel.MEDIUM, 'keywords': ['download', 'fetch', 'get']},
            
            # Processing risks
            'heavy_computation': {'level': RiskLevel.MEDIUM, 'keywords': ['process', 'analyze', 'compute']},
            'memory_intensive': {'level': RiskLevel.MEDIUM, 'keywords': ['video', 'large', 'batch']},
            
            # Privacy risks
            'personal_data': {'level': RiskLevel.HIGH, 'keywords': ['face', 'person', 'identity']},
            'sensitive_content': {'level': RiskLevel.HIGH, 'keywords': ['private', 'sensitive', 'confidential']}
        }
        
        self.mitigation_strategies = {
            RiskLevel.LOW: ['Monitor execution', 'Standard logging'],
            RiskLevel.MEDIUM: ['Enhanced monitoring', 'Resource limits', 'Timeout protection'],
            RiskLevel.HIGH: ['User confirmation required', 'Sandboxed execution', 'Detailed audit log'],
            RiskLevel.CRITICAL: ['Administrative approval', 'Full isolation', 'Legal review']
        }
    
    async def assess_risk(self, tool_name: str, parameters: Dict[str, Any]) -> RiskAssessment:
        """Assess the risk of executing a tool."""
        try:
            risk_factors = []
            max_risk_level = RiskLevel.LOW
            
            # Analyze tool name for risk patterns
            tool_lower = tool_name.lower()
            for risk_pattern, details in self.risk_patterns.items():
                if any(keyword in tool_lower for keyword in details['keywords']):
                    risk_factors.append(f"Tool type: {risk_pattern}")
                    if details['level'].value > max_risk_level.value:
                        max_risk_level = details['level']
            
            # Analyze parameters for risks
            param_risks = await self._analyze_parameter_risks(parameters)
            risk_factors.extend(param_risks['factors'])
            if param_risks['level'].value > max_risk_level.value:
                max_risk_level = param_risks['level']
            
            # Analyze execution context
            context_risks = await self._analyze_execution_context()
            risk_factors.extend(context_risks['factors'])
            if context_risks['level'].value > max_risk_level.value:
                max_risk_level = context_risks['level']
            
            # Determine if user confirmation is required
            requires_confirmation = max_risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
            
            # Calculate confidence
            confidence = self._calculate_risk_confidence(tool_name, risk_factors)
            
            # Get mitigation strategies
            mitigation = self.mitigation_strategies.get(max_risk_level, ['Monitor execution'])
            
            return RiskAssessment(
                risk_level=max_risk_level,
                risk_factors=risk_factors,
                mitigation_strategies=mitigation,
                confidence=confidence,
                requires_confirmation=requires_confirmation
            )
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            # Return high-risk assessment as safety fallback
            return RiskAssessment(
                risk_level=RiskLevel.HIGH,
                risk_factors=[f"Assessment error: {str(e)}"],
                mitigation_strategies=['User confirmation required', 'Enhanced monitoring'],
                confidence=0.5,
                requires_confirmation=True
            )
    
    async def _analyze_parameter_risks(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze parameters for risk indicators."""
        risk_factors = []
        risk_level = RiskLevel.LOW
        
        for key, value in parameters.items():
            key_lower = key.lower()
            value_str = str(value).lower()
            
            # File path risks
            if 'path' in key_lower or 'file' in key_lower:
                if any(danger in value_str for danger in ['system', 'windows', 'program files', '/etc']):
                    risk_factors.append(f"System path access: {key}")
                    risk_level = max(risk_level, RiskLevel.HIGH, key=lambda x: x.value)
                elif '..' in value_str:
                    risk_factors.append(f"Directory traversal risk: {key}")
                    risk_level = max(risk_level, RiskLevel.MEDIUM, key=lambda x: x.value)
            
            # Size-based risks
            if isinstance(value, (list, tuple)) and len(value) > 1000:
                risk_factors.append(f"Large data structure: {key} ({len(value)} items)")
                risk_level = max(risk_level, RiskLevel.MEDIUM, key=lambda x: x.value)
            
            # Content-based risks
            if any(sensitive in value_str for sensitive in ['password', 'secret', 'key', 'token']):
                risk_factors.append(f"Sensitive data in parameters: {key}")
                risk_level = max(risk_level, RiskLevel.HIGH, key=lambda x: x.value)
        
        return {'factors': risk_factors, 'level': risk_level}
    
    async def _analyze_execution_context(self) -> Dict[str, Any]:
        """Analyze current execution context for risks."""
        risk_factors = []
        risk_level = RiskLevel.LOW
        
        try:
            # Check system resources
            import psutil
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            if cpu_percent > 90:
                risk_factors.append(f"High CPU usage: {cpu_percent:.1f}%")
                risk_level = max(risk_level, RiskLevel.MEDIUM, key=lambda x: x.value)
            
            if memory_percent > 90:
                risk_factors.append(f"High memory usage: {memory_percent:.1f}%")
                risk_level = max(risk_level, RiskLevel.MEDIUM, key=lambda x: x.value)
            
            # Check time of day (higher risk during off-hours)
            current_hour = time.localtime().tm_hour
            if current_hour < 6 or current_hour > 22:
                risk_factors.append("Execution during off-hours")
                risk_level = max(risk_level, RiskLevel.MEDIUM, key=lambda x: x.value)
            
        except Exception as e:
            risk_factors.append(f"Context analysis error: {str(e)}")
            risk_level = RiskLevel.MEDIUM
        
        return {'factors': risk_factors, 'level': risk_level}
    
    def _calculate_risk_confidence(self, tool_name: str, risk_factors: List[str]) -> float:
        """Calculate confidence in risk assessment."""
        base_confidence = 0.8
        
        # Lower confidence if many risk factors (might be false positives)
        if len(risk_factors) > 5:
            base_confidence -= 0.2
        
        # Lower confidence for unknown tools
        if not any(pattern in tool_name.lower() for patterns in self.risk_patterns.values() 
                  for pattern in patterns['keywords']):
            base_confidence -= 0.1
        
        return max(0.1, base_confidence)

class CostPredictor:
    """Main cost prediction orchestrator."""
    
    def __init__(self, budget_threshold: float = 1.0, safety_threshold: float = 0.7):
        self.cost_engine = CostModelingEngine()
        self.risk_engine = RiskAssessmentEngine()
        self.budget_threshold = budget_threshold
        self.safety_threshold = safety_threshold
        
        # Budget tracking
        self.current_budget = 10.0  # $10 default budget
        self.spent_today = 0.0
        self.execution_queue: List[Dict[str, Any]] = []
        
    async def audit_before_execution(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive audit before tool execution."""
        tool_name = tool_call.get('tool_name', 'unknown')
        parameters = tool_call.get('parameters', {})
        
        try:
            # Estimate costs
            cost_estimate = await self.cost_engine.estimate_tool_cost(tool_name, parameters)
            
            # Assess risks
            risk_assessment = await self.risk_engine.assess_risk(tool_name, parameters)
            
            # Make execution decision
            decision = await self._make_execution_decision(cost_estimate, risk_assessment, tool_call)
            
            return {
                'approved': decision['approved'],
                'estimated_cost': cost_estimate,
                'risk_assessment': risk_assessment,
                'decision_reason': decision['reason'],
                'alternative_suggestion': decision.get('alternative'),
                'mitigation_required': decision.get('mitigation_required', False)
            }
            
        except Exception as e:
            logger.error(f"Audit failed for {tool_name}: {e}")
            return {
                'approved': False,
                'estimated_cost': CostEstimate(total_cost=999.0, confidence=0.1),
                'risk_assessment': RiskAssessment(
                    risk_level=RiskLevel.HIGH,
                    risk_factors=[f"Audit error: {str(e)}"],
                    mitigation_strategies=['Manual review required'],
                    confidence=0.1
                ),
                'decision_reason': f'Audit system error: {str(e)}',
                'mitigation_required': True
            }
    
    async def _make_execution_decision(self, cost_estimate: CostEstimate, 
                                     risk_assessment: RiskAssessment,
                                     tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Make intelligent execution decision."""
        
        # Check budget constraints
        if cost_estimate.total_cost > self.budget_threshold:
            alternative = await self._suggest_cheaper_alternative(tool_call)
            return {
                'approved': False,
                'reason': f'Cost ${cost_estimate.total_cost:.4f} exceeds threshold ${self.budget_threshold}',
                'alternative': alternative
            }
        
        # Check remaining budget
        if self.spent_today + cost_estimate.total_cost > self.current_budget:
            return {
                'approved': False,
                'reason': f'Insufficient budget. Remaining: ${self.current_budget - self.spent_today:.2f}',
                'alternative': await self._suggest_budget_friendly_alternative(tool_call)
            }
        
        # Check risk levels
        if risk_assessment.risk_level == RiskLevel.CRITICAL:
            return {
                'approved': False,
                'reason': 'Critical risk level requires administrative approval',
                'mitigation_required': True
            }
        
        if risk_assessment.risk_level == RiskLevel.HIGH and risk_assessment.requires_confirmation:
            return {
                'approved': False,
                'reason': 'High risk requires user confirmation',
                'mitigation_required': True,
                'confirmation_required': True
            }
        
        # Check confidence levels
        if (cost_estimate.confidence < 0.3 and cost_estimate.total_cost > 0.1) or \
           (risk_assessment.confidence < 0.3):
            return {
                'approved': False,
                'reason': 'Low confidence in cost/risk estimates requires manual review'
            }
        
        # Approve with potential mitigation
        mitigation_required = (
            risk_assessment.risk_level == RiskLevel.MEDIUM or
            cost_estimate.total_cost > self.budget_threshold * 0.5
        )
        
        return {
            'approved': True,
            'reason': 'Cost and risk within acceptable limits',
            'mitigation_required': mitigation_required
        }
    
    async def _suggest_cheaper_alternative(self, tool_call: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Suggest a cheaper alternative to the requested tool."""
        tool_name = tool_call.get('tool_name', '')
        
        # Mapping of expensive tools to cheaper alternatives
        alternatives = {
            'video_processing': 'image_processing',  # Process single frames instead
            'object_detection': 'simple_detection',  # Use simpler model
            'high_res_analysis': 'low_res_analysis',  # Reduce resolution
            'batch_processing': 'single_processing',  # Process one at a time
        }
        
        for expensive, cheap in alternatives.items():
            if expensive in tool_name.lower():
                return {
                    'tool_name': cheap,
                    'parameters': tool_call.get('parameters', {}),
                    'cost_reduction': '60-80%',
                    'quality_impact': 'Moderate reduction in accuracy/detail'
                }
        
        # Generic cost reduction suggestions
        return {
            'suggestion': 'Consider reducing input size, resolution, or batch size',
            'estimated_savings': '30-50%'
        }
    
    async def _suggest_budget_friendly_alternative(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest budget-friendly alternatives."""
        return {
            'options': [
                'Schedule for off-peak hours (50% cost reduction)',
                'Use cached results if available',
                'Reduce quality/resolution settings',
                'Process in smaller batches'
            ],
            'recommended': 'Use cached results or schedule for later'
        }
    
    async def request_user_confirmation(self, tool_call: Dict[str, Any], 
                                      cost_estimate: CostEstimate,
                                      risk_assessment: RiskAssessment) -> bool:
        """Request user confirmation for high-risk/cost operations."""
        
        confirmation_message = f"""
        ⚠️  CONFIRMATION REQUIRED ⚠️
        
        Tool: {tool_call.get('tool_name', 'Unknown')}
        Estimated Cost: ${cost_estimate.total_cost:.4f}
        Risk Level: {risk_assessment.risk_level.value.upper()}
        
        Risk Factors:
        {chr(10).join(f'• {factor}' for factor in risk_assessment.risk_factors)}
        
        Mitigation Strategies:
        {chr(10).join(f'• {strategy}' for strategy in risk_assessment.mitigation_strategies)}
        
        Do you want to proceed? (y/n):
        """
        
        # In a real implementation, this would prompt the user
        # For now, we'll return False for high-risk operations
        logger.warning(confirmation_message)
        
        # Simulate user decision based on risk level
        if risk_assessment.risk_level == RiskLevel.CRITICAL:
            return False
        elif risk_assessment.risk_level == RiskLevel.HIGH:
            return cost_estimate.total_cost < 0.1  # Only approve if cost is very low
        else:
            return True
    
    async def record_execution_result(self, tool_name: str, parameters: Dict[str, Any],
                                    estimated_cost: CostEstimate, actual_cost: float,
                                    execution_time: float, success: bool,
                                    resource_usage: Dict[str, float]):
        """Record actual execution results for learning."""
        
        # Create actual cost estimate
        actual_cost_estimate = CostEstimate(
            total_cost=actual_cost,
            compute_cost=actual_cost * 0.4,
            memory_cost=actual_cost * 0.2,
            time_cost=actual_cost * 0.3,
            api_cost=actual_cost * 0.1,
            confidence=1.0  # Actual measurement
        )
        
        # Update cost model
        self.cost_engine.add_execution_result(
            tool_name, parameters, actual_cost_estimate, 
            execution_time, success, resource_usage
        )
        
        # Update budget tracking
        if success:
            self.spent_today += actual_cost
        
        # Log learning
        cost_error = abs(estimated_cost.total_cost - actual_cost) / max(actual_cost, 0.001)
        logger.info(f"Cost prediction for {tool_name}: "
                   f"estimated=${estimated_cost.total_cost:.4f}, "
                   f"actual=${actual_cost:.4f}, "
                   f"error={cost_error:.1%}")
    
    def get_budget_status(self) -> Dict[str, float]:
        """Get current budget status."""
        return {
            'total_budget': self.current_budget,
            'spent_today': self.spent_today,
            'remaining': self.current_budget - self.spent_today,
            'utilization': self.spent_today / self.current_budget if self.current_budget > 0 else 0
        }
    
    def set_budget(self, new_budget: float):
        """Set daily budget limit."""
        self.current_budget = new_budget
        logger.info(f"Budget set to ${new_budget:.2f}")
    
    def reset_daily_spending(self):
        """Reset daily spending counter."""
        self.spent_today = 0.0
        logger.info("Daily spending counter reset")

# Integration with existing agent framework
class CostAwareAgent:
    """Mixin for agents to use cost prediction."""
    
    def __init__(self):
        self.cost_predictor = CostPredictor()
        
    async def execute_with_cost_control(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool with cost control."""
        start_time = time.time()
        
        # Audit before execution
        audit_result = await self.cost_predictor.audit_before_execution({
            'tool_name': tool_name,
            'parameters': parameters
        })
        
        if not audit_result['approved']:
            return {
                'success': False,
                'reason': audit_result['decision_reason'],
                'alternative': audit_result.get('alternative_suggestion'),
                'cost_estimate': audit_result['estimated_cost']
            }
        
        try:
            # Execute the tool (this would be the actual tool execution)
            result = await self._execute_tool(tool_name, parameters)
            
            execution_time = time.time() - start_time
            
            # Record results for learning
            await self.cost_predictor.record_execution_result(
                tool_name=tool_name,
                parameters=parameters,
                estimated_cost=audit_result['estimated_cost'],
                actual_cost=execution_time * 0.001,  # Simple cost model
                execution_time=execution_time,
                success=True,
                resource_usage={'cpu': 50.0, 'memory': 30.0}  # Would be measured
            )
            
            return {
                'success': True,
                'result': result,
                'cost_info': audit_result['estimated_cost'],
                'execution_time': execution_time
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Record failed execution
            await self.cost_predictor.record_execution_result(
                tool_name=tool_name,
                parameters=parameters,
                estimated_cost=audit_result['estimated_cost'],
                actual_cost=execution_time * 0.001,
                execution_time=execution_time,
                success=False,
                resource_usage={}
            )
            
            return {
                'success': False,
                'error': str(e),
                'cost_info': audit_result['estimated_cost'],
                'execution_time': execution_time
            }
    
    async def _execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Placeholder for actual tool execution."""
        # This would be implemented by the specific agent
        await asyncio.sleep(0.1)  # Simulate work
        return f"Result from {tool_name} with parameters {parameters}"
