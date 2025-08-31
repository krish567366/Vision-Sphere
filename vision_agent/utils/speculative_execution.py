"""
Speculative Tool Execution System
Predictive tool execution to reduce latency through parallel processing.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
from enum import Enum
from functools import wraps
import json
import hashlib
from functools import wraps

class PredictionConfidence(str, Enum):
    """Confidence levels for tool predictions."""
    HIGH = "high"      # >90% confidence
    MEDIUM = "medium"  # 70-90% confidence  
    LOW = "low"        # 50-70% confidence


@dataclass
class ToolPrediction:
    """Predicted tool execution."""
    tool_name: str
    args: Dict[str, Any]
    confidence: PredictionConfidence
    estimated_duration: float
    priority: int = 0  # Higher number = higher priority
    
    def __hash__(self):
        return hash((self.tool_name, json.dumps(self.args, sort_keys=True)))


@dataclass
class SpeculativeResult:
    """Result from speculative execution."""
    prediction: ToolPrediction
    result: Any
    execution_time: float
    success: bool
    error: Optional[str] = None


class ToolPredictor:
    """
    ML-based tool prediction system.
    Learns from agent execution patterns to predict likely tool calls.
    """
    
    def __init__(self):
        self.execution_patterns: Dict[str, List[Dict[str, Any]]] = {}
        self.tool_sequences: List[List[str]] = []
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def record_execution(self, 
                        query: str, 
                        executed_tools: List[Tuple[str, Dict[str, Any]]]):
        """Record actual tool execution for learning."""
        # Hash query for pattern matching
        query_hash = hashlib.md5(query.lower().encode()).hexdigest()[:16]
        
        # Store execution pattern
        if query_hash not in self.execution_patterns:
            self.execution_patterns[query_hash] = []
        
        pattern = {
            "tools": executed_tools,
            "timestamp": time.time(),
            "query_length": len(query),
            "tool_count": len(executed_tools)
        }
        self.execution_patterns[query_hash].append(pattern)
        
        # Keep only recent patterns
        if len(self.execution_patterns[query_hash]) > 10:
            self.execution_patterns[query_hash] = \
                self.execution_patterns[query_hash][-5:]
        
        # Store tool sequence for pattern analysis
        tool_sequence = [tool[0] for tool in executed_tools]
        self.tool_sequences.append(tool_sequence)
        
        # Keep sequence history manageable
        if len(self.tool_sequences) > 1000:
            self.tool_sequences = self.tool_sequences[-500:]
    
    def predict_tools(self, query: str, max_predictions: int = 3) -> List[ToolPrediction]:
        """
        Predict likely tool calls for a given query.
        
        Args:
            query: Input query to analyze
            max_predictions: Maximum number of predictions to return
            
        Returns:
            List of tool predictions sorted by confidence
        """
        predictions = []
        
        # Simple keyword-based predictions (can be enhanced with ML)
        query_lower = query.lower()
        
        # Image/video analysis keywords
        if any(keyword in query_lower for keyword in ['image', 'photo', 'picture', 'detect', 'identify']):
            if 'face' in query_lower:
                predictions.append(ToolPrediction(
                    tool_name="face_detection",
                    args={"confidence_threshold": 0.7},
                    confidence=PredictionConfidence.HIGH,
                    estimated_duration=2.0,
                    priority=3
                ))
            
            if any(keyword in query_lower for keyword in ['object', 'thing', 'item', 'detect']):
                predictions.append(ToolPrediction(
                    tool_name="object_detection",
                    args={"confidence_threshold": 0.5},
                    confidence=PredictionConfidence.HIGH,
                    estimated_duration=1.5,
                    priority=2
                ))
            
            if any(keyword in query_lower for keyword in ['classify', 'category', 'type', 'what is']):
                predictions.append(ToolPrediction(
                    tool_name="image_classification",
                    args={"top_k": 5},
                    confidence=PredictionConfidence.MEDIUM,
                    estimated_duration=1.0,
                    priority=1
                ))
        
        # Video analysis keywords
        if any(keyword in query_lower for keyword in ['video', 'movie', 'clip', 'footage']):
            predictions.append(ToolPrediction(
                tool_name="video_analysis",
                args={"extract_frames": True},
                confidence=PredictionConfidence.HIGH,
                estimated_duration=10.0,
                priority=3
            ))
        
        # Pattern-based predictions from historical data
        historical_predictions = self._predict_from_patterns(query)
        predictions.extend(historical_predictions)
        
        # Remove duplicates and sort by priority and confidence
        unique_predictions = list(set(predictions))
        unique_predictions.sort(
            key=lambda p: (p.priority, p.confidence.value == "high"), 
            reverse=True
        )
        
        return unique_predictions[:max_predictions]
    
    def _predict_from_patterns(self, query: str) -> List[ToolPrediction]:
        """Predict tools based on historical execution patterns."""
        predictions = []
        query_hash = hashlib.md5(query.lower().encode()).hexdigest()[:16]
        
        # Look for similar queries
        for pattern_hash, patterns in self.execution_patterns.items():
            if len(patterns) < 2:
                continue
            
            # Simple similarity check (can be enhanced)
            similarity = self._calculate_pattern_similarity(query_hash, pattern_hash)
            if similarity > 0.3:
                # Get most common tools from this pattern
                tool_frequency = {}
                for pattern in patterns[-3:]:  # Recent patterns
                    for tool_name, args in pattern["tools"]:
                        if tool_name not in tool_frequency:
                            tool_frequency[tool_name] = {"count": 0, "args": args}
                        tool_frequency[tool_name]["count"] += 1
                
                # Convert to predictions
                for tool_name, data in tool_frequency.items():
                    if data["count"] >= 2:  # Appeared in multiple patterns
                        confidence = PredictionConfidence.MEDIUM if data["count"] >= 3 else PredictionConfidence.LOW
                        predictions.append(ToolPrediction(
                            tool_name=tool_name,
                            args=data["args"],
                            confidence=confidence,
                            estimated_duration=2.0,  # Default
                            priority=1
                        ))
        
        return predictions
    
    def _calculate_pattern_similarity(self, hash1: str, hash2: str) -> float:
        """Calculate simple similarity between query hashes."""
        # Simple Hamming distance approximation
        if hash1 == hash2:
            return 1.0
        
        # Count matching characters
        matches = sum(c1 == c2 for c1, c2 in zip(hash1, hash2))
        return matches / len(hash1)


class SpeculativeToolRunner:
    """
    Advanced speculative execution system that predicts and pre-runs likely tools.
    """
    
    def __init__(self, 
                 max_speculative_tasks: int = 3,
                 speculation_timeout: float = 30.0):
        self.max_speculative_tasks = max_speculative_tasks
        self.speculation_timeout = speculation_timeout
        
        self.predictor = ToolPredictor()
        self.active_speculations: Dict[str, asyncio.Task] = {}
        self.completed_speculations: Dict[str, SpeculativeResult] = {}
        
        # Tool registry
        self.available_tools: Dict[str, Callable] = {}
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.stats = {
            "speculations_started": 0,
            "speculations_used": 0,
            "speculations_wasted": 0,
            "total_time_saved": 0.0
        }
    
    def register_tool(self, name: str, tool_func: Callable):
        """Register a tool for speculative execution."""
        self.available_tools[name] = tool_func
        self.logger.debug(f"Registered tool: {name}")
    
    async def run_with_speculation(self, 
                                  query: str,
                                  main_execution_func: Callable,
                                  context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Run main execution with speculative tool execution.
        
        Args:
            query: Input query for prediction
            main_execution_func: Main function to execute
            context: Additional context for prediction
            
        Returns:
            Result from main execution, potentially enhanced with speculative results
        """
        execution_id = f"exec_{int(time.time() * 1000000)}"
        
        # Start speculative execution
        speculative_tasks = await self._start_speculative_execution(
            query, execution_id, context
        )
        
        start_time = time.time()
        
        try:
            # Run main execution in parallel with speculation
            main_result = await main_execution_func()
            
            # Check if any speculative results can be used
            enhanced_result = await self._merge_speculative_results(
                main_result, execution_id
            )
            
            execution_time = time.time() - start_time
            self.logger.info(
                f"Execution completed in {execution_time:.2f}s with "
                f"{len(self.completed_speculations)} speculative results available"
            )
            
            return enhanced_result
            
        finally:
            # Clean up speculative tasks
            await self._cleanup_speculation(execution_id)
    
    async def _start_speculative_execution(self, 
                                         query: str,
                                         execution_id: str,
                                         context: Optional[Dict[str, Any]]) -> List[asyncio.Task]:
        """Start speculative tool execution based on predictions."""
        predictions = self.predictor.predict_tools(query, self.max_speculative_tasks)
        
        if not predictions:
            return []
        
        speculative_tasks = []
        
        for prediction in predictions:
            if prediction.tool_name in self.available_tools:
                task = asyncio.create_task(
                    self._execute_speculation(prediction, execution_id)
                )
                speculative_tasks.append(task)
                self.active_speculations[f"{execution_id}_{prediction.tool_name}"] = task
                
                self.stats["speculations_started"] += 1
                
                self.logger.debug(
                    f"Started speculation: {prediction.tool_name} "
                    f"(confidence: {prediction.confidence.value})"
                )
        
        return speculative_tasks
    
    async def _execute_speculation(self, 
                                  prediction: ToolPrediction,
                                  execution_id: str) -> SpeculativeResult:
        """Execute a single speculative tool call."""
        tool_func = self.available_tools[prediction.tool_name]
        start_time = time.time()
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                tool_func(**prediction.args),
                timeout=self.speculation_timeout
            )
            
            execution_time = time.time() - start_time
            
            speculative_result = SpeculativeResult(
                prediction=prediction,
                result=result,
                execution_time=execution_time,
                success=True
            )
            
            # Store result
            result_key = f"{execution_id}_{prediction.tool_name}"
            self.completed_speculations[result_key] = speculative_result
            
            return speculative_result
            
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            self.logger.warning(
                f"Speculation timeout for {prediction.tool_name} after {execution_time:.2f}s"
            )
            return SpeculativeResult(
                prediction=prediction,
                result=None,
                execution_time=execution_time,
                success=False,
                error="Timeout"
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.warning(f"Speculation error for {prediction.tool_name}: {e}")
            return SpeculativeResult(
                prediction=prediction,
                result=None,
                execution_time=execution_time,
                success=False,
                error=str(e)
            )
    
    async def _merge_speculative_results(self, 
                                       main_result: Any,
                                       execution_id: str) -> Any:
        """Merge speculative results with main execution result."""
        # This is a simplified merger - can be enhanced based on specific needs
        if hasattr(main_result, 'speculative_data'):
            speculative_data = {}
            
            for key, spec_result in self.completed_speculations.items():
                if key.startswith(execution_id) and spec_result.success:
                    tool_name = spec_result.prediction.tool_name
                    speculative_data[tool_name] = {
                        "result": spec_result.result,
                        "confidence": spec_result.prediction.confidence.value,
                        "execution_time": spec_result.execution_time
                    }
                    self.stats["speculations_used"] += 1
                    self.stats["total_time_saved"] += spec_result.prediction.estimated_duration
            
            main_result.speculative_data = speculative_data
        
        return main_result
    
    async def _cleanup_speculation(self, execution_id: str):
        """Clean up speculative execution resources."""
        # Cancel active tasks
        keys_to_remove = []
        for key, task in list(self.active_speculations.items()):
            if key.startswith(execution_id):
                task.cancel()
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self.active_speculations.pop(key, None)
        
        # Clean up completed results after some time
        asyncio.create_task(self._delayed_cleanup(execution_id))
    
    async def _delayed_cleanup(self, execution_id: str):
        """Delayed cleanup of speculation results."""
        await asyncio.sleep(300)  # Keep results for 5 minutes
        
        keys_to_remove = [
            key for key in self.completed_speculations.keys()
            if key.startswith(execution_id)
        ]
        
        for key in keys_to_remove:
            spec_result = self.completed_speculations.pop(key, None)
            if spec_result and not spec_result.success:
                self.stats["speculations_wasted"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get speculative execution statistics."""
        total_speculations = self.stats["speculations_started"]
        usage_rate = 0.0
        if total_speculations > 0:
            usage_rate = self.stats["speculations_used"] / total_speculations
        
        return {
            "total_speculations": total_speculations,
            "successful_speculations": self.stats["speculations_used"],
            "wasted_speculations": self.stats["speculations_wasted"],
            "usage_rate": usage_rate,
            "total_time_saved_seconds": self.stats["total_time_saved"],
            "active_speculations": len(self.active_speculations),
            "registered_tools": len(self.available_tools)
        }


class CostOptimizedRouter:
    """
    Cost optimization router for model selection based on complexity and budget.
    """
    
    def __init__(self):
        # Model cost per 1K tokens (approximate)
        self.model_costs = {
            "gpt-4o-mini": 0.000150,  # $0.15 per 1M tokens
            "gpt-4o": 0.005,          # $5 per 1M tokens  
            "o1-preview": 0.015,      # $15 per 1M tokens
            "claude-3-haiku": 0.00025,
            "claude-3-sonnet": 0.003,
            "claude-3-opus": 0.015
        }
        
        # Model capabilities (0-1 scale)
        self.model_capabilities = {
            "gpt-4o-mini": {"reasoning": 0.6, "creativity": 0.5, "accuracy": 0.7},
            "gpt-4o": {"reasoning": 0.8, "creativity": 0.8, "accuracy": 0.9},
            "o1-preview": {"reasoning": 0.95, "creativity": 0.7, "accuracy": 0.95},
            "claude-3-haiku": {"reasoning": 0.6, "creativity": 0.7, "accuracy": 0.7},
            "claude-3-sonnet": {"reasoning": 0.8, "creativity": 0.8, "accuracy": 0.85},
            "claude-3-opus": {"reasoning": 0.9, "creativity": 0.9, "accuracy": 0.9}
        }
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def route_request(self, 
                     complexity_score: float,
                     budget_per_request: float,
                     required_capabilities: Optional[Dict[str, float]] = None,
                     estimated_tokens: int = 1000) -> str:
        """
        Route request to optimal model based on complexity, budget, and capabilities.
        
        Args:
            complexity_score: Task complexity (0-1)
            budget_per_request: Maximum budget per request
            required_capabilities: Required capability minimums
            estimated_tokens: Estimated token count
            
        Returns:
            Optimal model name
        """
        # Calculate cost for each model
        model_scores = {}
        
        for model_name, cost_per_k in self.model_costs.items():
            estimated_cost = (estimated_tokens / 1000) * cost_per_k
            
            # Skip if over budget
            if estimated_cost > budget_per_request:
                continue
            
            capabilities = self.model_capabilities[model_name]
            
            # Check minimum capability requirements
            if required_capabilities:
                meets_requirements = all(
                    capabilities.get(req, 0) >= min_val
                    for req, min_val in required_capabilities.items()
                )
                if not meets_requirements:
                    continue
            
            # Calculate combined score
            # Higher complexity needs better reasoning
            capability_score = (
                capabilities["reasoning"] * complexity_score +
                capabilities["accuracy"] * 0.8 +
                capabilities["creativity"] * (1 - complexity_score)
            ) / 2.8
            
            # Cost efficiency (lower cost = higher score)
            max_cost = max(self.model_costs.values())
            cost_efficiency = 1 - (cost_per_k / max_cost)
            
            # Combined score (capability weighted more for complex tasks)
            complexity_weight = 0.6 + (complexity_score * 0.3)
            cost_weight = 1 - complexity_weight
            
            total_score = (
                capability_score * complexity_weight +
                cost_efficiency * cost_weight
            )
            
            model_scores[model_name] = {
                "score": total_score,
                "estimated_cost": estimated_cost,
                "capability_score": capability_score,
                "cost_efficiency": cost_efficiency
            }
        
        if not model_scores:
            # Fallback to cheapest model if budget is very tight
            cheapest_model = min(self.model_costs.keys(), key=lambda m: self.model_costs[m])
            self.logger.warning(f"Budget too low, using cheapest model: {cheapest_model}")
            return cheapest_model
        
        # Select model with highest score
        best_model = max(model_scores.keys(), key=lambda m: model_scores[m]["score"])
        
        self.logger.info(
            f"Selected {best_model} for complexity {complexity_score:.2f} "
            f"(estimated cost: ${model_scores[best_model]['estimated_cost']:.4f})"
        )
        
        return best_model
    
    def estimate_task_complexity(self, 
                                query: str,
                                task_type: str,
                                input_size: Optional[int] = None) -> float:
        """
        Estimate task complexity score (0-1).
        
        Args:
            query: Input query
            task_type: Type of task (image, video, text, etc.)
            input_size: Size of input data in bytes
            
        Returns:
            Complexity score (0-1)
        """
        base_complexity = 0.3  # Base complexity
        
        # Query complexity indicators
        query_lower = query.lower()
        complexity_keywords = {
            "analyze": 0.2,
            "complex": 0.3,
            "detailed": 0.2,
            "comprehensive": 0.3,
            "advanced": 0.2,
            "multi": 0.2,
            "batch": 0.3,
            "compare": 0.2,
            "video": 0.4,
            "track": 0.3
        }
        
        for keyword, weight in complexity_keywords.items():
            if keyword in query_lower:
                base_complexity += weight
        
        # Task type complexity
        task_complexity_map = {
            "image_classification": 0.2,
            "face_detection": 0.4,
            "object_detection": 0.5,
            "video_analysis": 0.8,
            "multi_modal": 0.9
        }
        
        base_complexity += task_complexity_map.get(task_type, 0.3)
        
        # Input size complexity
        if input_size:
            # Larger inputs are more complex
            size_mb = input_size / (1024 * 1024)
            if size_mb > 100:
                base_complexity += 0.3
            elif size_mb > 10:
                base_complexity += 0.2
            elif size_mb > 1:
                base_complexity += 0.1
        
        return min(1.0, base_complexity)


# Global instances
speculative_runner = SpeculativeToolRunner()
cost_optimizer = CostOptimizedRouter()


# Decorator for speculative execution
def enable_speculation(max_speculative_tasks: int = 3):
    """Decorator to enable speculative execution for agent methods."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract query from args (assumes first arg is query/input)
            query = str(args[0]) if args else "default_query"
            
            async def main_execution():
                return await func(*args, **kwargs)
            
            return await speculative_runner.run_with_speculation(
                query=query,
                main_execution_func=main_execution
            )
        
        return wrapper
    return decorator


async def initialize_speculative_system():
    """Initialize the speculative execution system."""
    logging.getLogger('SpeculativeSystem').info("Speculative Tool Execution System initialized")


async def record_tool_execution(query: str, executed_tools: List[Tuple[str, Dict[str, Any]]]):
    """Record tool execution for learning."""
    speculative_runner.predictor.record_execution(query, executed_tools)
