"""
Evaluation Framework for Memory Providers
Runs scenarios and calculates metrics
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import time
from collections import defaultdict


class MemoryEvaluator:
    """
    Evaluates memory providers against defined scenarios
    """
    
    def __init__(self, scenarios_path: str, llm_client=None):
        """
        Initialize evaluator
        
        Args:
            scenarios_path: Path to scenarios.json
            llm_client: Optional LLM client for semantic evaluation
        """
        with open(scenarios_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.scenarios = self.data['scenarios']
        self.config = self.data.get('evaluation_config', {})
        self.llm_client = llm_client
        
        # Results storage
        self.results = defaultdict(list)

    @staticmethod
    def _get_wait_seconds(provider, key: str, default_seconds: float = 0.5) -> float:
        """
        Read optional provider timing overrides.
        Provider can expose:
          - get_timing_config() -> dict
          - timing_config dict attribute
        """
        timing = {}
        if hasattr(provider, "get_timing_config"):
            try:
                timing = provider.get_timing_config() or {}
            except Exception:
                timing = {}
        elif hasattr(provider, "timing_config"):
            try:
                timing = getattr(provider, "timing_config") or {}
            except Exception:
                timing = {}

        value = timing.get(key, default_seconds)
        try:
            return float(value)
        except Exception:
            return float(default_seconds)
        
    def run_evaluation(self, provider, user_id: str = "test_user") -> Dict[str, Any]:
        """
        Run all scenarios for a provider
        
        Args:
            provider: Memory provider instance
            user_id: User ID for testing
            
        Returns:
            Dict with all evaluation results
        """
        print(f"\n{'='*60}")
        print(f"Evaluating Provider: {provider.get_name()}")
        print(f"{'='*60}\n")
        
        provider_results = {
            "provider": provider.get_name(),
            "capabilities": provider.get_capabilities(),
            "scenarios": [],
            "aggregate_metrics": {},
            "timestamp": datetime.now().isoformat()
        }
        
        for scenario in self.scenarios:
            print(f"Running scenario: {scenario['id']} - {scenario['name']}")
            
            # Reset provider state before each scenario
            provider.reset(user_id=user_id)
            time.sleep(self._get_wait_seconds(provider, "post_reset_sleep_s", 0.5))
            
            # Run scenario
            result = self.run_scenario(provider, scenario, user_id)
            provider_results["scenarios"].append(result)
            
            print(f"  ✓ Completed\n")
        
        # Calculate aggregate metrics
        provider_results["aggregate_metrics"] = self.calculate_aggregate_metrics(
            provider_results["scenarios"]
        )
        
        return provider_results
    
    def run_scenario(self, provider, scenario: Dict, user_id: str) -> Dict[str, Any]:
        """
        Run a single scenario
        
        Args:
            provider: Memory provider instance
            scenario: Scenario definition
            user_id: User ID
            
        Returns:
            Dict with scenario results
        """
        scenario_result = {
            "id": scenario["id"],
            "name": scenario["name"],
            "category": scenario["category"],
            "setup_success": True,
            "queries": [],
            "metrics": {}
        }
        
        # Setup phase: Store memories
        setup = scenario.get("setup", {})
        memories = setup.get("memories", [])
        
        stored_memories = {}
        for memory in memories:
            result = provider.store(
                content=memory["content"],
                metadata=memory.get("metadata", {}),
                user_id=user_id,
                memory_id=memory["id"]
            )
            
            if not result.get("success"):
                scenario_result["setup_success"] = False
                scenario_result["setup_error"] = result.get("error")
                return scenario_result
            
            stored_memories[memory["id"]] = result
        
        # Handle updates if specified
        updates = setup.get("updates", [])
        for update in updates:
            provider.update(
                memory_id=update["id"],
                new_content=update["new_content"],
                metadata=update.get("metadata", {}),
                user_id=user_id
            )
        
        # Handle deletions if specified
        deletions = setup.get("deletions", [])
        for memory_id in deletions:
            provider.delete(memory_id=memory_id, user_id=user_id)
        
        # Small delay after setup
        time.sleep(self._get_wait_seconds(provider, "post_setup_sleep_s", 0.5))
        
        # Query phase: Test retrieval
        queries = scenario.get("queries", [])
        
        for query_spec in queries:
            query_result = self.execute_query(
                provider, query_spec, scenario, user_id
            )
            scenario_result["queries"].append(query_result)
        
        # Calculate scenario metrics
        scenario_result["metrics"] = self.calculate_scenario_metrics(
            scenario, scenario_result["queries"]
        )
        
        return scenario_result
    
    def execute_query(self, provider, query_spec: Dict, scenario: Dict, 
                      user_id: str) -> Dict[str, Any]:
        """
        Execute a single query and measure results
        
        Args:
            provider: Memory provider
            query_spec: Query specification from scenario
            scenario: Full scenario context
            user_id: User ID
            
        Returns:
            Dict with query results and measurements
        """
        query_text = query_spec["query"]
        k = query_spec.get("k", 5)
        
        # Execute retrieval
        start_time = time.time()
        result = provider.retrieve(
            query=query_text,
            user_id=user_id,
            k=k
        )
        latency = time.time() - start_time
        
        retrieved_memories = result.get("memories", [])
        
        # Extract IDs and content
        retrieved_ids = [m["id"] for m in retrieved_memories]
        retrieved_contents = [m["content"] for m in retrieved_memories]
        
        # Expected values
        expected_ids = query_spec.get("expected_memory_ids", [])
        expected_content = query_spec.get("expected_content")
        should_not_include = query_spec.get("should_not_include", [])
        
        return {
            "query": query_text,
            "k": k,
            "retrieved_count": len(retrieved_memories),
            "retrieved_ids": retrieved_ids,
            "retrieved_contents": retrieved_contents,
            "expected_ids": expected_ids,
            "expected_content": expected_content,
            "should_not_include": should_not_include,
            "latency_ms": round(latency * 1000, 2),
            "raw_result": result
        }
    
    def calculate_scenario_metrics(self, scenario: Dict, 
                                   query_results: List[Dict]) -> Dict[str, Any]:
        """
        Calculate metrics for a scenario based on query results
        
        Args:
            scenario: Scenario definition
            query_results: List of query results
            
        Returns:
            Dict with calculated metrics
        """
        metrics = {
            "precision_at_k": [],
            "recall_at_k": [],
            "exact_matches": [],
            "semantic_relevance": [],
            "temporal_accuracy": [],
            "false_positives": [],
            "latency_ms": []
        }
        
        for query_result in query_results:
            retrieved = set(query_result["retrieved_ids"])
            expected = set(query_result["expected_ids"])
            should_not = set(query_result.get("should_not_include", []))
            
            # Precision@K
            if len(retrieved) > 0:
                true_positives = len(retrieved & expected)
                precision = true_positives / len(retrieved)
                metrics["precision_at_k"].append(precision)
            else:
                # No results retrieved
                if len(expected) == 0:
                    metrics["precision_at_k"].append(1.0)  # Correctly returned nothing
                else:
                    metrics["precision_at_k"].append(0.0)  # Should have returned something
            
            # Recall@K
            if len(expected) > 0:
                true_positives = len(retrieved & expected)
                recall = true_positives / len(expected)
                metrics["recall_at_k"].append(recall)
            else:
                # No expected results
                if len(retrieved) == 0:
                    metrics["recall_at_k"].append(1.0)  # Correctly returned nothing
                else:
                    metrics["recall_at_k"].append(0.0)  # False positives
            
            # Exact match (did we get exactly what was expected?)
            exact_match = (retrieved == expected) if expected else (len(retrieved) == 0)
            metrics["exact_matches"].append(1.0 if exact_match else 0.0)
            
            # False positives (zombie memories, deleted items)
            false_pos = len(retrieved & should_not)
            metrics["false_positives"].append(false_pos)
            
            # Temporal accuracy (for temporal scenarios)
            if scenario["category"] == "temporal_accuracy":
                temporal_acc = self.calculate_temporal_accuracy(query_result, scenario)
                metrics["temporal_accuracy"].append(temporal_acc)
            
            # Semantic relevance (requires LLM)
            if self.llm_client and query_result.get("expected_content"):
                relevance = self.calculate_semantic_relevance(query_result)
                metrics["semantic_relevance"].append(relevance)
            
            # Latency
            metrics["latency_ms"].append(query_result["latency_ms"])
        
        # Average metrics
        return {
            "avg_precision_at_k": self._safe_avg(metrics["precision_at_k"]),
            "avg_recall_at_k": self._safe_avg(metrics["recall_at_k"]),
            "exact_match_rate": self._safe_avg(metrics["exact_matches"]),
            "avg_semantic_relevance": self._safe_avg(metrics["semantic_relevance"]),
            "temporal_accuracy_rate": self._safe_avg(metrics["temporal_accuracy"]),
            "total_false_positives": sum(metrics["false_positives"]),
            "avg_latency_ms": self._safe_avg(metrics["latency_ms"]),
            "raw_metrics": metrics
        }
    
    def calculate_temporal_accuracy(self, query_result: Dict, 
                                    scenario: Dict) -> float:
        """
        Calculate temporal accuracy for time-based queries
        
        Args:
            query_result: Query result
            scenario: Scenario definition
            
        Returns:
            Accuracy score (0-1)
        """
        retrieved_ids = query_result["retrieved_ids"]
        expected_ids = query_result["expected_ids"]
        
        # Check if we got the right memories
        if not expected_ids:
            return 1.0 if not retrieved_ids else 0.0
        
        # For "most recent" queries, check if first result is correct
        query_spec = None
        for q in scenario.get("queries", []):
            if q["query"] == query_result["query"]:
                query_spec = q
                break
        
        if query_spec:
            temporal_pref = query_spec.get("temporal_preference")
            
            if temporal_pref == "most_recent":
                # First retrieved should be the expected one
                if retrieved_ids and retrieved_ids[0] in expected_ids:
                    return 1.0
                return 0.0
            
            elif temporal_pref == "historical":
                # Should retrieve older, not newer
                if set(retrieved_ids) == set(expected_ids):
                    return 1.0
                return 0.0
            
            elif temporal_pref == "all":
                # Should retrieve all in correct order
                expected_order = query_spec.get("expected_order", expected_ids)
                if retrieved_ids[:len(expected_order)] == expected_order:
                    return 1.0
                # Partial credit for having all the right memories
                if set(retrieved_ids) == set(expected_ids):
                    return 0.5
                return 0.0
        
        # Default: exact match
        return 1.0 if set(retrieved_ids) == set(expected_ids) else 0.0
    
    def calculate_semantic_relevance(self, query_result: Dict) -> float:
        """
        Calculate semantic relevance using LLM
        
        Args:
            query_result: Query result with retrieved content
            
        Returns:
            Relevance score (0-1)
        """
        if not self.llm_client:
            return 0.0
        
        # This would call the LLM to judge semantic similarity
        # For now, return a simple heuristic
        expected = query_result.get("expected_content", "").lower()
        retrieved = " ".join(query_result.get("retrieved_contents", [])).lower()
        
        # Simple keyword matching (replace with LLM call)
        if expected in retrieved:
            return 1.0
        
        # Check for partial matches
        expected_words = set(expected.split())
        retrieved_words = set(retrieved.split())
        
        if expected_words & retrieved_words:
            overlap = len(expected_words & retrieved_words) / len(expected_words)
            return overlap
        
        return 0.0
    
    def calculate_aggregate_metrics(self, scenario_results: List[Dict]) -> Dict[str, Any]:
        """
        Calculate aggregate metrics across all scenarios
        
        Args:
            scenario_results: List of scenario results
            
        Returns:
            Dict with aggregate metrics
        """
        # Group by category
        by_category = defaultdict(list)
        
        all_precision = []
        all_recall = []
        all_exact_match = []
        all_semantic = []
        all_temporal = []
        all_false_pos = []
        all_latency = []
        
        for result in scenario_results:
            category = result["category"]
            metrics = result.get("metrics", {})
            
            by_category[category].append(metrics)
            
            # Collect all metrics
            if "avg_precision_at_k" in metrics:
                all_precision.append(metrics["avg_precision_at_k"])
            if "avg_recall_at_k" in metrics:
                all_recall.append(metrics["avg_recall_at_k"])
            if "exact_match_rate" in metrics:
                all_exact_match.append(metrics["exact_match_rate"])
            if "avg_semantic_relevance" in metrics:
                all_semantic.append(metrics["avg_semantic_relevance"])
            if "temporal_accuracy_rate" in metrics:
                all_temporal.append(metrics["temporal_accuracy_rate"])
            if "total_false_positives" in metrics:
                all_false_pos.append(metrics["total_false_positives"])
            if "avg_latency_ms" in metrics:
                all_latency.append(metrics["avg_latency_ms"])
        
        return {
            "overall": {
                "avg_precision_at_k": self._safe_avg(all_precision),
                "avg_recall_at_k": self._safe_avg(all_recall),
                "exact_match_rate": self._safe_avg(all_exact_match),
                "avg_semantic_relevance": self._safe_avg(all_semantic),
                "temporal_accuracy_rate": self._safe_avg(all_temporal),
                "total_false_positives": sum(all_false_pos),
                "avg_latency_ms": self._safe_avg(all_latency),
                "total_scenarios": len(scenario_results),
                "successful_scenarios": sum(1 for r in scenario_results if r.get("setup_success"))
            },
            "by_category": {
                cat: self._aggregate_category_metrics(metrics_list)
                for cat, metrics_list in by_category.items()
            }
        }
    
    def _aggregate_category_metrics(self, metrics_list: List[Dict]) -> Dict[str, float]:
        """Aggregate metrics for a category"""
        precision = [m.get("avg_precision_at_k", 0) for m in metrics_list]
        recall = [m.get("avg_recall_at_k", 0) for m in metrics_list]
        exact_match = [m.get("exact_match_rate", 0) for m in metrics_list]
        
        return {
            "avg_precision_at_k": self._safe_avg(precision),
            "avg_recall_at_k": self._safe_avg(recall),
            "exact_match_rate": self._safe_avg(exact_match),
            "scenario_count": len(metrics_list)
        }
    
    def _safe_avg(self, values: List[float]) -> float:
        """Safely calculate average, handling empty lists"""
        if not values:
            return 0.0
        return sum(values) / len(values)
    
    def generate_report(self, results: Dict[str, Any], output_path: str = None) -> str:
        """
        Generate a human-readable report
        
        Args:
            results: Evaluation results
            output_path: Optional path to save report
            
        Returns:
            Report as string
        """
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append(f"MEMORY PROVIDER EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Provider: {results['provider']}")
        report_lines.append(f"Timestamp: {results['timestamp']}")
        report_lines.append("")
        
        # Capabilities
        report_lines.append("CAPABILITIES:")
        for cap, supported in results['capabilities'].items():
            status = "✓" if supported else "✗"
            report_lines.append(f"  {status} {cap}")
        report_lines.append("")
        
        # Overall metrics
        overall = results['aggregate_metrics']['overall']
        report_lines.append("OVERALL PERFORMANCE:")
        report_lines.append(f"  Precision@K:        {overall['avg_precision_at_k']:.2%}")
        report_lines.append(f"  Recall@K:           {overall['avg_recall_at_k']:.2%}")
        report_lines.append(f"  Exact Match Rate:   {overall['exact_match_rate']:.2%}")
        report_lines.append(f"  Semantic Relevance: {overall['avg_semantic_relevance']:.2%}")
        report_lines.append(f"  Temporal Accuracy:  {overall['temporal_accuracy_rate']:.2%}")
        report_lines.append(f"  False Positives:    {overall['total_false_positives']}")
        report_lines.append(f"  Avg Latency:        {overall['avg_latency_ms']:.2f}ms")
        report_lines.append(f"  Success Rate:       {overall['successful_scenarios']}/{overall['total_scenarios']}")
        report_lines.append("")
        
        # By category
        report_lines.append("PERFORMANCE BY CATEGORY:")
        for category, metrics in results['aggregate_metrics']['by_category'].items():
            report_lines.append(f"\n  {category.upper().replace('_', ' ')}:")
            report_lines.append(f"    Precision@K:      {metrics['avg_precision_at_k']:.2%}")
            report_lines.append(f"    Recall@K:         {metrics['avg_recall_at_k']:.2%}")
            report_lines.append(f"    Exact Match Rate: {metrics['exact_match_rate']:.2%}")
            report_lines.append(f"    Scenarios:        {metrics['scenario_count']}")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        # Save if path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
        
        return report_text
