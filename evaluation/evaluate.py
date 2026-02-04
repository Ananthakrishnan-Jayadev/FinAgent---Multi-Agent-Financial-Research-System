"""
FinAgent Evaluation Runner

Runs test queries through the pipeline and generates evaluation report.
"""

import json
import time
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.graph import create_graph
from evaluation.test_queries import TEST_QUERIES, get_test_queries


def run_single_evaluation(query_config: dict, graph, verbose: bool = True) -> dict:
    """
    Run a single query through the pipeline and collect metrics.
    
    Args:
        query_config: Test query configuration dict
        graph: Compiled LangGraph
        verbose: Print progress updates
        
    Returns:
        Evaluation result dict
    """
    query_id = query_config["id"]
    query = query_config["query"]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running: {query_id}")
        print(f"Query: {query}")
        print(f"{'='*60}")
    
    # Initial state
    initial_state = {
        "query": query,
        "company": None,
        "query_complexity": "complex",
        "research_plan": [],
        "raw_findings": [],
        "financial_data": None,
        "analysis": None,
        "report_draft": None,
        "quality_review": None,
        "risk_assessment": None,
        "revision_count": 0,
        "human_approved": None,
        "current_agent": "starting",
        "final_report": None,
        "errors": []
    }
    
    config = {"configurable": {"thread_id": f"eval-{query_id}-{int(time.time())}"}}
    
    # Track execution
    start_time = time.time()
    agents_executed = []
    
    result = {
        "query_id": query_id,
        "query": query,
        "expected_complexity": query_config["expected_complexity"],
        "expected_path": query_config["expected_path"],
        "category": query_config["category"],
        "success": False,
        "error": None,
        "metrics": {}
    }
    
    try:
        # Run the graph (no interrupts for evaluation)
        for event in graph.stream(initial_state, config, stream_mode="updates"):
            if isinstance(event, dict):
                for node_name, node_state in event.items():
                    if isinstance(node_state, dict):
                        current_agent = node_state.get("current_agent", node_name)
                    else:
                        current_agent = node_name
                    agents_executed.append(current_agent)
                    if verbose:
                        print(f"  → {current_agent}")
        
        # Get final state
        final_state = graph.get_state(config).values
        
        end_time = time.time()
        
        # Collect metrics
        result["success"] = True
        result["metrics"] = {
            "execution_time_seconds": round(end_time - start_time, 2),
            "agents_executed": agents_executed,
            "agent_count": len(agents_executed),
            "actual_complexity": final_state.get("query_complexity"),
            "company_detected": final_state.get("company"),
            "revision_count": final_state.get("revision_count", 0),
            "quality_score": (final_state.get("quality_review") or {}).get("overall_score"),
            "has_final_report": final_state.get("final_report") is not None,
            "report_length": len(final_state.get("final_report", "") or ""),
            "errors": final_state.get("errors", []),
            "path_taken": "simple_response" if "simple_response" in str(agents_executed) else "full_pipeline"
        }
        
        # Validate expectations
        result["metrics"]["complexity_match"] = (
            result["metrics"]["actual_complexity"] == query_config["expected_complexity"]
        )
        result["metrics"]["path_match"] = (
            result["metrics"]["path_taken"] == query_config["expected_path"]
        )
        
        # Store report preview
        final_report = final_state.get("final_report", "")
        result["report_preview"] = final_report[:500] if final_report else None
        
        if verbose:
            print(f"\n✓ Completed in {result['metrics']['execution_time_seconds']}s")
            print(f"  Complexity: {result['metrics']['actual_complexity']} (expected: {query_config['expected_complexity']})")
            print(f"  Path: {result['metrics']['path_taken']} (expected: {query_config['expected_path']})")
            print(f"  Revisions: {result['metrics']['revision_count']}")
            print(f"  Quality: {result['metrics']['quality_score']}/10")
        
    except Exception as e:
        result["success"] = False
        result["error"] = str(e)
        result["metrics"]["execution_time_seconds"] = round(time.time() - start_time, 2)
        if verbose:
            print(f"\n✗ Failed: {e}")
    
    return result


def run_full_evaluation(
    queries: list = None,
    output_file: str = None,
    verbose: bool = True
) -> dict:
    """
    Run evaluation on multiple queries and generate report.
    
    Args:
        queries: List of query configs (defaults to all TEST_QUERIES)
        output_file: Path to save JSON results
        verbose: Print progress
        
    Returns:
        Evaluation summary dict
    """
    if queries is None:
        queries = TEST_QUERIES
    
    # Create graph without interrupts for automated evaluation
    graph = create_graph(with_interrupts=False)
    
    print("\n" + "="*60)
    print("FINAGENT EVALUATION")
    print(f"Running {len(queries)} test queries")
    print("="*60)
    
    results = []
    start_time = time.time()
    
    for query_config in queries:
        result = run_single_evaluation(query_config, graph, verbose)
        results.append(result)
    
    total_time = time.time() - start_time
    
    # Generate summary
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    complexity_matches = [r for r in successful if r["metrics"].get("complexity_match")]
    path_matches = [r for r in successful if r["metrics"].get("path_match")]
    
    avg_quality = sum(
        r["metrics"].get("quality_score", 0) or 0 
        for r in successful 
        if r["metrics"].get("quality_score")
    ) / max(len([r for r in successful if r["metrics"].get("quality_score")]), 1)
    
    avg_revisions = sum(
        r["metrics"].get("revision_count", 0) 
        for r in successful
    ) / max(len(successful), 1)
    
    summary = {
        "evaluation_date": datetime.now().isoformat(),
        "total_queries": len(queries),
        "successful": len(successful),
        "failed": len(failed),
        "success_rate": f"{len(successful)/len(queries)*100:.1f}%",
        "complexity_accuracy": f"{len(complexity_matches)/max(len(successful),1)*100:.1f}%",
        "path_accuracy": f"{len(path_matches)/max(len(successful),1)*100:.1f}%",
        "average_quality_score": round(avg_quality, 1),
        "average_revisions": round(avg_revisions, 1),
        "total_time_seconds": round(total_time, 1),
        "average_time_per_query": round(total_time / len(queries), 1),
        "results": results
    }
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total Queries:        {summary['total_queries']}")
    print(f"Successful:           {summary['successful']}")
    print(f"Failed:               {summary['failed']}")
    print(f"Success Rate:         {summary['success_rate']}")
    print(f"Complexity Accuracy:  {summary['complexity_accuracy']}")
    print(f"Path Accuracy:        {summary['path_accuracy']}")
    print(f"Average Quality:      {summary['average_quality_score']}/10")
    print(f"Average Revisions:    {summary['average_revisions']}")
    print(f"Total Time:           {summary['total_time_seconds']}s")
    print(f"Avg Time/Query:       {summary['average_time_per_query']}s")
    print("="*60)
    
    # Save results if output file specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nResults saved to: {output_file}")
    
    return summary


def run_quick_evaluation(verbose: bool = True) -> dict:
    """Run evaluation on a small subset (3 queries) for quick testing."""
    quick_queries = [
        get_test_queries("complex")[0],  # One complex
        get_test_queries("simple")[0],   # One simple
        TEST_QUERIES[-1],                 # One edge case
    ]
    return run_full_evaluation(quick_queries, verbose=verbose)


# CLI
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FinAgent Evaluation Runner")
    parser.add_argument("--quick", action="store_true", help="Run quick evaluation (3 queries)")
    parser.add_argument("--full", action="store_true", help="Run full evaluation (all queries)")
    parser.add_argument("--output", type=str, help="Output file path for results JSON")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_evaluation(verbose=not args.quiet)
    elif args.full:
        output = args.output or f"evaluation/results/eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        run_full_evaluation(output_file=output, verbose=not args.quiet)
    else:
        print("Usage: python -m evaluation.evaluate --quick OR --full")
        print("  --quick  Run 3 test queries")
        print("  --full   Run all test queries")
        print("  --output PATH  Save results to JSON file")