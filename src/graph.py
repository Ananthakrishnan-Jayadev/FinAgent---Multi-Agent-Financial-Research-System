"""
FinAgent Graph Definition

Orchestrates all agents into a stateful, conditional workflow with:
- Conditional routing based on query complexity
- Quality check revision cycles
- Human-in-the-loop approval gate
"""

from typing import Literal
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from src.state import ResearchState
from src.agents.planner import planner_node
from src.agents.researcher import researcher_node
from src.agents.analyst import analyst_node
from src.agents.writer import writer_node
from src.agents.quality_checker import quality_checker_node, should_revise
from src.agents.risk_assessor import risk_assessor_node


# === Routing Functions ===

def route_by_complexity(state: ResearchState) -> Literal["researcher", "simple_response"]:
    """
    Route based on query complexity determined by Planner.
    
    Simple queries skip the full pipeline and get a quick response.
    Complex queries go through the full research workflow.
    """
    complexity = state.get("query_complexity", "complex")
    
    if complexity == "simple":
        return "simple_response"
    return "researcher"


def route_after_quality_check(state: ResearchState) -> Literal["human_approval", "writer"]:
    """
    Route based on quality check results.
    
    If passed or max revisions reached → proceed to human approval
    If failed and under revision limit → go back to writer for revision
    """
    decision = should_revise(state)
    
    if decision == "approve":
        return "human_approval"
    return "writer"  # Revise by going back to writer


# === Special Nodes ===

def simple_response_node(state: ResearchState) -> dict:
    """
    Handle simple queries with just financial data lookup.
    Bypasses the full analysis pipeline.
    """
    financial_data = state.get("financial_data")
    company = state.get("company", "Unknown")
    query = state.get("query", "")
    
    if financial_data and financial_data.get("success", True):
        # Build a simple response report
        price = financial_data.get("current_price", "N/A")
        market_cap = financial_data.get("market_cap")
        pe_ratio = financial_data.get("pe_ratio", "N/A")
        
        market_cap_str = f"${market_cap:,}" if market_cap else "N/A"
        
        simple_report = f"""# {company} - Quick Lookup

**Query:** {query}

## Key Data Points

| Metric | Value |
|--------|-------|
| Current Price | ${price} |
| Market Cap | {market_cap_str} |
| P/E Ratio | {pe_ratio} |
| Sector | {financial_data.get('sector', 'N/A')} |
| Industry | {financial_data.get('industry', 'N/A')} |

---

*Quick lookup by FinAgent. For detailed analysis, ask a more specific question.*
"""
        return {
            "final_report": simple_report,
            "current_agent": "simple_response_complete"
        }
    else:
        return {
            "final_report": f"Could not retrieve data for {company}.",
            "current_agent": "simple_response_failed",
            "errors": ["Financial data lookup failed for simple query"]
        }


def human_approval_node(state: ResearchState) -> dict:
    """
    Human-in-the-loop approval gate.
    
    In actual use, this node will be interrupted and the human
    can review the report before proceeding.
    
    For now, we auto-approve for testing purposes.
    The interrupt happens at the graph level, not in this node.
    """
    return {
        "human_approved": True,
        "current_agent": "human_approval_complete"
    }


def finalize_report_node(state: ResearchState) -> dict:
    """
    Finalize the report by combining the draft with risk assessment.
    """
    report_draft = state.get("report_draft", "")
    risk_assessment = state.get("risk_assessment", {})
    
    # Append risk assessment to the report if available
    if risk_assessment and risk_assessment.get("overall_risk_level"):
        risk_section = f"""

---

## Detailed Risk Assessment

**Overall Risk Level: {risk_assessment.get('overall_risk_level', 'N/A').upper()}**

### Risk by Category

| Category | Level | Assessment |
|----------|-------|------------|
| Market Risk | {risk_assessment.get('market_risk', {}).get('level', 'N/A').upper()} | {risk_assessment.get('market_risk', {}).get('assessment', 'N/A')} |
| Credit Risk | {risk_assessment.get('credit_risk', {}).get('level', 'N/A').upper()} | {risk_assessment.get('credit_risk', {}).get('assessment', 'N/A')} |
| Regulatory Risk | {risk_assessment.get('regulatory_risk', {}).get('level', 'N/A').upper()} | {risk_assessment.get('regulatory_risk', {}).get('assessment', 'N/A')} |
| Operational Risk | {risk_assessment.get('operational_risk', {}).get('level', 'N/A').upper()} | {risk_assessment.get('operational_risk', {}).get('assessment', 'N/A')} |
| Competitive Risk | {risk_assessment.get('competitive_risk', {}).get('level', 'N/A').upper()} | {risk_assessment.get('competitive_risk', {}).get('assessment', 'N/A')} |

### Key Risk Factors
{chr(10).join(f"- {factor}" for factor in risk_assessment.get('key_risk_factors', []))}

### Risk Mitigants
{chr(10).join(f"- {mitigant}" for mitigant in risk_assessment.get('risk_mitigants', []))}

### Risk Summary
{risk_assessment.get('risk_summary', 'No summary available.')}
"""
        final_report = report_draft + risk_section
    else:
        final_report = report_draft
    
    return {
        "final_report": final_report,
        "current_agent": "finalize_complete"
    }


# === Graph Builder ===

def create_graph(with_interrupts: bool = True) -> StateGraph:
    """
    Create the FinAgent workflow graph.
    
    Args:
        with_interrupts: If True, add human-in-the-loop interrupt points
        
    Returns:
        Compiled StateGraph
    """
    # Initialize the graph with our state schema
    workflow = StateGraph(ResearchState)
    
    # === Add all nodes ===
    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("quality_checker", quality_checker_node)
    workflow.add_node("human_approval", human_approval_node)
    workflow.add_node("risk_assessor", risk_assessor_node)
    workflow.add_node("finalize_report", finalize_report_node)
    workflow.add_node("simple_response", simple_response_node)
    
    # === Define edges ===
    
    # Start → Planner
    workflow.add_edge(START, "planner")
    
    # Planner → Conditional routing based on complexity
    workflow.add_conditional_edges(
        "planner",
        route_by_complexity,
        {
            "researcher": "researcher",
            "simple_response": "researcher"  # Even simple queries need data
        }
    )
    
    # Researcher → Check if simple or complex path
    def route_after_researcher(state: ResearchState) -> Literal["analyst", "simple_response"]:
        if state.get("query_complexity") == "simple":
            return "simple_response"
        return "analyst"
    
    workflow.add_conditional_edges(
        "researcher",
        route_after_researcher,
        {
            "analyst": "analyst",
            "simple_response": "simple_response"
        }
    )
    
    # Simple response → END
    workflow.add_edge("simple_response", END)
    
    # Complex path: Analyst → Writer
    workflow.add_edge("analyst", "writer")
    
    # Writer → Quality Checker
    workflow.add_edge("writer", "quality_checker")
    
    # Quality Checker → Conditional (approve or revise)
    workflow.add_conditional_edges(
        "quality_checker",
        route_after_quality_check,
        {
            "human_approval": "human_approval",
            "writer": "writer"  # Revision cycle!
        }
    )
    
    # Human Approval → Risk Assessor
    workflow.add_edge("human_approval", "risk_assessor")
    
    # Risk Assessor → Finalize
    workflow.add_edge("risk_assessor", "finalize_report")
    
    # Finalize → END
    workflow.add_edge("finalize_report", END)
    
    # === Compile with checkpointer ===
    checkpointer = MemorySaver()
    
    if with_interrupts:
        # Add interrupt before human approval for HITL
        compiled = workflow.compile(
            checkpointer=checkpointer,
            interrupt_before=["human_approval"]
        )
    else:
        compiled = workflow.compile(checkpointer=checkpointer)
    
    return compiled


def get_graph_visualization(graph) -> str:
    """
    Get a Mermaid diagram of the graph for visualization.
    """
    try:
        return graph.get_graph().draw_mermaid()
    except Exception as e:
        return f"Could not generate visualization: {e}"


# === Main runner for testing ===

def run_research(query: str, interrupt: bool = False) -> dict:
    """
    Run a research query through the full pipeline.
    
    Args:
        query: The research query
        interrupt: Whether to use human-in-the-loop interrupts
        
    Returns:
        Final state dict with results
    """
    graph = create_graph(with_interrupts=interrupt)
    
    # Initialize state
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
    
    # Thread ID for checkpointing
    config = {"configurable": {"thread_id": "test-run-1"}}
    
    # Run the graph and get final state
    final_state = graph.invoke(initial_state, config)
    
    return final_state


# Test function
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    print("=" * 60)
    print("FinAgent Graph Test")
    print("=" * 60)
    
    # Print the graph structure
    graph = create_graph(with_interrupts=False)
    print("\nGraph Visualization (Mermaid):")
    print("-" * 40)
    print(get_graph_visualization(graph))
    print("-" * 40)
    
    # Test with a complex query
    print("\n" + "=" * 60)
    print("Running COMPLEX query test...")
    print("=" * 60)
    
    query = "Analyze Goldman Sachs' financial health and outlook for 2025"
    print(f"\nQuery: {query}\n")
    print("Pipeline execution:")
    
    # Stream for progress, then get final state
    config = {"configurable": {"thread_id": "test-run-2"}}
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
    
    for state in graph.stream(initial_state, config):
        for node_name, state_update in state.items():
            current_agent = state_update.get("current_agent", node_name)
            print(f"  → {current_agent}")
    
    # Get the full final state
    final_state = graph.get_state(config).values
    
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    
    if final_state:
        print(f"Company: {final_state.get('company')}")
        print(f"Complexity: {final_state.get('query_complexity')}")
        print(f"Revision cycles: {final_state.get('revision_count')}")
        print(f"Quality Score: {final_state.get('quality_review', {}).get('overall_score')}/10")
        print(f"Errors: {final_state.get('errors', [])}")
        
        final_report = final_state.get('final_report')
        if final_report:
            print(f"\nFinal report length: {len(final_report)} characters")
            print("\nReport preview (first 500 chars):")
            print("-" * 40)
            print(final_report[:500])
            print("...")
    else:
        print("No result returned")