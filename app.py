"""
FinAgent - Multi-Agent Financial Research System
Streamlit Demo Interface
"""

import streamlit as st
import time
from datetime import datetime
from dotenv import load_dotenv

from src.state import ResearchState
from src.graph import create_graph, get_graph_visualization
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# === Page Config ===
st.set_page_config(
    page_title="FinAgent - AI Financial Research",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Custom CSS ===
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .agent-status {
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .agent-active {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .agent-complete {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .agent-pending {
        background-color: #f8f9fa;
        border-left: 4px solid #6c757d;
    }
</style>
""", unsafe_allow_html=True)


# === Persistent Graph with Shared Checkpointer ===
@st.cache_resource
def get_checkpointer():
    """Get a shared checkpointer that persists across reruns."""
    return MemorySaver()


def get_graph():
    """Get graph with the shared checkpointer."""
    from langgraph.graph import StateGraph, END, START
    from src.state import ResearchState
    from src.agents.planner import planner_node
    from src.agents.researcher import researcher_node
    from src.agents.analyst import analyst_node
    from src.agents.writer import writer_node
    from src.agents.quality_checker import quality_checker_node, should_revise
    from src.agents.risk_assessor import risk_assessor_node
    
    # Routing functions
    def route_by_complexity(state):
        complexity = state.get("query_complexity", "complex")
        if complexity == "simple":
            return "simple_response"
        return "researcher"
    
    def route_after_researcher(state):
        if state.get("query_complexity") == "simple":
            return "simple_response"
        return "analyst"
    
    def route_after_quality_check(state):
        decision = should_revise(state)
        if decision == "approve":
            return "human_approval"
        return "writer"
    
    # Special nodes
    def simple_response_node(state):
        financial_data = state.get("financial_data")
        company = state.get("company", "Unknown")
        query = state.get("query", "")
        
        if financial_data and financial_data.get("success", True):
            price = financial_data.get("current_price", "N/A")
            market_cap = financial_data.get("market_cap")
            pe_ratio = financial_data.get("pe_ratio", "N/A")
            market_cap_str = f"${market_cap:,}" if market_cap else "N/A"
            
            simple_report = f"""# {company} - Quick Lookup

**Query:** {query}

| Metric | Value |
|--------|-------|
| Current Price | ${price} |
| Market Cap | {market_cap_str} |
| P/E Ratio | {pe_ratio} |
| Sector | {financial_data.get('sector', 'N/A')} |

*Quick lookup by FinAgent.*
"""
            return {"final_report": simple_report, "current_agent": "simple_response_complete"}
        else:
            return {"final_report": f"Could not retrieve data for {company}.", "current_agent": "simple_response_failed"}
    
    def human_approval_node(state):
        return {"human_approved": True, "current_agent": "human_approval_complete"}
    
    def finalize_report_node(state):
        report_draft = state.get("report_draft", "")
        risk_assessment = state.get("risk_assessment", {})
        
        if risk_assessment and risk_assessment.get("overall_risk_level"):
            risk_section = f"""

---

## Detailed Risk Assessment

**Overall Risk Level: {risk_assessment.get('overall_risk_level', 'N/A').upper()}**

### Key Risk Factors
{chr(10).join(f"- {factor}" for factor in risk_assessment.get('key_risk_factors', []))}

### Risk Summary
{risk_assessment.get('risk_summary', 'No summary available.')}
"""
            final_report = report_draft + risk_section
        else:
            final_report = report_draft
        
        return {"final_report": final_report, "current_agent": "finalize_complete"}
    
    # Build graph
    workflow = StateGraph(ResearchState)
    
    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("quality_checker", quality_checker_node)
    workflow.add_node("human_approval", human_approval_node)
    workflow.add_node("risk_assessor", risk_assessor_node)
    workflow.add_node("finalize_report", finalize_report_node)
    workflow.add_node("simple_response", simple_response_node)
    
    workflow.add_edge(START, "planner")
    workflow.add_conditional_edges("planner", route_by_complexity, {"researcher": "researcher", "simple_response": "researcher"})
    workflow.add_conditional_edges("researcher", route_after_researcher, {"analyst": "analyst", "simple_response": "simple_response"})
    workflow.add_edge("simple_response", END)
    workflow.add_edge("analyst", "writer")
    workflow.add_edge("writer", "quality_checker")
    workflow.add_conditional_edges("quality_checker", route_after_quality_check, {"human_approval": "human_approval", "writer": "writer"})
    workflow.add_edge("human_approval", "risk_assessor")
    workflow.add_edge("risk_assessor", "finalize_report")
    workflow.add_edge("finalize_report", END)
    
    # Compile with shared checkpointer
    checkpointer = get_checkpointer()
    compiled = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_approval"]
    )
    
    return compiled


# === Session State Initialization ===
def init_session_state():
    if "research_state" not in st.session_state:
        st.session_state.research_state = None
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "current_agent" not in st.session_state:
        st.session_state.current_agent = None
    if "agent_history" not in st.session_state:
        st.session_state.agent_history = []
    if "final_report" not in st.session_state:
        st.session_state.final_report = None
    if "awaiting_approval" not in st.session_state:
        st.session_state.awaiting_approval = False
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    if "run_complete" not in st.session_state:
        st.session_state.run_complete = False


# === Agent Status Display ===
AGENT_DESCRIPTIONS = {
    "planner": ("🎯 Planner", "Analyzing query and creating research plan..."),
    "researcher": ("🔍 Researcher", "Gathering data from web and financial sources..."),
    "analyst": ("📊 Analyst", "Synthesizing findings into structured analysis..."),
    "writer": ("✍️ Writer", "Generating research report..."),
    "quality_checker": ("✅ Quality Checker", "Reviewing report quality..."),
    "human_approval": ("👤 Human Approval", "Awaiting your review..."),
    "risk_assessor": ("⚠️ Risk Assessor", "Evaluating risk factors..."),
    "finalize_report": ("📋 Finalize", "Preparing final report..."),
    "simple_response": ("⚡ Quick Response", "Generating quick data lookup..."),
}


def display_agent_status(agent_history: list, current_agent: str | None):
    all_agents = ["planner", "researcher", "analyst", "writer", 
                  "quality_checker", "human_approval", "risk_assessor", "finalize_report"]
    
    completed = set()
    for entry in agent_history:
        agent_name = entry.replace("_complete", "").replace("_failed", "")
        completed.add(agent_name)
    
    for agent in all_agents:
        icon, description = AGENT_DESCRIPTIONS.get(agent, ("❓", "Processing..."))
        
        if current_agent and agent in current_agent:
            st.markdown(f"""
            <div class="agent-status agent-active">
                <strong>{icon} {agent.replace('_', ' ').title()}</strong><br>
                <small>{description}</small>
            </div>
            """, unsafe_allow_html=True)
        elif agent in completed:
            st.markdown(f"""
            <div class="agent-status agent-complete">
                <strong>{icon} {agent.replace('_', ' ').title()}</strong> ✓
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="agent-status agent-pending">
                <strong>{icon} {agent.replace('_', ' ').title()}</strong>
            </div>
            """, unsafe_allow_html=True)


# === Sidebar ===
def render_sidebar():
    with st.sidebar:
        st.title("📊 FinAgent")
        st.caption("Multi-Agent Financial Research System")
        
        st.divider()
        
        st.subheader("About")
        st.markdown("""
        FinAgent uses **6 specialized AI agents** to research companies:
        
        1. **Planner** - Creates research strategy
        2. **Researcher** - Gathers data
        3. **Analyst** - Synthesizes findings
        4. **Writer** - Generates report
        5. **Quality Checker** - Reviews & revises
        6. **Risk Assessor** - Evaluates risks
        """)
        
        st.divider()
        st.subheader("🔀 Graph Structure")
        if st.button("Show Graph Diagram", use_container_width=True):
            st.code("""
        graph TD
            START --> planner
            planner -.-> researcher
            researcher -.-> analyst
            researcher -.-> simple_response
            analyst --> writer
            writer --> quality_checker
            quality_checker -.-> human_approval
            quality_checker -.-> writer
            human_approval --> risk_assessor
            risk_assessor --> finalize_report
            finalize_report --> END
            simple_response --> END
            """, language="mermaid")
            st.caption("Dotted lines = conditional edges")

        st.divider()
        
        st.subheader("Example Queries")
        st.markdown("""
        **Complex:**
        - "Analyze Goldman Sachs' financial health"
        - "What are the risks facing Tesla?"
        
        **Simple:**
        - "What is Apple's stock price?"
        """)
        
        st.divider()
        
        if st.button("🔄 Reset Session", use_container_width=True):
            # Clear checkpointer cache too
            get_checkpointer.clear()
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# === Main App ===
def main():
    init_session_state()
    render_sidebar()
    
    st.title("📊 FinAgent Research System")
    
    # Query input
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Enter your research query:",
            placeholder="e.g., Analyze Goldman Sachs' financial health",
            disabled=st.session_state.is_running or st.session_state.awaiting_approval
        )
    with col2:
        st.write("")
        st.write("")
        run_button = st.button(
            "🚀 Research",
            disabled=st.session_state.is_running or st.session_state.awaiting_approval or not query,
            use_container_width=True
        )
    
    st.divider()
    
    # Layout
    left_col, right_col = st.columns([1, 2])
    
    with left_col:
        st.subheader("Pipeline Status")
        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        with status_placeholder.container():
            if st.session_state.agent_history:
                display_agent_status(st.session_state.agent_history, st.session_state.current_agent)
            else:
                st.info("Enter a query to start")
    
    with right_col:
        st.subheader("Research Output")
        
        # === HUMAN APPROVAL STATE ===
        if st.session_state.awaiting_approval:
            st.warning("⏸️ **Awaiting Your Approval**")
            
            if st.session_state.research_state:
                draft = st.session_state.research_state.get("report_draft", "No draft available")
                quality = st.session_state.research_state.get("quality_review", {})
                
                st.markdown(f"**Quality Score:** {quality.get('overall_score', 'N/A')}/10")
                
                with st.expander("📄 View Draft Report", expanded=True):
                    st.markdown(draft[:3000] + "..." if len(str(draft)) > 3000 else draft)
            
            col_approve, col_reject = st.columns(2)
            with col_approve:
                if st.button("✅ Approve & Continue", use_container_width=True, type="primary"):
                    # Continue the graph
                    graph = get_graph()
                    config = {"configurable": {"thread_id": st.session_state.thread_id}}
                    
                    with st.spinner("Continuing pipeline..."):
                        try:
                            for event in graph.stream(None, config, stream_mode="updates"):
                                if isinstance(event, dict):
                                    for node_name, node_state in event.items():
                                        if isinstance(node_state, dict):
                                            current_agent = node_state.get("current_agent", node_name)
                                        else:
                                            current_agent = node_name
                                        st.session_state.agent_history.append(current_agent)
                            
                            final_state = graph.get_state(config)
                            st.session_state.research_state = final_state.values
                            st.session_state.final_report = final_state.values.get("final_report")
                            st.session_state.awaiting_approval = False
                            st.session_state.run_complete = True
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
            
            with col_reject:
                if st.button("❌ Reject", use_container_width=True):
                    st.session_state.awaiting_approval = False
                    st.session_state.agent_history = []
                    st.warning("Rejected. Start a new query.")
                    st.rerun()
        
        # === FINAL REPORT STATE ===
        elif st.session_state.final_report:
            st.success("✅ Research Complete!")
            
            if st.session_state.research_state:
                state = st.session_state.research_state
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Company", state.get("company", "N/A"))
                with m2:
                    st.metric("Complexity", str(state.get("query_complexity", "N/A")).title())
                with m3:
                    st.metric("Revisions", state.get("revision_count", 0))
                with m4:
                    quality = state.get("quality_review", {}) or {}
                    st.metric("Quality", f"{quality.get('overall_score', 'N/A')}/10")
            
            st.divider()
            
            tab_report, tab_analysis, tab_debug = st.tabs(["📄 Report", "📊 Analysis", "🔧 Debug"])
            
            with tab_report:
                st.markdown(st.session_state.final_report)
                st.download_button(
                    "📥 Download Report",
                    data=st.session_state.final_report,
                    file_name="finagent_report.md",
                    mime="text/markdown"
                )
            
            with tab_analysis:
                if st.session_state.research_state:
                    analysis = st.session_state.research_state.get("analysis", {}) or {}
                    if analysis:
                        st.markdown(f"**Health:** {analysis.get('financial_health_score', 'N/A').upper()}")
                        st.markdown(f"**Outlook:** {analysis.get('outlook', 'N/A').upper()}")
                        st.markdown(f"**Summary:** {analysis.get('summary', 'N/A')}")
            
            with tab_debug:
                st.json(st.session_state.research_state)
        
        # === RUN RESEARCH ===
        elif run_button and query:
            # Reset for new run
            st.session_state.thread_id = f"session-{datetime.now().strftime('%Y%m%d-%H%M%S%f')}"
            st.session_state.agent_history = []
            st.session_state.final_report = None
            st.session_state.research_state = None
            
            graph = get_graph()
            
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
            
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            progress_bar.progress(0)
            step = 0
            
            try:
                for event in graph.stream(initial_state, config, stream_mode="updates"):
                    if isinstance(event, dict):
                        for node_name, node_state in event.items():
                            if isinstance(node_state, dict):
                                current_agent = node_state.get("current_agent", node_name)
                            else:
                                current_agent = node_name
                            
                            st.session_state.current_agent = current_agent
                            st.session_state.agent_history.append(current_agent)
                            
                            step += 1
                            progress_bar.progress(min(step / 8, 0.95))
                            
                            with status_placeholder.container():
                                display_agent_status(st.session_state.agent_history, current_agent)
                            
                            time.sleep(0.1)
                    
                    # Check for interrupt
                    state_snapshot = graph.get_state(config)
                    if state_snapshot.next and "human_approval" in state_snapshot.next:
                        st.session_state.awaiting_approval = True
                        st.session_state.research_state = state_snapshot.values
                        st.rerun()
                
                # If we get here, no interrupt - check final state
                final_state = graph.get_state(config)
                st.session_state.research_state = final_state.values
                st.session_state.final_report = final_state.values.get("final_report")
                progress_bar.progress(1.0)
                st.rerun()
                
            except Exception as e:
                import traceback
                st.error(f"Error: {e}\n{traceback.format_exc()}")
        
        else:
            st.info("👆 Enter a query above to get started")


if __name__ == "__main__":
    main()