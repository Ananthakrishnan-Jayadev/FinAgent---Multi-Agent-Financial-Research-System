"""
FinAgent State Schema

This TypedDict defines the complete state that flows through the graph.
Every agent reads from and writes to this shared state.
"""

from typing import TypedDict, Annotated, Literal
from pydantic import BaseModel
import operator


# === Pydantic models for structured data within state ===

class ResearchSubtask(BaseModel):
    """A single research subtask from the Planner."""
    task_id: str
    description: str
    task_type: Literal["web_search", "financial_data", "analysis"]
    priority: Literal["high", "medium", "low"]


class ResearchFinding(BaseModel):
    """A single finding from the Researcher."""
    source: str
    content: str
    relevance: Literal["high", "medium", "low"]
    timestamp: str


class FinancialMetrics(BaseModel):
    """Structured financial data from yfinance."""
    ticker: str
    company_name: str
    current_price: float | None
    market_cap: float | None
    pe_ratio: float | None
    debt_to_equity: float | None
    revenue_growth: float | None
    profit_margin: float | None
    fifty_two_week_high: float | None
    fifty_two_week_low: float | None
    dividend_yield: float | None
    sector: str | None
    industry: str | None
    description: str | None


class AnalysisOutput(BaseModel):
    """Structured output from the Analyst."""
    key_findings: list[str]
    strengths: list[str]
    weaknesses: list[str]
    opportunities: list[str]
    threats: list[str]
    financial_health_score: Literal["strong", "moderate", "weak", "critical"]
    outlook: Literal["bullish", "neutral", "bearish"]
    summary: str


class QualityReview(BaseModel):
    """Output from the Quality Checker."""
    passed: bool
    overall_score: int  # 1-10
    unsupported_claims: list[str]
    missing_sections: list[str]
    factual_concerns: list[str]
    suggestions: list[str]
    revision_instructions: str | None


class RiskAssessment(BaseModel):
    """Output from the Risk Assessor."""
    overall_risk_level: Literal["low", "moderate", "high", "critical"]
    market_risk: str
    credit_risk: str
    regulatory_risk: str
    operational_risk: str
    competitive_risk: str
    key_risk_factors: list[str]
    risk_summary: str


# === Custom reducer for list accumulation ===
# This allows multiple nodes to append to the same list field

def add_to_list(existing: list, new: list | None) -> list:
    """Reducer that accumulates list items across node executions."""
    if new is None:
        return existing
    return existing + new


# === Main State Schema ===

class ResearchState(TypedDict):
    """
    Complete state for the FinAgent research pipeline.
    
    This state flows through all nodes in the graph.
    Each field is documented with which agent(s) read/write it.
    """
    
    # --- Input ---
    query: str  # Original user query (set once at start)
    
    # --- Planner outputs ---
    company: str | None  # Extracted company/ticker (Planner writes)
    query_complexity: Literal["simple", "complex"]  # For routing (Planner writes)
    research_plan: list[dict]  # List of subtasks (Planner writes)
    
    # --- Researcher outputs ---
    raw_findings: Annotated[list[dict], add_to_list]  # Accumulated research (Researcher appends)
    financial_data: dict | None  # Structured financial metrics (Researcher writes)
    
    # --- Analyst outputs ---
    analysis: dict | None  # AnalysisOutput as dict (Analyst writes)
    
    # --- Writer outputs ---
    report_draft: str | None  # Markdown report (Writer writes)
    
    # --- Quality Checker outputs ---
    quality_review: dict | None  # QualityReview as dict (QC writes)
    
    # --- Risk Assessor outputs ---
    risk_assessment: dict | None  # RiskAssessment as dict (Risk Assessor writes)
    
    # --- Control flow ---
    revision_count: int  # Tracks revision cycles (QC increments)
    human_approved: bool | None  # HITL gate status
    current_agent: str  # For UI progress tracking
    
    # --- Final output ---
    final_report: str | None  # Approved final report
    
    # --- Error tracking ---
    errors: Annotated[list[str], add_to_list]  # Accumulated errors from any node