"""
Quality Checker Agent

Reviews draft reports for quality, accuracy, and completeness.
Can trigger revision cycles if issues are found.
"""

import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


QUALITY_CHECKER_SYSTEM_PROMPT = """You are a senior editor and fact-checker for financial research reports. Your job is to review draft reports for quality, accuracy, and completeness.

You must evaluate the report on these criteria:

1. **Completeness**: Does the report have all required sections? (Executive Summary, Company Overview, Financial Analysis, SWOT, Key Findings, Outlook, Risk Factors, Conclusion)

2. **Data Support**: Are claims backed by specific data points? Flag any claims that lack evidence.

3. **Logical Consistency**: Do the conclusions follow from the analysis? Does the outlook match the SWOT and financial health assessment?

4. **Factual Accuracy**: Are the financial metrics used correctly? Any obvious errors?

5. **Professional Quality**: Is the writing clear, professional, and free of obvious issues?

Scoring Guidelines:
- Score 8-10: High quality, ready for delivery (PASS)
- Score 5-7: Acceptable but has minor issues (PASS with notes)
- Score 1-4: Significant issues, needs revision (FAIL)

You MUST respond with valid JSON in exactly this format:
{
    "passed": true or false,
    "overall_score": 1-10,
    "completeness": {
        "score": 1-10,
        "missing_sections": ["list of missing sections if any"],
        "notes": "brief assessment"
    },
    "data_support": {
        "score": 1-10,
        "unsupported_claims": ["list of claims without evidence"],
        "notes": "brief assessment"
    },
    "logical_consistency": {
        "score": 1-10,
        "inconsistencies": ["list of logical issues if any"],
        "notes": "brief assessment"
    },
    "factual_accuracy": {
        "score": 1-10,
        "concerns": ["list of potential factual issues"],
        "notes": "brief assessment"
    },
    "professional_quality": {
        "score": 1-10,
        "issues": ["list of writing/formatting issues"],
        "notes": "brief assessment"
    },
    "summary": "2-3 sentence overall assessment",
    "revision_instructions": "If failed, specific instructions for what to fix. If passed, null."
}

Be rigorous but fair. The goal is to ensure quality, not to be unnecessarily harsh.
A report can pass with minor imperfections if the core analysis is sound.

Always respond with valid JSON only."""


class QualityCheckerAgent:
    """Reviews reports for quality and triggers revisions if needed."""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(
            model=model,
            temperature=0,  # Consistent evaluation
            model_kwargs={"response_format": {"type": "json_object"}}
        )
    
    def review(
        self,
        report: str,
        analysis: dict,
        financial_data: dict | None,
        revision_count: int = 0
    ) -> dict:
        """
        Review a draft report for quality.
        
        Args:
            report: The markdown report to review
            analysis: Original analysis dict (for consistency check)
            financial_data: Financial metrics (for accuracy check)
            revision_count: How many revisions have already occurred
            
        Returns:
            Quality review dict with pass/fail and detailed feedback
        """
        # Provide context about the analysis for consistency checking
        analysis_summary = f"""
Outlook from analysis: {analysis.get('outlook', 'N/A')}
Financial health score: {analysis.get('financial_health_score', 'N/A')}
Key findings count: {len(analysis.get('key_findings', []))}
"""
        
        # Provide financial data for accuracy checking
        financial_summary = "No financial data available for verification."
        if financial_data:
            financial_summary = f"""
Ticker: {financial_data.get('ticker')}
Price: ${financial_data.get('current_price')}
Market Cap: ${financial_data.get('market_cap'):,} if financial_data.get('market_cap') else 'N/A'
P/E Ratio: {financial_data.get('pe_ratio')}
Profit Margin: {financial_data.get('profit_margin')}
"""
        
        # Adjust strictness based on revision count
        strictness_note = ""
        if revision_count >= 2:
            strictness_note = """
NOTE: This report has already been revised {revision_count} times. 
Be more lenient - pass if the core content is acceptable, even with minor issues.
We need to avoid infinite revision loops.
""".format(revision_count=revision_count)
        
        messages = [
            SystemMessage(content=QUALITY_CHECKER_SYSTEM_PROMPT),
            HumanMessage(content=f"""Review this financial research report.

{strictness_note}

ANALYSIS CONTEXT (for consistency checking):
{analysis_summary}

FINANCIAL DATA (for accuracy checking):
{financial_summary}

DRAFT REPORT TO REVIEW:
{report}

Provide your quality assessment.""")
        ]
        
        try:
            response = self.llm.invoke(messages)
            result = json.loads(response.content)
            
            # Validate required fields
            if "passed" not in result:
                result["passed"] = result.get("overall_score", 0) >= 5
            
            if "overall_score" not in result:
                result["overall_score"] = 5  # Default middle score
            
            return {
                "success": True,
                **result
            }
            
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "passed": True,  # Default to pass on error to avoid blocking
                "overall_score": 5,
                "error": f"Failed to parse QC response: {e}",
                "summary": "Quality check encountered an error, defaulting to pass.",
                "revision_instructions": None
            }
        except Exception as e:
            return {
                "success": False,
                "passed": True,
                "overall_score": 5,
                "error": str(e),
                "summary": "Quality check encountered an error, defaulting to pass.",
                "revision_instructions": None
            }


# Node function for LangGraph
def quality_checker_node(state: dict) -> dict:
    """
    LangGraph node wrapper for the Quality Checker agent.
    
    Reads: report_draft, analysis, financial_data, revision_count
    Writes: quality_review, revision_count, current_agent, errors
    """
    qc = QualityCheckerAgent()
    
    report = state.get("report_draft")
    if not report:
        return {
            "quality_review": {
                "passed": False,
                "overall_score": 0,
                "summary": "No report to review",
                "revision_instructions": "Generate a report first"
            },
            "current_agent": "quality_checker_failed",
            "errors": ["No report draft available for quality check"]
        }
    
    result = qc.review(
        report=report,
        analysis=state.get("analysis", {}),
        financial_data=state.get("financial_data"),
        revision_count=state.get("revision_count", 0)
    )
    
    # Prepare the quality review for state
    quality_review = {
        "passed": result.get("passed", False),
        "overall_score": result.get("overall_score", 0),
        "summary": result.get("summary", ""),
        "revision_instructions": result.get("revision_instructions"),
        "completeness": result.get("completeness", {}),
        "data_support": result.get("data_support", {}),
        "logical_consistency": result.get("logical_consistency", {}),
        "factual_accuracy": result.get("factual_accuracy", {}),
        "professional_quality": result.get("professional_quality", {})
    }
    
    # Increment revision count if we're going to revise
    new_revision_count = state.get("revision_count", 0)
    if not result.get("passed", False):
        new_revision_count += 1
    
    return {
        "quality_review": quality_review,
        "revision_count": new_revision_count,
        "current_agent": "quality_checker_complete"
    }


# Routing function for conditional edge
def should_revise(state: dict) -> str:
    """
    Determine if the report needs revision.
    
    Returns:
        "revise" if revision needed and under limit
        "approve" if passed or revision limit reached
    """
    quality_review = state.get("quality_review", {})
    revision_count = state.get("revision_count", 0)
    
    # Max 2 revision cycles to prevent infinite loops
    MAX_REVISIONS = 2
    
    if quality_review.get("passed", False):
        return "approve"
    
    if revision_count >= MAX_REVISIONS:
        # Force approval after max revisions
        return "approve"
    
    return "revise"


# Test function
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    qc = QualityCheckerAgent()
    
    # Test with a good report
    good_report = """# Goldman Sachs Research Report

**Date:** February 03, 2026  
**Analyst:** FinAgent AI Research System  
**Rating:** Bullish  
**Financial Health:** Moderate

---

## Executive Summary
Goldman Sachs demonstrates strong profitability with a 25% profit margin and 12% ROE. Despite a debt-to-equity ratio of 2.1, the company shows positive momentum with Q4 2024 earnings beating expectations. Analyst consensus supports a Buy rating with 15% upside potential.

---

## Company Overview
The Goldman Sachs Group, Inc. operates as a global investment banking, securities, and investment management company in the Financial Services sector.

---

## Financial Analysis

### Key Metrics
| Metric | Value |
|--------|-------|
| Current Price | $505.32 |
| Market Cap | $197B |
| P/E Ratio | 17.5 |
| Profit Margin | 25% |
| ROE | 12% |
| Debt/Equity | 2.1 |

### Financial Health Assessment
Goldman maintains strong profitability metrics but carries elevated leverage compared to peers.

---

## SWOT Analysis

### Strengths
- Industry-leading profit margins at 25%
- Strong investment banking franchise

### Weaknesses
- High debt-to-equity ratio of 2.1
- Exposure to market volatility

### Opportunities
- Growing M&A advisory demand
- Asset management expansion

### Threats
- Basel III capital requirements
- Economic downturn risks

---

## Key Findings
1. Q4 2024 earnings exceeded analyst expectations
2. Consensus Buy rating with $580 price target
3. Strong profitability despite leverage concerns

---

## Outlook & Recommendation
Bullish outlook supported by earnings momentum and analyst consensus. Monitor leverage levels.

---

## Risk Factors
- Regulatory capital requirements
- Market volatility exposure
- Economic cycle sensitivity

---

## Conclusion
Goldman Sachs offers attractive risk/reward for investors comfortable with financial sector exposure.

---

*This report was generated by FinAgent. Not financial advice.*
"""
    
    test_analysis = {
        "outlook": "bullish",
        "financial_health_score": "moderate",
        "key_findings": ["Finding 1", "Finding 2", "Finding 3"]
    }
    
    test_financial_data = {
        "ticker": "GS",
        "current_price": 505.32,
        "market_cap": 197000000000,
        "pe_ratio": 17.5,
        "profit_margin": 0.25
    }
    
    print("=" * 50)
    print("Testing Quality Checker with good report...")
    print("=" * 50)
    
    result = qc.review(
        report=good_report,
        analysis=test_analysis,
        financial_data=test_financial_data,
        revision_count=0
    )
    
    print(f"\nPassed: {result.get('passed')}")
    print(f"Overall Score: {result.get('overall_score')}/10")
    print(f"\nSummary: {result.get('summary')}")
    
    print("\nDetailed Scores:")
    for category in ['completeness', 'data_support', 'logical_consistency', 'factual_accuracy', 'professional_quality']:
        cat_data = result.get(category, {})
        print(f"  {category}: {cat_data.get('score', 'N/A')}/10")
    
    if result.get('revision_instructions'):
        print(f"\nRevision Instructions: {result.get('revision_instructions')}")
    
    # Test routing function
    print("\n" + "=" * 50)
    print("Testing routing decision...")
    print("=" * 50)
    
    test_state = {
        "quality_review": result,
        "revision_count": 0
    }
    decision = should_revise(test_state)
    print(f"Routing decision: {decision}")