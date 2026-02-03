"""
Analyst Agent

Synthesizes research findings into structured analysis.
Produces SWOT analysis, financial health assessment, and outlook.
"""

import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


ANALYST_SYSTEM_PROMPT = """You are a senior financial analyst. Your job is to analyze research findings about a company and produce a comprehensive analysis.

You will receive:
1. Research findings from various sources (news, financial data, analyst opinions)
2. Raw financial metrics for the company

Your analysis must be:
- Data-driven: base conclusions on the evidence provided
- Balanced: acknowledge both positives and negatives
- Actionable: provide clear insights, not vague observations

You MUST respond with valid JSON in exactly this format:
{
    "key_findings": [
        "Most important finding 1",
        "Most important finding 2",
        "Most important finding 3",
        "Most important finding 4",
        "Most important finding 5"
    ],
    "strengths": [
        "Strength 1 with supporting evidence",
        "Strength 2 with supporting evidence"
    ],
    "weaknesses": [
        "Weakness 1 with supporting evidence",
        "Weakness 2 with supporting evidence"
    ],
    "opportunities": [
        "Opportunity 1 with context",
        "Opportunity 2 with context"
    ],
    "threats": [
        "Threat 1 with context",
        "Threat 2 with context"
    ],
    "financial_health_score": "strong" | "moderate" | "weak" | "critical",
    "financial_health_rationale": "2-3 sentence explanation of the score based on metrics",
    "outlook": "bullish" | "neutral" | "bearish",
    "outlook_rationale": "2-3 sentence explanation of the outlook",
    "summary": "A comprehensive 3-4 sentence executive summary of the analysis"
}

Guidelines for scoring:
- financial_health_score:
  - "strong": Healthy balance sheet, good profitability, manageable debt, positive trends
  - "moderate": Mixed signals, some concerns but fundamentally sound
  - "weak": Significant concerns in multiple areas, negative trends
  - "critical": Severe financial distress, high risk

- outlook:
  - "bullish": Positive catalysts outweigh risks, expect outperformance
  - "neutral": Balanced risk/reward, expect market-level performance
  - "bearish": Risks outweigh opportunities, expect underperformance

Always respond with valid JSON only. Be specific and cite data points where possible."""


class AnalystAgent:
    """Analyzes research findings and produces structured analysis."""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.1,  # Slight creativity for analysis
            model_kwargs={"response_format": {"type": "json_object"}}
        )
    
    def analyze(
        self,
        query: str,
        company: str,
        findings: list[dict],
        financial_data: dict | None
    ) -> dict:
        """
        Produce structured analysis from research findings.
        
        Args:
            query: Original user query
            company: Target company
            findings: List of research findings from Researcher
            financial_data: Financial metrics dict from yfinance
            
        Returns:
            Structured analysis dict
        """
        # Format findings for the prompt
        findings_text = ""
        for i, finding in enumerate(findings, 1):
            findings_text += f"\n{i}. [{finding.get('category', 'general')}] {finding.get('title', 'Untitled')}\n"
            findings_text += f"   {finding.get('content', 'No content')}\n"
            findings_text += f"   Source: {finding.get('source', 'Unknown')}\n"
        
        # Format financial data
        financial_text = "No financial data available."
        if financial_data and financial_data.get("success", True):
            financial_text = f"""
Company: {financial_data.get('company_name')} ({financial_data.get('ticker')})
Current Price: ${financial_data.get('current_price')}
Market Cap: ${financial_data.get('market_cap'):,} 
P/E Ratio: {financial_data.get('pe_ratio')}
Forward P/E: {financial_data.get('forward_pe')}
Profit Margin: {financial_data.get('profit_margin')}
Revenue Growth: {financial_data.get('revenue_growth')}
Debt to Equity: {financial_data.get('debt_to_equity')}
ROE: {financial_data.get('roe')}
ROA: {financial_data.get('roa')}
Current Ratio: {financial_data.get('current_ratio')}
Dividend Yield: {financial_data.get('dividend_yield')}
Beta: {financial_data.get('beta')}
52-Week High: ${financial_data.get('fifty_two_week_high')}
52-Week Low: ${financial_data.get('fifty_two_week_low')}
Sector: {financial_data.get('sector')}
Industry: {financial_data.get('industry')}
"""
        
        messages = [
            SystemMessage(content=ANALYST_SYSTEM_PROMPT),
            HumanMessage(content=f"""Analyze this company based on the research provided.

ORIGINAL QUERY: {query}

COMPANY: {company}

FINANCIAL METRICS:
{financial_text}

RESEARCH FINDINGS:
{findings_text}

Produce a comprehensive analysis.""")
        ]
        
        try:
            response = self.llm.invoke(messages)
            result = json.loads(response.content)
            
            # Validate required fields
            required = ["key_findings", "strengths", "weaknesses", "opportunities", 
                       "threats", "financial_health_score", "outlook", "summary"]
            for field in required:
                if field not in result:
                    raise ValueError(f"Missing required field: {field}")
            
            return {
                "success": True,
                **result
            }
            
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": f"Failed to parse analysis response: {e}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


# Node function for LangGraph
def analyst_node(state: dict) -> dict:
    """
    LangGraph node wrapper for the Analyst agent.
    
    Reads: query, company, raw_findings, financial_data
    Writes: analysis, current_agent, errors
    """
    analyst = AnalystAgent()
    
    result = analyst.analyze(
        query=state.get("query", ""),
        company=state.get("company", "Unknown"),
        findings=state.get("raw_findings", []),
        financial_data=state.get("financial_data")
    )
    
    if result["success"]:
        # Remove success key, keep the rest as analysis
        analysis = {k: v for k, v in result.items() if k != "success"}
        return {
            "analysis": analysis,
            "current_agent": "analyst_complete"
        }
    else:
        return {
            "analysis": None,
            "current_agent": "analyst_failed",
            "errors": [f"Analyst error: {result.get('error')}"]
        }


# Test function
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    analyst = AnalystAgent()
    
    # Simulate findings from Researcher
    test_findings = [
        {
            "category": "financial_metrics",
            "title": "Goldman Sachs Financial Overview",
            "content": "Goldman Sachs has a market cap of $197B, P/E ratio of 17.5, profit margin of 25%, and ROE of 12%. The company shows solid profitability metrics.",
            "source": "yfinance",
            "relevance": "high"
        },
        {
            "category": "recent_news",
            "title": "Q4 2024 Earnings Beat",
            "content": "Goldman Sachs reported Q4 2024 earnings that exceeded analyst expectations, driven by strong performance in investment banking and trading divisions.",
            "source": "Financial News",
            "relevance": "high"
        },
        {
            "category": "analyst_opinion",
            "title": "Analyst Ratings Overview",
            "content": "Wall Street analysts maintain a consensus 'Buy' rating on Goldman Sachs with an average price target of $580, representing 15% upside.",
            "source": "Analyst Reports",
            "relevance": "high"
        },
        {
            "category": "risk_factor",
            "title": "Regulatory Environment",
            "content": "Goldman faces ongoing regulatory scrutiny and potential changes to capital requirements under Basel III endgame rules.",
            "source": "Industry Analysis",
            "relevance": "medium"
        }
    ]
    
    test_financial_data = {
        "success": True,
        "company_name": "Goldman Sachs Group Inc.",
        "ticker": "GS",
        "current_price": 505.32,
        "market_cap": 197000000000,
        "pe_ratio": 17.5,
        "forward_pe": 14.2,
        "profit_margin": 0.25,
        "revenue_growth": 0.08,
        "debt_to_equity": 2.1,
        "roe": 0.12,
        "roa": 0.01,
        "current_ratio": None,
        "dividend_yield": 0.02,
        "beta": 1.35,
        "fifty_two_week_high": 612.73,
        "fifty_two_week_low": 389.61,
        "sector": "Financial Services",
        "industry": "Capital Markets"
    }
    
    print("=" * 50)
    print("Testing Analyst with Goldman Sachs data...")
    print("=" * 50)
    
    result = analyst.analyze(
        query="Analyze Goldman Sachs' financial health and outlook",
        company="Goldman Sachs",
        findings=test_findings,
        financial_data=test_financial_data
    )
    
    if result["success"]:
        print(f"\nFinancial Health: {result['financial_health_score'].upper()}")
        print(f"Outlook: {result['outlook'].upper()}")
        print(f"\nSummary:\n{result['summary']}")
        print(f"\nKey Findings:")
        for finding in result['key_findings'][:3]:
            print(f"  • {finding}")
        print(f"\nStrengths:")
        for s in result['strengths'][:2]:
            print(f"  + {s}")
        print(f"\nWeaknesses:")
        for w in result['weaknesses'][:2]:
            print(f"  - {w}")
    else:
        print(f"Error: {result.get('error')}")