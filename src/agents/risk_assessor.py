"""
Risk Assessor Agent

Provides dedicated financial risk analysis.
Runs after human approval to add final risk evaluation.
"""

import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


RISK_ASSESSOR_SYSTEM_PROMPT = """You are a financial risk analyst specializing in corporate risk assessment. Your job is to evaluate the key risks facing a company based on research findings and financial data.

You must assess these risk categories:

1. **Market Risk**: Exposure to market movements, volatility, interest rates
2. **Credit Risk**: Counterparty risk, default exposure, credit quality
3. **Regulatory Risk**: Compliance requirements, regulatory changes, legal exposure
4. **Operational Risk**: Business execution, technology, talent, process risks
5. **Competitive Risk**: Market position threats, disruption, competitive dynamics

For each category, provide a brief but specific assessment based on the data provided.

You MUST respond with valid JSON in exactly this format:
{
    "overall_risk_level": "low" | "moderate" | "high" | "critical",
    "market_risk": {
        "level": "low" | "moderate" | "high",
        "assessment": "2-3 sentence specific assessment"
    },
    "credit_risk": {
        "level": "low" | "moderate" | "high",
        "assessment": "2-3 sentence specific assessment"
    },
    "regulatory_risk": {
        "level": "low" | "moderate" | "high",
        "assessment": "2-3 sentence specific assessment"
    },
    "operational_risk": {
        "level": "low" | "moderate" | "high",
        "assessment": "2-3 sentence specific assessment"
    },
    "competitive_risk": {
        "level": "low" | "moderate" | "high",
        "assessment": "2-3 sentence specific assessment"
    },
    "key_risk_factors": [
        "Top risk factor 1",
        "Top risk factor 2",
        "Top risk factor 3"
    ],
    "risk_mitigants": [
        "Key mitigating factor 1",
        "Key mitigating factor 2"
    ],
    "risk_summary": "3-4 sentence overall risk assessment"
}

Be specific and reference actual data points where possible. Avoid generic risk statements.
Always respond with valid JSON only."""


class RiskAssessorAgent:
    """Provides dedicated risk analysis for financial reports."""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.1,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
    
    def assess_risk(
        self,
        company: str,
        analysis: dict,
        financial_data: dict | None,
        findings: list[dict]
    ) -> dict:
        """
        Perform comprehensive risk assessment.
        
        Args:
            company: Target company
            analysis: Analysis dict from Analyst
            financial_data: Financial metrics
            findings: Research findings
            
        Returns:
            Risk assessment dict
        """
        # Format context for the LLM
        analysis_context = f"""
SWOT Analysis:
- Strengths: {', '.join(analysis.get('strengths', [])[:3])}
- Weaknesses: {', '.join(analysis.get('weaknesses', [])[:3])}
- Threats: {', '.join(analysis.get('threats', [])[:3])}

Financial Health: {analysis.get('financial_health_score', 'N/A')}
Outlook: {analysis.get('outlook', 'N/A')}
"""
        
        financial_context = "No financial data available."
        if financial_data:
            financial_context = f"""
Company: {financial_data.get('company_name')} ({financial_data.get('ticker')})
Sector: {financial_data.get('sector')}
Industry: {financial_data.get('industry')}
Market Cap: ${financial_data.get('market_cap'):,} if financial_data.get('market_cap') else 'N/A'
Debt to Equity: {financial_data.get('debt_to_equity')}
Beta: {financial_data.get('beta')}
Current Ratio: {financial_data.get('current_ratio')}
Profit Margin: {financial_data.get('profit_margin')}
"""
        
        findings_context = ""
        risk_findings = [f for f in findings if f.get('category') in ['risk_factor', 'industry_context']]
        for finding in risk_findings[:5]:
            findings_context += f"\n- {finding.get('title')}: {finding.get('content')[:200]}"
        
        if not findings_context:
            findings_context = "No specific risk-related findings available."
        
        messages = [
            SystemMessage(content=RISK_ASSESSOR_SYSTEM_PROMPT),
            HumanMessage(content=f"""Assess the risks for this company.

COMPANY: {company}

FINANCIAL DATA:
{financial_context}

ANALYSIS CONTEXT:
{analysis_context}

RISK-RELATED FINDINGS:
{findings_context}

Provide a comprehensive risk assessment.""")
        ]
        
        try:
            response = self.llm.invoke(messages)
            result = json.loads(response.content)
            
            return {
                "success": True,
                **result
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "overall_risk_level": "moderate",
                "risk_summary": "Risk assessment could not be completed due to an error."
            }


# Node function for LangGraph
def risk_assessor_node(state: dict) -> dict:
    """
    LangGraph node wrapper for the Risk Assessor agent.
    
    Reads: company, analysis, financial_data, raw_findings
    Writes: risk_assessment, current_agent, errors
    """
    assessor = RiskAssessorAgent()
    
    result = assessor.assess_risk(
        company=state.get("company", "Unknown"),
        analysis=state.get("analysis", {}),
        financial_data=state.get("financial_data"),
        findings=state.get("raw_findings", [])
    )
    
    if result.get("success", False):
        # Remove success key for state storage
        risk_assessment = {k: v for k, v in result.items() if k != "success"}
        return {
            "risk_assessment": risk_assessment,
            "current_agent": "risk_assessor_complete"
        }
    else:
        return {
            "risk_assessment": {
                "overall_risk_level": "moderate",
                "risk_summary": "Risk assessment encountered an error."
            },
            "current_agent": "risk_assessor_failed",
            "errors": [f"Risk Assessor error: {result.get('error')}"]
        }


# Test function
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    assessor = RiskAssessorAgent()
    
    test_analysis = {
        "strengths": [
            "Strong profitability with 25% profit margin",
            "Leading position in investment banking",
            "Diversified revenue streams"
        ],
        "weaknesses": [
            "High debt-to-equity ratio of 2.1",
            "Exposure to trading volatility",
            "Regulatory scrutiny"
        ],
        "threats": [
            "Basel III capital requirements",
            "Economic downturn impact",
            "Fintech disruption"
        ],
        "financial_health_score": "moderate",
        "outlook": "bullish"
    }
    
    test_financial_data = {
        "company_name": "Goldman Sachs Group Inc.",
        "ticker": "GS",
        "sector": "Financial Services",
        "industry": "Capital Markets",
        "market_cap": 197000000000,
        "debt_to_equity": 2.1,
        "beta": 1.35,
        "current_ratio": 0.8,
        "profit_margin": 0.25
    }
    
    test_findings = [
        {
            "category": "risk_factor",
            "title": "Regulatory Environment",
            "content": "Goldman faces ongoing regulatory scrutiny and Basel III endgame rules that may require additional capital buffers."
        },
        {
            "category": "industry_context",
            "title": "Market Conditions",
            "content": "Investment banking sector showing recovery with M&A activity increasing in 2024-2025."
        }
    ]
    
    print("=" * 50)
    print("Testing Risk Assessor...")
    print("=" * 50)
    
    result = assessor.assess_risk(
        company="Goldman Sachs",
        analysis=test_analysis,
        financial_data=test_financial_data,
        findings=test_findings
    )
    
    if result.get("success"):
        print(f"\nOverall Risk Level: {result.get('overall_risk_level', 'N/A').upper()}")
        
        print("\nRisk by Category:")
        for category in ['market_risk', 'credit_risk', 'regulatory_risk', 'operational_risk', 'competitive_risk']:
            cat_data = result.get(category, {})
            level = cat_data.get('level', 'N/A')
            print(f"  {category.replace('_', ' ').title()}: {level.upper()}")
        
        print(f"\nKey Risk Factors:")
        for factor in result.get('key_risk_factors', []):
            print(f"  • {factor}")
        
        print(f"\nRisk Mitigants:")
        for mitigant in result.get('risk_mitigants', []):
            print(f"  + {mitigant}")
        
        print(f"\nSummary: {result.get('risk_summary')}")
    else:
        print(f"Error: {result.get('error')}")