"""
Researcher Agent

Executes the research plan created by the Planner.
Uses search and financial tools to gather data.
"""

import json
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.tools.search_tools import SearchTools
from src.tools.financial_tools import FinancialTools


RESEARCHER_SYSTEM_PROMPT = """You are a financial researcher. You have been given research findings from various sources.
Your job is to synthesize these findings into a clean, structured summary.

Review all the data provided and extract the most relevant information.
Focus on facts, numbers, and concrete information. Avoid speculation.

You MUST respond with valid JSON in exactly this format:
{
    "findings": [
        {
            "category": "financial_metrics" | "recent_news" | "analyst_opinion" | "industry_context" | "risk_factor",
            "title": "Brief title for this finding",
            "content": "Detailed content of the finding",
            "source": "Where this came from",
            "relevance": "high" | "medium" | "low"
        }
    ],
    "data_quality": "high" | "medium" | "low",
    "gaps": ["List of information that could not be found or is missing"]
}

Be thorough but concise. Only include findings that are relevant to the research query.
Always respond with valid JSON only."""


class ResearcherAgent:
    """Executes research plans and gathers data."""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(
            model=model,
            temperature=0,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        self.search_tools = SearchTools()
        self.financial_tools = FinancialTools()
    
    def execute_task(self, task: dict, company: str) -> dict:
        """
        Execute a single research subtask.
        
        Args:
            task: Task dict with task_type, description
            company: Target company name/ticker
            
        Returns:
            dict with task results
        """
        task_type = task.get("task_type", "web_search")
        description = task.get("description", "")
        
        result = {
            "task_id": task.get("task_id"),
            "task_type": task_type,
            "description": description,
            "data": None,
            "success": False,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            if task_type == "financial_data":
                # Get comprehensive financial metrics
                metrics = self.financial_tools.get_company_metrics(company)
                price_history = self.financial_tools.get_price_history(company, period="6mo")
                earnings = self.financial_tools.get_recent_earnings(company)
                
                result["data"] = {
                    "metrics": metrics,
                    "price_history": price_history,
                    "earnings": earnings
                }
                result["success"] = metrics.get("success", False)
                
            elif task_type == "web_search":
                # Determine search type based on description
                description_lower = description.lower()
                
                if "news" in description_lower or "recent" in description_lower:
                    search_result = self.search_tools.search_financial_news(company)
                elif "analyst" in description_lower or "rating" in description_lower:
                    search_result = self.search_tools.search_company_analysis(company)
                elif "industry" in description_lower or "sector" in description_lower:
                    # Get company sector first, then search industry
                    metrics = self.financial_tools.get_company_metrics(company)
                    industry = metrics.get("industry", metrics.get("sector", company))
                    search_result = self.search_tools.search_industry_trends(industry)
                else:
                    # Generic search with the task description
                    search_result = self.search_tools.search(f"{company} {description}")
                
                result["data"] = search_result
                result["success"] = search_result.get("success", False)
                
            elif task_type == "analysis":
                # Analysis tasks don't fetch new data - they're handled by the Analyst
                # But we can do a targeted search based on the description
                search_result = self.search_tools.search(f"{company} {description}")
                result["data"] = search_result
                result["success"] = search_result.get("success", False)
            
            else:
                result["error"] = f"Unknown task type: {task_type}"
                
        except Exception as e:
            result["error"] = str(e)
            result["success"] = False
        
        return result
    
    def execute_plan(self, research_plan: list[dict], company: str) -> list[dict]:
        """
        Execute all tasks in the research plan.
        
        Args:
            research_plan: List of task dicts from Planner
            company: Target company
            
        Returns:
            List of task results
        """
        results = []
        
        # Sort by priority (high first)
        priority_order = {"high": 0, "medium": 1, "low": 2}
        sorted_plan = sorted(
            research_plan,
            key=lambda x: priority_order.get(x.get("priority", "low"), 2)
        )
        
        for task in sorted_plan:
            result = self.execute_task(task, company)
            results.append(result)
        
        return results
    
    def synthesize_findings(self, task_results: list[dict], query: str, company: str) -> dict:
        """
        Use LLM to synthesize raw task results into structured findings.
        
        Args:
            task_results: Raw results from execute_plan
            query: Original user query
            company: Target company
            
        Returns:
            Synthesized findings dict
        """
        # Prepare data summary for LLM
        data_summary = []
        
        for result in task_results:
            if not result.get("success"):
                continue
                
            task_type = result.get("task_type")
            data = result.get("data", {})
            
            if task_type == "financial_data":
                # Extract key metrics
                metrics = data.get("metrics", {})
                if metrics.get("success"):
                    data_summary.append({
                        "source": "Financial Data (yfinance)",
                        "type": "financial_metrics",
                        "content": {
                            "company": metrics.get("company_name"),
                            "ticker": metrics.get("ticker"),
                            "price": metrics.get("current_price"),
                            "market_cap": metrics.get("market_cap"),
                            "pe_ratio": metrics.get("pe_ratio"),
                            "forward_pe": metrics.get("forward_pe"),
                            "profit_margin": metrics.get("profit_margin"),
                            "revenue_growth": metrics.get("revenue_growth"),
                            "debt_to_equity": metrics.get("debt_to_equity"),
                            "roe": metrics.get("roe"),
                            "dividend_yield": metrics.get("dividend_yield"),
                            "sector": metrics.get("sector"),
                            "industry": metrics.get("industry"),
                            "description": metrics.get("description"),
                            "52_week_high": metrics.get("fifty_two_week_high"),
                            "52_week_low": metrics.get("fifty_two_week_low")
                        }
                    })
                
                price_history = data.get("price_history", {})
                if price_history.get("success"):
                    data_summary.append({
                        "source": "Price History (yfinance)",
                        "type": "price_performance",
                        "content": {
                            "period": price_history.get("period"),
                            "change_pct": price_history.get("period_change_pct"),
                            "period_high": price_history.get("period_high"),
                            "period_low": price_history.get("period_low")
                        }
                    })
                    
            elif task_type in ["web_search", "analysis"]:
                search_results = data.get("results", [])
                for item in search_results[:3]:  # Top 3 results per search
                    data_summary.append({
                        "source": item.get("url", "Web Search"),
                        "type": "web_search",
                        "title": item.get("title"),
                        "content": item.get("content")
                    })
        
        # Send to LLM for synthesis
        messages = [
            SystemMessage(content=RESEARCHER_SYSTEM_PROMPT),
            HumanMessage(content=f"""Research Query: {query}
Company: {company}

Raw Research Data:
{json.dumps(data_summary, indent=2, default=str)}

Synthesize these findings into a structured summary.""")
        ]
        
        try:
            response = self.llm.invoke(messages)
            return json.loads(response.content)
        except Exception as e:
            return {
                "findings": [],
                "data_quality": "low",
                "gaps": [f"Failed to synthesize findings: {e}"],
                "error": str(e)
            }


# Node function for LangGraph
def researcher_node(state: dict) -> dict:
    """
    LangGraph node wrapper for the Researcher agent.
    
    Reads: query, company, research_plan
    Writes: raw_findings, financial_data, current_agent, errors
    """
    researcher = ResearcherAgent()
    
    company = state.get("company")
    research_plan = state.get("research_plan", [])
    query = state.get("query", "")
    
    if not company:
        return {
            "current_agent": "researcher_failed",
            "errors": ["No company specified for research"]
        }
    
    if not research_plan:
        return {
            "current_agent": "researcher_failed", 
            "errors": ["No research plan provided"]
        }
    
    # Execute all research tasks
    task_results = researcher.execute_plan(research_plan, company)
    
    # Extract financial data separately (for direct state access)
    financial_data = None
    for result in task_results:
        if result.get("task_type") == "financial_data" and result.get("success"):
            financial_data = result.get("data", {}).get("metrics")
            break
    
    # Synthesize findings
    synthesis = researcher.synthesize_findings(task_results, query, company)
    
    # Convert findings to list of dicts for state
    findings = synthesis.get("findings", [])
    
    return {
        "raw_findings": findings,
        "financial_data": financial_data,
        "current_agent": "researcher_complete"
    }


# Test function
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    researcher = ResearcherAgent()
    
    # Simulate a research plan from Planner
    test_plan = [
        {
            "task_id": "task_1",
            "description": "Gather financial statements and key metrics",
            "task_type": "financial_data",
            "priority": "high"
        },
        {
            "task_id": "task_2", 
            "description": "Research recent news and developments",
            "task_type": "web_search",
            "priority": "high"
        },
        {
            "task_id": "task_3",
            "description": "Collect analyst opinions and ratings",
            "task_type": "web_search", 
            "priority": "medium"
        }
    ]
    
    company = "Goldman Sachs"
    query = "Analyze Goldman Sachs' financial health and outlook"
    
    print("=" * 50)
    print(f"Executing research plan for {company}...")
    print("=" * 50)
    
    # Execute plan
    results = researcher.execute_plan(test_plan, company)
    print(f"\nExecuted {len(results)} tasks")
    
    for r in results:
        status = "✓" if r["success"] else "✗"
        print(f"  {status} {r['task_id']}: {r['task_type']}")
    
    # Synthesize
    print("\n" + "=" * 50)
    print("Synthesizing findings...")
    print("=" * 50)
    
    synthesis = researcher.synthesize_findings(results, query, company)
    print(f"\nFound {len(synthesis.get('findings', []))} findings")
    print(f"Data quality: {synthesis.get('data_quality')}")
    
    if synthesis.get("gaps"):
        print(f"Gaps: {synthesis.get('gaps')}")
    
    print("\nTop findings:")
    for finding in synthesis.get("findings", [])[:3]:
        print(f"  [{finding.get('category')}] {finding.get('title')}")