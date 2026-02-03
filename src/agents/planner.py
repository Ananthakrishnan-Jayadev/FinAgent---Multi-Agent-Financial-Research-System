"""
Planner Agent

Takes user query and creates a structured research plan.
Determines query complexity for routing decisions.
"""

import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


PLANNER_SYSTEM_PROMPT = """You are a financial research planner. Your job is to analyze user queries about companies, stocks, or financial topics and create a structured research plan.

For each query, you must:
1. Identify the primary company or companies being asked about
2. Determine if this is a SIMPLE query (quick data lookup) or COMPLEX query (requires deep research)
3. Create a list of specific research subtasks

SIMPLE queries are things like:
- "What is Apple's stock price?"
- "What's the P/E ratio of Microsoft?"
- "Show me Tesla's market cap"

COMPLEX queries require multiple data sources and analysis:
- "Analyze JPMorgan's financial health"
- "Compare the top 3 Canadian banks"
- "What are the risks facing Goldman Sachs?"
- "Should I invest in Bank of America?"

You MUST respond with valid JSON in exactly this format:
{
    "company": "Primary company name or ticker",
    "additional_companies": ["other", "companies", "if any"],
    "query_complexity": "simple" or "complex",
    "research_plan": [
        {
            "task_id": "task_1",
            "description": "What to research",
            "task_type": "web_search" or "financial_data" or "analysis",
            "priority": "high" or "medium" or "low"
        }
    ],
    "reasoning": "Brief explanation of your planning decisions"
}

For SIMPLE queries, create 1-2 tasks focused on data retrieval.
For COMPLEX queries, create 4-6 tasks covering:
- Financial data gathering (always high priority)
- Recent news and developments
- Analyst opinions and ratings
- Industry/competitive context
- Risk factors (if relevant)

Always respond with valid JSON only. No other text."""


class PlannerAgent:
    """Creates research plans from user queries."""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(
            model=model,
            temperature=0,  # Deterministic for structured output
            model_kwargs={"response_format": {"type": "json_object"}}
        )
    
    def plan(self, query: str) -> dict:
        """
        Analyze query and create research plan.
        
        Args:
            query: User's research query
            
        Returns:
            dict with company, complexity, and research_plan
        """
        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=f"Create a research plan for this query:\n\n{query}")
        ]
        
        try:
            response = self.llm.invoke(messages)
            result = json.loads(response.content)
            
            # Validate required fields
            required = ["company", "query_complexity", "research_plan"]
            for field in required:
                if field not in result:
                    raise ValueError(f"Missing required field: {field}")
            
            # Normalize complexity value
            if result["query_complexity"] not in ["simple", "complex"]:
                result["query_complexity"] = "complex"  # Default to complex if unclear
            
            return {
                "success": True,
                **result
            }
            
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": f"Failed to parse LLM response as JSON: {e}",
                "raw_response": response.content if 'response' in dir() else None
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


# Node function for LangGraph
def planner_node(state: dict) -> dict:
    """
    LangGraph node wrapper for the Planner agent.
    
    Reads: query
    Writes: company, query_complexity, research_plan, current_agent, errors
    """
    planner = PlannerAgent()
    result = planner.plan(state["query"])
    
    if result["success"]:
        return {
            "company": result["company"],
            "query_complexity": result["query_complexity"],
            "research_plan": result["research_plan"],
            "current_agent": "planner_complete"
        }
    else:
        return {
            "company": None,
            "query_complexity": "complex",  # Default to complex on failure
            "research_plan": [],
            "current_agent": "planner_failed",
            "errors": [f"Planner error: {result['error']}"]
        }


# Test function
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    planner = PlannerAgent()
    
    # Test simple query
    print("=" * 50)
    print("Testing SIMPLE query...")
    print("=" * 50)
    result = planner.plan("What is JPMorgan's current stock price?")
    print(json.dumps(result, indent=2))
    
    # Test complex query
    print("\n" + "=" * 50)
    print("Testing COMPLEX query...")
    print("=" * 50)
    result = planner.plan("Analyze Goldman Sachs' financial health and outlook for 2025")
    print(json.dumps(result, indent=2))