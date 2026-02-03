"""
Search tools for the FinAgent Researcher.

Wraps Tavily API for web search with financial research focus.
"""

import os
from datetime import datetime
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()


class SearchTools:
    """Web search capabilities for financial research."""
    
    def __init__(self):
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY not found in environment")
        self.client = TavilyClient(api_key=api_key)
    
    def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "advanced",
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None
    ) -> dict:
        """
        Execute a web search using Tavily.
        
        Args:
            query: Search query string
            max_results: Number of results to return (default 5)
            search_depth: "basic" or "advanced" (advanced is slower but better)
            include_domains: Optional list of domains to restrict search to
            exclude_domains: Optional list of domains to exclude
            
        Returns:
            dict with 'results' list and 'query' echo
        """
        try:
            # Default to reputable financial news sources if none specified
            if include_domains is None:
                include_domains = []  # Let Tavily search broadly
            
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_domains=include_domains if include_domains else None,
                exclude_domains=exclude_domains
            )
            
            # Structure the results cleanly
            results = []
            for item in response.get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                    "score": item.get("score", 0)
                })
            
            return {
                "success": True,
                "query": query,
                "results": results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "query": query,
                "results": [],
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def search_financial_news(self, company: str, topic: str | None = None) -> dict:
        """
        Search for recent financial news about a company.
        
        Args:
            company: Company name or ticker
            topic: Optional specific topic (e.g., "earnings", "acquisitions")
            
        Returns:
            Search results focused on financial news
        """
        query = f"{company} financial news"
        if topic:
            query += f" {topic}"
        query += " 2024 2025"  # Bias toward recent news
        
        return self.search(
            query=query,
            max_results=5,
            search_depth="advanced"
        )
    
    def search_company_analysis(self, company: str) -> dict:
        """
        Search for analyst reports and company analysis.
        
        Args:
            company: Company name or ticker
            
        Returns:
            Search results focused on analysis and reports
        """
        query = f"{company} stock analysis analyst rating outlook"
        
        return self.search(
            query=query,
            max_results=5,
            search_depth="advanced"
        )
    
    def search_industry_trends(self, industry: str) -> dict:
        """
        Search for industry trends and sector analysis.
        
        Args:
            industry: Industry or sector name
            
        Returns:
            Search results about industry trends
        """
        query = f"{industry} industry trends outlook 2024 2025"
        
        return self.search(
            query=query,
            max_results=5,
            search_depth="advanced"
        )


# Quick test function
if __name__ == "__main__":
    tools = SearchTools()
    
    print("Testing financial news search...")
    result = tools.search_financial_news("JPMorgan Chase", "earnings")
    
    if result["success"]:
        print(f"Found {len(result['results'])} results:")
        for r in result["results"][:2]:
            print(f"  - {r['title'][:60]}...")
    else:
        print(f"Error: {result.get('error')}")