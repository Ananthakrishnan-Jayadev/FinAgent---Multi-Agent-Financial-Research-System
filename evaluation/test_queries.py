"""
FinAgent Evaluation Test Queries

Diverse set of queries to test different capabilities:
- Complex analysis queries
- Simple data lookups
- Comparison queries
- Risk-focused queries
- Different company types (banks, tech, etc.)
"""

TEST_QUERIES = [
    # === COMPLEX ANALYSIS (Full Pipeline) ===
    {
        "id": "complex_1",
        "query": "Analyze Goldman Sachs' financial health and outlook for 2025",
        "expected_complexity": "complex",
        "expected_path": "full_pipeline",
        "category": "single_company_analysis",
        "notes": "Major investment bank, should trigger full analysis"
    },
    {
        "id": "complex_2",
        "query": "What are the key risks facing Tesla right now?",
        "expected_complexity": "complex",
        "expected_path": "full_pipeline",
        "category": "risk_focused",
        "notes": "Risk-focused query on volatile stock"
    },
    {
        "id": "complex_3",
        "query": "Analyze JPMorgan Chase's competitive position in retail banking",
        "expected_complexity": "complex",
        "expected_path": "full_pipeline",
        "category": "competitive_analysis",
        "notes": "Requires industry context"
    },
    {
        "id": "complex_4",
        "query": "Should I invest in Microsoft based on their AI strategy?",
        "expected_complexity": "complex",
        "expected_path": "full_pipeline",
        "category": "investment_thesis",
        "notes": "Investment recommendation query"
    },
    {
        "id": "complex_5",
        "query": "Evaluate Bank of America's dividend sustainability",
        "expected_complexity": "complex",
        "expected_path": "full_pipeline",
        "category": "dividend_analysis",
        "notes": "Specific financial metric focus"
    },
    
    # === SIMPLE LOOKUPS (Quick Response Path) ===
    {
        "id": "simple_1",
        "query": "What is Apple's current stock price?",
        "expected_complexity": "simple",
        "expected_path": "simple_response",
        "category": "price_lookup",
        "notes": "Basic price query"
    },
    {
        "id": "simple_2",
        "query": "What's the P/E ratio of Amazon?",
        "expected_complexity": "simple",
        "expected_path": "simple_response",
        "category": "metric_lookup",
        "notes": "Single metric query"
    },
    {
        "id": "simple_3",
        "query": "What is NVIDIA's market cap?",
        "expected_complexity": "simple",
        "expected_path": "simple_response",
        "category": "metric_lookup",
        "notes": "Market cap lookup"
    },
    
    # === EDGE CASES ===
    {
        "id": "edge_1",
        "query": "Tell me about TD Bank",
        "expected_complexity": "complex",
        "expected_path": "full_pipeline",
        "category": "ambiguous",
        "notes": "Vague query, should default to complex"
    },
    {
        "id": "edge_2",
        "query": "Compare the profitability of Citigroup and Wells Fargo",
        "expected_complexity": "complex",
        "expected_path": "full_pipeline",
        "category": "comparison",
        "notes": "Multi-company comparison"
    },
]


def get_test_queries(category: str = None) -> list:
    """
    Get test queries, optionally filtered by category.
    
    Args:
        category: Filter by category (e.g., 'simple', 'complex', 'risk_focused')
        
    Returns:
        List of test query dicts
    """
    if category is None:
        return TEST_QUERIES
    
    if category in ["simple", "complex"]:
        return [q for q in TEST_QUERIES if q["expected_complexity"] == category]
    
    return [q for q in TEST_QUERIES if q["category"] == category]


def get_query_by_id(query_id: str) -> dict | None:
    """Get a specific test query by ID."""
    for q in TEST_QUERIES:
        if q["id"] == query_id:
            return q
    return None