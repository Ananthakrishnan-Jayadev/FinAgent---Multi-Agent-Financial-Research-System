"""
Financial data tools for the FinAgent Researcher.

Wraps yfinance for stock data, financial metrics, and company info.
"""

from datetime import datetime
import yfinance as yf


class FinancialTools:
    """Financial data retrieval using yfinance."""
    
    # Common ticker mappings for companies often referred to by name
    TICKER_MAP = {
        "jpmorgan": "JPM",
        "jpmorgan chase": "JPM",
        "jp morgan": "JPM",
        "goldman sachs": "GS",
        "goldman": "GS",
        "bank of america": "BAC",
        "wells fargo": "WFC",
        "citigroup": "C",
        "citi": "C",
        "morgan stanley": "MS",
        "td bank": "TD",
        "royal bank": "RY",
        "rbc": "RY",
        "scotiabank": "BNS",
        "bmo": "BMO",
        "cibc": "CM",
        "hsbc": "HSBC",
        "barclays": "BCS",
        "deutsche bank": "DB",
        "ubs": "UBS",
        "credit suisse": "CS",
        "apple": "AAPL",
        "microsoft": "MSFT",
        "google": "GOOGL",
        "amazon": "AMZN",
        "tesla": "TSLA",
        "nvidia": "NVDA",
        "meta": "META",
        "facebook": "META",
    }
    
    def _resolve_ticker(self, company_or_ticker: str) -> str:
        """Convert company name to ticker if needed."""
        lookup = company_or_ticker.lower().strip()
        return self.TICKER_MAP.get(lookup, company_or_ticker.upper())
    
    def get_company_metrics(self, company_or_ticker: str) -> dict:
        """
        Get comprehensive financial metrics for a company.
        
        Args:
            company_or_ticker: Company name or stock ticker
            
        Returns:
            dict with financial metrics or error info
        """
        ticker_symbol = self._resolve_ticker(company_or_ticker)
        
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info
            
            # Check if we got valid data
            if not info or info.get("regularMarketPrice") is None:
                # Try to get at least some data
                if info.get("shortName") is None:
                    return {
                        "success": False,
                        "ticker": ticker_symbol,
                        "error": f"Could not find data for ticker: {ticker_symbol}",
                        "timestamp": datetime.now().isoformat()
                    }
            
            metrics = {
                "success": True,
                "ticker": ticker_symbol,
                "company_name": info.get("shortName") or info.get("longName"),
                "current_price": info.get("regularMarketPrice") or info.get("currentPrice"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "peg_ratio": info.get("pegRatio"),
                "price_to_book": info.get("priceToBook"),
                "debt_to_equity": info.get("debtToEquity"),
                "current_ratio": info.get("currentRatio"),
                "quick_ratio": info.get("quickRatio"),
                "revenue": info.get("totalRevenue"),
                "revenue_growth": info.get("revenueGrowth"),
                "profit_margin": info.get("profitMargins"),
                "operating_margin": info.get("operatingMargins"),
                "roe": info.get("returnOnEquity"),
                "roa": info.get("returnOnAssets"),
                "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
                "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
                "fifty_day_avg": info.get("fiftyDayAverage"),
                "two_hundred_day_avg": info.get("twoHundredDayAverage"),
                "dividend_yield": info.get("dividendYield"),
                "dividend_rate": info.get("dividendRate"),
                "beta": info.get("beta"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "description": info.get("longBusinessSummary"),
                "website": info.get("website"),
                "employees": info.get("fullTimeEmployees"),
                "country": info.get("country"),
                "timestamp": datetime.now().isoformat()
            }
            
            return metrics
            
        except Exception as e:
            return {
                "success": False,
                "ticker": ticker_symbol,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_price_history(
        self,
        company_or_ticker: str,
        period: str = "1y"
    ) -> dict:
        """
        Get historical price data.
        
        Args:
            company_or_ticker: Company name or ticker
            period: Time period - 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max
            
        Returns:
            dict with price history data
        """
        ticker_symbol = self._resolve_ticker(company_or_ticker)
        
        try:
            ticker = yf.Ticker(ticker_symbol)
            hist = ticker.history(period=period)
            
            if hist.empty:
                return {
                    "success": False,
                    "ticker": ticker_symbol,
                    "error": "No historical data available",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Calculate some useful stats
            start_price = hist["Close"].iloc[0]
            end_price = hist["Close"].iloc[-1]
            high = hist["High"].max()
            low = hist["Low"].min()
            avg_volume = hist["Volume"].mean()
            
            return {
                "success": True,
                "ticker": ticker_symbol,
                "period": period,
                "start_date": hist.index[0].strftime("%Y-%m-%d"),
                "end_date": hist.index[-1].strftime("%Y-%m-%d"),
                "start_price": round(start_price, 2),
                "end_price": round(end_price, 2),
                "period_change_pct": round(((end_price - start_price) / start_price) * 100, 2),
                "period_high": round(high, 2),
                "period_low": round(low, 2),
                "avg_daily_volume": int(avg_volume),
                "data_points": len(hist),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "ticker": ticker_symbol,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_recent_earnings(self, company_or_ticker: str) -> dict:
        """
        Get recent earnings data.
        
        Args:
            company_or_ticker: Company name or ticker
            
        Returns:
            dict with earnings info
        """
        ticker_symbol = self._resolve_ticker(company_or_ticker)
        
        try:
            ticker = yf.Ticker(ticker_symbol)
            
            # Get earnings dates and history
            earnings_dates = ticker.earnings_dates
            quarterly_earnings = ticker.quarterly_earnings
            
            result = {
                "success": True,
                "ticker": ticker_symbol,
                "timestamp": datetime.now().isoformat()
            }
            
            # Recent earnings dates
            if earnings_dates is not None and not earnings_dates.empty:
                recent = earnings_dates.head(4).reset_index()
                result["upcoming_earnings"] = []
                for _, row in recent.iterrows():
                    result["upcoming_earnings"].append({
                        "date": row["Earnings Date"].strftime("%Y-%m-%d") if hasattr(row["Earnings Date"], "strftime") else str(row["Earnings Date"]),
                        "eps_estimate": row.get("EPS Estimate"),
                        "eps_actual": row.get("Reported EPS")
                    })
            
            # Quarterly earnings history
            if quarterly_earnings is not None and not quarterly_earnings.empty:
                result["quarterly_earnings"] = []
                for date, row in quarterly_earnings.tail(4).iterrows():
                    result["quarterly_earnings"].append({
                        "quarter": str(date),
                        "revenue": row.get("Revenue"),
                        "earnings": row.get("Earnings")
                    })
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "ticker": ticker_symbol,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def compare_companies(self, tickers: list[str]) -> dict:
        """
        Compare key metrics across multiple companies.
        
        Args:
            tickers: List of company names or tickers
            
        Returns:
            Comparison dict with metrics for each company
        """
        results = {
            "success": True,
            "companies": [],
            "timestamp": datetime.now().isoformat()
        }
        
        for ticker in tickers:
            metrics = self.get_company_metrics(ticker)
            if metrics["success"]:
                results["companies"].append({
                    "ticker": metrics["ticker"],
                    "name": metrics["company_name"],
                    "market_cap": metrics["market_cap"],
                    "pe_ratio": metrics["pe_ratio"],
                    "profit_margin": metrics["profit_margin"],
                    "roe": metrics["roe"],
                    "debt_to_equity": metrics["debt_to_equity"],
                    "revenue_growth": metrics["revenue_growth"]
                })
            else:
                results["companies"].append({
                    "ticker": ticker,
                    "error": metrics.get("error", "Unknown error")
                })
        
        return results


# Quick test function
if __name__ == "__main__":
    tools = FinancialTools()
    
    print("Testing company metrics...")
    result = tools.get_company_metrics("JPMorgan Chase")
    
    if result["success"]:
        print(f"Company: {result['company_name']} ({result['ticker']})")
        print(f"Price: ${result['current_price']}")
        print(f"Market Cap: ${result['market_cap']:,}" if result['market_cap'] else "N/A")
        print(f"P/E Ratio: {result['pe_ratio']}")
        print(f"Sector: {result['sector']}")
    else:
        print(f"Error: {result.get('error')}")
    
    print("\nTesting price history...")
    history = tools.get_price_history("JPM", period="6mo")
    if history["success"]:
        print(f"6-month change: {history['period_change_pct']}%")