import os
from typing import Optional

import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("news-server")

def _news_api_key() -> str:
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing NEWS_API_KEY in environment. Please set it in your .env file."
        )
    return api_key

BASE_URL = "https://newsapi.org/v2"

@mcp.tool()
def top_headlines(country: Optional[str] = None, category: Optional[str] = None, q: Optional[str] = None, page_size: int = 10, page: int = 1) -> dict:
    """Get top headlines.

    - country: 2-letter ISO 3166-1 code (e.g., us, gb). Mutually exclusive with sources.
    - category: business, entertainment, general, health, science, sports, technology
    - q: keywords or phrases to search for
    - page_size: number of results per page (max 100)
    - page: page index (1-based)
    """
    params = {
        "apiKey": _news_api_key(),
        "pageSize": page_size,
        "page": page,
    }
    if country:
        params["country"] = country
    if category:
        params["category"] = category
    if q:
        params["q"] = q

    response = requests.get(f"{BASE_URL}/top-headlines", params=params, timeout=30)
    response.raise_for_status()
    return response.json()


@mcp.tool()
def search_news(q: str, from_date: Optional[str] = None, to_date: Optional[str] = None, language: Optional[str] = None, sort_by: Optional[str] = None, page_size: int = 10, page: int = 1) -> dict:
    """Search all articles matching the query.

    - q: keywords or phrases to search for (required)
    - from_date: oldest article date (ISO 8601: YYYY-MM-DD)
    - to_date: newest article date (ISO 8601: YYYY-MM-DD)
    - language: 2-letter language code (e.g., en, es)
    - sort_by: relevancy, popularity, publishedAt
    - page_size: results per page (max 100)
    - page: page index (1-based)
    """
    params = {
        "apiKey": _news_api_key(),
        "q": q,
        "pageSize": page_size,
        "page": page,
    }
    if from_date:
        params["from"] = from_date
    if to_date:
        params["to"] = to_date
    if language:
        params["language"] = language
    if sort_by:
        params["sortBy"] = sort_by

    response = requests.get(f"{BASE_URL}/everything", params=params, timeout=30)
    response.raise_for_status()
    return response.json()


@mcp.tool()
def list_sources(category: Optional[str] = None, language: Optional[str] = None, country: Optional[str] = None) -> dict:
    """List the news sources and blogs available.

    - category: business, entertainment, general, health, science, sports, technology
    - language: 2-letter code
    - country: 2-letter code
    """
    params = {
        "apiKey": _news_api_key(),
    }
    if category:
        params["category"] = category
    if language:
        params["language"] = language
    if country:
        params["country"] = country

    response = requests.get(f"{BASE_URL}/top-headlines/sources", params=params, timeout=30)
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    mcp.run(transport="stdio")


