# src/utils/search_utils.py

import aiohttp
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import urllib.parse
import json

class WebSearchManager:
    """Web search manager using DuckDuckGo."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_url = "https://api.duckduckgo.com/"
        self.timeout = 10

    async def search(
        self,
        query: str,
        num_results: int = 10,
        categories: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform web search using DuckDuckGo.
        """
        try:
            params = {
                'q': query,
                'format': 'json',
                'no_html': 1,
                'no_redirect': 1,
                't': 'CustomRAG'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.base_url,
                    params=params,
                    timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = self._process_results(data, num_results)
                        return results
                    else:
                        self.logger.error(f"Search failed: {response.status}")
                        return []
                        
        except Exception as e:
            self.logger.error(f"Error performing web search: {e}")
            # Fallback to alternative search
            return await self._fallback_search(query, num_results)

    async def _fallback_search(
        self,
        query: str,
        num_results: int
    ) -> List[Dict[str, Any]]:
        """
        Fallback search using direct web scraping of DuckDuckGo HTML.
        """
        try:
            encoded_query = urllib.parse.quote(query)
            url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=self.timeout) as response:
                    if response.status == 200:
                        html = await response.text()
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        results = []
                        for result in soup.select('.result')[:num_results]:
                            title_elem = result.select_one('.result__title')
                            snippet_elem = result.select_one('.result__snippet')
                            
                            if title_elem and snippet_elem:
                                results.append({
                                    'title': title_elem.get_text(strip=True),
                                    'url': title_elem.a['href'] if title_elem.a else '',
                                    'snippet': snippet_elem.get_text(strip=True),
                                    'source': 'duckduckgo',
                                    'timestamp': datetime.now().isoformat()
                                })
                        
                        return results
                    return []
                    
        except Exception as e:
            self.logger.error(f"Fallback search failed: {e}")
            return []

    def _process_results(
        self,
        data: Dict[str, Any],
        num_results: int
    ) -> List[Dict[str, Any]]:
        """Process DuckDuckGo API results."""
        results = []
        
        # Process AbstractText if available
        if data.get('AbstractText'):
            results.append({
                'title': data.get('Heading', ''),
                'url': data.get('AbstractURL', ''),
                'snippet': data.get('AbstractText', ''),
                'source': 'duckduckgo_abstract',
                'timestamp': datetime.now().isoformat()
            })
        
        # Process RelatedTopics
        for topic in data.get('RelatedTopics', [])[:num_results]:
            if isinstance(topic, dict) and 'Text' in topic:
                results.append({
                    'title': topic.get('FirstURL', '').split('/')[-1],
                    'url': topic.get('FirstURL', ''),
                    'snippet': topic.get('Text', ''),
                    'source': 'duckduckgo_related',
                    'timestamp': datetime.now().isoformat()
                })
        
        return results[:num_results]

    async def search_with_context(
        self,
        query: str,
        documents: List[Dict[str, Any]] = None,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Perform web search with optional document context.
        """
        enhanced_query = query
        if documents:
            # Extract key terms from documents
            terms = self._extract_key_terms(documents)
            if terms:
                enhanced_query = f"{query} {' '.join(terms[:3])}"
        
        return await self.search(enhanced_query, max_results)

    def _extract_key_terms(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[str]:
        """Extract key terms from documents."""
        terms = set()
        for doc in documents:
            content = doc.get('content', '')
            if content:
                # Extract capitalized phrases
                import re
                capitalized = re.findall(
                    r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b',
                    content
                )
                terms.update(capitalized)
        return list(terms)