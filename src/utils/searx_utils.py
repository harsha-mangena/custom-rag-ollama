# src/utils/searx_utils.py

import aiohttp
import logging
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urljoin
import json
import asyncio
from bs4 import BeautifulSoup
import re
from datetime import datetime
from nltk.tokenize import sent_tokenize
import nltk
from config.settings import Settings

class SearXNGManager:
    """Enhanced SearXNG manager with document-aware search capabilities and public instances."""
    
    def __init__(self):
        """Initialize SearXNG manager."""
        # Public instances configuration
        self.instances = [
            "https://searx.be",
            "https://searx.fmac.xyz",
            "https://searx.tiekoetter.com",
            "https://searx.lyxx.ca",
            "https://searx.nicfab.eu"
        ]
        self.current_instance = None
        self.timeout = 10
        self.logger = logging.getLogger(__name__)
        
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
        except Exception as e:
            self.logger.warning(f"Failed to download NLTK data: {e}")

    async def _test_instance(self, url: str) -> bool:
        """Test if a SearXNG instance is accessible."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{url}/search",
                    params={"q": "test", "format": "json"},
                    timeout=self.timeout
                ) as response:
                    return response.status == 200
        except Exception:
            return False

    async def get_working_instance(self) -> Optional[str]:
        """Get a working SearXNG instance."""
        if self.current_instance:
            if await self._test_instance(self.current_instance):
                return self.current_instance

        # Try all instances until we find one that works
        for url in self.instances:
            if await self._test_instance(url):
                self.current_instance = url
                self.logger.info(f"Using SearXNG instance: {url}")
                return url

        return None

    async def search_with_context(
        self,
        documents: List[Dict[str, Any]],
        query: str,
        max_results_per_doc: int = 5,
        max_total_results: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Perform contextualized search using document content.
        
        Args:
            documents: List of processed documents
            query: Base search query
            max_results_per_doc: Maximum results per document context
            max_total_results: Maximum total results to return
            
        Returns:
            List of search results with relevance to documents
        """
        try:
            # Extract key concepts and terms from documents
            doc_contexts = self._extract_document_contexts(documents)
            
            # Generate search queries for each context
            search_tasks = []
            for context in doc_contexts:
                enhanced_query = self._build_enhanced_query(query, context)
                search_tasks.append(
                    self.search(
                        enhanced_query,
                        num_results=max_results_per_doc
                    )
                )
            
            # Execute searches concurrently
            all_results = await asyncio.gather(*search_tasks)
            
            # Flatten and deduplicate results
            unique_results = self._deduplicate_results(all_results)
            
            # Sort by relevance and limit
            sorted_results = self._sort_results_by_relevance(
                unique_results,
                doc_contexts,
                query
            )
            
            return sorted_results[:max_total_results]
            
        except Exception as e:
            self.logger.error(f"Contextualized search failed: {e}")
            return []

    async def search(
        self,
        query: str,
        num_results: int = 10,
        categories: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Perform search using SearXNG with fallback support."""
        try:
            instance_url = await self.get_working_instance()
            if not instance_url:
                raise Exception("No available SearXNG instances")

            params = {
                "q": query,
                "format": "json",
                "pageno": 1,
                "results": num_results,
                "language": "en",
                "categories": ",".join(categories) if categories else "general"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{instance_url}/search",
                    params=params,
                    timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = await self._process_results(data.get("results", []))
                        if not results:
                            # Try another instance if no results
                            self.current_instance = None
                            return await self.search(query, num_results, categories)
                        return results
                    else:
                        self.logger.warning(
                            f"Search failed for {instance_url}: {response.status}"
                        )
                        # Try another instance
                        self.current_instance = None
                        return await self.search(query, num_results, categories)
                        
        except Exception as e:
            self.logger.error(f"Error performing SearXNG search: {e}")
            return []

    def _extract_document_contexts(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract key contexts from documents."""
        contexts = []
        
        for doc in documents:
            try:
                content = doc.get('content', '')
                if not content:
                    continue
                
                # Extract sentences
                sentences = sent_tokenize(content)
                
                # Extract key terms (simple approach - can be enhanced)
                terms = set()
                for sentence in sentences:
                    # Extract potential key terms (e.g., capitalized phrases)
                    capitalized = re.findall(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b', sentence)
                    terms.update(capitalized)
                    
                    # Extract potential technical terms
                    technical = re.findall(r'\b\w+(?:-\w+)*\b', sentence.lower())
                    terms.update(term for term in technical if len(term) > 5)
                
                contexts.append({
                    'source': doc.get('source', ''),
                    'terms': list(terms),
                    'summary': ' '.join(sentences[:3]),  # Simple summary
                    'file_type': doc.get('file_type', '')
                })
                
            except Exception as e:
                self.logger.warning(f"Failed to extract context from document: {e}")
                continue
        
        return contexts

    def _build_enhanced_query(self, base_query: str, context: Dict[str, Any]) -> str:
        """Build enhanced search query using document context."""
        # Add key terms to query
        key_terms = ' OR '.join(f'"{term}"' for term in context['terms'][:5])
        enhanced_query = f"{base_query} ({key_terms})"
        
        # Add file type if relevant
        if context['file_type'] in ['pdf', 'docx', 'txt']:
            enhanced_query += f" filetype:{context['file_type']}"
        
        return enhanced_query

    async def _process_results(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process and enrich search results."""
        processed = []
        for result in results:
            try:
                # Basic result info
                processed_result = {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "snippet": result.get("content", ""),
                    "source": result.get("engine", "unknown"),
                    "timestamp": datetime.now().isoformat()
                }
                
                # Fetch and extract content if available
                if processed_result["url"]:
                    content = await self.fetch_content(processed_result["url"])
                    if content:
                        processed_result["full_content"] = content
                        # Extract main content section
                        processed_result["main_content"] = self._extract_main_content(content)
                
                processed.append(processed_result)
                
            except Exception as e:
                self.logger.warning(f"Failed to process result: {e}")
                continue
        
        return processed

    def _extract_main_content(self, html_content: str) -> str:
        """Extract main content from HTML."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'ads']):
                element.decompose()
            
            # Try to find main content
            main_content = None
            
            # Look for common main content containers
            for selector in ['main', 'article', '[role="main"]', '#content', '.content']:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            if not main_content:
                # Fallback to largest text container
                paragraphs = soup.find_all('p')
                if paragraphs:
                    main_content = max(paragraphs, key=lambda p: len(p.get_text()))
            
            if main_content:
                return main_content.get_text(separator=' ', strip=True)
            return soup.get_text(separator=' ', strip=True)
            
        except Exception as e:
            self.logger.warning(f"Failed to extract main content: {e}")
            return ""

    def _deduplicate_results(
        self,
        all_results: List[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Deduplicate results based on URL."""
        seen_urls = set()
        unique_results = []
        
        for results_group in all_results:
            for result in results_group:
                url = result.get('url')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    unique_results.append(result)
        
        return unique_results

    def _sort_results_by_relevance(
        self,
        results: List[Dict[str, Any]],
        doc_contexts: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """Sort results by relevance to documents and query."""
        
        for result in results:
            # Calculate relevance score
            score = 0
            content = (
                result.get('full_content', '') or 
                result.get('main_content', '') or 
                result.get('snippet', '')
            ).lower()
            
            # Query relevance
            if query.lower() in content:
                score += 5
            
            # Document context relevance
            for context in doc_contexts:
                for term in context['terms']:
                    if term.lower() in content:
                        score += 2
                    if term.lower() in result.get('title', '').lower():
                        score += 3
            
            result['relevance_score'] = score
        
        # Sort by relevance score
        return sorted(results, key=lambda x: x.get('relevance_score', 0), reverse=True)

    async def fetch_content(self, url: str) -> Optional[str]:
        """Fetch and extract content from URL."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=self.timeout) as response:
                    if response.status == 200:
                        return await response.text()
                    return None
        except Exception as e:
            self.logger.warning(f"Failed to fetch content from {url}: {e}")
            return None