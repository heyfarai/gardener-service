import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import httpx
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutoTitler:
    """Handles automatic titling and enrichment of topics using Claude Sonnet."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.model = "claude-3-sonnet-20240229"
        self.headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
    async def generate_topic_metadata(self, snippets: List[Dict[str, Any]]) -> Tuple[Optional[str], List[str], Optional[float], Optional[str]]:
        """
        Generate topic metadata using Claude Sonnet.
        
        Args:
            snippets: List of dictionaries containing 'text' and 'similarity' keys
            
        Returns:
            Tuple of (title, keywords, confidence, blurb)
        """
        if not snippets:
            return None, [], 0.0, None
            
        # Sort snippets by similarity and recency
        sorted_snippets = sorted(
            snippets,
            key=lambda x: (x.get('similarity', 0), x.get('timestamp', '')),
            reverse=True
        )
        
        # Take top 5-10 most relevant snippets
        context_snippets = sorted_snippets[:10]
        
        # Extract TF-IDF keyphrases (simplified for now)
        keyphrases = self._extract_keyphrases([s['text'] for s in context_snippets])
        
        # Prepare prompt for Claude
        prompt = self._build_prompt(context_snippets, keyphrases)
        
        try:
            # Call Claude API
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.api_url,
                    headers=self.headers,
                    json={
                        "model": self.model,
                        "max_tokens": 400,
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "temperature": 0.7
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                
                # Parse Claude's response
                content = result.get('content', [{}])[0].get('text', '').strip()
                return self._parse_response(content)
                
        except Exception as e:
            logger.error(f"Error generating topic metadata: {str(e)}")
            return None, [], 0.0, None
    
    def _extract_keyphrases(self, texts: List[str], top_n: int = 10) -> List[str]:
        """Extract keyphrases from text using TF-IDF (simplified version)."""
        # In a production environment, you might want to use a proper NLP library here
        # This is a simplified version that just returns the most frequent words
        from collections import Counter
        import re
        
        words = []
        for text in texts:
            # Simple word tokenization
            words.extend(re.findall(r'\b\w{3,}\b', text.lower()))
            
        # Remove stopwords and get most common words
        stopwords = set(['the', 'and', 'or', 'but', 'a', 'an', 'in', 'on', 'at', 'to', 'for'])
        word_counts = Counter([w for w in words if w not in stopwords])
        return [w[0] for w in word_counts.most_common(top_n)]
    
    def _build_prompt(self, snippets: List[Dict[str, Any]], keyphrases: List[str]) -> str:
        """Build the prompt for Claude."""
        snippets_text = "\n---\n".join([s['text'] for s in snippets])
        
        return f"""You are an expert at analyzing and summarizing topics from conversation snippets. 
        Your task is to analyze the following conversation snippets and create a concise, informative 
        title, keywords, and blurb that captures the main theme.
        
        Here are the keyphrases from the topic: {', '.join(keyphrases)}
        
        Here are the conversation snippets:
        {snippets_text}
        
        Please provide your response in the following JSON format:
        {{
            "title": "A clear, 3-6 word title in Title Case",
            "keywords": ["list", "of", "3-7", "relevant", "keywords"],
            "confidence": 0.0-1.0,  // Your confidence in the title and keywords
            "blurb": "A concise one-line description (max 160 chars) of the main theme"
        }}
        
        Only respond with valid JSON, no other text or explanation."""
    
    def _parse_response(self, response_text: str) -> Tuple[Optional[str], List[str], Optional[float], Optional[str]]:
        """Parse Claude's response into structured data."""
        try:
            data = json.loads(response_text)
            return (
                data.get('title'),
                data.get('keywords', []),
                float(data.get('confidence', 0.0)),
                data.get('blurb')
            )
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Error parsing Claude response: {str(e)}")
            return None, [], 0.0, None

# Singleton instance
auto_titler = AutoTitler()
