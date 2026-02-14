#!/usr/bin/env python3
"""
LLM API Client with Disk Caching
Supports Anthropic Claude and OpenAI GPT with async calls and caching.
"""

import os
import sys
import time
import json
import asyncio
import logging
from typing import List, Optional, Tuple
import diskcache as dc
import threading

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LLMClient:
    """
    Async LLM client with disk caching support.

    Supports:
    - Anthropic Claude (via anthropic library)
    - OpenAI GPT (via openai library)
    - Disk-based caching with diskcache
    - Retry logic with exponential backoff
    - Async batch processing
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        cache_path: str = ".cache",
        temperature: float = 0.8,
        wait_times: Tuple[int, ...] = (5, 10, 30, 60, 120),
    ):
        """
        Initialize LLM client.

        Args:
            provider: "anthropic" or "openai"
            model: Model name
            api_key: API key (if None, loads from environment)
            cache_path: Directory for cache storage
            temperature: Sampling temperature
            wait_times: Retry backoff times in seconds
        """
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature
        self.wait_times = wait_times

        # Track cache hits and API calls
        self.cache_hits = 0
        self.api_calls = 0

        # Load API key
        if api_key is None:
            # Try loading from config files first
            if self.provider == "anthropic":
                key_file = "config/anthropic_api_key.txt"
                if os.path.exists(key_file):
                    with open(key_file, 'r') as f:
                        api_key = f.read().strip()
                else:
                    api_key = os.environ.get("ANTHROPIC_API_KEY")
            elif self.provider == "openai":
                key_file = "config/openai_api_key.txt"
                if os.path.exists(key_file):
                    with open(key_file, 'r') as f:
                        api_key = f.read().strip()
                else:
                    api_key = os.environ.get("OPENAI_API_KEY")
            elif self.provider == "gemini":
                key_file = "config/gemini_api_key.txt"
                if os.path.exists(key_file):
                    with open(key_file, 'r') as f:
                        api_key = f.read().strip()
                else:
                    api_key = os.environ.get("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError(f"API key not found for provider: {self.provider}")

        self.api_key = api_key

        # Initialize cache
        cache_dir = os.path.join(cache_path, f"{self.provider}_cache")
        os.makedirs(cache_dir, exist_ok=True)

        cache_settings = dc.DEFAULT_SETTINGS.copy()
        cache_settings["eviction_policy"] = "none"
        cache_settings["size_limit"] = int(1e12)
        cache_settings["cull_limit"] = 0

        self.cache = dc.Cache(cache_dir, **cache_settings)
        self._lock = threading.Lock()

        # Initialize client
        self._client = None
        self._initialize_client()

        logger.info(f"LLM client initialized: {provider}/{model}")
        logger.info(f"Cache location: {cache_dir}")

    def _initialize_client(self):
        """Initialize the provider-specific client."""
        if self.provider == "anthropic":
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic library not installed. Run: pip install anthropic")

        elif self.provider == "openai":
            try:
                import openai
                self._client = openai.AsyncOpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai library not installed. Run: pip install openai")

        elif self.provider == "gemini":
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(self.model)
            except ImportError:
                raise ImportError("google.generativeai library not installed. Run: pip install google-generativeai")

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    async def ask(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_output: bool = False
    ) -> str:
        """
        Send a prompt to the LLM and get response.

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            json_output: Request JSON format response

        Returns:
            LLM response string
        """
        # Create cache key
        cache_key = (self.provider, self.model, system_prompt or "", prompt, json_output)

        # Check cache first (without lock for read)
        cached_response = self.cache.get(cache_key, default=None)
        if cached_response is not None:
            self.cache_hits += 1
            logger.warning(f"Cache hit: {cached_response[:50]}...")
            return cached_response

        # Cache miss - call API
        self.api_calls += 1
        logger.debug(f"Cache miss - calling API for: {prompt[:50]}...")
        response = await self._send_request(prompt, system_prompt, json_output)

        # Store in cache (with lock for write)
        with self._lock:
            self.cache[cache_key] = response

        return response

    async def _send_request(
        self,
        prompt: str,
        system_prompt: Optional[str],
        json_output: bool
    ) -> str:
        """Send request to LLM API with retry logic."""
        for attempt in range(len(self.wait_times)):
            try:
                if self.provider == "anthropic":
                    response = await self._call_anthropic(prompt, system_prompt, json_output)
                elif self.provider == "openai":
                    response = await self._call_openai(prompt, system_prompt, json_output)
                elif self.provider == "gemini":
                    response = await self._call_gemini(prompt, system_prompt, json_output)
                else:
                    raise ValueError(f"Unsupported provider: {self.provider}")

                return response

            except Exception as e:
                sleep_time = self.wait_times[attempt]
                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{len(self.wait_times)}): {e}. "
                    f"Retrying in {sleep_time}s..."
                )
                await asyncio.sleep(sleep_time)

        # Final attempt
        try:
            if self.provider == "anthropic":
                response = await self._call_anthropic(prompt, system_prompt, json_output)
            elif self.provider == "openai":
                response = await self._call_openai(prompt, system_prompt, json_output)
            elif self.provider == "gemini":
                response = await self._call_gemini(prompt, system_prompt, json_output)
            return response
        except Exception as e:
            logger.error(f"All retries failed: {e}")
            return ""

    async def _call_anthropic(
        self,
        prompt: str,
        system_prompt: Optional[str],
        json_output: bool
    ) -> str:
        """Call Anthropic API."""
        messages = [{"role": "user", "content": prompt}]

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": 8192,  # Anthropic requires max_tokens (API constraint), set to maximum
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        response = await self._client.messages.create(**kwargs)
        return response.content[0].text

    async def _call_openai(
        self,
        prompt: str,
        system_prompt: Optional[str],
        json_output: bool
    ) -> str:
        """Call OpenAI API."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }

        if json_output:
            kwargs["response_format"] = {"type": "json_object"}

        response = await self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    async def _call_gemini(
        self,
        prompt: str,
        system_prompt: Optional[str],
        json_output: bool
    ) -> str:
        """Call Google Gemini API."""
        # Combine system prompt and user prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        # print(f'full_prompt: {full_prompt}')
        # breakpoint()
        # Configure generation parameters
        generation_config = {
            "temperature": self.temperature,
        }

        if json_output:
            generation_config["response_mime_type"] = "application/json"

        response = await asyncio.to_thread(
            self._client.generate_content,
            full_prompt,
            generation_config=generation_config
        )

        return response.text

    async def ask_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        json_output: bool = False,
        max_concurrent: int = 10
    ) -> List[str]:
        """
        Send multiple prompts concurrently.

        Args:
            prompts: List of prompts
            system_prompt: System prompt for all requests
            json_output: Request JSON format
            max_concurrent: Maximum concurrent requests

        Returns:
            List of responses (same order as prompts)
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def ask_with_semaphore(prompt: str) -> str:
            async with semaphore:
                return await self.ask(prompt, system_prompt, json_output)

        tasks = [ask_with_semaphore(prompt) for prompt in prompts]
        responses = await asyncio.gather(*tasks)

        return responses

    def parse_json(self, response: str) -> Optional[dict]:
        """
        Parse JSON from LLM response.

        Args:
            response: LLM response string

        Returns:
            Parsed JSON dict or None if parsing fails
        """
        if not response or response == "":
            return None

        try:
            # Try direct parsing
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        try:
            content = response.strip()

            # Remove ```json ... ``` blocks
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                content = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                content = content[start:end].strip()

            return json.loads(content)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.debug(f"Response content: {response[:500]}")
            return None

    # def clear_cache(self):
    #     """Clear the entire cache."""
    #     self.cache.clear()
    #     logger.info("Cache cleared")

    def delete_cache_entry(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_output: bool = False
    ):
        """Delete a specific cache entry."""
        cache_key = (self.provider, self.model, system_prompt or "", prompt, json_output)
        if cache_key in self.cache:
            del self.cache[cache_key]
            logger.debug(f"Deleted cache entry for prompt: {prompt[:50]}...")

    def cache_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "volume": self.cache.volume(),
            "cache_hits": self.cache_hits,
            "api_calls": self.api_calls,
            "total_requests": self.cache_hits + self.api_calls
        }
    
    def clear_specific_cache(self, cache_key: str):
        """Clear a specific cache item."""
        self.cache.delete(cache_key)
        logger.info(f"Specific cache item cleared: {cache_key}")


    def __del__(self):
        """Cleanup: close cache properly."""
        if hasattr(self, 'cache'):
            self.cache.close()


# Standalone test
if __name__ == "__main__":
    async def test():
        print("Testing LLM Client with Caching")
        print("=" * 60)

        # Initialize client
        client = LLMClient(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            cache_path=".cache",
            temperature=0.8
        )

        # Test single request
        print("\n1. Testing single request (will call API)...")
        response1 = await client.ask("What is 2+2? Answer in one word.", json_output=False)
        print(f"Response: {response1}")

        # Test caching (same request)
        print("\n2. Testing cache (same request)...")
        response2 = await client.ask("What is 2+2? Answer in one word.", json_output=False)
        print(f"Response (from cache): {response2}")

        # Test JSON parsing
        print("\n3. Testing JSON output...")
        response3 = await client.ask(
            'Return JSON: {"result": 4}',
            json_output=True
        )
        parsed = client.parse_json(response3)
        print(f"Parsed JSON: {parsed}")

        # Test batch requests
        print("\n4. Testing batch requests...")
        prompts = [
            "What is the capital of France? One word.",
            "What is the capital of Japan? One word.",
            "What is the capital of USA? One word."
        ]
        responses = await client.ask_batch(prompts, max_concurrent=3)
        for p, r in zip(prompts, responses):
            print(f"  {p} -> {r}")

        # Cache stats
        print("\n5. Cache statistics:")
        stats = client.cache_stats()
        print(f"  Cached items: {stats['size']}")
        print(f"  Cache volume: {stats['volume']} bytes")

        print("\n" + "=" * 60)
        print("✅ Test complete!")

    # Run test
    asyncio.run(test())
