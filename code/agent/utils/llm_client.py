"""
LLM Client for Ollama (local LLM)
Handles chat completions with JSON schema validation
Uses requests instead of OpenAI client to avoid connection issues
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, ValidationError
import requests

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for Ollama local LLM API"""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize LLM client

        Args:
            api_key: API key (not required for Ollama, defaults to 'ollama')
            model: Model name to use (defaults to 'qwen2.5:72b')
            base_url: Ollama API base URL (defaults to 'http://localhost:11434/v1')
        """
        self.api_key = api_key or os.environ.get('OLLAMA_API_KEY', 'ollama')
        self.model = model or os.environ.get('OLLAMA_MODEL', 'qwen2.5:72b')
        base = base_url or os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434/v1')
        # Ensure base_url ends with /v1 for chat completions
        if not base.endswith('/v1'):
            base = base.rstrip('/') + '/v1'
        self.base_url = base
        self.session = requests.Session()
        # Disable proxy for local Ollama
        self.session.trust_env = False
        self.max_retries = 3
        self.retry_delay = 1.0

    def chat_completion(
        self,
        system_prompt: str,
        user_message: str,
        output_schema: Optional[BaseModel] = None,
        temperature: float = 0.3,
        max_tokens: int = 4000
    ) -> Dict[str, Any]:
        """
        Send chat completion request using requests
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        # Add JSON output instruction
        if output_schema:
            schema_instruction = "\n\nIMPORTANT: You must return ONLY valid JSON that conforms to the specified schema. Do not include any markdown formatting, code blocks, or explanatory text."
            messages[1]["content"] = messages[1]["content"] + schema_instruction

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=300.0
                )
                response.raise_for_status()
                data = response.json()

                content = data["choices"][0]["message"]["content"].strip()

                # Try to extract JSON if wrapped in markdown
                if "```json" in content:
                    start = content.find("```json") + 7
                    end = content.find("```", start)
                    content = content[start:end].strip()
                elif "```" in content:
                    start = content.find("```") + 3
                    end = content.find("```", start)
                    content = content[start:end].strip()

                # If no schema expected, return raw content
                if not output_schema:
                    return {"content": content}

                # Parse JSON
                parsed = None
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError as e:
                    fixed_content = self._try_fix_truncated_json(content)
                    if fixed_content != content:
                        try:
                            parsed = json.loads(fixed_content)
                        except json.JSONDecodeError:
                            parsed = None

                    if parsed is None:
                        if attempt < self.max_retries - 1:
                            time.sleep(self.retry_delay * (attempt + 1))
                            continue
                        raise ValueError(f"Failed to parse JSON response: {e}\nResponse: {content[:500]}")

                # Validate against schema
                try:
                    validated = output_schema(**parsed)
                    return validated.model_dump(mode='json')
                except ValidationError as e:
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (attempt + 1))
                        continue
                    raise ValueError(f"Schema validation failed: {e}\nParsed: {parsed}")

            except requests.exceptions.RequestException as e:
                logger.warning(f"LLM request failed (attempt {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                raise RuntimeError(f"LLM API request failed after {self.max_retries} attempts: {e}")
            except Exception as e:
                logger.warning(f"LLM request failed (attempt {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                raise

        raise RuntimeError("Failed to get valid response from LLM")

    def _try_fix_truncated_json(self, content: str) -> str:
        """Try to fix truncated/partial JSON by trimming to the last balanced point"""
        stripped = content.lstrip()
        if not stripped.startswith('{') and not stripped.startswith('['):
            return content

        stack = []
        in_string = False
        escape_next = False
        last_balanced = -1

        for i, ch in enumerate(content):
            if escape_next:
                escape_next = False
                continue
            if ch == '\\':
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch in '{[':
                stack.append(ch)
            elif ch in '}]':
                if stack:
                    stack.pop()
                else:
                    break
            if not stack:
                last_balanced = i

        fixed = content if last_balanced == -1 else content[:last_balanced + 1]
        fixed = fixed.rstrip()

        while fixed and fixed[-1] in {',', ' ', '\n', '\t', '\r'}:
            if fixed[-1] == ',':
                fixed = fixed[:-1]
                break
            fixed = fixed[:-1]

        closing = {'{': '}', '[': ']'}
        for opener in reversed(stack):
            fixed += closing.get(opener, '')

        return fixed

    def chat_completion_with_schema_description(
        self,
        system_prompt: str,
        user_message: str,
        schema_description: str,
        temperature: float = 0.3,
        max_tokens: int = 4000
    ) -> Dict[str, Any]:
        """Send chat completion with schema description"""
        schema_instruction = f"\n\nOutput Format:\n{schema_description}\n\nReturn ONLY valid JSON matching this schema."
        full_user_message = user_message + schema_instruction

        return self.chat_completion(
            system_prompt=system_prompt,
            user_message=full_user_message,
            output_schema=None,
            temperature=temperature,
            max_tokens=max_tokens
        )
