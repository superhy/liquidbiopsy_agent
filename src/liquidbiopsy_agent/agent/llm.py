from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


class LLMClient:
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None, model_path: Optional[str] = None) -> None:
        self.provider = provider or os.getenv("LIQUIDBIOPSY_LLM_PROVIDER", "local_llama_cpp")
        self.model = model or os.getenv("LIQUIDBIOPSY_LLM_MODEL", "")
        self.model_path = model_path or os.getenv("LIQUIDBIOPSY_LLM_MODEL_PATH", "")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.enabled = False
        self._llama = None

        if self.provider == "openai":
            self.enabled = self.api_key is not None and bool(self.model)
        elif self.provider == "local_llama_cpp":
            self.enabled = bool(self.model_path)

    def _load_llama(self):
        if self._llama is not None:
            return self._llama
        try:
            from llama_cpp import Llama
        except ImportError:
            return None
        n_ctx = int(os.getenv("LIQUIDBIOPSY_LLM_CTX", "4096"))
        n_threads = int(os.getenv("LIQUIDBIOPSY_LLM_THREADS", "4"))
        self._llama = Llama(model_path=self.model_path, n_ctx=n_ctx, n_threads=n_threads)
        return self._llama

    def complete(self, prompt: str) -> Optional[str]:
        if not self.enabled:
            return None
        if self.provider == "openai":
            try:
                import openai
            except ImportError:
                return None
            client = openai.OpenAI(api_key=self.api_key)
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=400,
                )
                return resp.choices[0].message.content
            except Exception:
                return None
        if self.provider == "local_llama_cpp":
            llama = self._load_llama()
            if llama is None:
                return None
            try:
                resp = llama.create_completion(prompt=prompt, max_tokens=400, temperature=0)
                return resp["choices"][0]["text"].strip()
            except Exception:
                return None
        return None


def safe_parse_json(text: Optional[str]) -> Optional[Dict[str, Any]]:
    if text is None:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None
