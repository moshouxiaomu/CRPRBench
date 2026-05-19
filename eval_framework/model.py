# model.py
"""
Unified model wrappers: every class implements .call(query: str) -> str
and can be plugged into main.py via --model / --model-kwargs.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import time
import random
import json
import logging

log = logging.getLogger(__name__)


# =========================
# Abstract Base Class
# =========================
class BaseModel(ABC):
    """Unified interface: implement .call(query) and return a string reply."""

    @abstractmethod
    def call(self, query: str) -> str:
        raise NotImplementedError


# =========================
# Utilities: retries & parsing
# =========================
def _with_retries(fn, *, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 10.0):
    """Exponential backoff retry decorator (minimal version)."""
    def wrapper(*args, **kwargs):
        last_exc = None
        for i in range(max_retries):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_exc = e
                delay = min(max_delay, base_delay * (2 ** i)) * (1 + random.random() * 0.2)
                log.warning("Model call failed (%s/%s): %s, retry in %.2fs", i + 1, max_retries, e, delay)
                time.sleep(delay)
        # Retries exhausted: re-raise the last exception
        raise last_exc
    return wrapper


def _coerce_text_from_openai_response(resp: Any) -> str:
    """
    Robustly extract text from various OpenAI SDK response objects:
    - Responses API: resp.output_text (convenience property)
    - Responses API: iterate resp.output[*].content[*].text
    - Chat Completions: resp.choices[0].message.content
    - Completions: resp.choices[0].text
    """
    # 1) Official SDK Responses API convenience property (preferred)
    if hasattr(resp, "output_text") and resp.output_text:
        return str(resp.output_text)

    # 2) Responses API low-level fields
    try:
        parts = []
        for item in getattr(resp, "output", []):
            for c in getattr(item, "content", []):
                # Be tolerant of different field names
                txt = getattr(c, "text", None) or getattr(c, "content", None)
                if isinstance(txt, str):
                    parts.append(txt)
        if parts:
            return "".join(parts)
    except Exception:
        pass

    # 3) Chat Completions
    try:
        msg = resp.choices[0].message
        return msg["content"] if isinstance(msg, dict) else msg.content
    except Exception:
        pass

    # 4) Completions
    try:
        return resp.choices[0].text
    except Exception:
        pass

    # 5) Fallback
    return str(resp)


# =========================
# OpenAI: Responses API (recommended)
# =========================
@dataclass
class OpenAIResponsesModel(BaseModel):
    """
    Generic wrapper for the OpenAI Responses API.
    Requires: pip install openai>=1.0.0
    See official platform documentation for Responses API / Quickstart.
    """
    model_name: str = "deepseek-chat"
    api_key: Optional[str] = "sk-a96a2e61901d4d70901dd97ac2dc3e23"          # Can also be configured via the OPENAI_API_KEY environment variable
    system_prompt: str = "You are an expert in chemistry."
    temperature: float = 0.2
    max_output_tokens: int = 2048
    base_url: Optional[str] = None         # For OpenAI-compatible gateways/proxies
    organization: Optional[str] = None
    project: Optional[str] = None
    timeout: float = 60.0
    max_retries: int = 3

    def __post_init__(self):
        try:
            from openai import OpenAI  # Official SDK
        except Exception as e:
            raise RuntimeError("Please `pip install openai>=1.0.0` to use OpenAIResponsesModel") from e

        # Initialize client (supports custom base_url/organization/project)
        kwargs = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.organization:
            kwargs["organization"] = self.organization
        if self.project:
            kwargs["project"] = self.project

        self._OpenAI = OpenAI
        self._client = OpenAI(**kwargs)

    @_with_retries
    def call(self, query: str) -> str:
        """
        Use the Responses API (recommended). The simplest form can pass a single
        string via `input`; to include a system message, submit a messages-like
        list as the value of "input".
        """
        # Minimal usage example (single string input):
        # resp = self._client.responses.create(
        #     model=self.model_name,
        #     input=query,
        #     temperature=self.temperature,
        #     max_output_tokens=self.max_output_tokens,
        #     timeout=self.timeout,
        # )

        # Use a "messages-like" structure to explicitly pass a system prompt
        payload = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user",   "content": [{"type": "text", "text": query}]},
        ]
        resp = self._client.responses.create(
            model=self.model_name,
            input=payload,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            timeout=self.timeout,
        )
        return _coerce_text_from_openai_response(resp)


# =========================
# OpenAI-Compatible: Chat Completions
# (common for local/private deployments supporting Chat Completions only)
# =========================
@dataclass
class OpenAIChatCompatModel(BaseModel):
    """
    For OpenAI-compatible services that implement Chat Completions only
    (e.g., vLLM/TGI/LM Studio). Connect via base_url + api_key.
    """
    model_name: str = "local-llm"
    base_url: str = "http://localhost:8000/v1"  # Root of the compatible endpoint
    api_key: str = "EMPTY"                       # Some services only require a non-empty value
    system_prompt: str = "You are an expert in chemistry."
    temperature: float = 0.2
    max_tokens: int = 2048
    timeout: float = 60.0
    max_retries: int = 3

    def __post_init__(self):
        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError("Please `pip install openai>=1.0.0` to use OpenAIChatCompatModel") from e
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    @_with_retries
    def call(self, query: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": query},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )
        return _coerce_text_from_openai_response(resp)


# =========================
# Generic HTTP JSON inference endpoint (custom services)
# =========================
@dataclass
class HTTPJsonModel(BaseModel):
    """
    Adapt to any HTTP JSON inference service:
      - POST {endpoint}
      - Request body: { "prompt": "...", **extra }
      - Response body: { "text": "..." } or { "output": "..." } or { "choices":[{"text": "..."}] }
    """
    endpoint: str = "http://localhost:9000/generate"
    headers: Optional[Dict[str, str]] = None
    extra: Optional[Dict[str, Any]] = None
    timeout: float = 60.0
    max_retries: int = 3

    def __post_init__(self):
        try:
            import requests  # Lazy import
        except Exception as e:
            raise RuntimeError("Please `pip install requests` to use HTTPJsonModel") from e

    @_with_retries
    def call(self, query: str) -> str:
        import requests
        payload = {"prompt": query}
        if self.extra:
            payload.update(self.extra)
        hdrs = {"Content-Type": "application/json"}
        if self.headers:
            hdrs.update(self.headers)
        r = requests.post(self.endpoint, data=json.dumps(payload), headers=hdrs, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        # Support multiple response schemas
        if isinstance(data, dict):
            if "text" in data and isinstance(data["text"], str):
                return data["text"]
            if "output" in data and isinstance(data["output"], str):
                return data["output"]
            if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
                ch0 = data["choices"][0]
                if isinstance(ch0, dict):
                    return ch0.get("text") or ch0.get("message", {}).get("content", "")
        return str(data)
    
@dataclass
class VLLMLocalModel(BaseModel):
    """
    Unified wrapper for running local weights via vLLM.
    Typical usage (main.py):
      --model model:VLLMLocalModel
      --model-kwargs '{"model_path":"/home/xshe/KG/chemical_model/model/LlaSMol/mistral",
                      "tensor_parallel_size":2,"gpu_memory_utilization":0.90,
                      "dtype":"bfloat16","max_tokens":512,"temperature":0.1}'
    """
    # ---- Basic configuration ----
    model_path: str = "/path/to/local/model"   # Your local model directory
    system_prompt: str = "You are an expert in chemistry."
    use_chat_template: bool = True             # Prefer tokenizer's chat template if available
    trust_remote_code: bool = True             # Required by some custom models

    # ---- Generation settings ----
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 512
    stop: Optional[List[str]] = None           # e.g., can include ["</SMILES>"]

    # ---- vLLM acceleration parameters ----
    tensor_parallel_size: int = 1              # Number of GPUs for tensor parallelism
    gpu_memory_utilization: float = 0.90       # Fraction of GPU memory to use
    dtype: str = "auto"                        # "auto" | "half" | "bfloat16" | "float16" etc.
    max_model_len: Optional[int] = None        # e.g., 8192 / 32768
    enforce_eager: bool = False                # True = easier debugging, slightly slower

    # Internal objects
    def __post_init__(self):
        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer
        except Exception as e:
            raise RuntimeError(
                "Please install dependencies first: `pip install vllm transformers` (and a CUDA-matched torch)."
            ) from e

        self._SamplingParams = SamplingParams
        self._AutoTokenizer = AutoTokenizer

        # 1) Load tokenizer (for chat templates)
        self._tokenizer = self._AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code
        )

        # 2) Initialize vLLM engine
        llm_kwargs = dict(
            model=self.model_path,
            tokenizer=self.model_path,                 # Explicit to avoid accidental downloads
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=self.trust_remote_code,
            enforce_eager=self.enforce_eager,
        )
        if self.max_model_len is not None:
            llm_kwargs["max_model_len"] = self.max_model_len

        self._llm = LLM(**llm_kwargs)

        # 3) Sampling params
        self._sparams = self._SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stop=self.stop or [],
        )

    def _build_prompt(self, query: str) -> str:
        """
        Prefer tokenizer.apply_chat_template; if unavailable, fall back to a generic text template.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": query},
        ]
        if self.use_chat_template:
            try:
                return self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True  # Append assistant prefix
                )
            except Exception:
                pass  # Fallback to a generic template

        # Simple fallback template (works for most instruction-tuned models)
        sys = self.system_prompt.strip()
        usr = query.strip()
        return f"System: {sys}\nUser: {usr}\nAssistant:"

    @_with_retries
    def call(self, query: str) -> str:
        prompt = self._build_prompt(query)
        # vLLM batch API; we send a single prompt here
        outs = self._llm.generate([prompt], self._sparams)
        # vLLM returns a list of RequestOutput
        try:
            return outs[0].outputs[0].text
        except Exception:
            # Fallback: stringify the whole object
            return str(outs)

@dataclass
class OpenAICompletionCompatModel(BaseModel):
    """
    For OpenAI-compatible services that implement plain Completions.
    Useful for local vLLM models whose tokenizer has no chat_template.
    """
    model_name: str = "local-llm"
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    system_prompt: str = "You are an expert in chemistry."
    temperature: float = 0.2
    max_tokens: int = 2048
    timeout: float = 60.0
    max_retries: int = 3

    # 可选：给 Mistral/LlaSMol 这种 instruction 模型一个稳定模板
    prompt_style: str = "mistral_inst"

    def __post_init__(self):
        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError("Please `pip install openai>=1.0.0` to use OpenAICompletionCompatModel") from e
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _build_prompt(self, query: str) -> str:
        system = (self.system_prompt or "").strip()
        user = query.strip()

        if self.prompt_style == "mistral_inst":
            # Mistral-Instruct 常见格式
            if system:
                return f"<s>[INST] {system}\n\n{user} [/INST]"
            return f"<s>[INST] {user} [/INST]"

        elif self.prompt_style == "plain":
            if system:
                return f"{system}\n\n{user}\n"
            return user

        elif self.prompt_style == "chatml":
            return (
                f"<|system|>\n{system}\n"
                f"<|user|>\n{user}\n"
                f"<|assistant|>\n"
            )

        else:
            # 通用 fallback
            if system:
                return f"System: {system}\nUser: {user}\nAssistant:"
            return f"User: {user}\nAssistant:"

    @_with_retries
    def call(self, query: str) -> str:
        prompt = self._build_prompt(query)
        resp = self._client.completions.create(
            model=self.model_name,
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )
        return _coerce_text_from_openai_response(resp)
# =========================
# Exported aliases
# =========================
__all__ = [
    "BaseModel",
    "OpenAIResponsesModel",
    "OpenAIChatCompatModel",
    "OpenAICompletionCompatModel",
    "HTTPJsonModel",
    "VLLMLocalModel",
]
