"""
Centralized LLM Configuration for the Knowledge Graph Pipeline.

All LLM access goes through OpenRouter (https://openrouter.ai), an
OpenAI-compatible gateway.  This module is the single source of truth for:

* resolving the model name / temperature / API key from config,
* configuring the global DSPy LM (used by the extractor and summarizer),
* building a LangChain chat model (used by the summarization path).
"""

import logging
import os
from typing import Any, Dict, Optional

import dspy
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"


def get_model_name(config: Dict[str, Any], purpose: Optional[str] = None) -> str:
    """Resolve the OpenRouter model id for a given purpose."""
    if not config or "llm" not in config:
        raise ValueError("Configuration missing 'llm' section")

    llm_cfg = config["llm"]
    if hasattr(llm_cfg, "model_dump"):
        llm_cfg = llm_cfg.model_dump()

    if purpose == "extraction":
        return llm_cfg.get("extraction_model") or llm_cfg.get("base_model")
    if purpose == "summarization":
        return llm_cfg.get("summarization_model") or llm_cfg.get("base_model")
    if purpose == "synthetic":
        return llm_cfg.get("base_model")
    return llm_cfg.get("base_model")


def get_temperature(config: Dict[str, Any]) -> float:
    """Get the configured LLM temperature (default 0.0)."""
    if not config or "llm" not in config:
        return 0.0
    llm_cfg = config["llm"]
    if hasattr(llm_cfg, "model_dump"):
        llm_cfg = llm_cfg.model_dump()
    return float(llm_cfg.get("temperature", 0.0))


def get_openrouter_api_key(config: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Resolve the OpenRouter API key from config or environment.

    Accepts both the correctly-spelled ``OPENROUTER_API_KEY`` and the
    commonly-mistyped ``OPENROUTE_API_KEY`` environment variables.
    """
    if config:
        infra = config.get("infra", {})
        if hasattr(infra, "model_dump"):
            infra = infra.model_dump()
        llm_cfg = config.get("llm", {})
        if hasattr(llm_cfg, "model_dump"):
            llm_cfg = llm_cfg.model_dump()
        val = infra.get("openrouter_api_key") or llm_cfg.get("openrouter_api_key")
        if val:
            return val.get_secret_value() if hasattr(val, "get_secret_value") else str(val)

    return os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTE_API_KEY")


def configure_dspy_lm(
    config: Dict[str, Any],
    purpose: Optional[str] = None,
    max_tokens: int = 2048,
) -> str:
    """Configure the global DSPy LM to use OpenRouter; return the model id.

    Shared by the entity extractor and the community summarizer.  litellm
    (which DSPy wraps) routes the ``openrouter/`` model prefix to the
    OpenRouter endpoint.
    """
    model = get_model_name(config, purpose=purpose)
    temperature = get_temperature(config)
    api_key = get_openrouter_api_key(config)

    if not api_key:
        raise ValueError(
            "OpenRouter API key not found (set OPENROUTER_API_KEY in .env)."
        )

    # litellm uses the 'openrouter/<model>' prefix to select the provider.
    litellm_model = model if model.startswith("openrouter/") else f"openrouter/{model}"

    try:
        lm = dspy.LM(
            model=litellm_model,
            api_key=api_key,
            api_base=OPENROUTER_API_BASE,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        dspy.configure(lm=lm)
        logger.info("Configured DSPy for OpenRouter with model %s", litellm_model)
    except Exception as e:
        logger.warning("Failed to configure DSPy LM: %s", e)

    return model


def get_langchain_llm(config: Dict[str, Any], purpose: Optional[str] = None) -> ChatOpenAI:
    """Build a LangChain chat model backed by OpenRouter.

    Used by the summarization path (``llm.ainvoke(...)``).
    """
    model = get_model_name(config, purpose=purpose)
    temperature = get_temperature(config)
    api_key = get_openrouter_api_key(config)

    if not api_key:
        raise ValueError("OpenRouter API key not found (set OPENROUTER_API_KEY in .env).")

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url=OPENROUTER_API_BASE,
    )
