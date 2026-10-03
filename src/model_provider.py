from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class ProviderConfig:
    """Provider configuration shared by agents and benchmark suites.

    Supported providers:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float = 0.0
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Normalize provider name and handle common aliases."""
    val = (value or "").strip().lower()
    if val in {"openai", "oai"}:
        return "openai"
    if val in {"custom", "local", "vllm"}:
        return "custom"
    if val in {"gemini", "google", "google-genai", "google_genai"}:
        return "gemini"
    if val in {"anthropic", "anthorpic", "claude"}:
        return "anthropic"
    if val in {"ollama"}:
        return "ollama"
    if val in {"openrouter", "open-router"}:
        return "openrouter"
    return val


def build_chat_model(config: ProviderConfig) -> Any:
    """Instantiate chat model for the selected provider.

    Supported providers:
    - openai -> ChatOpenAI
    - custom -> ChatOpenAI with custom base_url
    - gemini -> ChatGoogleGenerativeAI
    - anthropic -> ChatAnthropic
    - ollama -> ChatOllama
    - openrouter -> ChatOpenAI with OpenRouter base_url
    """
    provider = normalize_provider(config.provider)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=api_key,
        )

    if provider == "custom":
        from langchain_openai import ChatOpenAI

        base_url = config.base_url or os.getenv("CUSTOM_BASE_URL", "http://localhost:8000/v1")
        api_key = config.api_key or os.getenv("CUSTOM_API_KEY", "custom-key")
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            base_url=base_url,
            api_key=api_key,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = config.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            google_api_key=api_key,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")
        return ChatAnthropic(
            model=config.model_name,
            temperature=config.temperature,
            api_key=api_key,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        base_url = config.base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(
            model=config.model_name,
            temperature=config.temperature,
            base_url=base_url,
        )

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI

        base_url = config.base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        api_key = config.api_key or os.getenv("OPENROUTER_API_KEY")
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            base_url=base_url,
            api_key=api_key,
        )

    raise ValueError(f"Unsupported provider: {config.provider}")
