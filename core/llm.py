from __future__ import annotations

import json
import re

PROVIDERS: dict[str, list[str]] = {
    "OpenAI": ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini"],
    "Anthropic": ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"],
    "Google": ["gemini-2.5-pro", "gemini-2.5-flash"],
}

PROVIDER_KEY_HELP: dict[str, str] = {
    "OpenAI": "Create an API key at platform.openai.com (starts with sk-)",
    "Anthropic": "Create an API key at console.anthropic.com (starts with sk-ant-)",
    "Google": "Create an API key at aistudio.google.com",
}


def _extract_json(text: str) -> dict:
    if not text:
        raise ValueError("The model returned no content")
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    start = cleaned.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(cleaned[start : i + 1])
    raise ValueError("No JSON object found in the model's response")


def _call_openai(model: str, api_key: str, system: str, user: str) -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    return _extract_json(resp.choices[0].message.content or "")


def _call_anthropic(model: str, api_key: str, system: str, user: str) -> dict:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(
        block.text for block in resp.content if getattr(block, "type", None) == "text"
    )
    return _extract_json(text)


def _call_google(model: str, api_key: str, system: str, user: str) -> dict:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=model,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
        ),
    )
    return _extract_json(resp.text or "")


_DISPATCH = {
    "OpenAI": _call_openai,
    "Anthropic": _call_anthropic,
    "Google": _call_google,
}


def call_structured(provider: str, model: str, api_key: str, system: str, user: str) -> dict:
    if provider not in _DISPATCH:
        raise ValueError(f"Unknown provider: {provider}")
    return _DISPATCH[provider](model, api_key, system, user)
