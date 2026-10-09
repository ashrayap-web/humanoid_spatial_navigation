"""Thin, provider-agnostic wrapper for vision-language model calls (C7; C8 full later).

One entry point, :func:`ask_json`: images + prompt + JSON schema in, parsed dict out. Responses are
cached on disk by a hash of everything that determines them, so reruns are free. Only the
``anthropic`` provider is implemented; credentials come from the environment
(``ANTHROPIC_API_KEY``, ``ANTHROPIC_AUTH_TOKEN`` or an ``ant auth login`` profile).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np

from changedet.core.logging import get_logger

log = get_logger("llm")


class LLMUnavailable(RuntimeError):
    """No usable provider/credentials — callers should degrade gracefully."""


class LLMRefused(RuntimeError):
    """The model declined the request (``stop_reason == "refusal"``)."""


def credentials_available(provider: str) -> bool:
    """Best-effort check that a request could authenticate, without spending a call."""
    if provider != "anthropic":
        return False
    env = (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_PROFILE",
        "ANTHROPIC_FEDERATION_RULE_ID",
    )
    return any(os.environ.get(k) for k in env) or (Path.home() / ".config/anthropic").exists()


def parse_json(text: str) -> dict:
    """Parse a JSON object from model text, tolerating ```json fences and surrounding prose."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"No JSON object in model output: {text[:200]!r}") from None
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in model output: {text[:200]!r}") from e
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object, got {type(value).__name__}")
    return value


def encode_jpeg(image_rgb: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", image_rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()


def _cache_key(cfg, prompt: str, schema: dict, images: list[bytes]) -> str:
    h = hashlib.sha256()
    for part in (
        cfg.provider,
        cfg.model,
        str(cfg.effort),
        str(cfg.prompt_version),
        prompt,
        json.dumps(schema, sort_keys=True),
    ):
        h.update(part.encode())
        h.update(b"\0")
    for img in images:
        h.update(hashlib.sha256(img).digest())
    return h.hexdigest()[:32]


def _call_anthropic(cfg, prompt: str, schema: dict, images: list[bytes]) -> str:
    import anthropic

    content = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.standard_b64encode(img).decode(),
            },
        }
        for img in images
    ] + [{"type": "text", "text": prompt}]
    client = anthropic.Anthropic()
    try:
        response = client.beta.messages.create(
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",  # re-run a safety decline on Anthropic's recommended model
            output_config={
                "effort": cfg.effort,
                "format": {"type": "json_schema", "schema": schema},
            },
            messages=[{"role": "user", "content": content}],
        )
    except anthropic.AuthenticationError as e:
        raise LLMUnavailable(f"Anthropic authentication failed: {e}") from e
    except anthropic.PermissionDeniedError as e:
        raise LLMUnavailable(f"Anthropic permission denied: {e}") from e
    except anthropic.APIConnectionError as e:
        raise LLMUnavailable(f"Cannot reach the Anthropic API: {e}") from e
    except TypeError as e:  # the SDK raises this when no credentials can be resolved
        if "authentication" not in str(e):
            raise
        raise LLMUnavailable(f"No Anthropic credentials: {e}") from e
    if response.stop_reason == "refusal":
        raise LLMRefused(str(getattr(response, "stop_details", None)))
    texts = [b.text for b in response.content if b.type == "text"]
    if not texts:
        raise ValueError(f"No text in response (stop_reason={response.stop_reason})")
    return texts[0]


PROVIDERS = {"anthropic": _call_anthropic}


def ask_json(cfg, prompt: str, schema: dict, images: list[bytes], cache_dir: Path) -> dict:
    """Ask the configured VLM; returns the parsed JSON object (cached by content hash).

    Raises:
        LLMUnavailable: provider unknown or not authenticated.
        LLMRefused: the model declined.
        ValueError: the reply was not a JSON object.
    """
    if cfg.provider not in PROVIDERS:
        raise LLMUnavailable(f"vlm.provider={cfg.provider} is not implemented")
    key = _cache_key(cfg, prompt, schema, images)
    path = Path(cache_dir) / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())["parsed"]
    text = PROVIDERS[cfg.provider](cfg, prompt, schema, images)
    parsed = parse_json(text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": cfg.model, "raw": text, "parsed": parsed}, indent=2))
    return parsed
