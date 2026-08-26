"""Optional copy-polish via the Claude API.

The pipeline never requires this: when disabled (default), when the
`anthropic` package is missing, or when no credentials resolve, callers get
the deterministic template text back unchanged. Enable by setting
`ai.enabled: true` in config/platform.json and providing ANTHROPIC_API_KEY
(or an `ant auth login` profile).
"""

from __future__ import annotations

DEFAULT_MODEL = "claude-opus-5"


def polish(text: str, kind: str, config: dict) -> str:
    ai_cfg = (config or {}).get("ai") or {}
    if not ai_cfg.get("enabled"):
        return text
    try:
        import anthropic
    except ImportError:
        print("[ai] anthropic package not installed; using template text")
        return text

    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=ai_cfg.get("model", DEFAULT_MODEL),
            max_tokens=1024,
            system=(
                "You are the copy editor for a GTA VI fan news site. Rewrite the "
                "given draft to be punchy and readable for gaming fans. Keep every "
                "fact, name, number and confidence label (confirmed/rumored/"
                "speculated) exactly as given. Return only the rewritten text."
            ),
            messages=[{"role": "user", "content": f"Draft ({kind}):\n\n{text}"}],
        )
        parts = [b.text for b in response.content if b.type == "text"]
        return "".join(parts).strip() or text
    except Exception as e:  # any API failure falls back to the template text
        print(f"[ai] polish failed ({e.__class__.__name__}); using template text")
        return text
