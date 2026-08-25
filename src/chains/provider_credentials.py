"""Accept either provider API keys or the endpoint URLs dashboards expose."""

from urllib.parse import parse_qs, urlparse


def extract_provider_key(raw: str, provider: str) -> str:
    value = str(raw or "").strip().strip('"').strip("'")
    assignment = f"{provider.upper()}_API_KEY="
    if value.upper().startswith(assignment):
        value = value.split("=", 1)[1].strip().strip('"').strip("'")
    if provider == "alchemy" and value.upper().startswith("ALCHEMY_KEY="):
        value = value.split("=", 1)[1].strip().strip('"').strip("'")

    parsed = urlparse(value)
    if parsed.scheme in {"http", "https", "ws", "wss"}:
        if provider == "helius":
            return (parse_qs(parsed.query).get("api-key") or [""])[0].strip()
        if provider == "alchemy":
            parts = [part for part in parsed.path.split("/") if part]
            if len(parts) >= 2 and parts[-2] == "v2":
                return parts[-1].strip()
    return value


def normalize_provider_environment(environ) -> None:
    for variable, provider in (("HELIUS_API_KEY", "helius"), ("ALCHEMY_KEY", "alchemy")):
        if environ.get(variable):
            environ[variable] = extract_provider_key(environ[variable], provider)
