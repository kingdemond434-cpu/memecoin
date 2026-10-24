"""Loopback Phantom operator UI and prediction-market paper service.

Run separately from the production evidence writer. Its own SQLite ledger is
never spliced into historical validation receipts or real-fill evidence.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import secrets
import time
from pathlib import Path

from aiohttp import web

from src.execution.phantom import PhantomBridge
from src.execution.signer import SignerPolicy
from src.research.prediction_markets import PaperBroker, PredictionPaperRunner
from src.research.local_decisions import LocalDecisions
from src.research.behavioral_forensics import PublicWalletResearch
from src.research.market_robustness import cpcv_splits, reality_check
from src.research.intelligence_coverage import IntelligenceCoverage

ASSETS = Path(__file__).resolve().parents[2] / "dashboard" / "wallet"
PAPER_TASK = web.AppKey("paper_task", asyncio.Task)
FORENSICS_TASK = web.AppKey("forensics_task", asyncio.Task)
COVERAGE_TASK = web.AppKey("coverage_task", asyncio.Task)


def token_at(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            out.write(secrets.token_urlsafe(32))
    token = path.read_text(encoding="utf-8").strip()
    if len(token) < 32:
        raise ValueError("invalid gateway pairing token")
    return token


def make_app(broker, *, token, port=18082, bridge=None, start_runner=True):
    runner = PredictionPaperRunner(broker)
    forensics = PublicWalletResearch(broker)
    coverage = IntelligenceCoverage(broker)
    runner.condition_hints = lambda: [row[0] for row in broker.db.execute(
        "SELECT json_extract(payload,'$.market') FROM public_trades WHERE at>? ORDER BY at DESC LIMIT 8", (time.time() - 300,))]
    bridge = bridge or PhantomBridge()
    sessions = set()
    started_at = time.time()
    local_decisions = LocalDecisions(Path(os.getenv("MEMECOIN_LOCAL_DECISION_DIR", "data/research/local_decisions")))
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    @web.middleware
    async def boundary(request, handler):
        if request.host not in allowed_hosts:
            raise web.HTTPForbidden(text="loopback host required")
        origin = request.headers.get("Origin")
        if origin and origin != "http://" + request.host:
            raise web.HTTPForbidden(text="same origin required")
        if request.path.startswith("/api/"):
            bearer = request.headers.get("Authorization", "")
            server_auth = secrets.compare_digest(bearer, "Bearer " + token)
            browser_auth = request.cookies.get("memecoin_wallet", "") in sessions
            if request.path != "/api/pair" and not (server_auth or browser_auth):
                raise web.HTTPUnauthorized(text="gateway pairing required")
            if request.method != "GET" and not server_auth and origin != "http://" + request.host:
                raise web.HTTPForbidden(text="browser mutation requires same-origin request")
            if request.path == "/api/phantom/sign" and not server_auth:
                raise web.HTTPForbidden(text="only the configured execution signer can create signing requests")
        try:
            response = await handler(request)
        except (ValueError, KeyError, TypeError) as exc:
            response = web.json_response({"error": str(exc)[:200]}, status=400)
        except PermissionError as exc:
            response = web.json_response({"error": str(exc)[:200]}, status=403)
        except asyncio.TimeoutError:
            response = web.json_response({"error": "Phantom approval timed out; transaction was not submitted"}, status=408)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
        return response

    app = web.Application(middlewares=[boundary], client_max_size=16 * 1024)

    async def health(request):
        components = {name: not start_runner or app.get(key) is not None and not app[key].done()
                      for name, key in (("paper", PAPER_TASK), ("wallet_research", FORENSICS_TASK), ("coverage", COVERAGE_TASK))}
        healthy = all(components.values())
        return web.json_response({"status": "healthy" if healthy else "failed", "mode": "PAPER",
                                  "components": components,
                                  "live_submission_locked": True, "uptime_seconds": time.time() - started_at},
                                 status=200 if healthy else 503)

    async def status(request):
        return web.json_response({"prediction_markets": runner.report(), "phantom": bridge.report(),
                                  "local_decisions": local_decisions.report(),
                                  "public_wallet_research": forensics.report(),
                                  "global_coverage": coverage.report()})

    async def evaluate(request):
        data = await request.json()
        return web.json_response(local_decisions.evaluate(data["text"], data["questions"]))

    async def robustness(request):
        data = await request.json()
        if data["method"] == "cpcv":
            result = cpcv_splits(data["decision_at"], data["label_end"], **data.get("settings", {}))
        elif data["method"] == "reality_check":
            result = reality_check(data["strategy_returns"], data["control_returns"], **data["settings"])
        else: raise ValueError("unknown research diagnostic")
        # User-supplied diagnostic inputs are never inserted into an earned
        # evidence ledger or promoted into real execution authority.
        return web.json_response(result)

    async def pair(request):
        data = await request.json()
        if not secrets.compare_digest(str(data.get("token", "")), token):
            raise web.HTTPUnauthorized(text="invalid pairing token")
        if len(sessions) >= 8: sessions.clear()
        session = secrets.token_urlsafe(32)
        sessions.add(session)
        response = web.json_response({"paired": True})
        response.set_cookie("memecoin_wallet", session, httponly=True, samesite="Strict", max_age=86400)
        return response

    async def wallet_challenge(request):
        return web.json_response(bridge.challenge(str((await request.json())["address"])))

    async def wallet_connect(request):
        data = await request.json()
        return web.json_response({"address": bridge.connect(data["nonce"], data["signature_base64"])})

    async def wallet_disconnect(request):
        bridge.disconnect()
        return web.json_response({"disconnected": True})

    async def requests(request):
        return web.json_response({"requests": bridge.requests()})

    async def complete(request):
        data = await request.json()
        bridge.complete(data["id"], data.get("signed_transaction_base64", ""), reject=data.get("reject") is True)
        return web.json_response({"accepted": True})

    async def sign(request):
        data = await request.json()
        message = base64.b64decode(data["message_base64"], validate=True)
        if len(message) > 1232: raise ValueError("oversized Solana message")
        result = await bridge.sign(message, str(data["address"]))
        return web.json_response({"signature_base64": base64.b64encode(result).decode()})

    async def asset(request):
        name = request.match_info.get("name", "index.html")
        if name not in {"index.html", "wallet.js", "wallet.css"}: raise web.HTTPNotFound()
        return web.FileResponse(ASSETS / name)

    app.router.add_get("/health", health)
    app.router.add_get("/status", status)
    app.router.add_post("/api/pair", pair)
    app.router.add_get("/api/status", status)
    app.router.add_post("/api/decisions/evaluate", evaluate)
    app.router.add_post("/api/research/robustness", robustness)
    app.router.add_post("/api/phantom/challenge", wallet_challenge)
    app.router.add_post("/api/phantom/connect", wallet_connect)
    app.router.add_post("/api/phantom/disconnect", wallet_disconnect)
    app.router.add_get("/api/phantom/requests", requests)
    app.router.add_post("/api/phantom/complete", complete)
    app.router.add_post("/api/phantom/sign", sign)
    app.router.add_get("/", asset)
    app.router.add_get("/wallet/{name}", asset)

    async def startup(app):
        if start_runner:
            app[PAPER_TASK] = asyncio.create_task(runner.run(), name="prediction-paper")
            app[FORENSICS_TASK] = asyncio.create_task(forensics.run(), name="public-wallet-forensics")
            app[COVERAGE_TASK] = asyncio.create_task(coverage.run(), name="intelligence-coverage")

    async def cleanup(app):
        bridge.disconnect()
        tasks = [app[key] for key in (PAPER_TASK, FORENSICS_TASK, COVERAGE_TASK) if key in app]
        for task in tasks: task.cancel()
        if tasks: await asyncio.gather(*tasks, return_exceptions=True)
        broker.close()

    app.on_startup.append(startup)
    app.on_cleanup.append(cleanup)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18082)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535: parser.error("invalid loopback port")
    broker = PaperBroker(args.state_dir / "prediction-paper.sqlite")
    # This entry point is exclusively PAPER; it cannot be armed by inherited
    # secrets, environment acknowledgements, or a browser connection.
    os.environ.pop("ALLOW_LIVE_TRADING", None)
    os.environ.pop("SOLANA_PRIVATE_KEY", None)
    bridge = PhantomBridge(policy=SignerPolicy(), live_authority=lambda: False)
    app = make_app(broker, token=token_at(args.state_dir / "gateway.token"), port=args.port, bridge=bridge)
    web.run_app(app, host="127.0.0.1", port=args.port, print=None)


if __name__ == "__main__":
    main()
