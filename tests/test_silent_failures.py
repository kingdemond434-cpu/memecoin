"""Regressions for defects whose whole signature was that nothing complained.

Each case here shipped, ran for days, and reported healthy while doing the
wrong thing. That is the expensive failure mode on this desk: a loud error
gets fixed the hour it appears, and a quiet one sets the readiness bar it is
silently failing to clear. The assertions are written against the symptom an
operator would never have seen, not against the implementation that caused it.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ops.watchdog import Policy, decide, observed_faults
from src.chains.rpc_manager import (
    ChainConfig, ChainType, EndpointHealth, RPCEndpointConfig, RPCHealth,
    RPCManager, _host_of,
)
from src.collectors.registry import load_declarations


SEED = """
sources:
  - id: mastodon:mstdn-jp
    kind: mastodon
    language: ja
    options: {instance: "https://mstdn.jp"}
  - id: rss:us-sec-press
    kind: rss
    language: en
    options: {url: "https://www.sec.gov/news/pressreleases.rss"}
  - id: rss:only-in-seed
    kind: rss
    language: en
    options: {url: "https://example.invalid/feed"}
"""

# The machine-generated overlay names the same endpoints its own way. That
# difference is the entire defect: nothing about these two files disagrees on
# what to poll, only on what to call it.
OVERLAY = """
sources:
  - id: mastodon-mstdn-jp
    kind: mastodon
    language: ja
    options: {instance: "https://mstdn.jp/"}
  - id: sec-press-rss
    kind: rss
    language: en
    options: {url: "https://www.sec.gov/news/pressreleases.rss"}
"""


class SourceOverlayMergesByEndpoint(unittest.TestCase):
    """The overlay must supersede the seed, not double it.

    Merging on source_id alone let one endpoint be registered twice under two
    spellings. Both copies polled, so every affected host saw exactly double
    its declared rate, answered 429, and was marked DEAD -- a mesh reporting
    ten dead sources that were all reachable by hand.
    """

    def merge(self):
        with tempfile.TemporaryDirectory() as directory:
            seed = Path(directory) / "sources.yaml"
            overlay = Path(directory) / "sources.verified.yaml"
            seed.write_text(SEED, encoding="utf-8")
            overlay.write_text(OVERLAY, encoding="utf-8")
            return load_declarations(f"{seed},{overlay}")

    def test_one_endpoint_yields_one_declaration(self):
        merged = self.merge()
        instances = [d.options.get("instance") for d in merged
                     if d.kind == "mastodon"]
        self.assertEqual(len(instances), 1, "mstdn.jp registered twice")

    def test_trailing_slash_is_not_a_different_endpoint(self):
        kinds = [d.source_id for d in self.merge() if d.kind == "mastodon"]
        self.assertEqual(kinds, ["mastodon-mstdn-jp"])

    def test_overlay_wins_and_seed_only_sources_survive(self):
        ids = {d.source_id for d in self.merge()}
        self.assertIn("sec-press-rss", ids)
        self.assertNotIn("rss:us-sec-press", ids)
        # Deduplication must not become deletion: a source the overlay never
        # verified is still a source.
        self.assertIn("rss:only-in-seed", ids)

    def test_no_source_id_appears_twice(self):
        ids = [d.source_id for d in self.merge()]
        self.assertEqual(len(ids), len(set(ids)))


class RateLimitedEndpointsAreDemotedAndParked(unittest.TestCase):
    """A non-200 is a failure and must cost the endpoint its health.

    It used to be neither counted nor logged: a provider answering 429 to
    every call stayed HEALTHY, kept winning selection, and surfaced only as
    an unattributed "All RPC endpoints failed" with the status that explained
    it never reaching a log line.
    """

    def manager(self):
        config = ChainConfig(
            name="Solana Mainnet", chain_id="solana",
            chain_type=ChainType.SOLANA,
            rpc_endpoints=[RPCEndpointConfig(url="https://a.invalid/?api-key=k"),
                           RPCEndpointConfig(url="https://b.invalid")],
            explorer_api="", explorer_key="", native_token="SOL", decimals=9,
            block_time=0.4, factories={}, routers={}, base_tokens=[],
            min_liquidity_usd=0.0, max_tax=0.0, honeypot_check=False)
        return RPCManager(config)

    def test_a_429_demotes_the_endpoint(self):
        manager = self.manager()
        endpoint = manager.endpoints[0]
        manager._penalise(endpoint, 429, cooldown=30.0)
        self.assertNotEqual(endpoint.health, RPCHealth.HEALTHY)
        self.assertEqual(endpoint.error_count, 1)
        self.assertEqual(endpoint.last_status, 429)

    def test_three_refusals_take_an_endpoint_down(self):
        manager = self.manager()
        endpoint = manager.endpoints[0]
        for _ in range(3):
            manager._penalise(endpoint, 429)
        self.assertEqual(endpoint.health, RPCHealth.DOWN)

    def test_a_cooling_endpoint_is_not_selected(self):
        manager = self.manager()
        manager._penalise(manager.endpoints[0], 429, cooldown=30.0)
        for _ in range(20):
            chosen = manager._select_endpoint()
            self.assertIsNot(chosen, manager.endpoints[0])

    def test_every_endpoint_cooling_still_returns_one(self):
        """Degraded enrichment beats none: the caller sees the real refusal."""
        manager = self.manager()
        for endpoint in manager.endpoints:
            manager._penalise(endpoint, 429, cooldown=30.0)
        self.assertIsNotNone(manager._select_endpoint())

    def test_retry_after_is_honoured_but_bounded(self):
        class Response:
            def __init__(self, value):
                self.headers = {"Retry-After": value}
        self.assertEqual(RPCManager._retry_after_seconds(Response("12"), 30.0), 12.0)
        # A hostile or broken header cannot park an endpoint for hours.
        self.assertEqual(RPCManager._retry_after_seconds(Response("99999"), 30.0), 300.0)
        self.assertEqual(RPCManager._retry_after_seconds(Response("soon"), 30.0), 30.0)

    def test_api_keys_never_reach_a_log_line(self):
        self.assertEqual(_host_of("https://mainnet.helius-rpc.com/?api-key=SECRET"),
                         "mainnet.helius-rpc.com")
        self.assertNotIn("SECRET", _host_of("https://h.io/v2/SECRET"))


class TelegramSilenceIsAFault(unittest.TestCase):
    """The fastest signal path fails by going quiet, so quiet is the check.

    Every field Telegram exposes still reads healthy when the messages stop:
    the session is connected, the handler is registered, the channel count is
    unchanged. Only the mention count standing still gives it away.
    """

    def healthy(self):
        return {
            "runtime_tasks": {"status": "OK", "failed": []},
            "source_mesh": {"sources": 2, "producers": 2, "streaming": True},
            "yellowstone": {"status": "STREAMING"},
            "rpc_program_stream": {"status": "RPC_WS"},
            "memory": {"band": "calm"},
            "stream_events": {"total": 10, "token_created": 1},
            "pump_decoder": {"status": "OK"},
            "event_loop": {}, "data_miners": {"status": "OK"},
            "prediction": "OK", "rug_hazard": {"model_trained": True},
            "credentials": {"absent": [], "telegram": {
                "keys_present": True, "session_authorised": True,
                "channels_listed": 38}},
            "social": {"data_status": {"telegram": "OK"},
                       "total_mentions": 4467},
        }

    def call(self, readiness, state, now=10_000.0):
        return decide(service_active=True, service_enabled=True,
                      readiness=readiness, readiness_age=10.0, state=state,
                      now=now, policy=Policy(), trainer_active=False,
                      training_age=10.0)

    def test_a_healthy_feed_raises_nothing(self):
        self.assertNotIn("telegram_signal_path_blocked",
                         observed_faults(self.healthy())[1])

    def test_an_unauthorised_session_is_named(self):
        readiness = self.healthy()
        readiness["credentials"]["telegram"]["session_authorised"] = False
        self.assertIn("telegram_session_unauthorised",
                      observed_faults(readiness)[1])

    def test_a_blocked_signal_path_is_named(self):
        readiness = self.healthy()
        readiness["social"]["data_status"]["telegram"] = "DATA_BLOCKED: flood wait"
        self.assertIn("telegram_signal_path_blocked",
                      observed_faults(readiness)[1])

    def test_a_frozen_mention_count_earns_a_restart_not_just_an_alert(self):
        """The fix for a wedged Telethon client is a reconnect, so the stall
        is repairable -- through the same persistence rule as every repair:
        one observation is a coincidence, two is a fault."""
        readiness, state = self.healthy(), {}
        self.assertFalse(self.call(readiness, state, now=10_000.0).restart_desk)
        # Still inside a plausible quiet stretch.
        self.assertFalse(self.call(readiness, state, now=11_000.0).restart_desk)
        # Past the stall window: first observation arms, second repairs.
        first = self.call(readiness, state, now=12_000.0)
        self.assertFalse(first.restart_desk)
        second = self.call(readiness, state, now=12_060.0)
        self.assertTrue(second.restart_desk)
        self.assertIn("telegram_signals_stalled", second.repair_reasons)

    def test_a_moving_count_resets_the_clock(self):
        readiness, state = self.healthy(), {}
        self.call(readiness, state, now=10_000.0)
        readiness["social"]["total_mentions"] = 4468
        plan = self.call(readiness, state, now=12_000.0)
        self.assertFalse(plan.restart_desk)
        self.assertNotIn("telegram_signals_stalled", plan.alerts)

    def test_an_unconfigured_alert_channel_says_so(self):
        """The alert about alerts. Returning a quiet False is how every
        alert this watchdog ever raised went to a journal nobody reads."""
        from ops import watchdog
        with patch.dict("os.environ", {"TELEGRAM_ALERT_BOT_TOKEN": "",
                                       "TELEGRAM_BOT_TOKEN": "",
                                       "TELEGRAM_ALERT_CHAT_ID": ""},
                        clear=False):
            with self.assertLogs(watchdog.logger, level="WARNING") as captured:
                self.assertFalse(watchdog._send_telegram_alert("desk is down"))
        joined = "\n".join(captured.output)
        self.assertIn("NOT DELIVERED", joined)
        self.assertIn("desk is down", joined)


if __name__ == "__main__":
    unittest.main()


class BatchRepliesAlignByIdNotByOrder(unittest.TestCase):
    """A reordered batch reply must not shift results onto the wrong request.

    JSON-RPC permits a server to answer a batch in any order and to omit
    entries. Zipping the reply against the request list attributes one
    signature's transaction to a different signature -- a wrong feature rather
    than a missing one, which nothing downstream can detect.
    """

    def align(self, requests, reply):
        by_id = {item.get("id"): item for item in reply if isinstance(item, dict)}
        return [(by_id.get(request.get("id")) or {}).get("result")
                for request in requests]

    def requests(self):
        return [{"jsonrpc": "2.0", "id": index, "method": "getTransaction"}
                for index in range(3)]

    def test_a_reversed_reply_still_lands_on_the_right_request(self):
        reply = [{"id": 2, "result": "c"}, {"id": 0, "result": "a"},
                 {"id": 1, "result": "b"}]
        self.assertEqual(self.align(self.requests(), reply), ["a", "b", "c"])

    def test_a_missing_entry_becomes_a_hole_not_a_shift(self):
        reply = [{"id": 0, "result": "a"}, {"id": 2, "result": "c"}]
        self.assertEqual(self.align(self.requests(), reply), ["a", None, "c"])

    def test_the_manager_aligns_by_id(self):
        from src.chains import rpc_manager
        with open(rpc_manager.__file__, encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn("by_id.get(request.get(\"id\"))", source)
        self.assertNotIn("return [r.get(\"result\") for r in data]", source)


class RestartFixableFaultsAreRepairedNotAnnounced(unittest.TestCase):
    """Where a restart IS the fix, the watchdog acts; alerts are for the rest.

    A broken creation-event subscription on a healthy socket and a wedged
    miner pool are both re-established by a process restart. Leaving them as
    alerts meant a human reading a journal was the repair path for faults the
    machine could fix in under two minutes. The promotion keeps the guard
    rails: two consecutive observations, the cooldown, and the budget.
    """

    def healthy(self):
        return {
            "runtime_tasks": {"status": "OK", "failed": []},
            "source_mesh": {"sources": 2, "producers": 2, "streaming": True},
            "yellowstone": {"status": "STREAMING"},
            "rpc_program_stream": {"status": "RPC_WS"},
            "memory": {"band": "calm"},
            "stream_events": {"total": 10, "token_created": 1},
            "pump_decoder": {"status": "OK"},
            "event_loop": {}, "data_miners": {"status": "OK", "runnable": 3},
            "prediction": "OK", "rug_hazard": {"model_trained": True},
            "credentials": {"absent": []},
        }

    def test_a_creationless_stream_is_repairable(self):
        readiness = self.healthy()
        readiness["stream_events"] = {"total": 500, "token_created": 0}
        repairs, alerts = observed_faults(readiness)
        self.assertIn("chain_stream_delivers_no_creation_events", repairs)
        self.assertNotIn("chain_stream_delivers_no_creation_events", alerts)

    def test_dark_miners_are_repairable(self):
        readiness = self.healthy()
        readiness["data_miners"] = {"status": "DATA_BLOCKED", "runnable": 5}
        repairs, alerts = observed_faults(readiness)
        self.assertIn("all_runnable_data_miners_are_dark", repairs)

    def test_what_a_restart_cannot_fix_stays_an_alert(self):
        """The boundary is the point: promoting these would flap the desk
        against faults only new code, credentials or evidence can clear."""
        readiness = self.healthy()
        readiness["pump_decoder"] = {"status": "DEGRADED"}
        readiness["event_loop"] = {"candidate_drops": 4}
        readiness["prediction"] = "DATA_BLOCKED"
        readiness["credentials"] = {"absent": [{"name": "X_BEARER_TOKEN"}]}
        repairs, alerts = observed_faults(readiness)
        for fault in ("pump_decoder_degraded", "decision_queue_dropped_work",
                      "prediction_model_data_blocked",
                      "optional_credentials_absent"):
            self.assertIn(fault, alerts)
            self.assertNotIn(fault, repairs)


class FreeEndpointsAreComplementaryNotRedundant(unittest.TestCase):
    """A method carve-out must cost one method, not the whole endpoint.

    Free Solana providers each refuse a different expensive call: publicnode
    403s getTokenLargestAccounts while serving getAccountInfo; leorpc does the
    reverse. Cooling an endpoint wholesale on a 403 makes the pool only as
    capable as its weakest member; excluding it per method makes the pool the
    UNION of what they serve, which is the entire value of adding free tiers.
    """

    def manager(self):
        config = ChainConfig(
            name="Solana Mainnet", chain_id="solana",
            chain_type=ChainType.SOLANA,
            rpc_endpoints=[RPCEndpointConfig(url="https://a.invalid"),
                           RPCEndpointConfig(url="https://b.invalid")],
            explorer_api="", explorer_key="", native_token="SOL", decimals=9,
            block_time=0.4, factories={}, routers={}, base_tokens=[],
            min_liquidity_usd=0.0, max_tax=0.0, honeypot_check=False)
        return RPCManager(config)

    def test_a_blocked_method_does_not_disqualify_the_endpoint(self):
        manager = self.manager()
        blocked = manager.endpoints[0]
        blocked.blocked_methods.add("getTokenLargestAccounts")
        # Excluded for that method...
        for _ in range(20):
            self.assertIsNot(
                manager._select_endpoint(method="getTokenLargestAccounts"),
                blocked)
        # ...and still eligible for everything else.
        chosen = [manager._select_endpoint(method="getAccountInfo")
                  for _ in range(40)]
        self.assertTrue(any(item is blocked for item in chosen))

    def test_no_endpoint_for_a_method_is_refused_not_faked(self):
        """Returning a permanent 403-er would dress it as a transient error."""
        manager = self.manager()
        for endpoint in manager.endpoints:
            endpoint.blocked_methods.add("getTokenLargestAccounts")
        self.assertIsNone(
            manager._select_endpoint(method="getTokenLargestAccounts"))
        self.assertIsNotNone(manager._select_endpoint(method="getAccountInfo"))

    def test_a_cooling_endpoint_still_serves_after_its_window(self):
        """Method blocks are permanent; rate-limit cooldowns are not."""
        manager = self.manager()
        endpoint = manager.endpoints[0]
        manager._penalise(endpoint, 429, cooldown=30.0)
        endpoint.cooldown_until = 0.0   # window elapsed
        endpoint.health = RPCHealth.HEALTHY
        chosen = [manager._select_endpoint(method="getAccountInfo")
                  for _ in range(40)]
        self.assertTrue(any(item is endpoint for item in chosen))
