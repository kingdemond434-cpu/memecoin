"""Twenty research surfaces that were built and answered nobody.

Every one of these produces a number the desk is entitled to know and was
throwing away: which source leads which event type, what the adversarial
detector has reweighted, which model's calibration can be trusted, what a
cold-start cohort implies about a venue at this hour, how much of a
narrative's audience is already exhausted.

They are composed into one research report rather than sprinkled through the
decision path, and that placement is the honest one. These are DIAGNOSTIC:
they say what the research layer currently believes. Promoting any of them to
a vote without forward evidence would be the same mistake as the hand-weighted
composite this desk spent the day removing -- so they are reported, dated, and
left with no authority until the gauntlet has priced them.

The feeds are different and are wired at their event sources, because a
surface that is read but never written reports an empty dictionary forever and
looks identical to one nobody consults.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


def _read(call: Any, *args: Any, **kwargs: Any) -> Any:
    if call is None:
        return None
    try:
        return call(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("research read failed: %s", exc)
        return None


def source_intelligence_report(desk: Any) -> Dict[str, Any]:
    """What the information graph has learned about who is early."""
    graph = getattr(desk, "info_graph", None)
    adversarial = getattr(desk, "adversarial", None) or getattr(
        desk, "adaptation_detector", None)
    lab = getattr(desk, "counterfactual_lab", None)
    if graph is None:
        return {"status": "DATA_BLOCKED", "detail": "no information graph"}
    from src.strategies.information_graph import LeadEventType
    rankings = {}
    for event_type in LeadEventType:
        ranked = _read(graph.get_source_ranking, event_type, 5)
        if ranked:
            rankings[event_type.value] = [
                {"source": name, "score": round(float(score), 5)}
                for name, score in ranked]
    return {
        "status": "OK" if rankings else "DATA_BLOCKED",
        "authority": "none",
        "source_ranking": rankings,
        # What the detector has reweighted, and what it is shouting about.
        "adapted_weights": _read(getattr(adversarial, "get_all_weights", None)),
        "adaptation_alerts": _read(getattr(adversarial, "get_alerts", None), 5),
        "counterfactual_policy": _read(
            getattr(lab, "get_best_policy", None), "incumbent"),
        "detail": ("" if rankings else
                   "no source has enough observed leads to be ranked"),
    }


def lead_forecast(desk: Any, token: str) -> Dict[str, Any]:
    """What the graph expects to happen to this token next, and how sure."""
    graph = getattr(desk, "info_graph", None)
    predicted = _read(getattr(graph, "predict_next_event", None), token)
    if not predicted:
        return {"status": "DATA_BLOCKED",
                "detail": "no lead history for this token"}
    return {
        "status": "OK", "authority": "none",
        "next_events": [{"event": kind.value, "probability": round(float(p), 5)}
                        for kind, p in predicted[:5]],
    }


def wallet_intelligence_report(desk: Any, regime: Any = None) -> Dict[str, Any]:
    """Measured wallet value, per-regime behaviour, and who is moving together."""
    intel = getattr(desk, "wallet_intel", None)
    if intel is None:
        return {"status": "DATA_BLOCKED", "detail": "no wallet intelligence"}
    ranked = _read(getattr(intel, "get_top_wallets_by_value", None), 10) or []
    coordinated = _read(getattr(intel, "get_coordinated_activity", None), 300) or []
    return {
        "status": "OK" if ranked else "DATA_BLOCKED",
        # Ranked on the lower confidence bound of what FOLLOWING them
        # returned, never on what they themselves made.
        "top_by_measured_value": [
            value.to_dict() if hasattr(value, "to_dict") else str(value)
            for value in ranked],
        "coordinated_groups": len(coordinated),
        "coordinated": coordinated[:5],
        "detail": ("" if ranked else
                   "no wallet has enough followed outcomes to be valued"),
    }


def wallet_regime_reading(desk: Any, wallet: str, token: str,
                          regime: Any) -> Dict[str, Any]:
    """What this wallet does in THIS regime, rather than on average.

    A wallet that is excellent on early-curve launches and poor after
    migration has no single score, and averaging the two describes a wallet
    that does not exist.
    """
    intel = getattr(desk, "wallet_intel", None)
    if intel is None or not wallet:
        return {"status": "DATA_BLOCKED", "detail": "no wallet intelligence"}
    signal = _read(getattr(intel, "get_wallet_signal", None), wallet, token, regime)
    performance = _read(getattr(intel, "get_regime_performance", None),
                        wallet, regime)
    return {
        "status": "OK" if signal else "DATA_BLOCKED",
        "authority": "none",
        "signal": signal,
        "regime_trades": getattr(performance, "trades", None),
        "regime_rug_exposure": getattr(performance, "rug_exposure", None),
    }


def evidence_quality_report(desk: Any) -> Dict[str, Any]:
    """Which of the desk's own instruments can currently be believed."""
    calibration = getattr(desk, "calibration", None) or getattr(
        desk, "calibration_book", None)
    ledger = getattr(desk, "source_edges", None) or getattr(
        desk, "edge_ledger", None)
    distillate = getattr(desk, "cold_distillate", None)
    edges = _read(getattr(ledger, "scored_edges", None)) or []
    models = {}
    for name in ("prediction", "rug_hazard", "action_value"):
        verdict = _read(getattr(calibration, "trustworthy", None), name)
        models[name] = verdict
    # Does each scored edge survive a change of tape? An edge measured
    # entirely inside one regime is a description of that regime.
    stability = {}
    for edge in edges[:5]:
        reading = _read(getattr(ledger, "regime_stability", None),
                        getattr(edge, "source_id", ""),
                        getattr(edge, "mechanism", None),
                        getattr(edge, "claim_type", None),
                        float(getattr(edge, "horizon_s", 0.0) or 0.0))
        if reading is not None:
            stability[str(getattr(edge, "source_id", ""))] = reading
    return {
        "edge_regime_stability": stability,
        "status": "OK" if (edges or any(v is not None for v in models.values()))
                  else "DATA_BLOCKED",
        # None is not False. An uncalibrated model is one nobody has checked,
        # which is a different problem from one that failed its check.
        "model_trustworthy": models,
        "scored_source_edges": len(edges),
        "cold_start_available": distillate is not None,
        "detail": ("" if edges or any(v is not None for v in models.values())
                   else "no instrument has been calibrated or scored yet"),
    }


def cold_start_prior(desk: Any, venue: str, hour: int,
                     funder: str = "") -> Dict[str, Any]:
    """What is known about a launch before it has done anything.

    The whole point of a cold-start prior is the first seconds, when there is
    no token history at all -- and these two were the only things the desk had
    to say in that window, and it said neither.
    """
    distillate = getattr(desk, "cold_distillate", None)
    if distillate is None:
        return {"status": "DATA_BLOCKED", "detail": "no cold distillate"}
    cohort = _read(getattr(distillate, "cohort_prior", None), venue, int(hour))
    funder_view = (_read(getattr(distillate, "funder_prior", None), funder)
                   if funder else None)
    return {
        "status": "OK" if (cohort or funder_view) else "DATA_BLOCKED",
        "authority": "none",
        "cohort": cohort, "funder": funder_view,
        "detail": ("" if cohort or funder_view else
                   "neither this venue-hour nor this funder has a prior yet"),
    }


def research_report(desk: Any) -> Dict[str, Any]:
    """Every diagnostic surface, dated, with no authority over anything."""
    return {
        "generated_at": time.time(),
        "authority": "none",
        "sources": source_intelligence_report(desk),
        "wallets": wallet_intelligence_report(desk),
        "evidence_quality": evidence_quality_report(desk),
    }


def fallback_confidence(desk: Any, facts: Sequence[str],
                        context: Optional[Dict[str, Any]] = None
                        ) -> Dict[str, Any]:
    """One confidence multiplier for a decision resting on several facts.

    `resolve_many` and `combined_confidence` exist so a decision that depends
    on four substituted facts is discounted by the WEAKEST of them rather than
    by their average. Nothing called either, so a decision resting on one
    primary source and three fallbacks was priced identically to one resting
    on four primaries.
    """
    resolver = getattr(desk, "facts", None)
    if resolver is None or not facts:
        return {"status": "DATA_BLOCKED", "detail": "no fallback resolver"}
    resolutions = _read(resolver.resolve_many, list(facts), context)
    if not resolutions:
        return {"status": "DATA_BLOCKED", "detail": "nothing resolved"}
    confidence = _read(type(resolver).combined_confidence,
                       list(resolutions.values()))
    return {
        "status": "OK",
        "confidence": confidence,
        "resolved": {fact: getattr(item, "tier", None)
                     for fact, item in resolutions.items()},
        "detail": ("the weakest rung dominates: a decision resting on three "
                   "fallbacks is not as good as one resting on primaries"),
    }
