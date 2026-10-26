"""Fill-bound TCA and mature markouts, in explicit native raw units.

Adapts the owner's quant fill_markout/tca methods without MT5 units or gates.
An AMM reserve ratio is a marginal reference, not a size-executable quote.
Confirmation-observation markouts are not precision execution-time markouts.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import math

WSOL = "So11111111111111111111111111111111111111112"
HORIZONS = (5, 30, 300, 3600)


def decimal(value):
    if isinstance(value, bool):
        raise ValueError("boolean measurement")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError("invalid numeric measurement") from None
    if not result.is_finite():
        raise ValueError("nonfinite measurement")
    return result


def clock(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        raise ValueError("positive observation clock required")
    return float(value)


def reserve_reference(token, virtual_quote, virtual_base, *, observed_at, quote_mint=WSOL, source_ref, source_at=None):
    quote, base = decimal(virtual_quote), decimal(virtual_base)
    if quote <= 0 or base <= 0 or not token or not source_ref:
        raise ValueError("positive observed reserve ratio and provenance required")
    receipt = clock(observed_at)
    event = receipt if source_at is None else clock(source_at)
    if event > receipt:
        raise ValueError("source reference is from the future")
    return {"market": token, "quote_mint": quote_mint, "unit": "raw_quote_per_raw_base",
            "reference_price": str(quote / base), "reference_at": event,
            "available_at": clock(observed_at), "source_ref": source_ref,
            "kind": "AMM_MARGINAL_RESERVE_REFERENCE", "executable_for_size": False}


def normalize_attempt(token, attempt):
    """Do not infer fills, decimals, native input amounts or complete fees."""
    if not isinstance(attempt, dict):
        return {"status": "DATA_BLOCKED_MEASUREMENT_SCHEMA", "real_fill": None}
    if attempt.get("simulated") is True:
        return {"status": "SIMULATED_ATTEMPT_SEPARATE", "real_fill": False}
    if attempt.get("simulated") is not False:
        return {"status": "DATA_BLOCKED_SIMULATION_IDENTITY", "real_fill": None}
    if not (attempt.get("filled") is True and attempt.get("landed") is True and attempt.get("success") is True):
        return {"status": "NOT_A_VERIFIED_FILL", "real_fill": False}
    ctx = attempt.get("measurement_context") or {}
    if not isinstance(ctx, dict):
        return {"status": "DATA_BLOCKED_FILL_PROVENANCE", "real_fill": True}
    if (ctx.get("amount_proof") != "confirmed_owner_balance_deltas" or not ctx.get("signature")
            or attempt.get("signature") != ctx.get("signature")):
        return {"status": "DATA_BLOCKED_FILL_PROVENANCE", "real_fill": True}
    if ctx.get("output_mint") == token and ctx.get("input_mint") != token:
        side, quote = "buy", ctx.get("input_mint")
        quantity, notional = ctx.get("output_amount_raw"), ctx.get("input_amount_raw")
    elif ctx.get("input_mint") == token and ctx.get("output_mint") != token:
        side, quote = "sell", ctx.get("output_mint")
        quantity, notional = ctx.get("input_amount_raw"), ctx.get("output_amount_raw")
    else:
        return {"status": "DATA_BLOCKED_FILL_ASSET_IDENTITY", "real_fill": True}
    try:
        if any(type(value) is not int for value in (quantity, notional)):
            raise ValueError("native raw integer amounts required")
        qty, paid = decimal(quantity), decimal(notional)
        if qty <= 0 or paid <= 0 or not quote:
            raise ValueError("balance delta unavailable")
        at = clock(ctx.get("confirmation_observed_at"))
        arrival_at = clock(ctx.get("arrival_at"))
    except ValueError:
        return {"status": "DATA_BLOCKED_FILL_AMOUNTS_OR_CLOCK", "real_fill": True}
    return {"status": "NORMALIZED_CONFIRMED_FILL", "real_fill": True,
            "fill_id": ctx["signature"], "market": token, "quote_mint": quote,
            "unit": "raw_quote_per_raw_base", "side": side, "quantity": str(qty),
            "fill_price": str(paid / qty), "reference_at": at, "available_at": at,
            "clock_basis": "CONFIRMATION_RECEIPT; NOT_EXECUTION_TIME",
            "transaction_block_time": ctx.get("transaction_block_time"),
            "arrival_at": arrival_at, "decision_at": ctx.get("decision_at"),
            "external_cost_quote_raw": ctx.get("external_cost_quote_raw"),
            "external_costs_complete": ctx.get("external_costs_complete") is True,
            "known_network_fee_lamports": ctx.get("network_fee_lamports"),
            "missing_costs": ctx.get("missing_costs", ["tip/rent and cross-asset fee reconciliation"]),
            "source_ref": "solana:" + ctx["signature"] + "#slot=" + str(ctx.get("slot"))}


def analyze_fill(fill, tape, *, as_of, horizons=HORIZONS, max_reference_age=2.0):
    """No nearest/future quote lookup and no immature/absent horizon zeros."""
    now, at = clock(as_of), clock(fill["reference_at"])
    available = clock(fill["available_at"])
    qty, price = decimal(fill["quantity"]), decimal(fill["fill_price"])
    if qty <= 0 or price <= 0 or fill.get("side") not in {"buy", "sell"}:
        raise ValueError("positive fill amounts and explicit side required")
    if not fill.get("source_ref") or not fill.get("market") or not fill.get("quote_mint"):
        raise ValueError("fill identity and provenance required")
    if isinstance(max_reference_age, bool) or not math.isfinite(max_reference_age) or max_reference_age < 0:
        raise ValueError("nonnegative reference freshness required")
    result = {"status": "PARTIAL_MEASUREMENT", "as_of": now, "fill_id": fill.get("fill_id"),
              "real_fill": fill.get("real_fill"), "market": fill["market"], "quote_mint": fill["quote_mint"],
              "unit": fill["unit"], "clock_basis": fill.get("clock_basis"),
              "fill_price": str(price), "quantity": str(qty), "source_ref": fill["source_ref"],
              "reference_at": at, "available_at": available,
              "horizons": {}, "unidentified_components": ["spread versus impact; not inferred from forecast"],
              "external_costs_complete": False, "external_cost_quote_raw": None}
    if available > now or at > now:
        return {**result, "status": "DATA_BLOCKED_FUTURE_FILL"}
    if available < at:
        return {**result, "status": "DATA_BLOCKED_FILL_CLOCK_ORDER"}
    refs = []
    for row in tape:
        if (not isinstance(row, dict) or any(row.get(key) != fill.get(key) for key in ("market", "quote_mint", "unit"))
                or not row.get("source_ref") or row.get("kind") not in {"AMM_MARGINAL_RESERVE_REFERENCE", "MID_QUOTE"}):
            continue
        try:
            stamp, receipt = clock(row.get("reference_at")), clock(row.get("available_at"))
            value = decimal(row.get("reference_price"))
            if value <= 0 or receipt < stamp:
                continue
        except ValueError:
            continue
        refs.append((stamp, receipt, value, row))
    def reference(target):
        candidates = [r for r in refs if r[0] <= target and r[1] <= target and r[1] <= now and target-r[0] <= max_reference_age]
        return max(candidates, key=lambda r: (r[0], r[1], str(r[3]['source_ref']))) if candidates else None
    anchor = reference(at)
    sign = Decimal(1 if fill['side'] == 'buy' else -1)
    cost = None
    if fill.get('external_costs_complete') is True and fill.get('external_cost_quote_raw') is not None:
        cost = decimal(fill['external_cost_quote_raw'])
        if cost < 0:
            raise ValueError('nonnegative reconciled external cost required')
        result.update(external_costs_complete=True, external_cost_quote_raw=str(cost))
    result['missing_costs'] = [] if cost is not None else fill.get('missing_costs', ['external execution costs'])
    result['reference_at_confirmation'] = str(anchor[2]) if anchor else None
    result['reference_source_ref'] = anchor[3]['source_ref'] if anchor else None
    result['reference_kind'] = anchor[3].get('kind') if anchor else None
    result['effective_reference_deviation_quote'] = str(sign*(price-anchor[2])*qty) if anchor else None
    for label in ('decision', 'arrival'):
        target = fill.get(label + '_at')
        try:
            target = clock(target)
            selected = reference(target) if target <= at else None
        except ValueError:
            selected = None
        gross = sign*(price-selected[2])*qty if selected else None
        result[label + '_reference'] = str(selected[2]) if selected else None
        result[label + '_shortfall_quote_gross'] = str(gross) if gross is not None else None
        result[label + '_shortfall_quote_after_external_cost'] = str(gross+cost) if gross is not None and cost is not None else None
    for horizon in horizons:
        if type(horizon) is not int or horizon <= 0:
            raise ValueError('positive integer horizon required')
        if at + horizon > now:
            row = {'status': 'NOT_MATURE', 'reference_move_quote': None, 'net_markout_quote': None}
        else:
            later = reference(at + horizon)
            if not anchor or not later:
                row = {'status': 'DATA_BLOCKED_REFERENCE_TAPE', 'reference_move_quote': None, 'net_markout_quote': None}
            elif anchor[3].get('kind') != later[3].get('kind'):
                row = {'status': 'DATA_BLOCKED_REFERENCE_KIND_CHANGE', 'reference_move_quote': None, 'net_markout_quote': None}
            else:
                move, gross = sign*(later[2]-anchor[2])*qty, sign*(later[2]-price)*qty
                row = {'status': 'MEASURED' if cost is not None else 'GROSS_ONLY_COSTS_INCOMPLETE',
                       'reference_move_quote': str(move), 'gross_markout_quote': str(gross),
                       'net_markout_quote': str(gross-cost) if cost is not None else None,
                       'realized_reference_spread_quote': str(sign*(price-anchor[2])*qty-move),
                       'future_reference_source_ref': later[3]['source_ref']}
        result['horizons'][str(horizon)] = row
    if any(row['status'] == 'MEASURED' for row in result['horizons'].values()):
        result['status'] = 'MEASURED_AVAILABLE_HORIZONS'
    return result


def episode_measurements(episode, *, as_of):
    tape = [row['execution_reference'] for row in episode.market_observations
            if isinstance(row, dict) and isinstance(row.get('execution_reference'), dict)]
    results = []
    for attempt in episode.execution_attempts:
        try:
            fill = normalize_attempt(episode.token, attempt)
            result = analyze_fill(fill, tape, as_of=as_of) if fill['status'] == 'NORMALIZED_CONFIRMED_FILL' else fill
        except (ValueError, TypeError, KeyError):
            result = {'status': 'DATA_BLOCKED_MEASUREMENT_SCHEMA', 'real_fill': None}
        results.append(result)
    return {'schema': 'v1', 'as_of': as_of, 'attempts': len(results), 'results': results,
            'authority': 'MEASUREMENT_ONLY', 'economic_edge': 'UNMEASURED'}
