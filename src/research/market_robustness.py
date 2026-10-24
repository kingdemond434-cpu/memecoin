"""Quant-derived historical robustness diagnostics, without capital authority.

Source concepts: quant@921ed183, libs/validation/cpcv.py and reality_check.py.
Use actual label windows for purge and time-based embargo; reuse bootstrap
buffers to keep the work bounded. These diagnostics supplement the existing
candidate gauntlet and never replace its untouched forward requirements.
"""
from itertools import combinations
import math
import numpy as np


def cpcv_splits(decision_at, label_end, *, groups=6, test_groups=2, embargo_seconds=0):
    start, end = np.asarray(decision_at, dtype=float), np.asarray(label_end, dtype=float)
    if (start.ndim != 1 or start.shape != end.shape or not 4 <= groups <= 8
            or not 1 <= test_groups < groups or len(start) < groups
            or math.comb(groups, test_groups) > 70 or not 0 <= embargo_seconds <= 86400 * 30):
        raise ValueError("invalid bounded CPCV configuration")
    if len(start) ** 2 * math.comb(groups, test_groups) > 10_000_000:
        raise ValueError("CPCV overlap work budget exceeded")
    if not np.isfinite(start).all() or not np.isfinite(end).all() or np.any(end <= start) or np.any(np.diff(start) <= 0):
        raise ValueError("finite increasing decisions and later label windows required")
    chunks = np.array_split(np.arange(len(start)), groups)
    splits = []
    for selected in combinations(range(groups), test_groups):
        test = np.concatenate([chunks[i] for i in selected])
        mask = np.ones(len(start), dtype=bool)
        mask[test] = False
        # Every held-out interval purges any intersecting training label.
        # Embargo begins after its outcome becomes available, not its entry.
        for index in test:
            mask &= (end < start[index]) | (start > end[index] + embargo_seconds)
        train = np.flatnonzero(mask)
        splits.append({"train": train.tolist(), "test": test.tolist()})
    return {"status": "SPLITS_ONLY", "splits": splits, "n_splits": len(splits),
            "authority": "HISTORICAL_DIAGNOSTIC_ONLY", "forward_evidence": False}


def reality_check(strategy_returns, control_returns, *, declared_trials, evidence_refs,
                  costs_measured, iterations=1000, mean_block=7, seed=0):
    matrix, control = np.asarray(strategy_returns, dtype=float), np.asarray(control_returns, dtype=float)
    if matrix.ndim != 2 or control.shape != (len(matrix),): raise ValueError("paired daily return matrix required")
    n, k = matrix.shape
    if n < 30: return {"status": "DATA_BLOCKED", "reason": "30 daily portfolio observations required", "promotion_authority": False}
    if (n > 5000 or not 1 <= k <= 128 or not 100 <= iterations <= 10000
            or not 1 <= mean_block <= 30 or n * k * iterations > 20_000_000):
        raise ValueError("reality-check work budget exceeded")
    if not costs_measured or len(evidence_refs) != n or not all(evidence_refs):
        raise ValueError("native after-cost evidence required for every daily row")
    if declared_trials != k:
        return {"status": "DATA_BLOCKED", "reason": "all preregistered family trials must be represented", "promotion_authority": False}
    if not np.isfinite(matrix).all() or not np.isfinite(control).all() or np.any(matrix <= -1) or np.any(control <= -1):
        raise ValueError("invalid/ruin returns require separate survival analysis")
    values = np.log1p(matrix) - np.log1p(control[:, None])
    means = values.mean(axis=0)
    observed = max(0., float(means.max()))
    centered = values - means
    rng = np.random.default_rng(seed)
    buffer = np.empty_like(centered)
    indices = np.empty(n, dtype=np.int64)
    exceed = 0
    for _ in range(iterations):
        indices[0] = rng.integers(n)
        for t in range(1, n):
            indices[t] = rng.integers(n) if rng.random() < 1 / mean_block else (indices[t - 1] + 1) % n
        np.take(centered, indices, axis=0, out=buffer, mode="clip")
        exceed += max(0., float(buffer.mean(axis=0).max())) >= observed
    p_value = (exceed + 1) / (iterations + 1)
    return {"status": "RESEARCH_REJECT_NULL" if observed > 0 and p_value <= .05 else "NO_ESTABLISHED_FAMILY_EDGE",
            "p_value": p_value, "daily_observations": n, "family_trials": k,
            "mean_block": mean_block, "iterations": iterations,
            "best_mean_paired_log_growth": observed, "method": "White_centered_stationary_block_reality_check",
            "promotion_authority": False, "untouched_forward_evidence": False}
