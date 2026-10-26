"""Selection and dependence diagnostics adapted from the owner's quant methods.

These are measurements, not promotion gates. DSR is conditional on the supplied
trial universe and IID moment approximation; it is not a probability of profit.
An overlap count is only an upper bound on independence, never a measured ESS.
"""
from __future__ import annotations

import math
import statistics


def _finite(value):
    if isinstance(value, bool):
        raise ValueError('finite numeric observation required')
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('finite numeric observation required')
    return number


def effective_sample(intervals, event_ids):
    """All rows retained; report an interval-scheduling independence ceiling.

    Maximum disjoint half-open intervals and distinct events are necessary,
    insufficient conditions for independent evidence. Serial/common-factor
    dependence is deliberately unmeasured here, rather than assumed absent.
    """
    if len(intervals) != len(event_ids):
        raise ValueError('one interval and event identity per observation required')
    parsed = []
    for interval, identity in zip(intervals, event_ids):
        start, stop = map(_finite, interval)
        if start < 0 or stop <= start or not isinstance(identity, str) or not identity:
            raise ValueError('positive interval and explicit event identity required')
        parsed.append((stop, start))
    previous, disjoint = -math.inf, 0
    for stop, start in sorted(parsed):
        if start >= previous:
            previous, disjoint = stop, disjoint + 1
    return {'raw_rows': len(parsed), 'distinct_events': len(set(event_ids)),
        'maximum_disjoint_windows': disjoint,
        'independent_sample_upper_bound': min(disjoint, len(set(event_ids))),
        'effective_sample_size': None,
        'unmeasured_dependence': ['serial', 'common_factor', 'entity', 'regime'],
        'status': 'DEPENDENCE_PARTLY_MEASURED' if parsed else 'NO_OBSERVATIONS'}


def deflated_sharpe_ratio(values, *, trial_count=None, trial_sharpe_variance=None,
                         selection_evidence_ref=None):
    """Non-annualized per-observation SR/PSR and selection-adjusted PSR (DSR).

    Uses unbiased sample skew and Pearson kurtosis, matching the quant donor.
    The variance must describe the observed selection universe, not these
    winning returns. Missing trials/variance never mean one trial/zero cost.
    """
    rows = [_finite(value) for value in values]
    result = {'observations': len(rows), 'sharpe_per_observation': None,
        'probabilistic_sharpe_ratio_iid': None, 'deflated_sharpe_ratio_iid': None,
        'expected_maximum_trial_sharpe': None, 'trial_count': trial_count,
        'trial_sharpe_variance': trial_sharpe_variance,
        'selection_evidence_ref': selection_evidence_ref,
        'status': 'INSUFFICIENT_MOMENTS', 'authority': 'MEASUREMENT_ONLY',
        'assumptions': 'conditional IID moment approximation; dependence not certified',
        'economic_edge': 'UNPROVEN'}
    supplied = (trial_count is not None, trial_sharpe_variance is not None, selection_evidence_ref is not None)
    if any(supplied):
        if (not all(supplied) or type(trial_count) is not int or trial_count < 1
                or not isinstance(selection_evidence_ref, str) or not selection_evidence_ref.strip()):
            raise ValueError('complete trial count, variance and selection provenance required')
        variance = _finite(trial_sharpe_variance)
        if variance < 0 or (trial_count > 1 and variance == 0):
            raise ValueError('positive observed cross-trial variance required for multiple trials')
        result['trial_sharpe_variance'] = variance
    if len(rows) < 4:
        return result
    n = len(rows)
    mean = statistics.fmean(rows)
    centered = [value-mean for value in rows]
    # Rescale central moments to avoid gratuitous powers overflowing.
    scale = max(map(abs, centered))
    if scale == 0:
        return {**result, 'status': 'ZERO_VARIANCE_UNDEFINED'}
    centered = [value/scale for value in centered]
    m2, m3, m4 = [statistics.fmean(value**power for value in centered) for power in (2, 3, 4)]
    sr = (mean/scale)/math.sqrt(m2*n/(n-1))
    skew = math.sqrt(n*(n-1))/(n-2)*m3/m2**1.5
    kurtosis = 3+(n-1)/((n-2)*(n-3))*((n+1)*(m4/m2**2-3)+6)
    denominator = 1-skew*sr+(kurtosis-1)/4*sr**2
    if not math.isfinite(denominator) or denominator <= 0:
        return {**result, 'status': 'MOMENT_APPROXIMATION_UNDEFINED'}
    normal = statistics.NormalDist()
    multiplier = math.sqrt((n-1)/denominator)
    result.update(sharpe_per_observation=sr,
        probabilistic_sharpe_ratio_iid=normal.cdf(sr*multiplier),
        status='SELECTION_UNMEASURED')
    if all(supplied):
        gamma = .5772156649015329
        # inv_cdf(1-p) == -inv_cdf(p), without rounding large N to 1.
        expected = (math.sqrt(variance)*(-(1-gamma)*normal.inv_cdf(1/trial_count)
            -gamma*normal.inv_cdf(1/(trial_count*math.e)))) if trial_count > 1 else 0.
        result.update(expected_maximum_trial_sharpe=expected,
            deflated_sharpe_ratio_iid=normal.cdf((sr-expected)*multiplier),
            status='CONDITIONAL_SELECTION_MEASUREMENT')
    return result


def markout_diagnostics(observations, **selection):
    """Exactly one cost-complete reference markout per fill identity/window."""
    unique = {}
    for row in observations:
        key = row['fill_id']
        if key in unique and unique[key] != row:
            raise ValueError('conflicting duplicate fill markout')
        unique[key] = row
    rows = list(unique.values())
    return {'scope': 'cost-complete marginal reference markouts; not realized strategy returns',
        'geometry': effective_sample([row['interval'] for row in rows], [row['market'] for row in rows]),
        'selection': deflated_sharpe_ratio([row['net_return'] for row in rows], **selection),
        'duplicate_observations': len(observations)-len(rows), 'realized_profit': None}
