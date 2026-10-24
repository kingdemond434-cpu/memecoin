"""Our local typed decision layer: transparent rules and validated classifiers.

This is not a reproduction of Jev's weights or proprietary architecture. It
uses a small hashed-text logistic model, temperature calibration, chronological
holdouts and explicit abstention. It never creates market data or authorises
capital. Artifacts contain JSON numbers, never executable pickles.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

SCHEMA = "local-typed-decisions-v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def vectorizer(dim):
    return HashingVectorizer(n_features=dim, alternate_sign=False, ngram_range=(1, 2),
                             lowercase=True, norm="l2", strip_accents="unicode")


def probabilities(scores, temperature):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 1: scores = np.column_stack((np.zeros(len(scores)), scores))
    scores = scores / temperature
    scores -= scores.max(axis=1, keepdims=True)
    values = np.exp(scores)
    return values / values.sum(axis=1, keepdims=True)


def train(rows, *, question_id, primitive, options, now=None, dimensions=4096,
          min_train=200, min_calibration=60, min_test=60):
    now = time.time() if now is None else now
    if primitive not in {"choice", "score", "noul"}: raise ValueError("invalid primitive")
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", question_id): raise ValueError("invalid question id")
    if len(options) < 2 or len(options) > 16 or len(set(options)) != len(options): raise ValueError("invalid options")
    if primitive == "noul" and options != ["no", "yes"]: raise ValueError("noul options must be no, yes")
    if not 128 <= dimensions <= 16384: raise ValueError("invalid feature dimensions")
    rows = sorted(rows, key=lambda r: float(r["decision_at"]))
    if len(rows) < min_train + min_calibration + min_test: raise ValueError("insufficient labeled data")
    for r in rows:
        if r["label"] not in options or not r.get("group_id") or not r.get("source_ref"):
            raise ValueError("labels require declared options, independent groups and source evidence")
        decision, observed = float(r["decision_at"]), float(r["label_observed_at"])
        if not math.isfinite(decision + observed) or not decision < observed <= now:
            raise ValueError("label timing is invalid or not yet observed")
        if not isinstance(r["text"], str) or len(r["text"]) > 16000: raise ValueError("invalid text")
    cut1, cut2 = float(rows[int(len(rows) * .6)]["decision_at"]), float(rows[int(len(rows) * .8)]["decision_at"])
    train_rows = [r for r in rows if float(r["decision_at"]) < cut1 and float(r["label_observed_at"]) < cut1]
    cal_rows = [r for r in rows if cut1 <= float(r["decision_at"]) < cut2 and float(r["label_observed_at"]) < cut2]
    test_rows = [r for r in rows if float(r["decision_at"]) >= cut2]
    for subset, minimum in zip((train_rows, cal_rows, test_rows), (min_train, min_calibration, min_test)):
        if len(subset) < minimum: raise ValueError("insufficient data after purging unresolved labels at split boundaries")
    for earlier, later in ((train_rows, cal_rows), (train_rows, test_rows), (cal_rows, test_rows)):
        if {r["group_id"] for r in earlier} & {r["group_id"] for r in later}:
            raise ValueError("campaign/market groups overlap chronological folds")
        if {r["text"] for r in earlier} & {r["text"] for r in later}:
            raise ValueError("duplicate input text crosses chronological folds")
    labels = lambda subset: np.array([options.index(r["label"]) for r in subset])
    y_train, y_cal, y_test = map(labels, (train_rows, cal_rows, test_rows))
    if any(np.sum(y_train == i) < 10 for i in range(len(options))):
        raise ValueError("at least ten training labels per option are required")
    v = vectorizer(dimensions)
    x_train, x_cal, x_test = [v.transform([r["text"] for r in subset]) for subset in (train_rows, cal_rows, test_rows)]
    model = LogisticRegression(C=1.0, max_iter=500, random_state=0).fit(x_train, y_train)
    cal_scores = model.decision_function(x_cal)
    candidates = (.5, .75, 1., 1.5, 2., 3., 5.)
    temperature = min(candidates, key=lambda t: log_loss(y_cal, probabilities(cal_scores, t), labels=list(range(len(options)))))
    p_test = probabilities(model.decision_function(x_test), temperature)
    prior = (np.bincount(y_train, minlength=len(options)) + 1) / (len(y_train) + len(options))
    baseline = log_loss(y_test, np.tile(prior, (len(y_test), 1)), labels=list(range(len(options))))
    measured = log_loss(y_test, p_test, labels=list(range(len(options))))
    skill = 1 - measured / baseline
    bins = []
    confidence = p_test.max(axis=1)
    correct = p_test.argmax(axis=1) == y_test
    for lo in (.0, .5, .6, .7, .8, .9):
        hi = .5 if lo == 0 else lo + .1
        selected = (confidence >= lo) & (confidence < hi if hi < 1 else confidence <= hi)
        n = int(selected.sum())
        accuracy = float(correct[selected].mean()) if n else None
        lower = None
        if n:
            z = 1.96
            center = accuracy + z * z / (2 * n)
            spread = z * math.sqrt(accuracy * (1 - accuracy) / n + z * z / (4 * n * n))
            lower = (center - spread) / (1 + z * z / n)
        bins.append({"lo": lo, "hi": hi, "samples": n, "accuracy": accuracy, "accuracy_lower_95": lower})
    artifact = {"schema": SCHEMA, "question_id": question_id, "primitive": primitive, "options": options,
                "dimensions": dimensions, "coefficients": model.coef_.tolist(), "intercept": model.intercept_.tolist(),
                "temperature": temperature, "created_at": now, "training_end": cut1, "calibration_end": cut2,
                "dataset_sha256": digest(rows), "counts": [len(train_rows), len(cal_rows), len(test_rows)],
                "validation": {"status": "PASSED" if skill > 0 else "FAILED", "log_loss": measured,
                               "prior_log_loss": baseline, "log_loss_skill": skill, "bins": bins,
                               "trading_value": "UNMEASURED", "real_fill_evidence": False}}
    artifact["sha256"] = digest(artifact)
    return artifact


def rules(text):
    """Immediately usable deterministic annotations, not invented probabilities."""
    lowered = text.lower()
    flags = {"insider_claim": bool(re.search(r"\binsider\b|inside information|dev wallet|whale wallet|内幕|инсайд", lowered)),
             "guaranteed_return_claim": bool(re.search(r"guaranteed|risk.free|100[x×]|1000[x×]|必赚|稳赚", lowered)),
             "wallet_secret_request": bool(re.search(r"seed phrase|private key|recovery phrase|助记词", lowered)),
             "launch_claim": bool(re.search(r"launch|presale|fairlaunch|pump.fun|发射|预售", lowered)),
             "settlement_ambiguity": bool(re.search(r"discretion|ambiguous|consensus of credible|may remain open", lowered))}
    return {"status": "RULE_BASED_UNCALIBRATED", "flags": flags,
            "probabilities": None, "confidence": None, "trade_authority": False}


class LocalDecisions:
    def __init__(self, artifact_dir: Path | None = None):
        self.artifact_dir = Path(artifact_dir) if artifact_dir else None
        self.cache = {}

    def report(self):
        models = {}
        for name in ("source_quality", "launch_growth", "prediction_outcome"):
            try:
                artifact = self._load(name)
                models[name] = ({"status": artifact["validation"]["status"], "sha256": artifact["sha256"],
                                 "counts": artifact["counts"], "trading_value": "UNMEASURED"} if artifact else {"status": "DATA_BLOCKED_NO_EVIDENCED_MODEL"})
            except (ValueError, KeyError, TypeError, OSError):
                models[name] = {"status": "DATA_BLOCKED_ARTIFACT_INVALID"}
        return {"mode": "LOCAL_SHADOW", "paid_api_calls": 0, "models": models,
                "primitives": ["choice", "score", "noul"], "trade_authority": False}

    def _load(self, question_id):
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", question_id): raise ValueError("invalid question id")
        if not self.artifact_dir: return None
        path = self.artifact_dir / (question_id + ".json")
        if not path.exists(): return None
        stamp = path.stat().st_mtime_ns
        if question_id in self.cache and self.cache[question_id][0] == stamp: return self.cache[question_id][1]
        if path.stat().st_size > 6_000_000: raise ValueError("local model exceeds size limit")
        artifact = json.loads(path.read_text(encoding="utf-8"))
        checksum = artifact.pop("sha256")
        if digest(artifact) != checksum or artifact["schema"] != SCHEMA or artifact["question_id"] != question_id:
            raise ValueError("local decision artifact integrity failed")
        artifact["sha256"] = checksum
        dim = artifact["dimensions"]
        if not isinstance(dim, int) or not 128 <= dim <= 16384: raise ValueError("invalid model dimensions")
        weights, intercept = np.array(artifact["coefficients"]), np.array(artifact["intercept"])
        n = len(artifact["options"])
        if not 2 <= n <= 16 or weights.shape != (1 if n == 2 else n, dim) or intercept.shape != (len(weights),):
            raise ValueError("invalid local model shape")
        if not np.isfinite(weights).all() or not np.isfinite(intercept).all() or not .1 <= artifact["temperature"] <= 10:
            raise ValueError("invalid local model numbers")
        self.cache[question_id] = (stamp, artifact)
        return artifact

    def evaluate(self, text, questions):
        if not isinstance(text, str) or len(text) > 16000 or len(questions) > 16:
            raise ValueError("decision request exceeds local bounds")
        answers = {}
        for question_id in questions:
            artifact = self._load(question_id)
            if not artifact or artifact["validation"]["status"] != "PASSED":
                answers[question_id] = {"status": "DATA_BLOCKED", "reason": "no validated local model for this exact question", "trade_authority": False}
                continue
            x = vectorizer(artifact["dimensions"]).transform([text])
            scores = np.asarray(x @ np.array(artifact["coefficients"]).T).reshape(1, -1) + np.array(artifact["intercept"])
            if scores.shape[1] == 1: scores = np.column_stack((np.zeros(1), scores[:, 0]))
            p = probabilities(scores, artifact["temperature"])[0]
            best = int(p.argmax())
            support = next((b for b in artifact["validation"]["bins"] if b["lo"] <= p[best] <= b["hi"]), None)
            accepted = bool(support and support["samples"] >= 20 and support["accuracy_lower_95"] >= .6 and p[best] >= .7)
            answer = {"status": "SHADOW_SIGNAL" if accepted else "ABSTAIN", "primitive": artifact["primitive"],
                      "choice": artifact["options"][best] if accepted else None,
                      "probabilities": dict(zip(artifact["options"], map(float, p))),
                      "confidence": {"holdout_support": support, "calibration_is_domain_specific": True},
                      "model_sha256": artifact["sha256"], "trade_authority": False}
            if artifact["primitive"] == "noul": answer["noul"] = float(p[artifact["options"].index("yes")])
            if artifact["primitive"] == "score": answer["score"] = float(sum(i * q for i, q in enumerate(p)))
            answers[question_id] = answer
        return {"schema": SCHEMA, "rules": rules(text), "answers": answers, "paid_api_calls": 0, "trade_authority": False}


_engines = {}


def annotate_public_text(text):
    """Optional research annotation; failures cannot suppress source evidence."""
    root = os.getenv("MEMECOIN_LOCAL_DECISION_DIR", "data/research/local_decisions")
    engine = _engines.setdefault(root, LocalDecisions(Path(root)))
    try:
        return engine.evaluate(str(text)[:16000], ["source_quality", "launch_growth", "prediction_outcome"])
    except (ValueError, KeyError, TypeError, OSError) as exc:
        return {"schema": SCHEMA, "status": "DATA_BLOCKED", "reason": type(exc).__name__,
                "rules": rules(str(text)[:16000]), "paid_api_calls": 0, "trade_authority": False}
