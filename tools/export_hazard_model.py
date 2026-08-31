"""Export the promoted hazard artifact into a form Rust can evaluate.

Native inference exists so the T0 path does not leave Rust to ask Python
for a probability. The danger it introduces is a SECOND model that can
disagree with the one the promotion gate validated -- which is exactly why
``decide.rs`` says inference deliberately does not live there.

This removes that danger rather than accepting it. Nothing is retrained and
no structure is reimplemented from a description: the exact fitted
parameters are lifted out of the promoted joblib and written as data, so
the Rust evaluator is an arithmetic replay of the same numbers rather than
a second model that happens to agree.

Only the hazard heads are exported, because only they have passed. The
return model's last report reads REJECTED and has no artifact, so there is
nothing to port and porting it would mean inventing one.

The shape being exported, per head:

    p_raw   = sigmoid(intercept + coef . features)
    p_final = isotonic(p_raw)          piecewise-linear, clipped at the ends

Both models are LogisticRegression and both calibrators are
IsotonicRegression with ``out_of_bounds="clip"``, which is why an exact
port is possible at all -- a gradient-boosted forest would be a far larger
surface for the two implementations to drift apart on.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict

#: The artifact schema this exporter understands. A bump means the shape
#: changed and the export must be re-read rather than assumed.
SUPPORTED_SCHEMA = 1


def export(artifact: Dict[str, Any]) -> Dict[str, Any]:
    """Fitted parameters as plain data, with the hash of what produced it."""
    if artifact.get("schema_version") != SUPPORTED_SCHEMA:
        raise ValueError(
            f"unsupported hazard artifact schema {artifact.get('schema_version')}; "
            f"this exporter understands {SUPPORTED_SCHEMA}")
    validation = artifact.get("validation") or {}
    if validation.get("status") != "PASSED":
        raise ValueError(
            "refusing to export an artifact that did not pass validation "
            f"(status={validation.get('status')!r}); native inference must "
            "replay the promoted model, not an arbitrary one")

    features = list(artifact.get("feature_names") or ())
    heads: Dict[str, Any] = {}
    for name, model in (artifact.get("models") or {}).items():
        coef = [float(value) for value in model.coef_[0]]
        if len(coef) != len(features):
            raise ValueError(
                f"head {name} has {len(coef)} coefficients for "
                f"{len(features)} features; the ordering cannot be trusted")
        head: Dict[str, Any] = {
            "intercept": float(model.intercept_[0]),
            "coef": coef,
        }
        calibrator = (artifact.get("calibrators") or {}).get(name)
        if calibrator is not None:
            thresholds = getattr(calibrator, "X_thresholds_", None)
            values = getattr(calibrator, "y_thresholds_", None)
            if thresholds is None or values is None:
                raise ValueError(f"calibrator {name} exposes no thresholds")
            # "clip" is the only out-of-bounds policy this export encodes,
            # and the Rust side clamps identically. Any other policy would
            # silently change behaviour outside the fitted range.
            if getattr(calibrator, "out_of_bounds", "clip") != "clip":
                raise ValueError(
                    f"calibrator {name} uses out_of_bounds="
                    f"{calibrator.out_of_bounds!r}; only 'clip' is ported")
            head["calibrator"] = {
                "x": [float(value) for value in thresholds],
                "y": [float(value) for value in values],
            }
        heads[name] = head

    return {
        "schema": "hazard_native_v1",
        # The exact ordering the coefficients were fitted against. Rust
        # indexes by position, so a reordering here is a silent wrong
        # answer rather than an error -- it is written down and asserted.
        "feature_names": features,
        "heads": heads,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--out", type=Path,
                        default=Path("models/hazard_native.json"))
    args = parser.parse_args()

    candidates = sorted(glob.glob(str(args.model_dir / "rug-hazard-*.joblib")))
    if not candidates:
        print("no hazard artifact to export")
        return 1
    import joblib

    payload = export(joblib.load(candidates[-1]))
    payload["source_artifact"] = Path(candidates[-1]).name
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"exported {len(payload['heads'])} head(s) from "
          f"{payload['source_artifact']} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
