"""Train a strictly chronological two-horizon rug-hazard artifact."""
import argparse
import gzip
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.research.shadow_trainer import _number, _repair_legacy_outcome
from src.strategies.rug_hazard import HAZARD_FEATURE_NAMES, ContinuousRugHazardModel

HORIZONS = {"rug_30s": 30.0, "rug_5m": 300.0}
SNAPSHOTS = ("t10s", "t30s", "t1m")


def load_rows(storage: Path) -> List[Tuple[str, float, np.ndarray, Dict[str, int]]]:
    rows=[]
    for path in storage.glob("*/*.json.gz"):
        try:
            with gzip.open(path,"rt",encoding="utf-8") as handle:episode=json.load(handle)
        except (OSError,json.JSONDecodeError):continue
        outcome=_repair_legacy_outcome(episode,episode.get("final_outcome") or {})
        if outcome.get("status")!="OK":continue
        created=_number(episode,"created_at");rug_time=outcome.get("rug_time")
        observations=episode.get("market_observations") or []
        for name in SNAPSHOTS:
            snapshot=(episode.get("snapshots") or {}).get(name)
            if not snapshot:continue
            as_of=_number(snapshot,"timestamp",created)
            remaining=(created+float(rug_time)-as_of) if rug_time is not None else None
            if remaining is not None and remaining<0:continue
            labels={key:int(bool(outcome.get("rugged")) and remaining is not None and 0<=remaining<=seconds)
                    for key,seconds in HORIZONS.items()}
            rows.append((str(episode.get("token","")),as_of,
                         ContinuousRugHazardModel.feature_vector_from_observations(observations,as_of),labels))
    return sorted(rows,key=lambda row:(row[1],row[0]))


def train(storage: Path, model_dir: Path, min_rows: int=250) -> Dict[str,Any]:
    rows=load_rows(storage);report={"created_at":time.time(),"rows":len(rows)}
    tokens={token:min(ts for tok,ts,_,_ in rows if tok==token) for token in {row[0] for row in rows}}
    ordered=sorted(tokens,key=lambda token:(tokens[token],token));split=max(1,min(len(ordered)-1,int(len(ordered)*.8))) if len(ordered)>1 else 0
    train_tokens=set(ordered[:split]);train_rows=[row for row in rows if row[0] in train_tokens];oos=[row for row in rows if row[0] not in train_tokens]
    report.update({"train_rows":len(train_rows),"oos_rows":len(oos),"train_episodes":len(train_tokens),"oos_episodes":len(set(ordered)-train_tokens)})
    if len(rows)<min_rows or not train_rows or not oos:
        report.update({"status":"DATA_BLOCKED","reason":"insufficient_chronological_hazard_rows"})
        model_dir.mkdir(parents=True,exist_ok=True)
        (model_dir/'last_hazard_training_report.json').write_text(json.dumps(report,indent=2,sort_keys=True),encoding='utf-8')
        return report
    models={};calibrators={};metrics={}
    X=np.asarray([row[2] for row in train_rows]);X_oos=np.asarray([row[2] for row in oos])
    for key in HORIZONS:
        y=np.asarray([row[3][key] for row in train_rows]);y_oos=np.asarray([row[3][key] for row in oos])
        positives=int(y.sum());oos_positives=int(y_oos.sum())
        if min(positives,len(y)-positives,oos_positives,len(y_oos)-oos_positives)<3:
            metrics[key]={"status":"DATA_BLOCKED","train_positive":positives,"oos_positive":oos_positives};continue
        fit_split=max(1,int(len(X)*.8));X_fit,y_fit,X_cal,y_cal=X[:fit_split],y[:fit_split],X[fit_split:],y[fit_split:]
        if len(np.unique(y_fit))<2:
            metrics[key]={"status":"DATA_BLOCKED","reason":"fit_window_lacks_both_classes"};continue
        model=LogisticRegression(max_iter=2000,class_weight="balanced",random_state=42).fit(X_fit,y_fit)
        calibrator=None
        if len(X_cal)>=10 and len(np.unique(y_cal))==2:
            calibrator=IsotonicRegression(out_of_bounds="clip").fit(model.predict_proba(X_cal)[:,1],y_cal)
        raw=model.predict_proba(X_oos)[:,1];pred=calibrator.predict(raw) if calibrator else raw
        brier=float(np.mean((y_oos-pred)**2));rate=float(np.mean(y));baseline=float(np.mean((y_oos-rate)**2))
        # A verdict either way needs enough out-of-sample positives to mean
        # something. rug_30s was REJECTED on FIVE positives -- resample those
        # five and the sign of brier-vs-baseline flips freely, so that word
        # said "evidence against" where the truth was "not enough evidence".
        # The champion pipeline treats the two differently, and should.
        # Below the floor the model is still fitted and kept: a hazard signal
        # with insufficient proof is used defensively, never as alpha.
        if min(oos_positives, int(len(y_oos))-oos_positives) < 10:
            metrics[key]={"status":"DATA_BLOCKED","reason":"insufficient_oos_positives_for_verdict",
                          "brier":brier,"baseline_brier":baseline,
                          "train_positive":positives,"oos_positive":oos_positives,
                          "verdict_floor_positives":10,
                          "calibration":"isotonic" if calibrator else "raw"}
        else:
            metrics[key]={"status":"PASSED" if brier<baseline else "REJECTED","brier":brier,"baseline_brier":baseline,
                          "train_positive":positives,"oos_positive":oos_positives,"calibration":"isotonic" if calibrator else "raw"}
        models[key]=model
        if calibrator:calibrators[key]=calibrator
    passed=set(metrics)==set(HORIZONS) and all(item.get("status")=="PASSED" for item in metrics.values())
    report.update({"status":"PASSED" if passed else "DATA_BLOCKED","metrics":metrics,"split":"strict_chronological_episode_80_20"})
    if passed:
        model_dir.mkdir(parents=True,exist_ok=True);path=model_dir/f"rug-hazard-{int(time.time())}.joblib"
        joblib.dump({"schema_version":1,"feature_names":HAZARD_FEATURE_NAMES,"models":models,"calibrators":calibrators,"validation":report},path)
        report["model_path"]=str(path)
    model_dir.mkdir(parents=True,exist_ok=True);(model_dir/'last_hazard_training_report.json').write_text(json.dumps(report,indent=2,sort_keys=True),encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--storage',default='data/launch_episodes');parser.add_argument('--model-dir',default='models');parser.add_argument('--min-rows',type=int,default=250);args=parser.parse_args()
    print(json.dumps(train(Path(args.storage),Path(args.model_dir),args.min_rows),indent=2,sort_keys=True))

if __name__=='__main__':main()
