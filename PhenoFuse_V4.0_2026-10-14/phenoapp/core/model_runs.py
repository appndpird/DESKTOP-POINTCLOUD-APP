"""
Where trained models go (PhenoFuse 4.0) and the top-three mean ensemble of a training run.

    <root>/Models/Height models/<run>/      <root> = the flight folder (single-flight training) or the dataset folder
    <root>/Models/Biomass models/<run>/     <run>  = <YYYY-MM-DD_HHMM>_<target>_<modalities>

Each run folder holds the predictions table, the results JSON, one .joblib per family x learner (or per height
model), the top-three mean ensemble (`top3_ensemble.joblib`, members fitted on all plots, chosen by held-out
RMSE) and `run.json` with the inputs. Any of those files can be picked in the "Use a pretrained model" section
of the Height / Biomass tabs, on this flight or on another one.
"""
from __future__ import annotations
import os
import json
import time

KIND_DIR = {"height": "Height models", "biomass": "Biomass models"}


def models_root(root: str) -> str:
    return os.path.join(root, "Models")


def new_run_dir(root: str, kind: str, target: str, modalities, name: str = "") -> str:
    kind = "height" if kind.lower().startswith("h") else "biomass"
    tag = time.strftime("%Y-%m-%d_%H%M") + "_" + (name or target) + "_" + "+".join(sorted(m.lower() for m in modalities))
    d = os.path.join(models_root(root), KIND_DIR[kind], tag)
    os.makedirs(d, exist_ok=True)
    return d


def list_models(root: str, kind: str):
    """All .joblib files under <root>/Models/<kind>/ (newest run first)."""
    kind = "height" if kind.lower().startswith("h") else "biomass"
    base = os.path.join(models_root(root), KIND_DIR[kind])
    out = []
    if not os.path.isdir(base):
        return out
    for run in sorted(os.listdir(base), reverse=True):
        rd = os.path.join(base, run)
        if os.path.isdir(rd):
            out += [os.path.join(rd, f) for f in sorted(os.listdir(rd)) if f.endswith(".joblib")]
    return out


def top3_ensemble(results, fitted, key_fn, n=3, rmse_key="RMSE", table=None):
    """results: list of result dicts with 'cv' metrics; fitted: {key: (model, feats)}; key_fn(result) -> key.
    table: the training feature table (its medians are stored for features unusable on a new flight).
    Returns the mean-ensemble dict (core.ensembles format) of the n best by held-out RMSE, or None."""
    ok = [r for r in results if r.get("status") == "ok" and r.get("cv") and key_fn(r) in fitted]
    if not ok:
        return None
    ok = sorted(ok, key=lambda r: r["cv"].get(rmse_key, r["cv"].get("rmse", 1e99)))[:n]
    members = []
    for r in ok:
        model, feats = fitted[key_fn(r)]
        members.append({"family": r.get("family", r.get("key", "")), "learner": r.get("learner", ""), "model": model, "features": list(feats),
                        "cv": r["cv"]})
    best = ok[0]["cv"]
    medians = {}; ranges = {}
    if table is not None:
        from .ensembles import feature_medians, feature_ranges
        allf = sorted({f for m in members for f in m["features"]})
        medians = feature_medians(table, allf); ranges = feature_ranges(table, allf)
    return {"type": "mean_ensemble", "members": members, "feature_medians": medians, "feature_ranges": ranges,
            "nested_cv": {"R2": float(best.get("R2", best.get("r2", float("nan")))), "RMSE": float(best.get(rmse_key, best.get("rmse", float("nan")))),
                          "note": "score of the best member (held-out); the ensemble itself was not re-validated"}}


def save_run_meta(run_dir: str, **meta):
    meta = dict(meta); meta["created"] = time.strftime("%Y-%m-%d %H:%M")
    with open(os.path.join(run_dir, "run.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
