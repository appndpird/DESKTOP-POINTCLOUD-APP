"""
Two-stream network (PhenoApp v3 deep models): sparse 3D CNN on the plot point cloud + spectral transformer on
VNIR pixel spectra, late fusion -> plot biomass (kg/ha) or plant height (cm).

Runs in a Python environment that has torch (+ spconv for the LiDAR stream). It is launched by PhenoApp as a
subprocess (core.deep_models) but can be used on its own:

  predict:  python two_stream.py predict --data <dl_data dir> --weights <dir with two_stream_<target>_fold*.pt>
                                         --target biomass|height --out <csv> [--device auto|cuda|cpu] [--variant two_stream|lidar_only]
  train:    python two_stream.py train   --data <dl_data dir> --target biomass|height --out <dir> [--device auto] [--epochs 60]
                                         [--folds 10] [--variant two_stream|lidar_only|vnir_only]
  probe:    python two_stream.py probe   -> JSON with torch / cuda / spconv availability

Device: 'auto' = CUDA when available, else CPU. On CPU spconv uses its native algorithm (slower, same weights).
Input tensors: dl_data/index.csv + plot_<id>.npz written by core.deep_models.prepare_plot_tensors.
"""
import os, sys, json, math, argparse, time
import numpy as np

VOX = 0.02; GRID = (256, 96, 96); NPIX = 512


def probe():
    out = {"python": sys.executable, "torch": None, "cuda": False, "gpu": None, "spconv": False}
    try:
        import torch; out["torch"] = torch.__version__; out["cuda"] = bool(torch.cuda.is_available())
        if out["cuda"]: out["gpu"] = torch.cuda.get_device_name(0)
    except Exception as e:
        out["error"] = str(e)
    try:
        import spconv.pytorch  # noqa
        out["spconv"] = True
    except Exception:
        pass
    return out


def pick_device(req):
    import torch
    if req == "cuda" and not torch.cuda.is_available():
        print("[two_stream] CUDA requested but not available - using CPU", flush=True); req = "cpu"
    if req == "auto":
        req = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(req)


def build(variant, nds, device):
    import torch, torch.nn as nn
    use_spconv = variant != "vnir_only"
    if use_spconv:
        import spconv.pytorch as spconv
        algo = spconv.ConvAlgo.Native if device.type == "cpu" else None
        kw = {"algo": algo} if algo is not None else {}

        class LidarStream(nn.Module):
            def __init__(self, cin=4, w=32):
                super().__init__()
                def block(i, o, key): return spconv.SparseSequential(spconv.SubMConv3d(i, o, 3, padding=1, indice_key=key, bias=False, **kw), nn.BatchNorm1d(o), nn.ReLU())
                def down(i, o): return spconv.SparseSequential(spconv.SparseConv3d(i, o, 3, stride=2, padding=1, bias=False, **kw), nn.BatchNorm1d(o), nn.ReLU())
                self.net = spconv.SparseSequential(block(cin, w, "s1"), block(w, w, "s1"), down(w, 2 * w), block(2 * w, 2 * w, "s2"), down(2 * w, 4 * w), block(4 * w, 4 * w, "s3"), down(4 * w, 4 * w))
                self.out = nn.Sequential(nn.Linear(8 * w, 128), nn.ReLU(), nn.Dropout(0.2))
            def forward(self, coords, feats, bs):
                y = self.net(spconv.SparseConvTensor(feats, coords, spatial_shape=list(GRID), batch_size=bs)); f = y.features; b = y.indices[:, 0].long()
                mx = torch.full((bs, f.shape[1]), -1e4, device=f.device).scatter_reduce(0, b[:, None].expand_as(f), f, reduce="amax", include_self=True)
                mean = torch.zeros((bs, f.shape[1]), device=f.device).index_add_(0, b, f) / torch.bincount(b, minlength=bs).clamp(min=1)[:, None]
                return self.out(torch.cat([mx, mean], 1))
    else:
        LidarStream = None

    class SpecStream(nn.Module):
        def __init__(self, nb=172, d=64):
            super().__init__(); self.emb = nn.Sequential(nn.Linear(nb, d), nn.GELU(), nn.Linear(d, d))
            self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 128, dropout=0.1, batch_first=True), 2); self.out = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Dropout(0.2))
        def forward(self, spec): return self.out(self.enc(self.emb(spec)).mean(1))

    class TwoStream(nn.Module):
        def __init__(self):
            super().__init__(); self.variant = variant
            self.lidar = LidarStream() if variant != "vnir_only" else None; self.spec = SpecStream() if variant != "lidar_only" else None
            self.ds_emb = nn.Embedding(nds, 8); d = 128 * (2 if variant == "two_stream" else 1) + 8
            self.head = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))
        def forward(self, coords, feats, spec, ds, bs):
            parts = []
            if self.lidar is not None: parts.append(self.lidar(coords, feats, bs))
            if self.spec is not None: parts.append(self.spec(spec))
            parts.append(self.ds_emb(ds)); return self.head(torch.cat(parts, 1)).squeeze(1)
    return TwoStream().to(device)


def make_sample(z, train, rng, excl, maxpts=16000):
    import torch
    pts = z["points"].astype(np.float32); pix = z["pixels"].astype(np.float32)
    if train:
        if rng.random() < 0.5: pts[:, 0] = -pts[:, 0]; pts[:, 1] = -pts[:, 1]
        if rng.random() < 0.5: pts[:, 1] = -pts[:, 1]
        keep = rng.random(len(pts)) > rng.uniform(0, 0.3); pts = pts[keep]; pts[:, :2] += rng.normal(0, 0.01, (len(pts), 2)).astype(np.float32)
        pix = pix[rng.choice(len(pix), len(pix), replace=True)] * rng.uniform(0.9, 1.1)
    if len(pts) > maxpts:
        r_ = rng if train else np.random.default_rng(len(pts) * 7919 + int(abs(pts[:, 2].sum()) * 1000) % 100000)   # deterministic per plot at inference
        pts = pts[r_.choice(len(pts), maxpts, replace=False)]
    cx = np.clip(np.floor(pts[:, 0] / VOX + GRID[0] / 2), 0, GRID[0] - 1); cy = np.clip(np.floor(pts[:, 1] / VOX + GRID[1] / 2), 0, GRID[1] - 1)
    ch = np.clip(np.floor(np.clip(pts[:, 2], -0.1, None) / VOX + 5), 0, GRID[2] - 1)
    coords = np.stack([cx, cy, ch], 1).astype(np.int32)
    feats = np.stack([pts[:, 2], np.log1p(np.clip(pts[:, 3], 0, 20)), pts[:, 4] / 3.0, np.ones(len(pts), np.float32)], 1).astype(np.float32)
    sp = np.log10(np.clip(pix / 10000.0, 1e-4, None)); valid = (pix > 0) & ~excl[None, :]
    mu = (sp * valid).sum(1, keepdims=True) / np.maximum(valid.sum(1, keepdims=True), 1); sd = np.sqrt(((sp - mu) ** 2 * valid).sum(1, keepdims=True) / np.maximum(valid.sum(1, keepdims=True), 1)) + 1e-6
    sp = np.where(valid, (sp - mu) / sd, 0).astype(np.float32)
    return torch.from_numpy(coords), torch.from_numpy(feats), torch.from_numpy(sp)


def batches(idx, data_dir, ds_id, train, rng, excl, bs=16, device=None, tcol=None):
    import torch
    order = rng.permutation(len(idx)) if train else np.arange(len(idx))
    for s in range(0, len(order), bs):
        rows = idx.iloc[order[s:s + bs]]
        if train and len(rows) == 1: continue
        C, F, S, D, Y, I = [], [], [], [], [], []
        for k, (_, r) in enumerate(rows.iterrows()):
            z = np.load(os.path.join(data_dir, r.file)); c, f, sp = make_sample(z, train, rng, excl)
            C.append(torch.cat([torch.full((len(c), 1), k, dtype=torch.int32), c], 1)); F.append(f); S.append(sp); D.append(ds_id.get(r.dataset, 0)); Y.append(float(r[tcol]) if tcol and tcol in r and np.isfinite(r[tcol]) else np.nan); I.append(int(r.name))
        yield (torch.cat(C).to(device), torch.cat(F).to(device), torch.stack(S).to(device), torch.tensor(D).to(device), torch.tensor(Y, dtype=torch.float32), np.array(I))


def metrics(p, y):
    e = p - y; mape = float(np.mean(np.abs(e) / np.maximum(np.abs(y), 1e-9)) * 100)
    return dict(R2=float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)), RMSE=float(np.sqrt(np.mean(e ** 2))), MAE=float(np.abs(e).mean()), r=float(np.corrcoef(p, y)[0, 1]) if np.std(p) > 0 else float("nan"), accuracy=100 - mape, n=int(len(y)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["probe", "predict", "train"]); ap.add_argument("--data"); ap.add_argument("--weights"); ap.add_argument("--target", default="biomass")
    ap.add_argument("--out"); ap.add_argument("--device", default="auto"); ap.add_argument("--variant", default="two_stream"); ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--folds", type=int, default=10)
    a = ap.parse_args()
    if a.mode == "probe":
        print(json.dumps(probe())); return
    import torch, torch.nn as nn, torch.nn.functional as Fn, pandas as pd
    dev = pick_device(a.device); print(f"[two_stream] device: {dev} ({torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'})", flush=True)
    if a.variant != "vnir_only":
        try: import spconv.pytorch  # noqa
        except Exception: print("[two_stream] spconv not installed - falling back to the VNIR-only stream", flush=True); a.variant = "vnir_only"
    idx = pd.read_csv(os.path.join(a.data, "index.csv")); tcol = "biomass_kg_ha" if a.target == "biomass" else "height_cm"
    wl = np.load(os.path.join(a.data, idx.file.iloc[0]))["wavelengths"]; excl = np.zeros(len(wl), bool)
    for lo, hi in ((0, 415), (755, 770), (928, 962)): excl |= (wl >= lo) & (wl <= hi)
    TRAIN_DSETS = ["trial_b_anthesis", "trial_b_maturity", "trial_a_anthesis", "trial_a_maturity"]     # embedding ids of the bundled pretrained models (two trials x two stages, sorted)
    rng = np.random.default_rng(0); ylog = a.target == "biomass"
    if a.mode == "predict":
        files = sorted([f for f in os.listdir(a.weights) if f.startswith(f"two_stream_{a.target}_fold") and f.endswith(".pt")]) if a.variant == "two_stream" else \
                sorted([f for f in os.listdir(a.weights) if f.startswith(f"{a.variant}_{a.target}_fold") and f.endswith(".pt")])
        if not files: raise SystemExit(f"no weights for {a.variant}/{a.target} in {a.weights}")
        # target normalisation of the pretrained models (fixed from the 2025 training set)
        norm = {"biomass": (9.239603, 0.227668), "height": (84.656250, 5.091641)}[a.target]
        ds_map = {d: i for i, d in enumerate(TRAIN_DSETS)}
        # map a new dataset to the closest training embedding by stage keyword
        ds_id = {d: ds_map.get(d, 2 if "anth" in str(d).lower() else 3) for d in idx.dataset.unique()}
        preds = np.zeros((len(files), len(idx)))
        for k, f in enumerate(files):
            sd = torch.load(os.path.join(a.weights, f), map_location="cpu"); nds = int(sd["ds_emb.weight"].shape[0])
            if nds == 2:   # single-trial models: embedding 0 = anthesis, 1 = maturity
                ds_id = {d: (1 if "mat" in str(d).lower() else 0) for d in idx.dataset.unique()}
            model = build(a.variant, nds, dev)
            model.load_state_dict({n: (t.float() if t.is_floating_point() else t) for n, t in sd.items()}); model.eval()
            with torch.no_grad():
                for C, F, S, D, Y, I in batches(idx, a.data, ds_id, False, rng, excl, device=dev):
                    p = model(C, F, S, D, len(I)).cpu().numpy() * norm[1] + norm[0]; preds[k, I] = np.exp(p) if ylog else p
            print(f"[two_stream] fold model {k+1}/{len(files)} done", flush=True)
        out = idx[["dataset", "Plot_ID"] + ([c for c in ("Plot", "Variety") if c in idx.columns])].copy()
        out[f"{tcol}_pred"] = preds.mean(0); out[f"{tcol}_pred_sd_folds"] = preds.std(0)
        if tcol in idx.columns and idx[tcol].notna().any():
            m = idx[tcol].notna().to_numpy(); out.loc[m, f"{tcol}_measured"] = idx.loc[m, tcol].values
            mm = metrics(out.loc[m, f"{tcol}_pred"].to_numpy(), idx.loc[m, tcol].to_numpy(float)); print("[two_stream] vs measured:", json.dumps({k_: round(v, 3) for k_, v in mm.items()}), flush=True)
        out.to_csv(a.out, index=False); print(f"[two_stream] predictions -> {a.out}", flush=True); return
    # ---------------- train (k-fold CV on the given data, then a model on all plots)
    from sklearn.model_selection import KFold
    os.makedirs(a.out, exist_ok=True); idx = idx[idx[tcol].notna()].reset_index(drop=True); y_all = idx[tcol].to_numpy(float)
    yt = np.log(y_all) if ylog else y_all; ymu, ysd = float(yt.mean()), float(yt.std()); ds_id = {d: i for i, d in enumerate(sorted(idx.dataset.unique()))}
    def train_one(rows, epochs):
        model = build(a.variant, max(len(ds_id), 1), dev); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        steps = epochs * math.ceil(len(rows) / 16); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
        for ep in range(epochs):
            model.train()
            for C, F, S, D, Y, I in batches(rows, a.data, ds_id, True, rng, excl, device=dev, tcol=tcol):
                yn = ((torch.log(Y) if ylog else Y) - ymu) / ysd; loss = Fn.smooth_l1_loss(model(C, F, S, D, len(I)), yn.to(dev), beta=0.5)
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step(); sched.step()
        return model
    pred = np.full(len(idx), np.nan); folds = min(a.folds, len(idx))
    for fold, (tr, te) in enumerate(KFold(folds, shuffle=True, random_state=0).split(idx)):
        t = time.time(); model = train_one(idx.iloc[tr], a.epochs); model.eval()
        with torch.no_grad():
            for C, F, S, D, Y, I in batches(idx.iloc[te], a.data, ds_id, False, rng, excl, device=dev):
                p = model(C, F, S, D, len(I)).cpu().numpy() * ysd + ymu; pred[I] = np.exp(p) if ylog else p
        torch.save(model.state_dict(), os.path.join(a.out, f"{a.variant}_{a.target}_fold{fold}.pt"))
        print(f"[two_stream] fold {fold+1}/{folds}: {json.dumps({k_: round(v, 3) for k_, v in metrics(pred[te], y_all[te]).items()})} ({time.time()-t:.0f}s)", flush=True)
    m = metrics(pred, y_all); print("[two_stream] CV:", json.dumps({k_: round(v, 3) for k_, v in m.items()}), flush=True)
    out = idx[["dataset", "Plot_ID"] + [c for c in ("Plot", "Variety") if c in idx.columns]].copy(); out[f"{tcol}_measured"] = y_all; out[f"{tcol}_predcv"] = pred
    out.to_csv(os.path.join(a.out, f"{a.variant}_{a.target}_cv_predictions.csv"), index=False)
    json.dump({"variant": a.variant, "target": a.target, "folds": folds, "epochs": a.epochs, "device": str(dev), "cv": m, "norm": [ymu, ysd], "datasets": list(ds_id)}, open(os.path.join(a.out, f"{a.variant}_{a.target}_cv.json"), "w"), indent=1)
    print(f"[two_stream] outputs -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
