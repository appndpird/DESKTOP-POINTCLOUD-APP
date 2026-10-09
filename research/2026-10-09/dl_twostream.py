"""
Two-stream network for plot biomass (and Muresk height) on Dataset_2026-10-09 tensors: sparse 3D convolution on the
classified plot cloud + a spectral transformer on VNIR pixel spectra, late fusion -> regression.

LiDAR stream : points voxelised at 2 cm (x, y, h), features per voxel = [h, intensity_norm, return_number, 1];
               4 SubMConv3d blocks with two stride-2 SparseConv3d downsamplings, global max + mean pooling -> 128-d.
VNIR stream  : 512 pixel spectra x 172 bands (reflectance, log, SNV per pixel). INVALID BANDS ARE NEVER USED: a band is
               masked when the per-plot band_mask (vnir_band_usable, v3.0.1 NaN rule) is 0, when the pixel value is 0
               (clipped), or when it lies in an excluded range; masked bands are zero after SNV, and the mask itself is
               appended so the embedding knows which bands were available. A plot whose cube is not reflectance (AGT
               2025-11-13) has every band masked: its spectral branch contributes nothing beyond the trial/stage embedding.
Fusion       : concat (+ trial/stage embedding) -> MLP -> target (log-biomass, standardised).
Variants     : two_stream, lidar_only, vnir_only.   Validation: 10-fold CV stratified by dataset.
Usage: python dl_twostream.py [--target biomass|height] [--epochs 100] [--variants two_stream,lidar_only,vnir_only]
       (default 100 epochs per fold, OneCycle schedule; 10 folds x 3 variants)
"""
import os, sys, json, time, math, argparse, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import spconv.pytorch as spconv
from sklearn.model_selection import StratifiedKFold
ap = argparse.ArgumentParser(); ap.add_argument("--target", default="biomass"); ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--variants", default="two_stream,lidar_only,vnir_only"); ap.add_argument("--folds", type=int, default=10); ap.add_argument("--maxpts", type=int, default=16000)
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()
B = r"D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment"
OUT = os.path.join(B, "Biomass_Height_2026-10-09", "two_stream"); DATA = os.path.join(OUT, "dl_data"); RES = os.path.join(OUT, "results"); MOD = os.path.join(OUT, "models")
os.makedirs(RES, exist_ok=True); os.makedirs(MOD, exist_ok=True)
dev = torch.device(args.device if torch.cuda.is_available() else "cpu"); torch.manual_seed(0); np.random.seed(0)
t0 = time.time(); LOGF = open(os.path.join(OUT, f"dl_{args.target}.log"), "a")
def log(*a):
    s = f"[{time.time()-t0:5.0f}s] " + " ".join(str(x) for x in a); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
log(f"device {dev}; torch {torch.__version__}; spconv {spconv.__version__ if hasattr(spconv, '__version__') else ''}")
VOX = 0.02; GRID = (256, 96, 96)      # x (along plot, 5.1 m), y (across, 1.9 m), h (1.9 m) at 2 cm

idx = pd.read_csv(os.path.join(DATA, "index.csv"))
tcol = "biomass_kg_ha" if args.target == "biomass" else "height_cm"
idx = idx[idx[tcol].notna()].reset_index(drop=True)
log(f"target {tcol}: {len(idx)} samples; per dataset {idx.dataset.value_counts().to_dict()}; plots with usable VNIR {int((idx.n_usable_bands > 0).sum())}")
wl = np.load(os.path.join(DATA, idx.file[0]))["wavelengths"]
EXCL = np.zeros(len(wl), bool)
for lo, hi in ((0, 415), (755, 770), (928, 962)): EXCL |= (wl >= lo) & (wl <= hi)
DSETS = sorted(idx.dataset.unique()); ds_id = {d: i for i, d in enumerate(DSETS)}

class PlotSet(torch.utils.data.Dataset):
    def __init__(self, rows, train): self.gidx = rows.index.to_numpy(); self.rows = rows.reset_index(drop=True); self.train = train
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows.iloc[i]; z = np.load(os.path.join(DATA, r.file))
        pts = z["points"].astype(np.float32); pix = z["pixels"].astype(np.float32)
        bmask = z["band_mask"].astype(bool) if "band_mask" in z.files else np.ones(len(wl), bool)
        if self.train:
            if np.random.rand() < 0.5: pts[:, 0] = -pts[:, 0]; pts[:, 1] = -pts[:, 1]          # 180 deg rotation about z
            if np.random.rand() < 0.5: pts[:, 1] = -pts[:, 1]                                   # mirror across the row axis
            keep = np.random.rand(len(pts)) > np.random.uniform(0, 0.3); pts = pts[keep]
            pts[:, :2] += np.random.normal(0, 0.01, (len(pts), 2)).astype(np.float32)
            pix = pix[np.random.choice(len(pix), len(pix), replace=True)] * np.random.uniform(0.9, 1.1)
        if len(pts) > args.maxpts: pts = pts[np.random.choice(len(pts), args.maxpts, replace=False)]
        cx = np.clip(np.floor(pts[:, 0] / VOX + GRID[0] / 2), 0, GRID[0] - 1); cy = np.clip(np.floor(pts[:, 1] / VOX + GRID[1] / 2), 0, GRID[1] - 1)
        ch = np.clip(np.floor(np.clip(pts[:, 2], -0.1, None) / VOX + 5), 0, GRID[2] - 1)
        coords = np.stack([cx, cy, ch], 1).astype(np.int32)
        feats = np.stack([pts[:, 2], np.log1p(np.clip(pts[:, 3], 0, 20)), pts[:, 4] / 3.0, np.ones(len(pts), np.float32)], 1).astype(np.float32)
        # spectra: reflectance -> log -> SNV per pixel over the VALID bands only; invalid bands (band_mask 0, clipped 0, excluded) -> 0
        valid = (pix > 0) & bmask[None, :] & ~EXCL[None, :]
        sp = np.log10(np.clip(pix / 10000.0, 1e-4, None))
        nv = np.maximum(valid.sum(1, keepdims=True), 1)
        mu = (sp * valid).sum(1, keepdims=True) / nv; sd = np.sqrt(((sp - mu) ** 2 * valid).sum(1, keepdims=True) / nv) + 1e-6
        sp = np.where(valid, (sp - mu) / sd, 0).astype(np.float32)
        sp = np.concatenate([sp, valid.astype(np.float32)], 1)                                   # (512, 344): values + availability mask
        return dict(coords=torch.from_numpy(coords), feats=torch.from_numpy(feats), spec=torch.from_numpy(sp), ds=ds_id[r.dataset], y=float(r[tcol]), i=int(self.gidx[i]))

def collate(batch):
    coords = torch.cat([torch.cat([torch.full((len(b["coords"]), 1), k, dtype=torch.int32), b["coords"]], 1) for k, b in enumerate(batch)])
    feats = torch.cat([b["feats"] for b in batch]); spec = torch.stack([b["spec"] for b in batch])
    return coords, feats, spec, torch.tensor([b["ds"] for b in batch]), torch.tensor([b["y"] for b in batch], dtype=torch.float32), torch.tensor([b["i"] for b in batch])

class LidarStream(nn.Module):
    def __init__(self, cin=4, w=32):
        super().__init__()
        def block(i, o, key): return spconv.SparseSequential(spconv.SubMConv3d(i, o, 3, padding=1, indice_key=key, bias=False), nn.BatchNorm1d(o), nn.ReLU())
        def down(i, o): return spconv.SparseSequential(spconv.SparseConv3d(i, o, 3, stride=2, padding=1, bias=False), nn.BatchNorm1d(o), nn.ReLU())
        self.net = spconv.SparseSequential(block(cin, w, "s1"), block(w, w, "s1"), down(w, 2 * w), block(2 * w, 2 * w, "s2"), down(2 * w, 4 * w), block(4 * w, 4 * w, "s3"), down(4 * w, 4 * w))
        self.out = nn.Sequential(nn.Linear(8 * w, 128), nn.ReLU(), nn.Dropout(0.2))
    def forward(self, coords, feats, bs):
        x = spconv.SparseConvTensor(feats, coords, spatial_shape=list(GRID), batch_size=bs); y = self.net(x)
        f = y.features; b = y.indices[:, 0].long()
        mx = torch.full((bs, f.shape[1]), -1e4, device=f.device).scatter_reduce(0, b[:, None].expand_as(f), f, reduce="amax", include_self=True)
        mean = torch.zeros((bs, f.shape[1]), device=f.device).index_add_(0, b, f) / torch.bincount(b, minlength=bs).clamp(min=1)[:, None]
        return self.out(torch.cat([mx, mean], 1))

class SpecStream(nn.Module):
    def __init__(self, nb=172, d=64):
        super().__init__()
        self.emb = nn.Sequential(nn.Linear(2 * nb, d), nn.GELU(), nn.Linear(d, d))        # values + availability mask
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 128, dropout=0.1, batch_first=True), 2)
        self.out = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Dropout(0.2))
    def forward(self, spec):  # (B, 512, 344)
        return self.out(self.enc(self.emb(spec)).mean(1))

class TwoStream(nn.Module):
    def __init__(self, variant, nds):
        super().__init__(); self.variant = variant
        self.lidar = LidarStream() if variant != "vnir_only" else None; self.spec = SpecStream() if variant != "lidar_only" else None
        self.ds_emb = nn.Embedding(nds, 8); d = 128 * (2 if variant == "two_stream" else 1) + 8
        self.head = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))
    def forward(self, coords, feats, spec, ds, bs):
        parts = []
        if self.lidar is not None: parts.append(self.lidar(coords, feats, bs))
        if self.spec is not None: parts.append(self.spec(spec))
        parts.append(self.ds_emb(ds)); return self.head(torch.cat(parts, 1)).squeeze(1)

def metrics(p, y):
    e = p - y; mape = float(np.mean(np.abs(e) / y) * 100)
    return dict(R2=float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)), RMSE=float(np.sqrt(np.mean(e ** 2))), rRMSE=float(np.sqrt(np.mean(e ** 2)) / y.mean() * 100), MAE=float(np.abs(e).mean()),
                bias=float(e.mean()), r=float(np.corrcoef(p, y)[0, 1]) if np.std(p) > 0 else np.nan, MAPE=mape, accuracy=100 - mape)

y_all = idx[tcol].to_numpy(float); ylog = args.target == "biomass"
yt = np.log(y_all) if ylog else y_all; ymu, ysd = yt.mean(), yt.std()
skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=0)
rows, preds = [], []
for variant in args.variants.split(","):
    pred = np.full(len(idx), np.nan)
    for fold, (tr, te) in enumerate(skf.split(idx, idx.dataset)):
        model = TwoStream(variant, len(DSETS)).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=args.epochs * math.ceil(len(tr) / 16))
        dl_tr = torch.utils.data.DataLoader(PlotSet(idx.iloc[tr], True), batch_size=16, shuffle=True, collate_fn=collate, num_workers=0, drop_last=False)
        dl_te = torch.utils.data.DataLoader(PlotSet(idx.iloc[te], False), batch_size=16, shuffle=False, collate_fn=collate, num_workers=0)
        for ep in range(args.epochs):
            model.train()
            for coords, feats, spec, ds, y, _ in dl_tr:
                coords, feats, spec, ds, y = coords.to(dev), feats.to(dev), spec.to(dev), ds.to(dev), y.to(dev)
                yn = ((torch.log(y) if ylog else y) - ymu) / ysd
                loss = F.smooth_l1_loss(model(coords, feats, spec, ds, len(y)), yn, beta=0.5)
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step(); sched.step()
        model.eval(); out = []
        with torch.no_grad():
            for coords, feats, spec, ds, y, ii in dl_te:
                p = model(coords.to(dev), feats.to(dev), spec.to(dev), ds.to(dev), len(y)).cpu().numpy() * ysd + ymu
                p = np.exp(p) if ylog else p; pred[ii.numpy()] = p
        torch.save(model.state_dict(), os.path.join(MOD, f"dl_{args.target}_{variant}_fold{fold}.pt"))
        m = metrics(pred[te], y_all[te]); log(f"{variant} fold {fold+1}/{args.folds}: R2 {m['R2']:.3f} RMSE {m['RMSE']:.0f}  (nan preds {int(np.isnan(pred[te]).sum())})")
    for d in DSETS:
        sel = (idx.dataset == d).to_numpy(); m = metrics(pred[sel], y_all[sel])
        rows.append(dict(scheme=f"{args.folds}fold_pooled", target=tcol, variant=variant, dataset=d, n=int(sel.sum()), **m)); log(f"  {variant} {d}: R2 {m['R2']:.3f} RMSE {m['RMSE']:.1f} acc {m['accuracy']:.1f}%")
    m = metrics(pred, y_all); rows.append(dict(scheme=f"{args.folds}fold_pooled", target=tcol, variant=variant, dataset="ALL", n=len(idx), **m)); log(f"  {variant} ALL: R2 {m['R2']:.3f} RMSE {m['RMSE']:.1f}")
    preds.append(pd.DataFrame(dict(variant=variant, dataset=idx.dataset, Plot_ID=idx.Plot_ID, Plot=idx.Plot, Variety=idx.Variety, measured=y_all, predicted=pred)))
    pd.DataFrame(rows).to_csv(os.path.join(RES, f"dl_{args.target}_comparison.csv"), index=False)
    pd.concat(preds).to_csv(os.path.join(RES, f"dl_{args.target}_per_plot_predictions.csv"), index=False)
json.dump(dict(target=tcol, epochs=args.epochs, folds=args.folds, maxpts=args.maxpts, variants=args.variants, device=str(dev), n=len(idx),
               band_policy="per-plot band_mask (vnir_band_usable) AND pixel > 0 AND not excluded; masked bands zero after SNV; availability mask appended to the embedding"),
          open(os.path.join(RES, f"dl_{args.target}_settings.json"), "w"), indent=1)
log("DL DONE")
