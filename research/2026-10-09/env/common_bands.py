import numpy as np, pandas as pd, os
D = r"D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment\Biomass_Height_2026-10-09\dataset"
fl = [("muresk", "2025-09-30"), ("muresk", "2025-11-21"), ("agt", "2025-09-22")]        # reflectance flights only
frac = {}
for tr, f in fl:
    bu = pd.read_csv(os.path.join(D, tr, f, "vnir_band_usable.csv")); gt = pd.read_csv(os.path.join(D, tr, f, "ground_truth.csv"))
    bu = bu[bu.Plot_ID.isin(gt.Plot_ID)]; wcols = [c for c in bu.columns if c.replace(".", "").isdigit()]
    frac[f] = (bu[wcols].to_numpy(float) > 0.5).mean(0)
wl = np.array([float(c) for c in wcols]); F = pd.DataFrame(frac, index=wl)
ex = np.zeros(len(wl), bool)
for lo, hi in ((0, 415), (755, 770), (928, 962)): ex |= (wl >= lo) & (wl <= hi)
for thr in (1.0, 0.95, 0.90):
    ok = (F.min(1).to_numpy() >= thr) & ~ex
    segs = []; start = None
    for i, o in enumerate(ok):
        if o and start is None: start = i
        if (not o or i == len(ok) - 1) and start is not None:
            end = i if o else i - 1; segs.append(f"{wl[start]:.0f}-{wl[end]:.0f}"); start = None
    print(f"usable on >= {thr*100:.0f} % of the GT plots of all three reflectance flights: {ok.sum()} bands: " + ", ".join(segs) + " nm")
ok = (F.min(1).to_numpy() >= 0.95) & ~ex
out = pd.DataFrame({"wavelength_nm": wl, "common_95": ok.astype(int), "common_100": ((F.min(1).to_numpy() >= 1.0) & ~ex).astype(int), "excluded_range": ex.astype(int),
                    **{f"usable_frac_{f}": F[f].round(3).to_numpy() for _, f in fl}})
out.to_csv(os.path.join(D, "vnir_common_bands.csv"), index=False); print("written dataset\\vnir_common_bands.csv")