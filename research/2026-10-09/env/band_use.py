import numpy as np, pandas as pd, os
D = r"D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment\Biomass_Height_2026-10-09\dataset"
flights = [("muresk", "2025-09-30", "Muresk anthesis"), ("muresk", "2025-11-21", "Muresk maturity"), ("agt", "2025-09-22", "AGT anthesis"), ("agt", "2025-11-13_f1", "AGT maturity f1")]
BU = pd.read_csv(os.path.join(D, "muresk", "2025-09-30", "vnir_band_usable.csv"))
wcols = [c for c in BU.columns if c.replace(".", "").isdigit()]; wl = np.array([float(c) for c in wcols])
ex = np.zeros(len(wl), bool)
for lo, hi in ((0, 415), (755, 770), (928, 962)): ex |= (wl >= lo) & (wl <= hi)
print(f"bands in cube: {len(wl)} ({wl.min():.0f}-{wl.max():.0f} nm, step {np.median(np.diff(wl)):.1f} nm)")
print(f"fixed exclusions: {ex.sum()} bands -> 0-415 nm: {((wl<=415)).sum()} bands ({wl[wl<=415].min():.0f}-{wl[wl<=415].max():.0f}); 755-770: {(((wl>=755)&(wl<=770))).sum()} ({wl[(wl>=755)&(wl<=770)].min():.0f}-{wl[(wl>=755)&(wl<=770)].max():.0f}); 928-962: {(((wl>=928)&(wl<=962))).sum()} ({wl[(wl>=928)&(wl<=962)].min():.0f}-{wl[(wl>=928)&(wl<=962)].max():.0f})")
print(f"bands available after fixed exclusions: {(~ex).sum()}")
GT = {}
for tr, fl, name in flights:
    bu = pd.read_csv(os.path.join(D, tr, fl, "vnir_band_usable.csv")); gt = pd.read_csv(os.path.join(D, tr, fl, "ground_truth.csv"))
    bu = bu[bu.Plot_ID.isin(gt.Plot_ID)]; M = bu[wcols].to_numpy(float) > 0.5; M[:, ex] = False
    frac = M.mean(0); allp = (frac == 1).sum(); none = (frac == 0).sum(); part = len(wl) - allp - none
    masked_some = wl[(frac < 1) & ~ex]
    rng = f"{masked_some.min():.0f}-{masked_some.max():.0f} nm" if len(masked_some) else "-"
    print(f"{name:16s} GT plots {len(bu):3d}: usable on every plot {allp:3d} bands | on no plot {none:3d} | plot-dependent {part:3d} ({rng}); mean usable bands per plot {M.sum(1).mean():.0f}")
    if part:
        for lo, hi in ((415, 520), (520, 600), (600, 700), (700, 755), (770, 928), (962, 1001)):
            sel = (wl > lo) & (wl <= hi) & ~ex
            if sel.any(): print(f"      {lo:4d}-{hi:4d} nm: usable on {100*frac[sel].mean():5.1f} % of plot-band cells")