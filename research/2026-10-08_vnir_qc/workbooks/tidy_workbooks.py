"""Tidy the per-trial workbooks: drop duplicated design columns from the join (Range_x/Range_y ...),
recompute the low-valid index list in the Flights sheet, and save alongside CSV copies."""
import os, glob, numpy as np, pandas as pd
DS = r"F:\Ibrahim's Workspace2\Biomass Experiment\Biomass Dataset"
for xlsx in glob.glob(os.path.join(DS, "*", "*_plot_traits_indices.xlsx")):
    x = pd.ExcelFile(xlsx); sheets = {sh: x.parse(sh) for sh in x.sheet_names}; x.close()
    for sh, df in list(sheets.items()):
        drop = []
        for c in list(df.columns):
            if c.endswith("_y"):
                drop.append(c)
            elif c.endswith("_x"):
                base = c[:-2]
                if base not in df.columns:
                    df.rename(columns={c: base}, inplace=True)
                else:
                    drop.append(c)
        if drop:
            df.drop(columns=drop, inplace=True)
        sheets[sh] = df
    fl = sheets.get("Flights")
    if fl is not None and "flight" in fl.columns:
        for i, row in fl.iterrows():
            k = row["flight"]; V = sheets.get(f"{k}_VNIR"); Q = sheets.get(f"{k}_VNIR_QC")
            if V is not None and Q is not None:
                # coverage = fraction of region pixels valid at 740/800 nm (bands that are never clipped)
                if "NDRE740_valid" in Q.columns:
                    cov = Q["NDRE740_valid"].values
                    V["vnir_coverage"] = cov
                    sheets[f"{k}_VNIR"] = V
                else:
                    cov = np.ones(len(V))
                covered = cov >= 0.5
                low = [c for c in V.columns if f"{c}_valid" in Q.columns and Q.loc[covered, f"{c}_valid"].mean() < 0.6]
                fl.loc[i, "plots_with_vnir"] = int(covered.sum())
                fl.loc[i, "vnir_indices_low_valid"] = ", ".join(low[:20]) + (" ..." if len(low) > 20 else "")
                fl.loc[i, "n_vnir_indices_low_valid"] = len(low)
        sheets["Flights"] = fl
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        for sh, df in sheets.items():
            df.to_excel(xw, sheet_name=sh[:31], index=False)
    print("tidied:", xlsx, {sh: df.shape for sh, df in sheets.items() if sh not in ("README",)})
