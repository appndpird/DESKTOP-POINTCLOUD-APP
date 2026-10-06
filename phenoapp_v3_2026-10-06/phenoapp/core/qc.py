"""
Point-cloud quality checks for drone LiDAR.

Runs a battery of checks equivalent to the free LAStools reporting tools
(lasinfo / lasvalidate) implemented directly with laspy, plus survey-grade
checks LAStools does not do out of the box (GCP vertical consistency).
If a LAStools bin folder is available, the genuine lasinfo64/lasvalidate64
(the free, LGPL-licensed subset of LAStools) are run as well and their
output appended — the paid tools (lasground, lasoverlap, ...) are NEVER
invoked, so no license is required and no watermarking/perturbation occurs.

Why not use LAStools for the traits themselves? We validated (Busselton
fodder trial, 4 surveyed GCPs) that laspy/PDAL geometry agrees with the
raw LAS to millimetres — the accuracy bottleneck is the ground model and
the sward itself, not the point reader. LAStools' paid processing tools
additionally perturb coordinates when unlicensed. So: free LAStools for
*reporting*, laspy/PDAL for *processing*.

Public API
----------
run_qc(las_path, gcp_csv=None, lastools_dir=None, sample_target=30_000_000,
       progress_cb=None) -> dict (structured report)
format_report(report) -> str
"""

from __future__ import annotations
import os
import subprocess
import numpy as np


def _grade(ok: bool, warn: bool = False) -> str:
    return "PASS" if ok else ("WARN" if warn else "FAIL")


def default_lastools_dir() -> str | None:
    """Locate the bundled free-LAStools folder (frozen build or source)."""
    import sys
    cands = []
    if getattr(sys, "frozen", False):
        cands.append(os.path.join(sys._MEIPASS, "lastools_free"))
        cands.append(os.path.join(os.path.dirname(sys.executable),
                                  "lastools_free"))
    here = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    cands.append(os.path.join(here, "lastools_free"))
    for c in cands:
        if os.path.isfile(os.path.join(c, "lasinfo64.exe")):
            return c
    return None


def run_qc(las_path: str, gcp_csv: str | None = None,
           lastools_dir: str | None = None,
           sample_target: int = 30_000_000,
           progress_cb=None) -> dict:
    import laspy

    if not lastools_dir:
        lastools_dir = default_lastools_dir()

    def _p(pct, msg):
        if progress_cb: progress_cb(pct, msg)

    rep: dict = {"las_path": las_path, "checks": {}}

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    _p(2, "Reading header...")
    with laspy.open(las_path) as f:
        h = f.header
        n = h.point_count
        rep["header"] = {
            "version": str(h.version),
            "point_format": int(h.point_format.id),
            "n_points": int(n),
            "generating_software": str(h.generating_software),
            "mins": [float(v) for v in h.mins],
            "maxs": [float(v) for v in h.maxs],
            "scales": [float(v) for v in h.scales],
        }
        try:
            crs = h.parse_crs()
            rep["header"]["crs"] = str(crs.to_epsg() or crs.name) if crs else "MISSING"
        except Exception:
            rep["header"]["crs"] = "unparseable"
        try:
            by_return = np.asarray(h.number_of_points_by_return).tolist()
        except Exception:
            by_return = []
        rep["header"]["points_by_return"] = [int(v) for v in by_return[:5]]

    rep["checks"]["crs"] = {
        "grade": _grade(rep["header"]["crs"] not in ("MISSING", "unparseable")),
        "detail": f"CRS: {rep['header']['crs']}"}

    # ------------------------------------------------------------------
    # Point scan (chunked; subsampled stats above sample_target points)
    # ------------------------------------------------------------------
    step = max(1, int(np.ceil(n / max(sample_target, 1))))
    zs, inten = [], []
    cls_counts = {}
    n_single = 0; n_seen = 0
    scan_nonzero = 0; psid_nonzero = 0; rgb_nonzero = 0
    gps_min, gps_max = np.inf, -np.inf
    has_rgb = False; has_gps = False

    with laspy.open(las_path) as f:
        dims = set(f.header.point_format.dimension_names)
        has_rgb = {"red", "green", "blue"} <= dims
        has_gps = "gps_time" in dims
        done = 0
        for pts in f.chunk_iterator(5_000_000):
            sl = slice(None, None, step)
            zs.append(np.asarray(pts.z)[sl])
            inten.append(np.asarray(pts.intensity)[sl])
            c = np.asarray(pts.classification)[sl]
            u, cn = np.unique(c, return_counts=True)
            for uu, cc in zip(u, cn):
                cls_counts[int(uu)] = cls_counts.get(int(uu), 0) + int(cc)
            rn = np.asarray(pts.return_number)[sl]
            nr = np.asarray(pts.number_of_returns)[sl]
            n_single += int((nr <= 1).sum()); n_seen += len(rn)
            if "scan_angle_rank" in dims:
                scan_nonzero += int((np.asarray(pts.scan_angle_rank)[sl] != 0).sum())
            elif "scan_angle" in dims:
                scan_nonzero += int((np.asarray(pts.scan_angle)[sl] != 0).sum())
            if "point_source_id" in dims:
                psid_nonzero += int((np.asarray(pts.point_source_id)[sl] != 0).sum())
            if has_rgb:
                rgb_nonzero += int(((np.asarray(pts.red)[sl] != 0) |
                                    (np.asarray(pts.green)[sl] != 0) |
                                    (np.asarray(pts.blue)[sl] != 0)).sum())
            if has_gps:
                g = np.asarray(pts.gps_time)[sl]
                if len(g):
                    gps_min = min(gps_min, float(g.min()))
                    gps_max = max(gps_max, float(g.max()))
            done += len(pts)
            _p(2 + int(58 * done / n), f"Scanning points ({done:,}/{n:,})")

    z = np.concatenate(zs); it = np.concatenate(inten)

    # density over the data footprint (grid-cell based, robust to empty bbox)
    hdr = rep["header"]
    area = max((hdr["maxs"][0] - hdr["mins"][0]) *
               (hdr["maxs"][1] - hdr["mins"][1]), 1e-6)
    rep["density_per_m2_bbox"] = float(n / area)

    # ---- z sanity: spikes above/below the bulk ----
    p1, p50, p99 = np.percentile(z, [1, 50, 99])
    lo_out = int((z < p1 - 10).sum() * step)
    hi_out = int((z > p99 + 10).sum() * step)
    rep["z"] = {"p1": float(p1), "p50": float(p50), "p99": float(p99),
                "min": hdr["mins"][2], "max": hdr["maxs"][2],
                "far_low_outliers": lo_out, "far_high_outliers": hi_out}
    rep["checks"]["z_outliers"] = {
        "grade": _grade(lo_out + hi_out < n * 1e-4, warn=lo_out + hi_out < n * 1e-3),
        "detail": (f"{lo_out + hi_out:,} points >10 m outside the p1-p99 band "
                   f"(z p1={p1:.2f}, p99={p99:.2f})")}

    # ---- returns / classification / attribute population ----
    single_frac = n_single / max(n_seen, 1)
    rep["returns"] = {"single_return_fraction": float(single_frac)}
    rep["checks"]["multi_return"] = {
        "grade": _grade(single_frac < 0.999, warn=True),
        "detail": (f"{single_frac*100:.1f}% single returns"
                   + (" - no canopy-penetration analysis possible; ground under "
                      "dense canopy will be interpolated, not measured"
                      if single_frac >= 0.999 else ""))}
    rep["classification_counts"] = cls_counts
    only_zero = set(cls_counts) <= {0, 1}
    rep["checks"]["classification"] = {
        "grade": _grade(not only_zero, warn=True),
        "detail": ("no ground classification present (all class "
                   f"{sorted(cls_counts)}) - height normalization must build "
                   "its own ground model" if only_zero else
                   f"classes present: {sorted(cls_counts)}")}
    rep["checks"]["scan_angle"] = {
        "grade": _grade(scan_nonzero > 0, warn=True),
        "detail": ("scan angles all zero - flightline geometry not recoverable"
                   if scan_nonzero == 0 else "scan angles populated")}
    rep["checks"]["flightline_ids"] = {
        "grade": _grade(psid_nonzero > 0, warn=True),
        "detail": ("point source IDs all zero - per-flightline overlap QC "
                   "impossible on this merged file" if psid_nonzero == 0
                   else "point source IDs populated")}
    if has_rgb:
        rep["checks"]["rgb"] = {
            "grade": _grade(rgb_nonzero > 0, warn=True),
            "detail": ("RGB fields present but all black (wasted 6 bytes/pt)"
                       if rgb_nonzero == 0 else "RGB populated")}
    if has_gps and np.isfinite(gps_min):
        # Detect the 'unix nanoseconds' bug (values ~1.7e18) and week-time
        # confusion; healthy adjusted-GPS times are ~1e8..4e9 in 2015-2040.
        weird = gps_max > 1e12 or gps_min < 0
        rep["gps_time"] = {"min": gps_min, "max": gps_max}
        rep["checks"]["gps_time"] = {
            "grade": _grade(not weird, warn=True),
            "detail": (f"GPS time range {gps_min:.3g}..{gps_max:.3g} is not a "
                       "valid LAS GPS time encoding (looks like Unix "
                       "nanoseconds)" if weird else
                       f"GPS time range {gps_min:.1f}..{gps_max:.1f} OK")}

    # ---- intensity ----
    rep["intensity"] = {"p1": float(np.percentile(it, 1)),
                        "p50": float(np.percentile(it, 50)),
                        "p99": float(np.percentile(it, 99))}

    # ------------------------------------------------------------------
    # GCP vertical consistency (the strongest geometric check available)
    # ------------------------------------------------------------------
    if gcp_csv and os.path.exists(gcp_csv):
        _p(65, "Checking GCPs...")
        try:
            rep["gcp"] = _gcp_check(las_path, gcp_csv)
            spread = rep["gcp"]["pairwise_resid_max_abs_m"]
            rough = rep["gcp"]["max_local_roughness_m"]
            rep["checks"]["gcp_consistency"] = {
                "grade": _grade(spread < 0.05, warn=spread < 0.10),
                "detail": (f"pairwise vertical residuals max |{spread*1000:.0f}| mm "
                           f"across {rep['gcp']['n_gcps']} GCPs; local surface "
                           f"noise ~{rough*100:.1f} cm. Constant datum offset "
                           f"{rep['gcp']['mean_offset_m']:+.2f} m (ellipsoid vs "
                           "orthometric is expected and not an error).")}
        except Exception as e:
            rep["checks"]["gcp_consistency"] = {
                "grade": "WARN", "detail": f"GCP check failed: {e}"}

    # ------------------------------------------------------------------
    # Optional: genuine free LAStools reports
    # ------------------------------------------------------------------
    if lastools_dir:
        _p(80, "Running free LAStools reports...")
        rep["lastools"] = {}
        for tool, args in (("lasinfo64", ["-nc", "-stdout"]),
                           ("lasvalidate64", ["-stdout"])):
            exe = os.path.join(lastools_dir, tool + ".exe")
            if not os.path.isfile(exe):
                exe = os.path.join(lastools_dir, tool.replace("64", "") + ".exe")
            if os.path.isfile(exe):
                try:
                    cf = 0x08000000 if os.name == "nt" else 0
                    pr = subprocess.run([exe, "-i", las_path] + args,
                                        capture_output=True, text=True,
                                        timeout=600, creationflags=cf)
                    rep["lastools"][tool] = (pr.stdout or pr.stderr)[-8000:]
                except Exception as e:
                    rep["lastools"][tool] = f"failed: {e}"

    _p(100, "QC complete")
    overall = "PASS"
    for c in rep["checks"].values():
        if c["grade"] == "FAIL":
            overall = "FAIL"; break
        if c["grade"] == "WARN":
            overall = "WARN"
    rep["overall"] = overall
    return rep


def _gcp_check(las_path: str, gcp_csv: str, radius: float = 0.35) -> dict:
    """Relative vertical consistency of the cloud against surveyed GCPs.

    The comparison uses *differences* between GCP pairs, so it is immune to
    the constant ellipsoid-vs-orthometric datum offset.
    CSV columns expected (flexible header match): easting, northing, height.
    """
    import laspy
    import csv as _csv

    pts = []
    with open(gcp_csv, newline="") as f:
        all_rows = list(_csv.reader(f))
    # tolerate metadata preambles (e.g. Emlid exports): find the first row
    # that contains easting/northing/height-like column names
    header_i = None
    ie = in_ = ih = None
    for ri, row in enumerate(all_rows):
        cols = [c.strip().lower() for c in row]
        def col(*names):
            for nm in names:
                for i, c in enumerate(cols):
                    if nm in c:
                        return i
            return None
        e_, n_, h_ = col("east"), col("north"), col("height", "elev")
        if e_ is not None and n_ is not None and h_ is not None:
            header_i, ie, in_, ih = ri, e_, n_, h_
            break
    if header_i is None:
        raise RuntimeError(
            "GCP CSV has no easting/northing/height header row "
            f"(first row: {all_rows[0] if all_rows else 'empty'})")
    for row in all_rows[header_i + 1:]:
        try:
            pts.append((float(row[ie]), float(row[in_]), float(row[ih])))
        except (ValueError, IndexError):
            continue
    if len(pts) < 2:
        raise RuntimeError(f"Only {len(pts)} usable GCP rows in {gcp_csv}")

    acc = [[] for _ in pts]
    with laspy.open(las_path) as f:
        for chunk in f.chunk_iterator(5_000_000):
            x = np.asarray(chunk.x); y = np.asarray(chunk.y)
            z = np.asarray(chunk.z)
            for i, (gx, gy, _) in enumerate(pts):
                m = (np.abs(x - gx) < radius) & (np.abs(y - gy) < radius)
                if m.any():
                    acc[i].append(z[m])

    med = []; rough = []
    for i, blocks in enumerate(acc):
        if not blocks:
            raise RuntimeError(f"No LiDAR points within {radius} m of GCP {i+1}")
        zz = np.concatenate(blocks)
        med.append(float(np.median(zz)))
        rough.append(float(np.std(zz)))

    resids = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            ldz = med[i] - med[j]
            gdz = pts[i][2] - pts[j][2]
            resids.append(ldz - gdz)
    offs = [m - p[2] for m, p in zip(med, pts)]
    return {
        "n_gcps": len(pts),
        "pairwise_resid_max_abs_m": float(np.max(np.abs(resids))),
        "pairwise_resid_rms_m": float(np.sqrt(np.mean(np.square(resids)))),
        "mean_offset_m": float(np.mean(offs)),
        "max_local_roughness_m": float(np.max(rough)),
    }


def format_report(rep: dict) -> str:
    L = []
    L.append(f"POINT CLOUD QC REPORT — overall: {rep['overall']}")
    L.append(f"File: {rep['las_path']}")
    h = rep["header"]
    L.append(f"LAS {h['version']} fmt {h['point_format']} | "
             f"{h['n_points']:,} points | software: {h['generating_software']}")
    L.append(f"CRS: {h['crs']} | density (bbox): "
             f"{rep['density_per_m2_bbox']:.0f} pts/m2")
    z = rep.get("z", {})
    if z:
        L.append(f"Z: min {z['min']:.2f} | p1 {z['p1']:.2f} | p50 {z['p50']:.2f}"
                 f" | p99 {z['p99']:.2f} | max {z['max']:.2f}")
    L.append("")
    L.append("Checks:")
    for name, c in rep["checks"].items():
        L.append(f"  [{c['grade']:<4}] {name}: {c['detail']}")
    if "gcp" in rep:
        g = rep["gcp"]
        L.append("")
        L.append(f"GCPs ({g['n_gcps']}): pairwise vertical RMS "
                 f"{g['pairwise_resid_rms_m']*1000:.0f} mm, max "
                 f"{g['pairwise_resid_max_abs_m']*1000:.0f} mm; "
                 f"datum offset {g['mean_offset_m']:+.2f} m; "
                 f"surface noise {g['max_local_roughness_m']*100:.1f} cm")
    if rep.get("lastools"):
        for tool, out in rep["lastools"].items():
            L.append("")
            L.append(f"--- {tool} output ---")
            L.append(out)
    return "\n".join(L)
