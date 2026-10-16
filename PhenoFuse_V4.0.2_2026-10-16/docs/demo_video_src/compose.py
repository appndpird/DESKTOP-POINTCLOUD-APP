"""Compose the demo: title / explanation cards, recorded frames with caption bar and highlight, zoom-ins, QA figure, narration -> mp4.
Run with the system Python 3.13 (cv2 + PIL + imageio_ffmpeg)."""
import json, os, sys, wave, subprocess, glob
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "PhenoFuse_VNIR_reflectance_calibration_demo.mp4")
OUTW, OUTH, VH, FPS = 1280, 880, 800, 25
NAR = json.load(open(os.path.join(HERE, "narration.json"), encoding="utf-8")); ST = json.load(open(os.path.join(HERE, "steps.json")))
F_B = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 27); F_R = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 18)
F_T = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 46); F_T2 = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 24); F_M = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 22); F_C = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 22)
FFMPEG = __import__("imageio_ffmpeg").get_ffmpeg_exe()
frames_by = {}
for s in ST["steps"]: frames_by.setdefault(s["step"], []).append(s)

def wav_pcm(p):
    with wave.open(p) as w:
        assert w.getframerate() == 22050 and w.getsampwidth() == 2 and w.getnchannels() == 1; return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
def nar(name):
    p = os.path.join(HERE, "tts", name + ".wav"); return wav_pcm(p) if os.path.exists(p) else np.zeros(22050, np.int16)
def caption(canvas, head, sub):
    im = Image.new("RGB", (OUTW, OUTH), (12, 14, 20)); im.paste(Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)), (0, 0)); d = ImageDraw.Draw(im)
    d.rectangle([0, VH, OUTW, OUTH], fill=(12, 14, 20)); d.line([0, VH, OUTW, VH], fill=(60, 160, 90), width=3)
    d.text((24, VH + 10), head, font=F_B, fill=(255, 255, 255)); d.text((24, VH + 46), sub, font=F_R, fill=(175, 180, 190)); return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
def fit(img, w=OUTW, h=VH, bg=18):
    ih, iw = img.shape[:2]; sc = min(w / iw, h / ih); nw, nh = int(iw * sc), int(ih * sc); r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    c = np.full((h, w, 3), bg, np.uint8); x0, y0 = (w - nw) // 2, (h - nh) // 2; c[y0:y0 + nh, x0:x0 + nw] = r; return c, sc, x0, y0
def highlight(canvas, rect, sc, x0, y0, color=(0, 200, 255)):
    x, y, w, h = rect; cv2.rectangle(canvas, (int(x0 + x * sc) - 4, int(y0 + y * sc) - 4), (int(x0 + (x + w) * sc) + 4, int(y0 + (y + h) * sc) + 4), color, 3); return canvas
def zoom(img, rect, pad=30):
    x, y, w, h = rect; H, W = img.shape[:2]; x0, y0, x1, y1 = max(x - pad, 0), max(y - pad, 0), min(x + w + pad, W), min(y + h + pad, H)
    # keep the 16:10 aspect of the stage
    cw, ch = x1 - x0, y1 - y0; want = OUTW / VH
    if cw / ch > want: nh = int(cw / want); y0 = max(0, y0 - (nh - ch) // 2); y1 = min(H, y0 + nh)
    else: nw = int(ch * want); x0 = max(0, x0 - (nw - cw) // 2); x1 = min(W, x0 + nw)
    return img[y0:y1, x0:x1]
def card(title, lines, mono=None, image=None):
    im = Image.new("RGB", (OUTW, VH), (16, 20, 28)); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, OUTW, 8], fill=(60, 160, 90)); d.text((56, 44), title, font=F_T, fill=(255, 255, 255)); y = 130
    for ln in lines:
        bullet = ln.startswith("- "); d.text((56 + (24 if bullet else 0), y), ("•  " + ln[2:]) if bullet else ln, font=F_C, fill=(220, 224, 230) if not ln.startswith("# ") else (120, 200, 150)); y += 36
    if mono:
        y += 10; d.rectangle([56, y, 700, y + 32 * len(mono) + 24], fill=(28, 34, 44), outline=(70, 80, 95)); yy = y + 12
        for ln in mono: d.text((76, yy), ln, font=F_M, fill=(200, 240, 200)); yy += 32
    c = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    if image is not None:
        ih, iw = image.shape[:2]; sc = min(540 / iw, 520 / ih); r = cv2.resize(image, (int(iw * sc), int(ih * sc))); c[150:150 + r.shape[0], OUTW - 60 - r.shape[1]:OUTW - 60] = r
    return c
def title_card(big, small):
    im = Image.new("RGB", (OUTW, VH), (16, 20, 28)); d = ImageDraw.Draw(im)
    w1 = d.textlength(big, font=F_T); d.text(((OUTW - w1) / 2, VH / 2 - 70), big, font=F_T, fill=(255, 255, 255))
    for i, ln in enumerate(small.split(" · ")): w2 = d.textlength(ln, font=F_T2); d.text(((OUTW - w2) / 2, VH / 2 + 10 + 34 * i), ln, font=F_T2, fill=(120, 200, 150))
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

segments = []   # (name, list of stage frames (VH x OUTW), audio pcm)
def seg(name, stage_frames, hold=0.6):
    a = nar(name); need = int(np.ceil(len(a) / 22050 * FPS)) + int(hold * FPS)
    fr = list(stage_frames)
    if len(fr) < need:   # repeat the last frame (or spread the frames evenly over the narration)
        if len(fr) > 1: idx = np.linspace(0, len(fr) - 1, need).astype(int); fr = [fr[i] for i in idx]
        else: fr = fr + [fr[-1]] * (need - len(fr))
    segments.append((name, fr, a))
def shot(step, n=0, hl="rect", zoom_sec=None, pre_full=1.0):
    s = frames_by[step][n]; img = cv2.imread(os.path.join(HERE, "frames", s["png"])); c, sc, x0, y0 = fit(img); frames = []
    if hl and s.get(hl): c = highlight(c.copy(), s[hl], sc, x0, y0)
    full = c; frames = [full]
    if zoom_sec:
        z, *_ = fit(zoom(img, s["rect"])); frames = [full] * int(pre_full * FPS) + [z] * int(zoom_sec * FPS)
    return frames
# ---- build
seg("title", [title_card(*NAR["title"][:2])])
spec_img = cv2.imread(os.path.join(HERE, "radiance_vs_reflectance.png"))
seg("why", [card("Radiance versus reflectance", ["- Radiance: light that reached the sensor = sun x surface x atmosphere", "- Reflectance: the surface alone, 0-1, comparable across dates and sensors",
                                                  "- A radiance cube keeps the O2-A (760 nm) and water-vapour (940 nm) dips", "- NIR about equal to red, NDVI about 0: every index is meaningless", "- PhenoFuse features assume reflectance"], image=spec_img)])
seg("tab", shot("tab", 0, zoom_sec=4))
seg("check_rad", shot("check_rad", 1, hl="label", zoom_sec=6))
seg("targets", shot("targets", 0, zoom_sec=5))
seg("csv", [card("Target reflectance spectra CSV", ["Where the spectrum comes from (best first):", "- the panel's calibration certificate (Spectralon, Group 8 tarps, Headwall / GRYFN kit panels)",
                                                     "- a field spectroradiometer measurement of the tarp (ASD FieldSpec, Spectral Evolution)", "- a nominal value for a grey flat panel: number in the polygon's refl column, no CSV",
                                                     "CSV: wavelength_nm + one reflectance column (0-1) per target, any step:"],
                 mono=["wavelength_nm,panel_50,panel_05", "400,0.492,0.051", "410,0.494,0.051", "...", "1000,0.501,0.053"])])
seg("qgis", [card("Target polygons in QGIS", ["Layer > Create Layer > New Shapefile Layer", "- geometry: Polygon; CRS: the cube's (EPSG:7850 for DPIRD WA)", "- fields: name (text), refl (decimal)",
                                              "- digitise one polygon per target, well inside the panel", "- avoid edges, shadows and glint", "- name = CSV column (panel_50), or refl = nominal value",
                                              "Save, then pick the shapefile under Target polygons", "and the CSV under Target spectra CSV", "Two targets of different brightness: gain + offset"])])
seg("refcube", shot("refcube", 0, zoom_sec=5))
cal = frames_by.get("calibrate", [])
cal_frames = [fit(cv2.imread(os.path.join(HERE, "frames", s["png"])))[0] for s in cal] or shot("refcube", 0)
seg("calibrate", cal_frames)
seg("calib_done", shot("calib_done", 0, hl=None, zoom_sec=None))
qa = cv2.imread(ST["qa_png"]); seg("qa", [fit(qa, bg=255)[0]])
seg("check_ok", shot("check_ok", 0, hl="label", zoom_sec=5))
feat = frames_by.get("features", []); feat_frames = [fit(cv2.imread(os.path.join(HERE, "frames", s["png"])))[0] for s in feat] + shot("features_done", 0, hl="rect")
seg("features", feat_frames)
seg("outro", [title_card(*NAR["outro"][:2])], hold=1.5)
# ---- write
raw = os.path.join(HERE, "video_raw.mp4"); wavp = os.path.join(HERE, "narration_all.wav")
raw = os.path.join(HERE, "video_raw.avi"); fourcc = cv2.VideoWriter_fourcc(*"MJPG"); vw = cv2.VideoWriter(raw, fourcc, FPS, (OUTW, OUTH)); assert vw.isOpened(); audio = []
for name, frs, a in segments:
    head, sub = NAR[name][0], NAR[name][1]; n = 0
    for fr in frs:
        fr = np.ascontiguousarray(fr[:, :, :3], dtype=np.uint8)
        if fr.shape[:2] != (VH, OUTW): fr = cv2.resize(fr, (OUTW, VH))
        out_fr = np.ascontiguousarray(caption(fr, head, sub)[:, :, :3], dtype=np.uint8)
        assert out_fr.shape == (OUTH, OUTW, 3), (name, out_fr.shape)
        ok = vw.write(out_fr); n += 1
    print(f"  segment {name}: {n} frames ({n / FPS:.1f} s), audio {len(a) / 22050:.1f} s", flush=True)
    pad = n / FPS * 22050 - len(a); audio.append(a); audio.append(np.zeros(max(int(pad), 0), np.int16))
vw.release(); pcm = np.concatenate(audio)
with wave.open(wavp, "wb") as w: w.setnchannels(1); w.setsampwidth(2); w.setframerate(22050); w.writeframes(pcm.tobytes())
subprocess.run([FFMPEG, "-y", "-i", raw, "-i", wavp, "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-shortest", OUT], check=True, capture_output=True)
print("->", OUT, f"{os.path.getsize(OUT)/1e6:.1f} MB, {len(pcm)/22050/60:.1f} min")
