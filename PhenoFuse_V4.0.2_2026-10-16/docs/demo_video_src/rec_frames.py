"""Record the reflectance check + empirical calibration workflow in the live PhenoFuse window (on screen, RDP desktop).
Writes frames/<step>_<n>.png (desktop grab cropped to the window) and steps.json with the rectangles of the calibration group."""
import os, sys, time, json, warnings; warnings.filterwarnings("ignore")
APP = sys.argv[1]; sys.path.insert(0, APP)
HERE = os.path.dirname(os.path.abspath(__file__)); FR = os.path.join(HERE, "frames"); os.makedirs(FR, exist_ok=True)
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import QRect, Qt
app = QApplication([])
# non-modal message boxes so the script keeps running; we grab the desktop while they are open
_open_boxes = []
def _nonmodal(icon):
    def f(parent, title, text, *a, **k):
        b = QMessageBox(parent); b.setIcon(icon); b.setWindowTitle(title); b.setText(text); b.setStandardButtons(QMessageBox.Ok); b.setModal(False); b.show(); _open_boxes.append(b); return QMessageBox.Ok
    return staticmethod(f)
QMessageBox.warning = _nonmodal(QMessageBox.Warning); QMessageBox.information = _nonmodal(QMessageBox.Information); QMessageBox.critical = _nonmodal(QMessageBox.Critical)
from phenoapp.app import MainWindow
from phenoapp.core.project import state
BE = "F:/Ibrahim's Workspace2/Biomass Experiment/2025_NUE_AGT_I_DPIRD"
RAD = f"{BE}/2025-11-13/flight2/20251113_AGT_NUE_Bejoording_Bejoording_Gobi_flight2_VNIR_Orthomosaic.bin"
REF = f"{BE}/2025-09-22/20250922_AGT_NUE_Bejoording_Bejoording_Gobi_VNIR_Orthomosaic.bin"
GRID = f"{BE}/2025-11-13/flight2/aligned_grid_corrected.shp"
win = MainWindow(); win.resize(1600, 940); win.move(40, 30); win.show(); app.processEvents()
s = state(); s.grid_path = GRID; s.saved_grid = GRID
tabs = win.findChild(type(win.centralWidget()).__mro__[0]) if False else None
# switch to the VNIR tab
from PyQt5.QtWidgets import QTabWidget
main_tabs = [t for t in win.findChildren(QTabWidget) if t.count() > 10][0]
for i in range(main_tabs.count()):
    if "VNIR" in main_tabs.tabText(i): main_tabs.setCurrentIndex(i); break
ip = win.tab_spectral.indices; win.tab_spectral.tabs.setCurrentIndex(0)
def pump(sec):
    t = time.time()
    while time.time() - t < sec: app.processEvents(); time.sleep(0.03)
pump(1.5)
steps = []; screen = app.primaryScreen()
from PyQt5.QtGui import QPainter
def rect_of(widget):
    tl = widget.mapTo(win, widget.rect().topLeft()); return [tl.x(), tl.y(), widget.width(), widget.height()]
def grab(step, n=0, extra=None):
    """render the window itself (works without an active desktop) and paste any open message box at its position."""
    pump(0.4); pix = win.grab(); origin = win.mapToGlobal(win.rect().topLeft())
    if _open_boxes:
        painter = QPainter(pix)
        for b in _open_boxes:
            if b.isVisible():
                bp = b.grab(); gp = b.mapToGlobal(b.rect().topLeft()); painter.drawPixmap(gp.x() - origin.x(), gp.y() - origin.y(), bp)
        painter.end()
    p = os.path.join(FR, f"{step}_{n:03d}.png"); pix.save(p); d = dict(step=step, n=n, png=os.path.basename(p), win=[pix.width(), pix.height()]); d.update(extra or {}); steps.append(d); return p
def close_boxes():
    for b in _open_boxes: b.close()
    _open_boxes.clear(); pump(0.3)
cal_group = ip.btn_cal.parentWidget(); chk_rect = lambda: rect_of(ip.lbl_check)
# 1. the tab, empty
grab("tab", 0, dict(rect=rect_of(cal_group)))
# 2. pick the radiance cube -> automatic check + warning
ip.ed_vnir.setText(RAD); state().vnir_path = RAD; ip.refresh_catalogue(); pump(0.5); ip._check_cube(quiet=True); pump(0.5)
grab("check_rad", 0, dict(rect=rect_of(cal_group), label=chk_rect()))
ip._check_cube(); pump(1.0); grab("check_rad", 1, dict(rect=rect_of(cal_group), label=chk_rect())); close_boxes()
# 3. reference option A: targets (fields enabled)
ip.cb_cal_mode.setCurrentIndex(0); pump(0.5); ip.ed_cal_targets.setText("F:/.../calibration_targets_2025-11-13.shp"); ip.ed_cal_spectra.setText("F:/.../panel_spectra.csv"); pump(0.3)
grab("targets", 0, dict(rect=rect_of(cal_group))); ip.ed_cal_targets.setText(""); ip.ed_cal_spectra.setText("")
# 4. reference option B: calibrated cube of the same trial
ip.cb_cal_mode.setCurrentIndex(1); ip.ed_cal_ref.setText(REF); pump(0.5); grab("refcube", 0, dict(rect=rect_of(cal_group)))
# 5. calibrate (real run on flight2)
ip._calibrate(); t0 = time.time(); n = 0; last = 0
while not ip.btn_cal.isEnabled() and time.time() - t0 < 1200:
    pump(0.2)
    if time.time() - last > 2.0: grab("calibrate", n, dict(rect=rect_of(ip.progress), t=round(time.time() - t0, 1))); n += 1; last = time.time()
pump(1.0); grab("calib_done", 0, dict(rect=rect_of(cal_group))); close_boxes(); pump(0.5)
grab("check_ok", 0, dict(rect=rect_of(cal_group), label=chk_rect(), cube=ip.ed_vnir.text()))
# 6. VNIR features on the calibrated cube (real run)
ip.cb_v3_cubes.setChecked(False); ip._run_v3(); t0 = time.time(); n = 0; last = 0
while not ip.btn_v3.isEnabled() and time.time() - t0 < 1800:
    pump(0.2)
    if time.time() - last > 3.0: grab("features", n, dict(rect=rect_of(ip.progress), t=round(time.time() - t0, 1))); n += 1; last = time.time()
pump(1.0); grab("features_done", 0, dict(rect=rect_of(ip.out))); close_boxes()
json.dump(dict(steps=steps, calibrated_cube=ip.ed_vnir.text(), qa_png=os.path.splitext(ip.ed_vnir.text())[0] + "_qa.png", readme=os.path.splitext(ip.ed_vnir.text())[0] + "_README.txt"), open(os.path.join(HERE, "steps.json"), "w"), indent=1)
print("frames:", len(steps), "->", FR); win.close()
