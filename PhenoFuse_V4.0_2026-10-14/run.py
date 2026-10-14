"""Top-level launcher: `python run.py`. Writes any uncaught error
(Python exception OR native segfault) to phenoapp_crash.log next to the EXE."""
import os, sys, faulthandler, traceback

_logdir  = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) \
           else os.path.dirname(os.path.abspath(__file__))
_logpath = os.path.join(_logdir, "phenoapp_crash.log")
try:
    _logfh = open(_logpath, "w", buffering=1, encoding="utf-8")
except Exception:
    _logfh = sys.stderr
faulthandler.enable(file=_logfh, all_threads=True)

def _excepthook(exctype, value, tb):
    _logfh.write("=== UNCAUGHT EXCEPTION (main thread) ===\n")
    traceback.print_exception(exctype, value, tb, file=_logfh)
    _logfh.flush()
    sys.__excepthook__(exctype, value, tb)
sys.excepthook = _excepthook

# Worker QThreads run their own loops; install a thread-level handler too.
import threading
def _thread_excepthook(args):
    _logfh.write(f"=== UNCAUGHT EXCEPTION ({args.thread.name}) ===\n")
    traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback,
                              file=_logfh)
    _logfh.flush()
threading.excepthook = _thread_excepthook

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_logfh.write("=== run.py boot OK ===\n")
_logfh.flush()
from phenoapp.app import main
main()
