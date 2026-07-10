"""Top-level launcher: `python run.py`."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phenoapp.app import main
main()
