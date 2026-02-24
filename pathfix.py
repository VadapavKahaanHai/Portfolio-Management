# pathfix.py — Ensures the project root is always on sys.path
# Import this as the FIRST thing in any module: import pathfix
#
# Works correctly regardless of:
#   - Current working directory
#   - How the script was invoked (python main.py, python -m, double-click)
#   - Windows or Linux

import sys
import os

# Resolve the absolute path of THIS file (pathfix.py lives in project root)
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Also change working directory to project root so relative paths
# (data/, models/, outputs/) always resolve correctly
os.chdir(_PROJECT_ROOT)
