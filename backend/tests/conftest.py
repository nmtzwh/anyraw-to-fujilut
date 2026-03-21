import sys
from pathlib import Path

# Ensure backend package is importable from repo root
backend_path = Path(__file__).resolve().parents[1]
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))
