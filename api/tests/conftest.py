import sys
from pathlib import Path

# Allow tests to import diagnose directly from api/src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
