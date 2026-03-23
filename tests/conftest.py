from pathlib import Path
import os


TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
MPL_CACHE_DIR = REPO_ROOT / ".pytest_mpl_cache"

MPL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
