"""
Download the LibriSpeech test-clean dataset.

Author: SemBind-Audio Research
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path
import urllib.request
import tarfile

URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
ARCHIVE = RAW_DIR / "test-clean.tar.gz"

RAW_DIR.mkdir(parents=True, exist_ok=True)

print("Downloading LibriSpeech test-clean...")

urllib.request.urlretrieve(URL, ARCHIVE)

print("Extracting...")

with tarfile.open(ARCHIVE) as tar:
    tar.extractall(RAW_DIR)

print("Done!")

print(f"Dataset extracted to:\n{RAW_DIR}")