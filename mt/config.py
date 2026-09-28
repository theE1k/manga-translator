"""Paths and tunables. Override any value with an environment variable of the same name."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = Path(os.environ.get("MT_MODELS", ROOT / "models"))
FONTS = Path(os.environ.get("MT_FONTS", ROOT / "fonts"))
CACHE = Path(os.environ.get("MT_CACHE", ROOT / ".cache"))

# Keep third-party model downloads inside the project (it lives on the external disk).
os.environ.setdefault("EASYOCR_MODULE_PATH", str(CACHE / "easyocr"))

DETECTOR_ONNX = MODELS / "comictextdetector.pt.onnx"
LAMA_ONNX = MODELS / "lama_fp32.onnx"

# Font used for rendering. Any .otf/.ttf/.ttc works; drop a manga font in fonts/ and point here.
FONT_PATH = Path(os.environ.get("MT_FONT", FONTS / "NotoSansCJKsc-Bold.otf"))
FONT_INDEX = int(os.environ.get("MT_FONT_INDEX", 0))  # face index inside a .ttc collection

# Translation through a local Ollama server.
OLLAMA_URL = os.environ.get("MT_OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("MT_OLLAMA_MODEL", "qwen3:4b-instruct")

DET_SIZE = 1024
DET_CONF = float(os.environ.get("MT_DET_CONF", 0.4))
DET_NMS = 0.35
MASK_THRESH = 0.3

LANG_NAMES = {
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Simplified Chinese",
    "zh-TW": "Traditional Chinese",
    "en": "English",
}
