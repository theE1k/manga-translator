"""Manga translator engine: detect -> OCR -> translate -> inpaint -> typeset."""
from . import config  # noqa: F401  (sets cache env vars before any model import)
from .types import TextBlock
from .pipeline import Pipeline

__all__ = ["TextBlock", "Pipeline"]
