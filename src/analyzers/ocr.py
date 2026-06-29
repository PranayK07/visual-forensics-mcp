"""OCR confidence extraction via Tesseract.

The Tesseract binary is an optional, local dependency. If it is not installed
the engine degrades gracefully: ``available`` is False, no OCR metrics are
produced, and the pipeline emits a warning instead of failing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import pytesseract

from ..utils.config import Config
from ..utils.image_ops import ensure_gray
from ..utils.logging import get_logger

logger = get_logger("ocr")


@dataclass
class OCRResult:
    ocr_confidence: float  # mean word confidence, 0.0-1.0 (0 if no words)
    word_count: int
    char_count: int
    available: bool


class OCREngine:
    """Stateful wrapper that checks Tesseract availability once."""

    def __init__(self, config: Config):
        self.config = config
        self.language = config.get("ocr.language", "eng")
        self.psm = int(config.get("ocr.psm", 6))
        self.oem = int(config.get("ocr.oem", 3))
        self.min_conf = float(config.get("ocr.min_word_confidence", 0))
        self.timeout = int(config.get("ocr.timeout", 20))
        self.available = False
        self.version: str | None = None

        cmd = config.get("ocr.tesseract_cmd", "") or ""
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd

        try:
            self.version = str(pytesseract.get_tesseract_version())
            self.available = True
            logger.info("Tesseract available: %s", self.version)
        except Exception as exc:
            logger.warning("Tesseract not available; OCR disabled: %s", exc)
            self.available = False

    def _config_string(self) -> str:
        return f"--oem {self.oem} --psm {self.psm}"

    def analyze(self, image: np.ndarray) -> OCRResult:
        """Return OCR confidence and counts for one image (e.g. a tile)."""
        if not self.available:
            return OCRResult(0.0, 0, 0, available=False)

        gray = ensure_gray(image)
        if gray.size == 0:
            return OCRResult(0.0, 0, 0, available=True)

        try:
            data = pytesseract.image_to_data(
                gray,
                lang=self.language,
                config=self._config_string(),
                output_type=pytesseract.Output.DICT,
                timeout=self.timeout,
            )
        except Exception as exc:  # pragma: no cover - runtime/runtime-timeout
            logger.warning("OCR failed on tile: %s", exc)
            return OCRResult(0.0, 0, 0, available=True)

        confidences: list[float] = []
        char_count = 0
        word_count = 0
        for text, conf in zip(data.get("text", []), data.get("conf", [])):
            try:
                conf_val = float(conf)
            except (TypeError, ValueError):
                continue
            token = (text or "").strip()
            if conf_val < 0 or not token:
                continue  # -1 conf marks non-text boxes
            if conf_val < self.min_conf:
                continue
            confidences.append(conf_val)
            word_count += 1
            char_count += len(token)

        if not confidences:
            return OCRResult(0.0, 0, 0, available=True)

        mean_conf = float(np.mean(confidences)) / 100.0  # normalise to 0-1
        return OCRResult(
            ocr_confidence=mean_conf,
            word_count=word_count,
            char_count=char_count,
            available=True,
        )
