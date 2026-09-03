import logging
import os
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)

# Cached OCR Reader singleton
_OCR_READER = None


def get_ocr_reader(languages: Optional[Sequence[str]] = None):
    """Lazily initialize and cache the EasyOCR Reader instance."""
    global _OCR_READER
    if _OCR_READER is None:
        try:
            import easyocr
            import torch

            lang_list = list(languages) if languages else ["en"]
            use_gpu = torch.cuda.is_available()
            logger.info(f"Initializing EasyOCR reader with languages={lang_list}, GPU={use_gpu}...")
            print(f"[OCR] Initializing EasyOCR engine (languages: {lang_list}, GPU: {use_gpu})...")
            _OCR_READER = easyocr.Reader(lang_list, gpu=use_gpu, verbose=False)
        except ImportError:
            _OCR_READER = "UNAVAILABLE"
    return _OCR_READER


def run_ocr(image_input: Union[str, Path, np.ndarray], languages: Optional[Sequence[str]] = None) -> str:
    """Extract text from an image path or numpy image array using EasyOCR or PaddleOCR."""
    reader = get_ocr_reader(languages)

    if reader != "UNAVAILABLE" and reader is not None:
        try:
            target = str(image_input) if isinstance(image_input, (str, Path)) else image_input
            results = reader.readtext(target, detail=0)
            return "\n".join(results).strip()
        except Exception as e:
            logger.warning(f"EasyOCR failed: {e}")

    # Fallback to PaddleOCR if available
    try:
        from paddleocr import PaddleOCR

        ocr = PaddleOCR(lang="en")
        target = str(image_input) if isinstance(image_input, (str, Path)) else image_input
        result = ocr.ocr(target)
        page_texts = []
        if result:
            for res in result:
                if isinstance(res, list):
                    for line in res:
                        if len(line) > 1 and isinstance(line[1], tuple):
                            page_texts.append(line[1][0])
                        elif isinstance(line, str):
                            page_texts.append(line)
        return "\n".join(page_texts).strip()
    except Exception as e:
        logger.warning(f"PaddleOCR fallback failed or unavailable: {e}")

    # Fallback to pytesseract if available
    try:
        import pytesseract
        from PIL import Image

        img = Image.open(str(image_input)) if isinstance(image_input, (str, Path)) else Image.fromarray(image_input)
        return pytesseract.image_to_string(img).strip()
    except Exception as e:
        logger.warning(f"PyTesseract fallback failed or unavailable: {e}")

    raise RuntimeError(
        "No working OCR library found. Please install easyocr (pip install easyocr) or paddleocr."
    )


class PDFLoader:
    """PDF & Document Loader with Intelligent OCR Support.

    Supports single files (.pdf, .jpg, .png, etc.) or entire directories.

    Modes:
      - 'auto': Extracts direct digital text first; automatically runs OCR on scanned/empty pages or images.
      - 'force': Forces OCR extraction on every page / image.
      - 'off': Uses direct text extraction only (no OCR).
    """

    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

    def __init__(
        self,
        pdf_path: Union[str, Path],
        ocr_mode: str = "auto",
        ocr_languages: Optional[Sequence[str]] = None,
        min_chars_threshold: int = 30,
        render_dpi: int = 200,
    ):
        self.target_path = Path(pdf_path)
        if not self.target_path.exists():
            raise FileNotFoundError(f"File or directory not found at {self.target_path}")

        self.ocr_mode = (ocr_mode or os.getenv("OCR_MODE", "auto")).lower()
        self.ocr_languages = ocr_languages or ["en"]
        self.min_chars_threshold = min_chars_threshold
        self.render_dpi = render_dpi

    def load(self) -> str:
        """Load and extract text from a single file or directory of documents."""
        if self.target_path.is_dir():
            return self._load_directory()

        suffix = self.target_path.suffix.lower()
        if suffix in self.IMAGE_EXTENSIONS:
            return self._load_image(self.target_path)
        elif suffix == ".pdf":
            return self._load_pdf(self.target_path)
        else:
            return self._load_pdf(self.target_path)

    def _load_directory(self) -> str:
        """Load and process all supported PDF and image files in a directory."""
        supported_exts = {".pdf", *self.IMAGE_EXTENSIONS}
        files = sorted(
            [
                f
                for f in self.target_path.iterdir()
                if f.is_file() and f.suffix.lower() in supported_exts
            ]
        )

        if not files:
            raise FileNotFoundError(
                f"No supported PDF or image files found in {self.target_path}"
            )

        print(
            f"[PDFLoader] Discovered {len(files)} document(s) in directory '{self.target_path.name}'"
        )
        documents_text: List[str] = []

        for file in files:
            print(f"\n[PDFLoader] Loading file: {file.name}")
            if file.suffix.lower() in self.IMAGE_EXTENSIONS:
                text = self._load_image(file)
            else:
                text = self._load_pdf(file)

            if text.strip():
                documents_text.append(f"=== Document: {file.name} ===\n{text}")

        full_text = "\n\n".join(documents_text)
        if not full_text.strip():
            raise ValueError(
                f"No text could be extracted from documents in {self.target_path}"
            )

        print(
            f"\n[PDFLoader] Finished directory loading: {len(documents_text)} documents processed ({len(full_text)} total chars)."
        )
        return full_text

    def _load_image(self, file_path: Path) -> str:
        """Run OCR directly on an image file."""
        print(f"[OCR] Running OCR on image file: {file_path.name}")
        extracted = run_ocr(file_path, self.ocr_languages)
        if not extracted:
            logger.warning(f"No text could be extracted via OCR from {file_path}")
            return ""
        print(f"[OCR] Extracted {len(extracted)} chars from {file_path.name}")
        return extracted

    def _load_pdf(self, file_path: Path) -> str:
        """Extract text from PDF pages using direct extraction and/or OCR."""
        try:
            import pymupdf

            doc = pymupdf.open(str(file_path))
        except ImportError:
            return self._load_pdf_fallback_pypdf(file_path)

        pages_text: List[str] = []
        total_pages = len(doc)
        print(
            f"[PDFLoader] Processing '{file_path.name}' ({total_pages} pages, mode='{self.ocr_mode}')"
        )

        for page_idx in range(total_pages):
            page_num = page_idx + 1
            page = doc[page_idx]
            page_content = ""

            if self.ocr_mode == "force":
                print(f"  Page {page_num}/{total_pages}: Running OCR (Forced)...")
                page_content = self._ocr_pymupdf_page(page)
            elif self.ocr_mode == "off":
                page_content = (page.get_text() or "").strip()
                print(
                    f"  Page {page_num}/{total_pages}: Direct text extracted ({len(page_content)} chars)"
                )
            else:
                # "auto" mode: try direct text first, fallback to OCR if empty or scanned
                direct_text = (page.get_text() or "").strip()
                if len(direct_text) >= self.min_chars_threshold:
                    print(
                        f"  Page {page_num}/{total_pages}: Direct text extracted ({len(direct_text)} chars)"
                    )
                    page_content = direct_text
                else:
                    print(
                        f"  Page {page_num}/{total_pages}: Low/no embedded text ({len(direct_text)} chars) -> Running OCR..."
                    )
                    ocr_text = self._ocr_pymupdf_page(page)
                    if ocr_text:
                        print(
                            f"  Page {page_num}/{total_pages}: OCR extracted ({len(ocr_text)} chars)"
                        )
                        page_content = ocr_text
                    else:
                        page_content = direct_text

            if page_content.strip():
                pages_text.append(page_content.strip())

        doc.close()

        full_text = "\n\n".join(pages_text)
        if not full_text.strip():
            logger.warning(f"No text extracted from PDF: {file_path}")
            return ""

        print(
            f"[PDFLoader] Completed '{file_path.name}': {len(pages_text)} pages extracted ({len(full_text)} chars)."
        )
        return full_text

    def _ocr_pymupdf_page(self, page) -> str:
        """Render a PyMuPDF page to image array and run OCR."""
        try:
            pix = page.get_pixmap(dpi=self.render_dpi)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            if pix.n == 4:
                img = img[:, :, :3]

            return run_ocr(img, self.ocr_languages)
        except Exception as e:
            logger.error(f"Error during OCR extraction on page: {e}")
            return ""

    def _load_pdf_fallback_pypdf(self, file_path: Path) -> str:
        """Fallback loader using pypdf if PyMuPDF is unavailable."""
        from pypdf import PdfReader

        reader = PdfReader(str(file_path))
        pages_text: List[str] = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages_text.append(text)

        full_text = "\n\n".join(pages_text)
        return full_text