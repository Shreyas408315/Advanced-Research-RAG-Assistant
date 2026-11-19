"""Page-preserving PDF extraction and resilient multi-document indexing."""
from __future__ import annotations
from pathlib import Path
import hashlib
from pypdf import PdfReader
from .core import Passage, chunk_page


def load_pdf(path: Path, chunk_words: int = 220, overlap_words: int = 40) -> list[Passage]:
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        try:
            reader.decrypt('')
        except Exception as exc:
            raise ValueError(f'Encrypted PDF not readable: {path.name}') from exc
    passages = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ''
        passages.extend(chunk_page(text, path.name, page_num, digest, chunk_words, overlap_words))
    return passages


def load_directory(folder: Path, chunk_words: int = 220, overlap_words: int = 40):
    paths = sorted(Path(folder).glob('*.pdf'))
    passages, errors = [], []
    for path in paths:
        try:
            chunks = load_pdf(path, chunk_words, overlap_words)
            if not chunks:
                errors.append(f'{path.name}: no selectable text (scanned PDFs require OCR)')
            passages.extend(chunks)
        except Exception as exc:
            errors.append(f'{path.name}: {exc}')
    return passages, errors
