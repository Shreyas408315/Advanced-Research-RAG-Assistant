"""Pure, independently testable document, fusion, compression, and citation utilities."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import re
from typing import Iterable


@dataclass(frozen=True)
class Passage:
    id: str
    text: str
    source: str
    page: int
    doc_id: str

    def metadata(self) -> dict:
        return {'source': self.source, 'page': self.page, 'doc_id': self.doc_id}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w]+", text.lower(), flags=re.UNICODE)


def chunk_page(text: str, source: str, page: int, doc_id: str,
               chunk_words: int = 220, overlap_words: int = 40) -> list[Passage]:
    if chunk_words < 1 or not 0 <= overlap_words < chunk_words:
        raise ValueError('Require chunk_words > overlap_words >= 0')
    words = text.split()
    if not words:
        return []
    result = []
    step = chunk_words - overlap_words
    for start in range(0, len(words), step):
        part = words[start:start + chunk_words]
        if not part:
            break
        key = f'{doc_id}|{page}|{start}'
        pid = hashlib.sha256(key.encode()).hexdigest()[:24]
        result.append(Passage(pid, ' '.join(part), source, page, doc_id))
        if start + chunk_words >= len(words):
            break
    return result


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    if k < 1:
        raise ValueError('k must be positive')
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, pid in enumerate(dict.fromkeys(ranking), 1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


def compress(passages: list[Passage], query: str, max_words: int = 1200,
             max_words_per_passage: int = 270) -> list[Passage]:
    """Extractive, metadata-safe compression; never invents evidence."""
    if max_words < 1 or max_words_per_passage < 1:
        return []
    terms = set(tokenize(query))
    output: list[Passage] = []
    remaining = max_words
    for p in passages:
        if remaining <= 0:
            break
        sentences = re.split(r'(?<=[.!?])\s+', p.text)
        if len(sentences) == 1:
            selected = p.text.split()[:min(remaining, max_words_per_passage)]
        else:
            scored = sorted(enumerate(sentences), key=lambda x: (-len(set(tokenize(x[1])) & terms), x[0]))
            budget = min(remaining, max_words_per_passage)
            chosen = []
            for idx, sentence in scored:
                tokens = sentence.split()
                if len(tokens) <= budget:
                    chosen.append((idx, sentence))
                    budget -= len(tokens)
                elif not chosen and budget:
                    chosen.append((idx, ' '.join(tokens[:budget])))
                    budget = 0
                if budget == 0:
                    break
            selected = ' '.join(text for _, text in sorted(chosen)).split()
        if selected:
            output.append(Passage(p.id, ' '.join(selected), p.source, p.page, p.doc_id))
            remaining -= len(selected)
    return output


def evidence_block(passages: list[Passage]) -> str:
    return '\n\n'.join(f'[S{i}] {p.source}, page {p.page}, chunk {p.id}\n{p.text}'
                       for i, p in enumerate(passages, 1))


def check_citations(answer: str, passages: list[Passage]) -> tuple[bool, list[str]]:
    """Syntax check only, not semantic claim verification."""
    used = re.findall(r'\[S\d+\]', answer)
    allowed = {f'[S{i}]' for i in range(1, len(passages) + 1)}
    invalid = sorted(set(used) - allowed)
    return bool(used) and not invalid, invalid
