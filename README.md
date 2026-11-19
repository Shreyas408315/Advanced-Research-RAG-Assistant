# Advanced Research RAG Assistant

End-to-end multi-PDF research RAG: `pypdf` → page-aware chunks → BGE embeddings / ChromaDB and BM25 → RRF → cross-encoder → extractive context compression → Gemini → inline source citations.

## Quick start (Python 3.11 or 3.12 recommended)

```bash
python -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env  # on Windows use: copy .env.example .env
```

Set your actual Gemini API key in `.env` (never commit it). Download the embedding and reranking models on the first run (internet required). Some models and Gemini calls may incur download/API costs.

Put at least two **text-based, selectable** research PDFs in `data/papers/`.

```bash
python cli.py index
python cli.py ask "Compare the research methods and findings across the papers"
python cli.py ask "What methods do these papers use?" --retrieve-only
streamlit run app.py
pytest -q
```

Rebuild all stored chunks using `python cli.py index --replace`. For scanned/image-only PDFs, perform OCR beforehand. The UI also accepts multiple uploaded PDFs.

## Retrieval evaluation

After indexing, obtain actual chunk IDs with `python cli.py ask "..." --retrieve-only` (the query will print candidates and IDs). Manually label questions and all known relevant IDs in `eval/questions.jsonl` using the example file.

```bash
python cli.py evaluate --dataset eval/questions.jsonl --mode bm25
python cli.py evaluate --dataset eval/questions.jsonl --mode semantic
python cli.py evaluate --dataset eval/questions.jsonl --mode hybrid
```

The CLI reports recall@5, MRR@5, and hit@5. For a fair benchmark, expand the labeled relevant set and hold out questions from index construction. No automatic faithfulness metric is claimed; validate citations and grounding by human review. Reranking is used in answering; this evaluation command measures first-stage retrieval, not reranked retrieval.

## Design notes and limitations

- One `Passage` always references one source PDF, one 1-based page number, and a deterministic content hash ID. Compression reuses IDs and page metadata.
- The BM25 index is reconstructed from Chroma's persisted documents on startup; it is not separately persisted. Practical for a few hundred/thousand chunks; optimize for a large corpus.
- Query rewriting requires Gemini and is skipped for `--retrieve-only` or falls back to the original question if rewriting fails.
- The embedding model is English-first; change models for other languages and then use `index --replace`.
- `--replace` resets the entire collection. Removed/modified source PDFs do not automatically remove old chunks without `--replace`.
- Indexing is idempotent for unchanged files. Renaming an identical PDF leaves its original source filename until reindexing.
- Retrieved snippets are evidence, not automatically verified statements. A citation syntax check verifies only inline IDs; it does not prove that every factual claim is entailed by the evidence. Inspect the cited passages.
- Passage extraction is based on PDF text; complicated reading orders, tables, formulas, and image-only pages may require specialized parsing.
- No access control or prompt-injection guarantees. Treat paper contents as untrusted; do not deploy this local prototype as a multi-tenant public service without hardening.

## Project layout

- `research_rag/core.py` — chunks, IDs, RRF, compression, citation checks
- `research_rag/ingestion.py` — multi-PDF extraction
- `research_rag/retrieval.py` — BGE + ChromaDB, BM25, reranking
- `research_rag/pipeline.py` — query rewriting and Gemini synthesis
- `research_rag/evaluation.py` — retrieval metrics
- `cli.py` — index, ask, evaluate
- `app.py` — Streamlit front end
- `tests/` — deterministic unit tests
