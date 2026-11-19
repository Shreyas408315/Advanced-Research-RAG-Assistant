"""Index, inspect, ask and evaluate from the command line."""
import argparse
import json
from pathlib import Path
from config import DB_DIR, PAPERS_DIR, EMBED_MODEL, RERANK_MODEL, GEMINI_MODEL
from research_rag.ingestion import load_directory
from research_rag.retrieval import Retriever, Reranker
from research_rag.pipeline import Gemini, ResearchAssistant


def main():
    parser = argparse.ArgumentParser(description='Advanced research RAG assistant')
    sub = parser.add_subparsers(dest='action', required=True)
    index = sub.add_parser('index', help='Load PDFs and index in Chroma')
    index.add_argument('--folder', default=PAPERS_DIR)
    index.add_argument('--replace', action='store_true')
    ask = sub.add_parser('ask', help='Answer a question')
    ask.add_argument('question')
    ask.add_argument('--retrieval-mode', choices=['hybrid', 'bm25', 'semantic'], default='hybrid')
    ask.add_argument('--no-rewrite', action='store_true')
    ask.add_argument('--no-rerank', action='store_true')
    ask.add_argument('--retrieve-only', action='store_true')
    ask.add_argument('--top-k', type=int, default=6)
    ev = sub.add_parser('evaluate', help='Evaluate JSONL questions with ground truth relevant chunk IDs')
    ev.add_argument('--dataset', default='eval/questions.jsonl')
    ev.add_argument('--mode', choices=['hybrid', 'bm25', 'semantic'], default='hybrid')
    args = parser.parse_args()
    if args.action == 'index':
        chunks, errors = load_directory(Path(args.folder))
        for error in errors:
            print('WARNING:', error)
        if not chunks and not args.replace:
            print('No readable PDF chunks found; nothing indexed.')
            return
        retriever = Retriever(DB_DIR, EMBED_MODEL)
        n = retriever.index(chunks, replace=args.replace)
        print(f'Indexed {n} new chunks. Total chunks: {len(retriever.chunks)}')
        return
    retriever = Retriever(DB_DIR, EMBED_MODEL)
    if not retriever.chunks:
        parser.error('No chunks indexed. Add PDFs to data/papers and run: python cli.py index')
    if args.action == 'evaluate':
        from research_rag.evaluation import evaluate
        dataset = [json.loads(line) for line in Path(args.dataset).read_text(encoding='utf-8').splitlines() if line.strip()]
        print(json.dumps(evaluate(retriever, dataset, args.mode), indent=2))
        return
    generator = None if args.retrieve_only else Gemini(GEMINI_MODEL)
    reranker = None if args.no_rerank else Reranker(RERANK_MODEL)
    assistant = ResearchAssistant(retriever, generator, reranker)
    answer = assistant.ask(args.question, rewrite=not args.no_rewrite and not args.retrieve_only,
                           rerank=not args.no_rerank, retrieval_mode=args.retrieval_mode,
                           top_k=args.top_k)
    print('\n' + answer.text + '\n')
    for i, p in enumerate(answer.sources, 1):
        print(f'[S{i}] {p.source} p.{p.page} | chunk={p.id}')
        if args.retrieve_only:
            print('   ', p.text[:350], '\n')
    if answer.citation_warning:
        print('\nWARNING:', answer.citation_warning)


if __name__ == '__main__':
    main()
