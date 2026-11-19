"""Retrieval evaluation with known relevant chunk IDs (not LLM-judged)."""

def evaluate(retriever, dataset: list[dict], mode: str = 'hybrid', k: int = 5):
    if not dataset:
        raise ValueError('Evaluation dataset is empty')
    total_recall, total_mrr, total_hit = 0.0, 0.0, 0.0
    records = []
    for row in dataset:
        truth = set(row['relevant_ids'])
        if not truth:
            raise ValueError('Every evaluation question needs at least one relevant chunk ID')
        retrieved = retriever.search([row['question']], n_each=max(15, k), n_fused=k, mode=mode)
        ids = [p.id for p in retrieved]
        hits = truth.intersection(ids)
        recall = len(hits) / len(truth)
        reciprocal_rank = next((1 / (i + 1) for i, pid in enumerate(ids) if pid in truth), 0.0)
        total_recall += recall
        total_mrr += reciprocal_rank
        total_hit += bool(hits)
        records.append({'question': row['question'], 'recall': recall, 'rr': reciprocal_rank, 'hit': bool(hits)})
    size = len(dataset)
    return {f'recall@{k}': total_recall / size, f'mrr@{k}': total_mrr / size,
            f'hit@{k}': total_hit / size, 'queries': size, 'details': records}
