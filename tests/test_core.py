import pytest
from research_rag.core import chunk_page, compress, Passage, rrf, check_citations, evidence_block
from research_rag.evaluation import evaluate


def test_chunking_preserves_pages_and_ids():
    a = chunk_page('one two three four five six seven', 'study.pdf', 3, 'sha', 4, 1)
    assert [p.text for p in a] == ['one two three four', 'four five six seven']
    assert all(p.page == 3 and p.source == 'study.pdf' for p in a)
    assert a == chunk_page('one two three four five six seven', 'study.pdf', 3, 'sha', 4, 1)


def test_rrf_combines_rankings():
    result = rrf([['a', 'b'], ['b', 'c']])
    assert result[0][0] == 'b'
    assert len(result) == 3


def test_rrf_deduplicates_within_rankings():
    assert rrf([['x', 'x']]) == [('x', 1/61)]


def test_compression_keeps_traceability():
    p = Passage('c1', 'Cats have whiskers. Dogs can bark. Cats chase mice.', 'paper.pdf', 7, 'doc')
    out = compress([p], 'cats whiskers', max_words=7, max_words_per_passage=7)
    assert len(out) == 1 and out[0].page == 7 and out[0].id == 'c1'
    assert 'whiskers' in out[0].text
    assert '[S1] paper.pdf, page 7' in evidence_block(out)


def test_citation_validation():
    passage = Passage('x', 'Evidence', 'study.pdf', 1, 'sha')
    assert check_citations('Claim [S1]', [passage])[0]
    assert not check_citations('Claim [S2]', [passage])[0]
    assert not check_citations('Claim without citation', [passage])[0]


def test_evaluation():
    p1 = Passage('x', 'foo', 'x.pdf', 1, 'doc')
    p2 = Passage('y', 'bar', 'y.pdf', 1, 'doc')
    class FakeRetriever:
        def search(self, *args, **kwargs):
            return [p1, p2]
    result = evaluate(FakeRetriever(), [{'question': 'test', 'relevant_ids': ['y']}], k=2)
    assert result['recall@2'] == 1.0
    assert result['mrr@2'] == 0.5
