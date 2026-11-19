"""Optional Streamlit UI. Run: streamlit run app.py"""
from pathlib import Path
import tempfile
import streamlit as st
from config import DB_DIR, PAPERS_DIR, EMBED_MODEL, RERANK_MODEL, GEMINI_MODEL
from research_rag.ingestion import load_directory
from research_rag.retrieval import Retriever, Reranker
from research_rag.pipeline import Gemini, ResearchAssistant

st.set_page_config(page_title='Research RAG Assistant', page_icon='📚', layout='wide')
st.title('📚 Advanced Research RAG Assistant')
st.caption('Hybrid BM25 + BGE search • RRF fusion • cross-encoder reranking • extractive compression • cited Gemini answers')

@st.cache_resource

def retriever():
    return Retriever(DB_DIR, EMBED_MODEL)

@st.cache_resource

def reranker():
    return Reranker(RERANK_MODEL)

with st.sidebar:
    st.header('Papers')
    uploaded = st.file_uploader('Upload readable PDFs', type=['pdf'], accept_multiple_files=True)
    if st.button('Save & index uploaded PDFs', disabled=not uploaded):
        folder = Path(PAPERS_DIR)
        folder.mkdir(parents=True, exist_ok=True)
        for item in uploaded:
            # Path(name).name removes traversal components; unique names prevent overwrites.
            import hashlib
            file_hash = hashlib.sha256(item.getvalue()).hexdigest()[:10]
            path = folder / f'{file_hash}_{Path(item.name).name}'
            path.write_bytes(item.getvalue())
        passages, errors = load_directory(folder)
        for error in errors:
            st.warning(error)
        with st.spinner('Building index...'):
            n = retriever().index(passages)
        st.success(f'Indexed {n} new chunks')
    st.metric('Indexed chunks', len(retriever().chunks))
    mode = st.selectbox('Retrieval', ['hybrid', 'semantic', 'bm25'])
    rewrite = st.checkbox('Query rewriting', True)
    use_rerank = st.checkbox('Cross-encoder reranking', True)
    top_k = st.slider('Number of evidence chunks', 2, 12, 6)

question = st.text_area('Ask a research question', placeholder='How do the papers compare their methods and results?')
if st.button('Research', type='primary') and question.strip():
    try:
        with st.spinner('Retrieving and synthesizing evidence...'):
            assistant = ResearchAssistant(retriever(), Gemini(GEMINI_MODEL), reranker() if use_rerank else None)
            answer = assistant.ask(question, rewrite=rewrite, rerank=use_rerank,
                                   retrieval_mode=mode, top_k=top_k)
        st.markdown(answer.text)
        if answer.citation_warning:
            st.warning(answer.citation_warning)
        with st.expander('Queries used'):
            st.write(answer.queries)
        st.subheader('Evidence')
        for i, p in enumerate(answer.sources, 1):
            with st.expander(f'[S{i}] {p.source} — page {p.page}'):
                st.caption(f'Chunk ID: {p.id}')
                st.write(p.text)
    except Exception as exc:
        st.error(f'Unable to complete request: {exc}')
