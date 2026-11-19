from pathlib import Path
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env')
DB_DIR = os.getenv('CHROMA_PATH', str(ROOT / 'data' / 'chroma'))
PAPERS_DIR = os.getenv('PAPERS_PATH', str(ROOT / 'data' / 'papers'))
EMBED_MODEL = os.getenv('EMBED_MODEL', 'BAAI/bge-small-en-v1.5')
RERANK_MODEL = os.getenv('RERANK_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
