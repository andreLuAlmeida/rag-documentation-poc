from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DOCUMENTS_DIR = BASE_DIR / "documents"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "documentacao_tecnica"

EMBEDDING_MODEL = "gemini-embedding-2"

GEMINI_MODEL = "gemini-3.5-flash-lite"

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150

TOP_K = 4