"""
FAISS Retriever — loads local FAISS indices and performs similarity search.
"""

from typing import Optional, List
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import INDEX_DIR, DOMAINS, TOP_K_RETRIEVAL
from indexer import _get_embeddings, indices_exist


# In-memory cache of loaded indices
_vectorstores = {}


def _load_index(domain: str) -> FAISS:
    """Load a FAISS index from disk and cache it."""
    if not indices_exist():
        raise RuntimeError(
            "FAISS indices not found. "
            "Run 'python main.py --build-index' first."
        )

    if domain not in _vectorstores:
        embeddings = _get_embeddings()
        path = str(INDEX_DIR / domain)
        _vectorstores[domain] = FAISS.load_local(
            path,
            embeddings,
            allow_dangerous_deserialization=True  # We trust our own indices
        )
    return _vectorstores[domain]


def _normalise_domain(company: Optional[str]) -> str:
    """Map company name to a domain index."""
    if not company or company.strip().lower() in ("none", ""):
        return "all"
    low = company.strip().lower()
    if low in DOMAINS:
        return low
    for d in DOMAINS:
        if d in low or low in d:
            return d
    return "all"


def get_relevant_context(
    query: str,
    company: Optional[str] = None,
    k: int = TOP_K_RETRIEVAL
) -> str:
    """
    Retrieve top-k relevant chunks from FAISS and format them.

    Args:
        query:   The support ticket issue.
        company: Target domain (hackerrank, claude, visa). Falls back to 'all'.
        k:       Number of chunks to retrieve.

    Returns:
        A formatted string of retrieved documents with metadata.
    """
    domain = _normalise_domain(company)
    vectorstore = _load_index(domain)

    # Perform similarity search
    docs: List[Document] = vectorstore.similarity_search(query, k=k)

    if not docs:
        return "No relevant documentation found."

    # Format chunks into a context string
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        header = f"--- [Doc {i}: {meta.get('title', 'Untitled')} ({meta.get('domain', '')} - {meta.get('category', '')})] ---"
        parts.append(f"{header}\n{doc.page_content}\n")

    return "\n".join(parts)
