"""
FAISS Indexer — builds local vector databases for RAG using sentence-transformers.
"""

import time
from typing import Callable, Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config import INDEX_DIR, DOMAINS, EMBEDDING_MODEL
from loader import load_domain_corpus, load_all_corpora


# Cache the embeddings model so we don't reload it
_embeddings: Optional[HuggingFaceEmbeddings] = None

def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        # This will download the model automatically on first run
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def indices_exist() -> bool:
    """Check if all domain indices exist."""
    for domain in DOMAINS:
        if not (INDEX_DIR / domain).exists():
            return False
    return (INDEX_DIR / "all").exists()


def build_all_indices(progress_cb: Optional[Callable[[str], None]] = None) -> None:
    """
    Build FAISS indices for all domains individually, plus one combined index.
    """
    embeddings = _get_embeddings()

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    all_docs = []

    for domain in DOMAINS:
        start_t = time.time()
        if progress_cb:
            progress_cb(f"Loading '{domain}' corpus…")

        docs = load_domain_corpus(domain)
        if not docs:
            if progress_cb:
                progress_cb(f"⚠️  No docs found for '{domain}'.")
            continue

        all_docs.extend(docs)

        if progress_cb:
            progress_cb(f"Vectorizing {len(docs)} chunks for '{domain}'…")

        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(str(INDEX_DIR / domain))

        if progress_cb:
            elapsed = time.time() - start_t
            progress_cb(f"✅ Built '{domain}' index in {elapsed:.1f}s.")

    # Build combined index
    if all_docs:
        start_t = time.time()
        if progress_cb:
            progress_cb(f"Vectorizing combined index ({len(all_docs)} chunks)…")

        all_store = FAISS.from_documents(all_docs, embeddings)
        all_store.save_local(str(INDEX_DIR / "all"))

        if progress_cb:
            elapsed = time.time() - start_t
            progress_cb(f"✅ Built 'all' index in {elapsed:.1f}s.")
