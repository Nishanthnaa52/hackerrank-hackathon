"""
Corpus Loader — reads all markdown files from data/ and chunks them.
Adds domain and category metadata so we can filter during retrieval.
"""

import re
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import DATA_DIR, DOMAINS, CHUNK_SIZE, CHUNK_OVERLAP


def _extract_title(text: str, filename: str) -> str:
    """Pull the first markdown heading, else fall back to filename."""
    match = re.search(r"^#\s+(.+)", text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return filename.replace("-", " ").replace(".md", "").strip()


def _derive_category(rel_path: Path) -> str:
    """Turn  screen/managing-tests/article.md  into  screen/managing-tests"""
    parts = rel_path.parts[:-1]
    if not parts:
        return "general"
    return "/".join(parts)


def load_domain_corpus(domain: str) -> List[Document]:
    """
    Load all .md files for a single domain and chunk them.

    Returns:
        List of chunked LangChain Documents with metadata.
    """
    domain_dir = DATA_DIR / domain
    if not domain_dir.exists():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )

    docs: List[Document] = []
    for md_file in sorted(domain_dir.rglob("*.md")):
        text = md_file.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            continue

        rel = md_file.relative_to(domain_dir)
        title = _extract_title(text, md_file.stem)
        category = _derive_category(rel)

        # Base document
        doc = Document(
            page_content=text,
            metadata={
                "source": str(rel),
                "domain": domain,
                "category": category,
                "title": title,
            },
        )

        # Chunk it
        chunks = splitter.split_documents([doc])
        docs.extend(chunks)

    return docs


def load_all_corpora() -> List[Document]:
    """Load and chunk all domains."""
    all_docs = []
    for domain in DOMAINS:
        all_docs.extend(load_domain_corpus(domain))
    return all_docs
