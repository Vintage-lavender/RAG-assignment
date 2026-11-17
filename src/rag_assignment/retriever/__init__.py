from pathlib import Path

from .bm25 import BM25Retriever
from .base import BaseRetriever, DocEntry, RetrievalResult
from rag_assignment.paths import INDEX_DIR


RETRIEVERS: dict[str, type[BaseRetriever]] = {"lexical": BM25Retriever}

def get_retriever(
    mode: str = "lexical",
    corpus: list[DocEntry] | None = None,
    index_dir: str | Path | None = None,
    **kwargs,
) -> BaseRetriever:
    if mode not in RETRIEVERS:
        raise ValueError(f"Unknown mode: {mode}")
    retriever = RETRIEVERS[mode](**kwargs)
    if corpus is None:
        return retriever
    
    index_dir = Path(index_dir) if index_dir is not None else INDEX_DIR / mode
    if index_dir.exists() and any(index_dir.iterdir()):
        print(f"Load {mode} index from {index_dir}")
        retriever.load(index_dir, corpus)
    else:
        print(f"Build {mode} index")
        retriever.build_index(corpus)
        retriever.save(index_dir)
    
    return retriever


__all__ = [
    "BaseRetriever", "RetrievalResult", "BM25Retriever",
    "RETRIEVERS", "get_retriever"
]