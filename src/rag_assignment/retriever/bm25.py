import json
import bm25s
from pathlib import Path

from .base import BaseRetriever, DocEntry, RetrievalResult

META_FILE = "bm25_retriever_meta.json"


class BM25Retriever(BaseRetriever):
    def __init__(self, k1: float = 1.5, b: float = 0.75, stopwords: str = "en"):
        self.k1 = k1
        self.b = b
        self.stopwords = stopwords
        self.model: bm25s.BM25 | None = None
        self.corpus: list[DocEntry] | None = None

    @staticmethod
    def _doc_to_index_text(doc: DocEntry) -> str:
        return f"{doc.title} {doc.text}"

    def _tokenize(self, texts: list[str], show_progress: bool):
        return bm25s.tokenize(texts, stopwords=self.stopwords, show_progress=show_progress)

    def _meta(self) -> dict:
        return {
            "k1": self.k1,
            "b": self.b,
            "stopwords": self.stopwords,
            "num_docs": len(self.corpus),
        }

    def _check_ready(self) -> None:
        if self.model is None or self.corpus is None:
            raise RuntimeError("인덱스가 없습니다. build_index(corpus) 또는 load(path, corpus) 를 먼저 호출하세요.")

    # ----- 인터페이스 구현 -----
    def build_index(self, corpus: list[DocEntry]) -> None:
        self.corpus = corpus
        tokens = self._tokenize([self._doc_to_index_text(d) for d in corpus], show_progress=True)
        self.model = bm25s.BM25(k1=self.k1, b=self.b)
        self.model.index(tokens)

    def save(self, index_dir: str) -> None:
        self._check_ready()
        index_dir = Path(index_dir)
        self.model.save(index_dir)
        (index_dir / META_FILE).write_text(json.dumps(self._meta(), indent=2))

    def load(self, index_dir: str, corpus: list[DocEntry]) -> None:
        index_dir = Path(index_dir)
        meta = json.loads((index_dir / META_FILE).read_text())

        self.k1, self.b, self.stopwords = meta["k1"], meta["b"], meta["stopwords"]
        self.corpus = corpus
        actual = self._meta()
        if actual != meta:
            raise ValueError(f"저장된 인덱스와 corpus 가 다릅니다.\n  saved : {meta}\n  corpus: {actual}")

        self.model = bm25s.BM25.load(index_dir)

    def batch_retrieve(self, queries: list[str], k: int) -> list[list[RetrievalResult]]:
        self._check_ready()
        k = min(k, len(self.corpus))
        show = len(queries) > 1

        query_tokens = self._tokenize(queries, show_progress=show)
        indices, scores = self.model.retrieve(query_tokens, k=k, show_progress=show)

        # indices 는 pid 가 아니라 corpus 리스트에서의 위치
        return [
            [RetrievalResult(self.corpus[i], float(s)) for i, s in zip(row_idx, row_score)]
            for row_idx, row_score in zip(indices, scores)
        ]
