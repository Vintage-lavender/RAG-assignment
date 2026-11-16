import json
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .base import BaseRetriever, DocEntry, RetrievalResult

META_FILE = "dense_retriever_meta.json"
INDEX_FILE = "index.faiss"
BASE_MODEL_NAME_OR_PATH = "sentence-transformers/all-MiniLM-L6-v2"


class DenseRetriever(BaseRetriever):
    def __init__(
        self,
        model_name_or_path: str = BASE_MODEL_NAME_OR_PATH,
        batch_size: int = 256,
        query_prompt: str = "",     # 모델에 따라 필요 (예: e5 → "query: ")
        doc_prompt: str = "",       # 모델에 따라 필요 (예: e5 → "passage: ")
        fp16: bool = True,          
        chunk_size: int = 100_000,
    ):
        self.model_name_or_path = model_name_or_path
        self.batch_size = batch_size
        self.query_prompt = query_prompt
        self.doc_prompt = doc_prompt
        self.chunk_size = chunk_size

        self.model = SentenceTransformer(model_name_or_path)
        if fp16 and self.model.device.type == "cuda":
            self.model.half()

        self.index: faiss.Index | None = None
        self.corpus: list[DocEntry] | None = None

    @staticmethod
    def _doc_to_index_text(doc: DocEntry) -> str:
        return f"{doc.title} {doc.text}"

    def _encode(self, texts: list[str], prompt: str, show_progress: bool) -> np.ndarray:
        emb = self.model.encode(
            texts,
            prompt=prompt or None,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return np.ascontiguousarray(emb, dtype=np.float32)

    def _meta(self) -> dict:
        return {
            "model": self.model_name_or_path,
            "query_prompt": self.query_prompt,
            "doc_prompt": self.doc_prompt,
            "dim": self.model.get_embedding_dimension(),
            "num_docs": len(self.corpus),
        }

    def _check_ready(self) -> None:
        if self.index is None or self.corpus is None:
            raise RuntimeError("인덱스가 없습니다. build_index(corpus) 또는 load(path, corpus) 를 먼저 호출하세요.")

    # ----- 인터페이스 구현 -----
    @torch.inference_mode()
    def build_index(self, corpus: list[DocEntry]) -> None:
        self.corpus = corpus
        self.index = faiss.IndexFlatIP(self.model.get_embedding_dimension())

        for start in tqdm(range(0, len(corpus), self.chunk_size), desc="Encode corpus (chunks)"):
            chunk = corpus[start: start + self.chunk_size]
            texts = [self._doc_to_index_text(d) for d in chunk]
            self.index.add(self._encode(texts, self.doc_prompt, show_progress=True))

    def save(self, index_dir: str) -> None:
        self._check_ready()
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_dir / INDEX_FILE))
        (index_dir / META_FILE).write_text(json.dumps(self._meta(), indent=2))

    def load(self, index_dir: str, corpus: list[DocEntry]) -> None:
        index_dir = Path(index_dir)
        meta = json.loads((index_dir / META_FILE).read_text())

        self.corpus = corpus
        actual = self._meta()
        if actual != meta:
            raise ValueError(f"저장된 인덱스와 현재 설정/corpus 가 다릅니다.\n  saved : {meta}\n  actual: {actual}")

        self.index = faiss.read_index(str(index_dir / INDEX_FILE))

    @torch.inference_mode()
    def batch_retrieve(self, queries: list[str], k: int) -> list[list[RetrievalResult]]:
        self._check_ready()
        k = min(k, self.index.ntotal)

        q_emb = self._encode(queries, self.query_prompt, show_progress=len(queries) > 1)
        scores, indices = self.index.search(q_emb, k)

        # indices 는 pid 가 아니라 corpus 리스트에서의 위치
        return [
            [RetrievalResult(self.corpus[i], float(s)) for i, s in zip(row_idx, row_score) if i >= 0]
            for row_idx, row_score in zip(indices, scores)
        ]
