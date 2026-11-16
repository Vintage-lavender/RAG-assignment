from abc import abstractmethod, ABC
from typing import NamedTuple

from rag_assignment.utils.load_utils import DocEntry


class RetrievalResult(NamedTuple):
    doc: DocEntry
    score: float


class BaseRetriever(ABC):
    @abstractmethod
    def build_index(self, corpus: list[DocEntry]) -> None:
        raise NotImplementedError

    def save(self, index_dir: str) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support save()")

    def load(self, index_dir: str, corpus: list[DocEntry]) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support load()")

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        return self.batch_retrieve([query], k)[0]

    @abstractmethod
    def batch_retrieve(self, queries: list[str], k: int) -> list[list[RetrievalResult]]:
        raise NotImplementedError