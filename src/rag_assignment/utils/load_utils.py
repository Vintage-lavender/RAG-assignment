from dataclasses import dataclass
from tqdm import tqdm
from pathlib import Path
import json


# ===== Entry Definition ==================================================
@dataclass
class QAEntry:
    qid: int
    question: str
    answers: list[str]


@dataclass
class QrelEntry:
    qid: int
    gold: list[int]
    weak_pos: list[int]
    hard_neg: list[int]


@dataclass
class DocEntry:
    pid: int
    title: str
    text: str
# ===== Entry Definition ==================================================


def load_qa_dataset(filepath: Path) -> list[QAEntry]:
    qa_dataset = []
    with filepath.open("r", encoding="utf-8") as f:
        for entry in tqdm(f, desc=f"Load {filepath}..."):
            qa_entry = QAEntry(**json.loads(entry))
            qa_dataset.append(qa_entry)
    return qa_dataset


def load_corpus(filepath: Path) -> list[DocEntry]:
    corpus = []
    with filepath.open("r", encoding="utf-8") as f:
        for entry in tqdm(f, desc=f"Load {filepath}..."):
            doc_entry = DocEntry(**json.loads(entry))
            corpus.append(doc_entry)
    return corpus


def load_qrels(filepath: Path) -> list[QrelEntry]:
    qrels = []
    with filepath.open("r", encoding="utf-8") as f:
        for entry in tqdm(f, desc=f"Load {filepath}..."):
            qrel_entry = QrelEntry(**json.loads(entry))
            qrels.append(qrel_entry)
    return qrels


if __name__ == "__main__":
    from rag_assignment.paths import DATA_DIR
    qa_dataset = load_qa_dataset(DATA_DIR / "dev_qa.jsonl")
    print(len(qa_dataset))
    print(qa_dataset[0])