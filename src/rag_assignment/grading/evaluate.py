# ========== !DO NOT CHANGE THIS FILE! ==========
import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from tqdm import tqdm

from .metrics import KVALUES, MetricResult, compute_metrics
from rag_assignment.utils import DocEntry, QAEntry, load_qa_dataset


@dataclass
class InferenceResult:
    qid: int
    question: str
    pred_answer: str
    answers: list[str]
    metrics: MetricResult


# ===== Load ==================================================
def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_pid2doc(corpus: list[DocEntry] | str | Path, pids: set[int]) -> dict[int, DocEntry]:
    if not isinstance(corpus, (str, Path)):
        return {d.pid: d for d in corpus if d.pid in pids}

    pid2doc = {}
    with Path(corpus).open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Load needed passages"):
            entry = json.loads(line)
            if entry["pid"] in pids:
                pid2doc[entry["pid"]] = DocEntry(**entry)
    return pid2doc
# ===== Load ==================================================


# ===== Evaluate ==================================================
def evaluate(
    output_dir: str | Path,
    qa_entries: list[QAEntry],
    corpus: list[DocEntry] | str | Path,
    kvalues: list[int] = KVALUES,
    save: bool = True,
) -> tuple[dict, list[InferenceResult]]:
    output_dir = Path(output_dir)
    retrieval = {r["qid"]: r for r in _read_jsonl(output_dir / "retrieval.jsonl")}
    predictions = {p["qid"]: p for p in _read_jsonl(output_dir / "predictions.jsonl")}

    # ----- 제출물 검증 -----
    missing = [qa.qid for qa in qa_entries if qa.qid not in predictions or qa.qid not in retrieval]
    if missing:
        raise ValueError(f"결과가 없는 qid {len(missing)}개: {missing[:10]} ...")

    max_k = max(kvalues)
    short = [qa.qid for qa in qa_entries if 0 < len(retrieval[qa.qid]["pids"]) < max_k]
    if short:
        raise ValueError(f"검색 결과가 {max_k}개 미만인 qid {len(short)}개: {short[:10]} ... (retrieval_k >= {max_k} 로 실행하세요)")

    needed = {pid for qa in qa_entries for pid in retrieval[qa.qid]["pids"][:max_k]}
    pid2doc = _load_pid2doc(corpus, needed)
    unknown = needed - pid2doc.keys()
    if unknown:
        raise ValueError(f"corpus 에 없는 pid {len(unknown)}개: {sorted(unknown)[:10]} ...")

    # ----- 질문별 채점 -----
    results: list[InferenceResult] = []
    for qa in tqdm(qa_entries, desc="Evaluate"):
        pred = predictions[qa.qid]["text"]
        passages = [pid2doc[pid].text for pid in retrieval[qa.qid]["pids"][:max_k]]
        metrics = compute_metrics(passages, pred, qa.answers, kvalues)
        results.append(InferenceResult(qa.qid, qa.question, pred, qa.answers, metrics))

    # ----- 요약 -----
    n = len(results)
    summary = {
        "num_questions": n,
        "span_em": sum(r.metrics.span_em for r in results) / n,
        "f1": sum(r.metrics.f1 for r in results) / n,
        "recall_at_k": {k: sum(r.metrics.recall_at_k[k] for r in results) / n for k in kvalues},
        "closed_book": all(len(retrieval[qa.qid]["pids"]) == 0 for qa in qa_entries),
        "tokens": {
            "input_mean": sum(predictions[qa.qid]["n_input_tokens"] for qa in qa_entries) / n,
            "output_mean": sum(predictions[qa.qid]["n_output_tokens"] for qa in qa_entries) / n
        }
    }
    eff_path = output_dir / "efficiency.json"
    if eff_path.exists():
        summary["efficiency"] = json.loads(eff_path.read_text())

    if save:
        with (output_dir / "inference_results.jsonl").open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    return summary, results


def print_summary(summary: dict) -> None:
    print(f"#questions : {summary['num_questions']}")
    print(f"span-EM    : {summary['span_em']:.4f}")
    print(f"F1         : {summary['f1']:.4f}")
    if summary["closed_book"]:
        print("recall@k   : - (closed-book)")
    else:
        print("recall@k   : " + "  ".join(f"@{k}={v:.4f}" for k, v in summary["recall_at_k"].items()))
    tok = summary["tokens"]
    print(f"tokens/q   : input={tok['input_mean']:.1f}  output={tok['output_mean']:.1f}")

    if "efficiency" in summary:
        eff = summary["efficiency"]
        stages = "  ".join(f"{k}={v:.1f}s" for k, v in eff["stage_seconds"].items())
        print(f"time       : {stages}  (total {eff['total_seconds']:.1f}s on {eff['device']})")
# ===== Evaluate ==================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG pipeline 결과 채점")
    parser.add_argument("--output_dir", required=True, help="retrieval.jsonl / predictions.jsonl 이 있는 폴더")
    parser.add_argument("--qa", required=True, help="정답 파일 (예: data/dev_qa.jsonl)")
    parser.add_argument("--corpus", required=True, help="corpus.jsonl 경로")
    args = parser.parse_args()

    summary, _ = evaluate(args.output_dir, load_qa_dataset(Path(args.qa)), Path(args.corpus))
    print_summary(summary)
