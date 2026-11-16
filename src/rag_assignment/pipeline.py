from pathlib import Path
from dataclasses import asdict
import json

from rag_assignment.retriever import BaseRetriever
from rag_assignment.generator import HFGenerator, GenerationResult
from rag_assignment.grading import EfficiencyTracker, EfficiencyResult
from rag_assignment.utils import QAEntry
from rag_assignment.paths import OUTPUTS_DIR


class RAGPipeline:
    def __init__(
        self,
        retriever: BaseRetriever,
        generator: HFGenerator,
        retrieval_k: int = 100,
        context_k: int = 5
    ):
        self.retriever = retriever
        self.generator = generator
        self.retrieval_k = retrieval_k
        self.context_k = context_k

    def run(
        self,
        qa_entries: list[QAEntry],
        use_context: bool = True,
        output_dir: Path = OUTPUTS_DIR,
    ) -> list[GenerationResult]:
        questions = [qa_entry.question for qa_entry in qa_entries]
        retrieval_results = [[] for _ in questions]
        context_list = None
        tracker = EfficiencyTracker(num_questions=len(questions))

        # 1. retrieval
        if use_context:
            with tracker.measure("retrieval"):
                retrieval_results = self.retriever.batch_retrieve(questions, self.retrieval_k)

            # 2. slice the context
            context_list = []
            for ret_results in retrieval_results:
                sliced = ret_results[:self.context_k]
                context_list.append([s.doc for s in sliced])
            
        # 3. generate
        with tracker.measure("generation"):
            generation_results = self.generator.batch_generate(questions, context_list)

        # 4. collect results
        retrieval_records = [
            {"qid": qa.qid, "pids": [r.doc.pid for r in ret_results], "scores": [r.score for r in ret_results]}
            for qa, ret_results in zip(qa_entries, retrieval_results)
        ]

        predictions = [
            {"qid": qa.qid, "question": qa.question, **gen._asdict()}
            for qa, gen in zip(qa_entries, generation_results)
        ]
        efficiency = tracker.result()

        # 5. save results
        self.save_results(retrieval_records, predictions, efficiency, output_dir)
        return generation_results
    
    @staticmethod
    def save_results(
        retrieval_records: list[dict] | None,
        predictions: list[dict],
        efficiency: EfficiencyResult,
        output_dir: Path
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with (output_dir / "retrieval.jsonl").open("w", encoding="utf-8") as f:
            for rec in retrieval_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        
        with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(json.dumps(pred, ensure_ascii=False) + "\n")
        
        (output_dir / "efficiency.json").write_text(json.dumps(asdict(efficiency), indent=2, ensure_ascii=False))