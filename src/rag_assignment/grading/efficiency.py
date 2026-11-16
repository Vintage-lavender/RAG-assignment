# ========== !DO NOT CHANGE THIS FILE! ==========
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch


@dataclass
class EfficiencyResult:
    num_questions: int
    device: str                                         # 측정 환경 확인용
    stage_seconds: dict[str, float]                     # 단계별 총 소요 시간
    seconds_per_question: dict[str, float]              # 단계별 질문당 평균 시간
    total_seconds: float

def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _device_name() -> str:
    return torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"


class EfficiencyTracker:
    def __init__(self, num_questions: int):
        self.num_questions = num_questions
        self.stage_seconds: dict[str, float] = {}

    @contextmanager
    def measure(self, stage: str):
        _sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            _sync()
            self.stage_seconds[stage] = self.stage_seconds.get(stage, 0.0) + time.perf_counter() - start

    def result(self) -> EfficiencyResult:
        n = max(self.num_questions, 1)
        return EfficiencyResult(
            num_questions=self.num_questions,
            device=_device_name(),
            stage_seconds=dict(self.stage_seconds),
            seconds_per_question={k: v / n for k, v in self.stage_seconds.items()},
            total_seconds=sum(self.stage_seconds.values()),
        )
