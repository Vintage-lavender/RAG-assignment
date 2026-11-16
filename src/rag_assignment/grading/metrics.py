# ========== !DO NOT CHANGE THIS FILE! ==========
import re
import string
import regex
from collections import Counter
from dataclasses import dataclass


KVALUES = [1, 5, 10, 20, 100]

@dataclass
class MetricResult:
    span_em: bool
    f1: float
    recall_at_k: dict[int, bool]


class SimpleTokenizer(object):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens


def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(" " if ch in exclude else ch for ch in text)

    def lower(text):
        return text.lower()

    def replace_num(text):
        word_to_number = {
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9"
        }

        pattern = re.compile(r'\b(' + '|'.join(word_to_number.keys()) + r')\b')
        text = pattern.sub(lambda x: word_to_number[x.group()], text)

        return text

    return replace_num(
        white_space_fix(
            remove_articles(
                remove_punc(lower(s))
            )
        )
    )


def has_answer(a_pred: str, a_true: list[str], tokenizer=SimpleTokenizer()) -> bool:
    """Check if a document contains an answer string."""
    a_pred = normalize_answer(a_pred)
    pred_tokens = tokenizer.tokenize(a_pred, uncased=True)

    for ans in a_true:
        ans = normalize_answer(ans)
        ans = tokenizer.tokenize(ans, uncased=True)
        # ignore empty answer after being normalized
        if not ans:
            continue

        for i in range(0, len(pred_tokens) - len(ans) + 1):
            if ans == pred_tokens[i: i + len(ans)]:
                return True
    return False


def _token_f1(pred_tokens: list[str], ans_tokens: list[str]) -> float:
    common = Counter(pred_tokens) & Counter(ans_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ans_tokens)
    return 2 * precision * recall / (precision + recall)


def f1_score(a_pred: str, a_true: list[str]) -> float:
    """SQuAD-style F1"""
    pred_tokens = normalize_answer(a_pred).split()
    ans_tokens_list = [normalize_answer(ans).split() for ans in a_true]

    if not pred_tokens:
        return 0.0

    scores = [
        _token_f1(pred_tokens, ans_tokens)
        for ans_tokens in ans_tokens_list
        if ans_tokens
    ]
    return max(scores, default=0.0)


### Retriever
def answer_recall_at_k(
    passages: list[str],
    a_true: list[str],
    kvalues: list[int] = KVALUES
) -> dict[int, bool]:
    """answer recall@k per question."""
    first_hit = None
    for rank, text in enumerate(passages[: max(kvalues)]):
        if has_answer(text, a_true):
            first_hit = rank
            break
    return {k: first_hit is not None and first_hit < k for k in kvalues}


### Return all metrics
def compute_metrics(
    passages: list[str],
    a_pred: str,
    a_true: list[str],
    kvalues: list[int] = KVALUES
) -> MetricResult:
    if isinstance(a_true, str):
        a_true = [a_true]
    
    # generator
    span_em = has_answer(a_pred, a_true)
    f1 = f1_score(a_pred, a_true)
    # retriever
    recall_at_k = answer_recall_at_k(passages, a_true, kvalues)

    return MetricResult(
        span_em=span_em,
        f1=f1,
        recall_at_k=recall_at_k
    )


# if __name__ == "__main__":
#     trues = ["Paris", "the City of Light"]
#     assert compute_metrics("It is Paris.", trues).span_em
#     assert not compute_metrics("Lyon", trues).span_em
#     assert not has_answer("anything", ["the"])                      # 빈 정답은 매칭 안 됨
#     assert abs(f1_score("Paris France", trues) - 2 / 3) < 1e-9      # 후보별 F1 의 max
#     assert answer_recall_at_k(["Lyon is big", "Paris is nice"], trues, [1, 2]) == {1: False, 2: True}
#     print("ok")