import re

from rag_assignment.utils.load_utils import DocEntry


# ===== System Prompt ==================================================
SYSTEM_PROMPT = (
    "You are a helpful assistant that answers the question. "
    "Answer with a short phrase only, without any explanation."
)
# ===== System Prompt ==================================================


# ===== Base Template ==================================================
PURE_TEMPLATE = """\
Question: {question}
Answer: """

RAG_TEMPLATE = """\
{retrieved_passages}\n
Context is provided above. Read the context carefully and answer the question based on it.\n
Question: {question}
Answer: """
# ===== Base Template ==================================================


def format_text(i: int, entry: DocEntry) -> str:
    return f"[{i}] Title: {entry.title} | {entry.text}"


def build_messages(question: str, passages: list[DocEntry], use_context: bool = True) -> list[dict[str, str]]:
    if not use_context:
        user_message = PURE_TEMPLATE.format(question=question)
    else:
        psgs_text_list = [format_text(i, psg) for i, psg in enumerate(passages, start=1)]
        retrieved_passages = f"\n".join(psgs_text_list)
        user_message = RAG_TEMPLATE.format(retrieved_passages=retrieved_passages, question=question)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]


_THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
_ANSWER_PREFIX = re.compile(r"^\s*(the\s+)?(final answer|answer|a)\s*[:：]\s*", flags=re.IGNORECASE)


def postprocess_answer(text: str) -> str:
    text = _THINK_BLOCK.sub("", text)
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    answer = _ANSWER_PREFIX.sub("", lines[0])
    answer = answer.strip().strip("*").strip().strip("\"'“”‘’").strip()
    return answer.rstrip(".").strip()



if __name__ == "__main__":
    from rag_assignment.paths import DATA_DIR
    from rag_assignment.utils.load_utils import load_corpus
    passages = load_corpus(DATA_DIR / "small_corpus.jsonl")[:5]
    question = "Let's build prompt!"
    
    pure_prompt = build_messages(question, passages, False)
    rag_prompt = build_messages(question, passages)

    print("===== Pure =====")
    print(pure_prompt)
    print("===== RAG =====")
    print(rag_prompt)