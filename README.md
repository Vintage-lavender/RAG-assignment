# RAG Assignment (NQ-open)

## Structure

```bash
.
├── main.ipynb
├── pyproject.toml
├── README.md
└── src
    └── rag_assignment
        ├── generator
        │   ├── hf_generator.py
        │   ├── __init__.py
        │   └── prompt.py
        ├── grading
        │   ├── efficiency.py
        │   ├── evaluate.py
        │   ├── __init__.py
        │   └── metrics.py
        ├── __init__.py
        ├── paths.py
        ├── pipeline.py
        ├── retriever
        │   ├── base.py
        │   ├── bm25.py
        │   ├── dense.py
        │   └── __init__.py
        └── utils
            ├── __init__.py
            └── load_utils.py

6 directories, 19 files
```


| 경로 | 설명 |
|---|---|
| `main.ipynb` | 베이스라인 실행 / 평가 |
| `src/rag_assignment/retriever/` | 검색기 (BM25, dense) |
| `src/rag_assignment/generator/` | 생성기 (Hugging Face LLM), 프롬프트 |
| `src/rag_assignment/pipeline.py` | 검색 -> 생성 파이프라인, 결과 저장 |
| `src/rag_assignment/grading/` | 평가 / 시간 측정 코드 **수정 금지** |
| `outputs/` | 실행 결과 (`retrieval.jsonl`, `predictions.jsonl`, `efficiency.json`, 평가 결과) |


## Data

데이터는 Hugging Face Hub [`NLPlab-skku/RAG-assignment`](https://huggingface.co/datasets/NLPlab-skku/RAG-assignment)에서 받습니다.

`main.ipynb` 의 **데이터 다운로드** 셀을 실행하면 Colab 로컬 디스크 `/content/data` 에 저장됩니다.

| 파일 | 설명 |
|---|---|
| `data/corpus.jsonl` | Wikipedia passage 1,087,303개 `{"pid", "title", "text"}` |
| `data/dev_qa.jsonl` | dev 1,000문항 `{"qid", "question", "answers"}` |
| `data/train_qa.jsonl` | train 58,880문항 `{"qid", "question", "answers"}` |
| `data/train_qrels.jsonl` | train 문항별 관련 passage `{"qid", "gold", "weak_pos", "hard_neg"}` |
| `data/index/{lexical,dense}/` | 베이스라인 인덱스 (BM25 / `all-MiniLM-L6-v2`) |

- 전체 약 2.9GB입니다.
- 직접 빌드한 인덱스를 세션 너머로 남기고 싶다면 `index_dir` 을 Google Drive 경로로 지정하세요.
- retriever 모델/설정을 바꾸면 제공된 인덱스와 맞지 않으므로 새 `index_dir` 에 인덱스를 다시 빌드해야 합니다.