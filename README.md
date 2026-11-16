# RAG Assignment (NQ-open)

## Setup (Google Colab)

1. 메뉴 `런타임 > 런타임 유형 변경 > T4 GPU`
2. (Llama·Gemma 등 gated 모델을 쓸 경우) Hugging Face에서 모델 라이선스 동의 후 토큰 발급 →
   Colab 왼쪽 🔑(Secrets) 탭에 `HF_TOKEN` 이름으로 추가 → **노트북 액세스** 켜기
3. `main.ipynb` 의 **0. 환경 세팅** 셀 실행 (세션이 새로 시작될 때마다)

## Data

데이터는 Hugging Face Hub [`NLPlab-skku/RAG-assignment`](https://huggingface.co/datasets/NLPlab-skku/RAG-assignment) (revision `v1`)에서 받습니다.
`main.ipynb` 의 **데이터 다운로드** 셀을 실행하면 Colab 로컬 디스크 `/content/data` 에 저장됩니다 (Google Drive 용량을 쓰지 않음).

| 파일 | 설명 |
|---|---|
| `data/corpus.jsonl` | Wikipedia passage 1,087,303개 `{"pid", "title", "text"}` |
| `data/dev_qa.jsonl` | dev 1,000문항 `{"qid", "question", "answers"}` |
| `data/train_qa.jsonl` | train 58,880문항 `{"qid", "question", "answers"}` |
| `data/train_qrels.jsonl` | train 문항별 관련 passage `{"qid", "gold", "weak_pos", "hard_neg"}` |
| `data/index/{lexical,dense}/` | 베이스라인 인덱스 (BM25 / `all-MiniLM-L6-v2`) |

- 전체 약 2.9GB입니다. 로컬 디스크는 세션이 끝나면 지워지므로 **세션마다 다시 받아야** 합니다.
- 직접 빌드한 인덱스를 세션 너머로 남기고 싶다면 `index_dir` 을 Google Drive 경로로 지정하세요.
- retriever 모델·설정을 바꾸면 제공된 인덱스와 맞지 않으므로 새 `index_dir` 에 인덱스를 다시 빌드해야 합니다.

## Structure

| 경로 | 설명 |
|---|---|
| `main.ipynb` | 베이스라인 실행 / 평가 |
| `src/rag_assignment/retriever/` | 검색기 (BM25, dense) |
| `src/rag_assignment/generator/` | 생성기 (Hugging Face LLM), 프롬프트 |
| `src/rag_assignment/pipeline.py` | 검색 -> 생성 파이프라인, 결과 저장 |
| `src/rag_assignment/grading/` | 평가 / 시간 측정 코드 **수정 금지** |
| `outputs/` | 실행 결과 (`retrieval.jsonl`, `predictions.jsonl`, `efficiency.json`, 평가 결과) |
