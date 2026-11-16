from typing import NamedTuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prompt import build_messages, postprocess_answer, DocEntry


class GenerationResult(NamedTuple):
    text: str               
    raw_text: str           
    n_input_tokens: int     
    n_output_tokens: int    


class HFGenerator:
    def __init__(
        self,
        model_name_or_path: str,
        max_seq_length: int = 2048,     
        max_ans_length: int = 32,       
        batch_size: int = 8,
        fp16: bool = True,              
    ):
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.max_ans_length = max_ans_length
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if (fp16 and self.device.type == "cuda") else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, dtype=dtype).to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        # use left padding & truncation
        self.tokenizer.padding_side = "left"        
        self.tokenizer.truncation_side = "left"     
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        eos = self.model.generation_config.eos_token_id
        self.eos_token_ids = set(eos if isinstance(eos, list) else [eos]) | {self.tokenizer.eos_token_id}

    # ----- 내부 유틸 -----
    def _to_prompt(self, messages: list[dict[str, str]]) -> str:
        try:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
            merged = [
                {"role": m["role"], "content": f"{system}\n\n{m['content']}" if m["role"] == "user" and system else m["content"]}
                for m in messages if m["role"] != "system"
            ]
            return self.tokenizer.apply_chat_template(merged, tokenize=False, add_generation_prompt=True)

    def _count_output_tokens(self, ids: list[int]) -> int:
        for i, t in enumerate(ids):
            if t in self.eos_token_ids:
                return i + 1
        return len(ids)

    @torch.inference_mode()
    def _generate_batch(self, prompts: list[str]) -> list[GenerationResult]:
        # convert text to tokens
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            add_special_tokens=False,
        ).to(self.device)

        out = self.model.generate(
            **enc,
            max_new_tokens=self.max_ans_length,
            do_sample=False,                            # use greedy decoding
            temperature=None, top_p=None, top_k=None,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        gen_ids = out[:, enc["input_ids"].shape[1]:].tolist()
        n_inputs = enc["attention_mask"].sum(dim=1).tolist()

        results = []
        # convert tokens to text
        for ids, n_in in zip(gen_ids, n_inputs):
            raw = self.tokenizer.decode(ids, skip_special_tokens=True)
            results.append(GenerationResult(postprocess_answer(raw), raw, int(n_in), self._count_output_tokens(ids)))
        return results

    # ----- 공개 메서드 -----
    def batch_generate(
        self,
        questions: list[str],
        passages_list: list[list[DocEntry]] | None = None,
    ) -> list[GenerationResult]:
        use_context = passages_list is not None
        if not use_context:
            passages_list = [[] for _ in questions]
        assert len(questions) == len(passages_list), "The length of questions doesn't match with passages_list."

        prompts = [self._to_prompt(build_messages(q, p, use_context)) for q, p in zip(questions, passages_list)]

        results: list[GenerationResult] = []
        for start in tqdm(range(0, len(prompts), self.batch_size), desc="Generate", disable=len(prompts) <= self.batch_size):
            results.extend(self._generate_batch(prompts[start: start + self.batch_size]))
        return results

    def generate(self, question: str, passages: list[DocEntry], use_context: bool = True) -> GenerationResult:
        return self.batch_generate([question], [passages] if use_context else None)[0]
