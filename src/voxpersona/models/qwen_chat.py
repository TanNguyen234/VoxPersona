import gc
import threading
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextIteratorStreamer


class QwenChat:
    def __init__(
        self,
        model_id: str,
        system_prompt: str,
        max_history: int = 10,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
    ) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.system_prompt = system_prompt
        self.max_history = max_history
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.conversation_history: List[Dict[str, str]] = []

    def call_stream(self, user_input: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
        ] + self.conversation_history + [
            {"role": "user", "content": user_input},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = dict(
            **model_inputs,
            streamer=streamer,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
        )

        with torch.no_grad():
            thread = threading.Thread(target=lambda: self.model.generate(**generation_kwargs))
            thread.start()

            response = ""
            for new_text in streamer:
                print(new_text, end="", flush=True)
                response += new_text

        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response})
        self.conversation_history = self.conversation_history[-self.max_history :]

        return response.strip()

    def clear_gpu(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
