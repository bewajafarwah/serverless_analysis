from cog import BasePredictor, Input
import torch
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import os

MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
MODEL_CACHE = "llama-cache"
auth_secret = os.environ['HUGGINGFACE_TOKEN']

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

class Predictor(BasePredictor):
    def setup(self):
        print("Loading Pipeline...")
        self._model = LlamaForCausalLM.from_pretrained(
            MODEL_ID, 
            use_auth_token=auth_secret, 
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        )
        self._tokenizer = LlamaTokenizer.from_pretrained(
            MODEL_ID, 
            use_auth_token=auth_secret,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        )
    
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="What is the meaning of life?",
        ),
        temperature: float = Input(
            description="",
            default=0.1,
        ),
        top_p: float = Input(
            description="",
            default=0.75,
        ),
        top_k: int = Input(
            description="",
            default=40,
        ),
        num_beams: int = Input(
            description="",
            default=1,
        ),
        max_length: int = Input(
            description="",
            default=512
        ),
    ) -> str:
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            repetition_penalty=1.2,
            max_length=max_length,
        )

        prompt_wrapped = f"{B_INST} {B_SYS} {DEFAULT_SYSTEM_PROMPT} {E_SYS} {prompt} {E_INST}"
        inputs = self._tokenizer(
            prompt_wrapped, return_tensors="pt", truncation=True, padding=False, max_length=1056
        )
        input_ids = inputs["input_ids"].to("cuda")

        with torch.no_grad():
            generation_output = self._model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_length=max_length,
                early_stopping=True,
            )

            decoded_output = []
            for beam in generation_output.sequences:
                decoded_output.append(self._tokenizer.decode(beam, skip_special_tokens=True).replace(prompt_wrapped, ""))

            return " ".join(decoded_output)