import os

import modal


DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
CACHE_DIR = "./model_cache"


def download_model():
    import torch
    from transformers import  LlamaForCausalLM, LlamaTokenizer 

    LlamaForCausalLM.from_pretrained(
        MODEL_ID,
        token=os.environ["HUGGINGFACE_TOKEN"],
        torch_dtype=torch.float16,
        cache_dir=CACHE_DIR,
    )

    LlamaTokenizer.from_pretrained(
            MODEL_ID,
            token=os.environ["HUGGINGFACE_TOKEN"],
            cache_dir=CACHE_DIR,
        )

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "accelerate==0.22.0",
        "bitsandbytes==0.41.1",
        "einops==0.6.1",
        "faker==19.3.1",
        "peft==0.5.0",
        "protobuf==3.20.3",
        "safetensors==0.3.3",
        "scipy==1.10.1",
        "sentencepiece==0.1.99",
        "transformers==4.32.1",
    ).pip_install(
        "torch==2.0.1+cu117",
        find_links="https://download.pytorch.org/whl/torch_stable.html",
    )
    .run_function(download_model, secret=modal.Secret.from_name("my-huggingface-secret"))
)

stub = modal.Stub("llama2-7b-chat", image=image)


@stub.cls(secret=modal.Secret.from_name("my-huggingface-secret"), gpu="A10G", container_idle_timeout=60)
class LlamaModel:
    def __enter__(self) -> None:
        import torch
        from transformers import  LlamaForCausalLM, LlamaTokenizer 

        self.model = LlamaForCausalLM.from_pretrained(
            MODEL_ID,
            token=os.environ["HUGGINGFACE_TOKEN"],
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=CACHE_DIR,
            local_files_only=True,
        )

        self.tokenizer = LlamaTokenizer.from_pretrained(
            MODEL_ID,
            token=os.environ["HUGGINGFACE_TOKEN"],
            cache_dir=CACHE_DIR,
            local_files_only=True,
        )
    
    @modal.method()
    def inference(self, 
                  prompt,
                  temperature=0.1,
                  top_p=0.75,
                  top_k=40,
                  num_beams=1,
                  max_length=512,
                  do_sample=True):
        import torch
        from transformers import GenerationConfig, TextIteratorStreamer
        
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            repetition_penalty=1.2,
            max_length=max_length,
            do_sample=do_sample,
        )

        prompt_wrapped = f"{B_INST} {B_SYS} {DEFAULT_SYSTEM_PROMPT} {E_SYS} {prompt} {E_INST}"
        inputs = self.tokenizer(
            prompt_wrapped,
            return_tensors="pt",
            truncation=True,
            padding=False,
            max_length=1056,
        )

        input_ids = inputs["input_ids"].to("cuda")

        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_length=max_length,
            )

            decoded_output = []
            for beam in generation_output.sequences:
                decoded_output.append(self.tokenizer.decode(beam, skip_special_tokens=True).replace(prompt_wrapped, ""))

            return " ".join(decoded_output)
        
        



from pydantic import BaseModel
class ItemReq(BaseModel):
    prompt: str

@stub.function()
@modal.asgi_app()
def app():
    from fastapi import FastAPI

    web_app = FastAPI()
    model = LlamaModel()

    @web_app.post("/infer")
    def infer(request: ItemReq):
        from fastapi.responses import Response

        prompt = request.prompt

        response = model.inference.remote(prompt)

        return {"response" : response}
    
    return web_app