import shutil
import argparse
from pathlib import Path
import os

import torch
from transformers import  LlamaForCausalLM, LlamaTokenizer

MODEL_CACHE_DIR ="llama-cache"
auth_secret= os.environ['HUGGINGFACE_TOKEN']


def download_model(MODEL_ID):
    model_cache_path = Path(MODEL_CACHE_DIR)

    if model_cache_path.exists():
        shutil.rmtree(model_cache_path)
        
    model_cache_path.mkdir(parents=True, exist_ok=True)


    model = LlamaForCausalLM.from_pretrained(
            MODEL_ID, 
            use_auth_token=auth_secret, 
            torch_dtype=torch.float16,
            cache_dir=model_cache_path,
        )
        
    tokenizer = LlamaTokenizer.from_pretrained(
            MODEL_ID, 
            use_auth_token=auth_secret,
            cache_dir=model_cache_path,
        )

# ---------------------------------------------------------------------------- #
#                                Parse Arguments                               #
# ---------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--model_id", type=str,
    default="meta-llama/Llama-2-7b-chat-hf",
    help="URL of the model to download."
)

if __name__ == "__main__":
    args = parser.parse_args()
    download_model(args.model_id)