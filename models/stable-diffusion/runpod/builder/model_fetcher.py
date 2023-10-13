
import shutil
import argparse
from pathlib import Path


from diffusers import StableDiffusionPipeline

MODEL_CACHE_DIR = "diffusers-cache"


def download_model(model_id: str):
    '''
    Downloads the model from the URL passed in.
    '''
    model_cache_path = Path(MODEL_CACHE_DIR)

    if model_cache_path.exists():
        shutil.rmtree(model_cache_path)
        
    model_cache_path.mkdir(parents=True, exist_ok=True)
  
    StableDiffusionPipeline.from_pretrained(
        model_id,
        cache_dir=model_cache_path,
    )


# ---------------------------------------------------------------------------- #
#                                Parse Arguments                               #
# ---------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--model_id", type=str,
    default="stabilityai/stable-diffusion-2-1",
    help="URL of the model to download."
)

if __name__ == "__main__":
    args = parser.parse_args()
    download_model(args.model_id)
