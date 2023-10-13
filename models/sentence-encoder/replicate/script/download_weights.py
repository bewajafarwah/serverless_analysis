import shutil
import argparse
from pathlib import Path


from huggingface_hub import snapshot_download

MODEL_CACHE_DIR = "model-cache"


def download_model(model_id: str):
    '''
    Downloads the model from the URL passed in.
    '''
    model_cache_path = Path(MODEL_CACHE_DIR)

    if model_cache_path.exists():
        shutil.rmtree(model_cache_path)
        
    model_cache_path.mkdir(parents=True, exist_ok=True)
  
    snapshot_download(
        model_id,
        local_dir=model_cache_path
    )


# ---------------------------------------------------------------------------- #
#                                Parse Arguments                               #
# ---------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--model_id", type=str,
    default="hfarwah/universal-sentence-encoder",
    help="URL of the model to download."
)

if __name__ == "__main__":
    args = parser.parse_args()
    download_model(args.model_id)
