import modal



REPO_ID = "hfarwah/universal-sentence-encoder"
MODEL_DIR = "model-cache"

def download_model():
    from huggingface_hub import snapshot_download
    
    snapshot_download(repo_id=REPO_ID, local_dir=MODEL_DIR)


dockerhub_image = modal.Image.from_registry(
    "tensorflow/tensorflow:latest-gpu"
    ).pip_install(
        "tensorflow_hub",
        "huggingface_hub",
        "numpy"
    ).run_function(download_model)


stub = modal.Stub("sentence-encoder", image=dockerhub_image)


@stub.cls(gpu="T4", container_idle_timeout=60)
class SentenceSimilarity:
    def __enter__(self):
        import tensorflow_hub as hub 
        
        self.model = hub.load(MODEL_DIR)

    @modal.method()
    def inference(self, source: str):
        source_embeddings = self.model([source]).numpy()

        return source_embeddings
    
@stub.local_entrypoint()
def entrypoint(source: str):
    import time
    ss = SentenceSimilarity()

    start_time = time.time()
    similarities = ss.inference.remote(source)
    end_time = time.time() - start_time

    print(similarities, end_time)

from pydantic import BaseModel
class ItemReq(BaseModel):
    input: str



@stub.function()
@modal.asgi_app()
def app():
    from fastapi import FastAPI

    web_app = FastAPI()
    model = SentenceSimilarity()

    @web_app.post("/infer")
    def infer(request: ItemReq):
        input = request.input
        embeddings = model.inference.remote(input)
        return {"embeddings" : embeddings.tolist()}
    
    return web_app