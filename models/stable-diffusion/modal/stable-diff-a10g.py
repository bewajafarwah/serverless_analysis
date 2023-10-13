import io
import time
import pathlib

import modal

stub = modal.Stub("stable-diff-a10g")

MODEL_ID = "stabilityai/stable-diffusion-2-1"
CACHE_DIR = "./model_cache"


def download_model():
    import diffusers
    import torch

    diffusers.StableDiffusionPipeline.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16,
        cache_dir=CACHE_DIR
      )
    
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "accelerate",
        "diffusers[torch]>=0.15.1",
        "ftfy",
        "torchvision",
        "transformers~=4.25.1",
        "triton",
        "safetensors",
    )
    .pip_install(
        "torch==2.0.1+cu117",
        find_links="https://download.pytorch.org/whl/torch_stable.html",
    )
    .run_function(download_model)
)

stub.image = image

@stub.cls(gpu="A10G", container_idle_timeout=60)
class StableDiffusion:
    def __enter__(self):
        import diffusers
        import torch

        self.pipe = diffusers.StableDiffusionPipeline.from_pretrained(
                        MODEL_ID, 
                        torch_dtype=torch.float16,
                        cache_dir=CACHE_DIR,
                        local_files_only=True
                    )
        self.pipe.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to("cuda")


    @modal.method()
    def inference(self, prompt: str, num_inference_steps: int=50, guidance_scale: float=7.5, num_outputs: int=1):
        import torch

        with torch.inference_mode():
            images = self.pipe(
                [prompt] * num_outputs,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            ).images
        
        images_outputs = []
        for image in images:
            with io.BytesIO() as buf:
                image.save(buf, format="PNG")
                images_outputs.append(buf.getvalue())

        return images_outputs
    

@stub.local_entrypoint()
def entrypoint(prompt: str, num_outputs: int=1, num_inference_steps: int=50):
    print(f"prompt => {prompt}, num_inference_steps => {num_inference_steps}, num_outputs => {num_outputs}")

    dir = pathlib.Path("/tmp/stable-diff-serve")
    if not dir.exists():
        dir.mkdir(exist_ok=True, parents=True)
    
    sd = StableDiffusion()

    start_time = time.time()
    images = sd.inference.remote(prompt, num_inference_steps=num_inference_steps, num_outputs=num_outputs)
    end_time = time.time()

    print(f"Sample took: {(end_time-start_time):.2f}s")

    for j, image_bytes in enumerate(images):
        output_path = dir / f"output_{j}.png"
        print(f"Saving it to {output_path}")
        with open(output_path, "wb") as f:
            f.write(image_bytes)
