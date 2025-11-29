import runpod
import torch
from diffusers import ZImagePipeline
from io import BytesIO
import base64
import os

SIZE = int(os.environ.get("SIZE", "1024"))
INFERENCE_STEPS = int(os.environ.get("INFERENCE_STEPS", "9"))
GUIDANCE_SCALE = float(os.environ.get("GUIDANCE_SCALE", "0.0"))
SEED = int(os.environ.get("SEED", "42"))

assert (
    torch.cuda.is_available()
), "CUDA is not available. Make sure you have a GPU instance."


def load_model():
    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    pipe = pipe.to("cuda")
    return pipe

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def stable_diffusion_handler(event):
    global model

    if "model" not in globals():
        model = load_model()

    prompt = event["input"].get("prompt")

    if not prompt:
        return {"error": "No prompt provided for image generation."}

    try:
        image = model(
            prompt=prompt,
            height=SIZE,
            width=SIZE,
            num_inference_steps=INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            generator=torch.Generator("cuda").manual_seed(SEED),
        ).images[0]
        image_base64 = image_to_base64(image)

        return {"image": image_base64, "prompt": prompt}

    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": stable_diffusion_handler})