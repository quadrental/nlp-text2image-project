from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline
import torch

app = FastAPI()
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

class TextRequest(BaseModel):
    text: str
    context: str = ""

@app.post("/generate")
def generate_image(request: TextRequest):
    prompt = f"{request.text} {request.context}".strip()
    try:
        image = pipe(prompt).images[0]
        image.save("output.png")
        return {"message": "Image generated and saved as output.png", "prompt": prompt}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
