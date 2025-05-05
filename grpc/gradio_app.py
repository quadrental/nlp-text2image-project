import gradio as gr
import grpc
import text2image_pb2
import text2image_pb2_grpc
import base64
import json
import asyncio
from PIL import Image
import io
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.middleware.cors import CORS

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(CORS, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Define request model for FastAPI
class ImageRequest(BaseModel):
    text: str
    context: str = ""

async def generate_image(text, context):
    async with grpc.aio.insecure_channel('localhost:50051') as channel:
        stub = text2image_pb2_grpc.TextToImageServiceStub(channel)
        try:
            response = await stub.GenerateImage(text2image_pb2.ImageRequest(text=text, context=context))
            json_response = json.loads(response.response_json)
            if json_response['status_code'] == 200:
                image_data = base64.b64decode(json_response['image_base64'])
                image = Image.open(io.BytesIO(image_data))
                return image, json_response
            return f"Error: {json_response['error']} (Status Code: {json_response['status_code']})", json_response
        except Exception as e:
            json_response = {
                "prompt": f"{text} {context}".strip(),
                "status": "error",
                "status_code": 500,
                "image_base64": "",
                "error": str(e)
            }
            return f"Error: {str(e)} (Status Code: 500)", json_response

# FastAPI endpoint for JSON responses
@app.post("/generate")
async def generate_image_api(request: ImageRequest):
    result, json_response = await generate_image(request.text, request.context)
    if json_response['status_code'] != 200:
        raise HTTPException(status_code=json_response['status_code'], detail=json_response)
    return json_response

# Gradio interface function
def gradio_generate(text, context):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result, json_response = loop.run_until_complete(generate_image(text, context))
    loop.close()
    return result

# Create Gradio interface
interface = gr.Interface(
    fn=gradio_generate,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter your prompt (e.g., 'cat in forest')", label="Prompt"),
        gr.Textbox(lines=2, placeholder="Add context (optional)", label="Context")
    ],
    outputs=gr.Image(type="pil", label="Generated Image"),
    title="Text-to-Image Generator (gRPC)",
    description="Generate images using a gRPC-based Stable Diffusion service.",
    examples=[["a cat in forest", "sunset"], ["a futuristic city", "night"]]
)

# Mount Gradio to FastAPI
app = gr.mount_gradio_app(app, interface, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)