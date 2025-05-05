import io
import base64
import torch
import grpc
import asyncio
from concurrent import futures
from PIL import Image
from diffusers import StableDiffusionPipeline
from functools import partial
from transformers import CLIPTokenizer
import logging

import text2image_pb2
import text2image_pb2_grpc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextToImageService(text2image_pb2_grpc.TextToImageServiceServicer):
    def __init__(self):
        try:
            self.model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            logger.info(f"Model loaded on {self.device}")
            self.executor = futures.ThreadPoolExecutor(max_workers=4)
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise

    def truncate_prompt(self, prompt, max_tokens=77):
        """Truncate prompt to fit within the model's token limit."""
        try:
            tokens = self.tokenizer(prompt, return_tensors="pt", truncation=False).input_ids
            token_count = tokens.shape[1]
            if token_count > max_tokens:
                truncated_tokens = tokens[:, :max_tokens-1]
                truncated_prompt = self.tokenizer.decode(truncated_tokens[0], skip_special_tokens=True)
                logger.info(f"Prompt truncated from {token_count} tokens ({len(prompt)} chars) to {max_tokens-1} tokens ({len(truncated_prompt)} chars)")
                return truncated_prompt, max_tokens-1
            logger.info(f"Prompt unchanged: {token_count} tokens ({len(prompt)} chars)")
            return prompt, token_count
        except Exception as e:
            logger.error(f"Error truncating prompt: {str(e)}")
            fallback_prompt = prompt[:100]
            return fallback_prompt, len(self.tokenizer(fallback_prompt, return_tensors="pt").input_ids[0])

    def generate_image_sync(self, prompt, negative_prompt):
        """Synchronous function to generate image, to be run in a thread."""
        if not prompt:
            return None, "error: Empty prompt provided"
        try:
            truncated_prompt, _ = self.truncate_prompt(prompt)
            logger.info(f"Generating image for prompt: {truncated_prompt}")
            image = self.model(
                truncated_prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                num_inference_steps=15,
                safety_checker=False
            ).images[0]
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return base64_str, "success"
        except Exception as e:
            logger.error(f"Image generation failed: {str(e)}")
            return None, f"error: {str(e)}"

    async def GenerateImage(self, request, context):
        raw_prompt = f"{request.text} {request.context}".strip()
        negative_prompt = request.negative_prompt
        try:
            truncated_prompt, token_count = self.truncate_prompt(raw_prompt)
            loop = asyncio.get_running_loop()
            base64_str, status = await loop.run_in_executor(
                self.executor,
                partial(self.generate_image_sync, truncated_prompt, negative_prompt)
            )
            if status.startswith("error"):
                context.set_details(status)
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                return text2image_pb2.ImageResponse(
                    prompt=truncated_prompt,
                    image_base64="",
                    status="error",
                    status_code=400,
                    response_json='{"status": "error", "status_code": 400, "error": "' + status + '"}'
                )
            return text2image_pb2.ImageResponse(
                prompt=truncated_prompt,
                image_base64=base64_str,
                status="success",
                status_code=200,
                response_json='{"status": "success", "status_code": 200, "image_base64": "' + base64_str + '", "prompt": "' + truncated_prompt + '"}'
            )
        except Exception as e:
            logger.error(f"Unexpected error in GenerateImage: {str(e)}")
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return text2image_pb2.ImageResponse(
                prompt=raw_prompt,
                image_base64="",
                status="error",
                status_code=500,
                response_json='{"status": "error", "status_code": 500, "error": "' + str(e) + '"}'
            )

async def serve():
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    text2image_pb2_grpc.add_TextToImageServiceServicer_to_server(TextToImageService(), server)
    server.add_insecure_port('[::]:50051')
    print("gRPC server running on port 50051...")
    await server.start()
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(serve())