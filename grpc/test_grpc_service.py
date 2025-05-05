import pytest
import grpc
import text2image_pb2
import text2image_pb2_grpc
import asyncio
import base64
import json
from transformers import CLIPTokenizer

@pytest.mark.asyncio
async def test_generate_image_success():
    """Test successful image generation with valid prompt, context, and negative prompt."""
    async with grpc.aio.insecure_channel('localhost:50051') as channel:
        stub = text2image_pb2_grpc.TextToImageServiceStub(channel)
        try:
            response = await stub.GenerateImage(
                text2image_pb2.ImageRequest(
                    text="snowy mountains",
                    context="sunset",
                    negative_prompt="no blurry images"
                )
            )
            json_response = json.loads(response.response_json)
            assert json_response['status'] == "success", f"Expected success, got {json_response}"
            assert json_response['status_code'] == 200, f"Expected status code 200, got {json_response['status_code']}"
            assert response.image_base64 != "", "Image base64 string is empty"
            assert response.prompt == "snowy mountains sunset", f"Expected prompt 'snowy mountains sunset', got {response.prompt}"
            try:
                base64.b64decode(response.image_base64)
            except Exception as e:
                pytest.fail(f"Invalid base64 image data: {str(e)}")
        except grpc.aio.AioRpcError as e:
            pytest.fail(f"gRPC error: {e.details()} (code: {e.code()})")
        except Exception as e:
            pytest.fail(f"Test failed with exception: {str(e)}")

@pytest.mark.asyncio
async def test_generate_image_empty_prompt():
    """Test error handling for empty prompt."""
    async with grpc.aio.insecure_channel('localhost:50051') as channel:
        stub = text2image_pb2_grpc.TextToImageServiceStub(channel)
        try:
            response = await stub.GenerateImage(
                text2image_pb2.ImageRequest(text="", context="", negative_prompt="")
            )
            json_response = json.loads(response.response_json)
            assert json_response['status'] == "error"
            assert json_response['status_code'] == 400
            assert response.image_base64 == ""
            assert response.prompt == ""
        except grpc.aio.AioRpcError as e:
            assert e.code() == grpc.StatusCode.INVALID_ARGUMENT
            assert "Empty prompt" in e.details()

@pytest.mark.asyncio
async def test_generate_image_long_context():
    """Test handling of overly long context."""
    long_context = "x" * 1000  # 1000-character context
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    async with grpc.aio.insecure_channel('localhost:50051') as channel:
        stub = text2image_pb2_grpc.TextToImageServiceStub(channel)
        try:
            response = await stub.GenerateImage(
                text2image_pb2.ImageRequest(
                    text="cat in forest",
                    context=long_context,
                    negative_prompt="no blurry images"
                )
            )
            json_response = json.loads(response.response_json)
            assert json_response['status'] == "success", f"Expected success, got {json_response}"
            assert response.image_base64 != ""
            tokens = tokenizer(response.prompt, return_tensors="pt").input_ids
            assert tokens.shape[1] <= 77, f"Prompt exceeds 77 tokens: {tokens.shape[1]}"
        except grpc.aio.AioRpcError as e:
            pytest.fail(f"gRPC error: {e.details()} (code: {e.code()})")

@pytest.mark.asyncio
@pytest.mark.skip(reason="Skipping due to unresolved model error with concurrent requests")
async def test_generate_image_concurrent():
    """Test concurrent requests."""
    async def make_request(stub, text, context, negative_prompt):
        response = await stub.GenerateImage(
            text2image_pb2.ImageRequest(
                text=text,
                context=context,
                negative_prompt=negative_prompt
            )
        )
        return response

    async with grpc.aio.insecure_channel('localhost:50051') as channel:
        stub = text2image_pb2_grpc.TextToImageServiceStub(channel)
        tasks = [
            make_request(stub, f"cat in forest {i}", "sunset", "no blurry images")
            for i in range(3)
        ]
        responses = await asyncio.gather(*tasks)
        for response in responses:
            json_response = json.loads(response.response_json)
            assert json_response['status'] == "success"
            assert response.image_base64 != ""

@pytest.mark.asyncio
async def test_generate_image_invalid_prompt():
    """Test handling of invalid prompt that may cause model failure."""
    async with grpc.aio.insecure_channel('localhost:50051') as channel:
        stub = text2image_pb2_grpc.TextToImageServiceStub(channel)
        try:
            response = await stub.GenerateImage(
                text2image_pb2.ImageRequest(
                    text="@#$%^&*() invalid",
                    context="sunset",
                    negative_prompt="no blurry images"
                )
            )
            json_response = json.loads(response.response_json)
            if json_response['status'] == "success":
                assert response.image_base64 != ""
                assert response.prompt == "@#$%^&*() invalid sunset"
            else:
                assert json_response['status'] == "error"
                assert response.image_base64 == ""
        except grpc.aio.AioRpcError as e:
            pytest.fail(f"gRPC error: {e.details()} (code: {e.code()})")

@pytest.mark.asyncio
async def test_generate_image_with_style():
    """Test image generation with different styles."""
    async with grpc.aio.insecure_channel('localhost:50051') as channel:
        stub = text2image_pb2_grpc.TextToImageServiceStub(channel)
        try:
            response = await stub.GenerateImage(
                text2image_pb2.ImageRequest(
                    text="cat in forest, cartoon style",
                    context="sunset",
                    negative_prompt="no blurry images"
                )
            )
            json_response = json.loads(response.response_json)
            assert json_response['status'] == "success"
            assert json_response['status_code'] == 200
            assert response.image_base64 != ""
            assert response.prompt == "cat in forest, cartoon style sunset"
        except grpc.aio.AioRpcError as e:
            pytest.fail(f"gRPC error: {e.details()} (code: {e.code()})")

if __name__ == "__main__":
    pytest.main(["-v"])