import grpc
import text2image_pb2
import text2image_pb2_grpc
import base64
import asyncio
import time
import json

async def generate_image(stub, text, context, index):
    try:
        response = await stub.GenerateImage(text2image_pb2.ImageRequest(text=text, context=context))
        json_response = json.loads(response.response_json)
        print(f"Request {index} Status: {json_response['status']} (Code: {json_response['status_code']})")
        if json_response['status_code'] == 200:
            with open(f"output_{index}.png", "wb") as f:
                f.write(base64.b64decode(json_response['image_base64']))
            print(f"Request {index} Image saved as output_{index}.png")
        else:
            print(f"Request {index} Error: {json_response['error']}")
        return json_response['status']
    except Exception as e:
        print(f"Request {index} failed: {str(e)}")
        return "error"

async def run_concurrent_requests(num_requests=3):
    async with grpc.aio.insecure_channel('localhost:50051') as channel:
        stub = text2image_pb2_grpc.TextToImageServiceStub(channel)
        tasks = [
            generate_image(stub, f"a cat in forest {i}", "sunset", i)
            for i in range(num_requests)
        ]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(run_concurrent_requests(num_requests=3))
    print(f"Total time: {time.time() - start_time:.2f} seconds")