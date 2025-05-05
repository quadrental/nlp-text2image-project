import asyncio
import grpc
import text2image_pb2
import text2image_pb2_grpc
import time
import matplotlib.pyplot as plt
import numpy as np

async def measure_response_time(stub, text, context, request_num):
    start_time = time.time()
    response = await stub.GenerateImage(text2image_pb2.ImageRequest(text=text, context=context))
    end_time = time.time()
    return request_num, end_time - start_time

async def run_performance_test(num_requests):
    async with grpc.aio.insecure_channel('localhost:50051') as channel:
        stub = text2image_pb2_grpc.TextToImageServiceStub(channel)
        tasks = [
            measure_response_time(stub, f"cat in forest {i}", "sunset", i)
            for i in range(num_requests)
        ]
        results = await asyncio.gather(*tasks)
        return dict(results)

def plot_performance(results):
    request_nums = list(results.keys())
    response_times = list(results.values())
    plt.figure(figsize=(8, 6))
    plt.plot(request_nums, response_times, 'b-o', label='Response Time')
    plt.title('Response Time vs. Number of Concurrent Requests')
    plt.xlabel('Request Number')
    plt.ylabel('Response Time (seconds)')
    plt.grid(True)
    plt.legend()
    plt.savefig('performance_plot.png')

if __name__ == "__main__":
    num_requests = 10
    results = asyncio.run(run_performance_test(num_requests))
    plot_performance(results)
    print(f"Performance test completed. Check 'performance_plot.png' for results.")