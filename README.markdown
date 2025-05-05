# Text-to-Image Generator with gRPC and Gradio

## Setup
1. **Prerequisites:**
   - Docker installed on your system.
   - Python 3.9+ (for local development).
   - Git (for version control and repository setup).

2. **Installation:**
   - Clone the repository:
     ```
     git clone https://github.com/quadrental/nlp-text2image-project.git
     cd nlp-text2image-project
     ```
   - Build and run the Docker container using the provided `deploy.sh` script:
     ```
     chmod +x deploy.sh
     ./deploy.sh
     ```
   - Alternatively, for local development:
     - Create a virtual environment: `python -m venv venv`
     - Activate it: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
     - Install dependencies: `pip install -r requirements.txt` (create `requirements.txt` with: `diffusers torch transformers pillow gradio fastapi uvicorn grpcio grpcio-tools matplotlib numpy`)
     - Start the gRPC server: `python grpc_server.py`
     - Start the Gradio app: `python gradio_app.py`

3. **Running the Project:**
   - Access the Gradio UI at `http://localhost:7860` after starting the app.
   - Ensure the gRPC server is running on `localhost:50051` for client and test interactions.

## Usage
- **Gradio Interface:**
  - Open `http://localhost:7860` in your browser.
  - Enter a prompt (e.g., "cat in forest").
  - Add optional context (e.g., "sunset") and negative prompts (e.g., "no blurry images").
  - Select an art style (realistic, oil painting, cartoon, cyberpunk).
  - Click submit to generate an image.
- **API Endpoint:**
  - Use `POST /generate` with a JSON payload (e.g., `{"text": "cat in forest", "context": "sunset"}`) at `http://localhost:7860/generate`.
- **gRPC Client:**
  - Run `python client.py` to generate images concurrently (saves as `output_*.png`).
- **Performance Test:**
  - Run `python test_performance.py` to measure response times and generate a plot (`performance_plot.png`).

## Architecture
- **gRPC Server (`grpc_server.py`):**
  - Implements `TextToImageService` using Stable Diffusion to generate images based on text prompts.
  - Runs on port `50051` and returns base64-encoded images.
- **Gradio App (`gradio_app.py`):**
  - Provides a web interface and FastAPI endpoint, integrating with the gRPC server.
  - Exposes the generator at `/` and API at `/generate`.
- **Client (`client.py`):**
  - Demonstrates concurrent image generation requests to the gRPC server.
- **Communication:**
  - gRPC handles requests between the client/app and server, with `text2image.proto` defining the interface.

## Model Sources
- **Stable Diffusion:** `runwayml/stable-diffusion-v1-5` from Hugging Face.
- **Tokenizer:** `openai/clip-vit-large-patch14` for prompt truncation.

## Limitations
- **Token Limit:** Prompts are truncated to 77 tokens to fit the model’s capacity.
- **Hardware Dependency:** Requires GPU (CUDA) for optimal performance; falls back to CPU if unavailable, which may slow processing.
- **Concurrent Requests:** Limited by server thread pool (max 10 workers); performance degrades with high load.
- **Image Quality:** Complex or ambiguous prompts may yield inconsistent results due to model constraints.
- **No Local File I/O in Browser:** Pygame or file-based operations are unsupported in web contexts.
- **Error Handling:** May fail with invalid inputs (e.g., empty prompts) or resource exhaustion.