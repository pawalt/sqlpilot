"""
vLLM inference server for synthetic data generation.

Serves Qwen3-235B-A22B-Instruct-2507-FP8 as an OpenAI-compatible API endpoint.

Usage:
    # Deploy the server (runs as a web endpoint)
    modal deploy datagen/vllm_server.py

    # The server URL will be printed, use it in modal_datagen.py
"""

import modal

MODEL_ID = "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"

# Volume to cache model weights (avoids re-downloading on cold starts)
model_cache = modal.Volume.from_name("sqlpilot-model-cache", create_if_missing=True)

# vLLM image with CUDA support
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "vllm>=0.6.0",
        "torch",
        "transformers",
        "huggingface_hub",
        "fastapi",
    )
)

app = modal.App("sqlpilot-vllm")

# How long to keep the server warm after last request (10 minutes)
IDLE_TIMEOUT = 600


@app.cls(
    image=vllm_image,
    gpu="H100:4",  # 4x H100 for the 235B MoE model
    volumes={"/root/.cache/huggingface": model_cache},
    timeout=3600,  # 1 hour max request time
    scaledown_window=IDLE_TIMEOUT,
)
@modal.concurrent(max_inputs=32)
class VLLMServer:
    @modal.enter()
    def start_engine(self):
        """Initialize vLLM engine on container startup."""
        from vllm import AsyncLLMEngine, AsyncEngineArgs

        print(f"Loading model: {MODEL_ID}")

        engine_args = AsyncEngineArgs(
            model=MODEL_ID,
            tensor_parallel_size=4,
            dtype="auto",
            max_model_len=4096,
            gpu_memory_utilization=0.85,
            max_num_seqs=32,
            trust_remote_code=True,
        )

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        print(f"Model loaded: {MODEL_ID}")

        # Commit the model cache so subsequent starts are faster
        model_cache.commit()

    @modal.asgi_app()
    def serve(self):
        """Serve an OpenAI-compatible API."""
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, StreamingResponse
        from vllm.sampling_params import SamplingParams
        import uuid
        import time
        import json

        fastapi_app = FastAPI(title="SQLPilot vLLM Server")

        @fastapi_app.get("/health")
        async def health():
            return {"status": "ok", "model": MODEL_ID}

        @fastapi_app.get("/v1/models")
        async def list_models():
            return {
                "object": "list",
                "data": [
                    {
                        "id": MODEL_ID,
                        "object": "model",
                        "owned_by": "sqlpilot",
                    }
                ],
            }

        @fastapi_app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            """OpenAI-compatible chat completions endpoint."""
            data = await request.json()
            
            messages = data.get("messages", [])
            max_tokens = data.get("max_tokens", 1024)
            temperature = data.get("temperature", 0.7)
            top_p = data.get("top_p", 0.9)
            
            # Build prompt from messages
            prompt = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
                elif role == "user":
                    prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
                elif role == "assistant":
                    prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"
            
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            
            request_id = str(uuid.uuid4())
            
            # Generate
            results = []
            async for output in self.engine.generate(prompt, sampling_params, request_id):
                results.append(output)
            
            if not results:
                return JSONResponse(
                    status_code=500,
                    content={"error": "No output generated"}
                )
            
            final_output = results[-1]
            generated_text = final_output.outputs[0].text
            
            # Format as OpenAI response
            response = {
                "id": f"chatcmpl-{request_id}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": generated_text,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(generated_text.split()),
                    "total_tokens": len(prompt.split()) + len(generated_text.split()),
                },
            }
            
            return JSONResponse(content=response)

        return fastapi_app


# Local entrypoint for testing
@app.local_entrypoint()
def main():
    """Print the deployment URL."""
    print("Deploy this app to get a URL:")
    print("  modal deploy datagen/vllm_server.py")
    print()
    print("Then use the URL in your datagen code:")
    print('  client = OpenAI(base_url="https://YOUR-APP-URL.modal.run/v1")')
