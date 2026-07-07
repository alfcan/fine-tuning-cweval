#!/usr/bin/env python3
import os
import sys
import argparse
import uvicorn
import torch
import uuid
import time
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

model = None
tokenizer = None
device = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/v1/models")
def get_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "local-model",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local"
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    global model, tokenizer, device
    try:
        data = await request.json()
        messages = data.get("messages", [])
        temperature = data.get("temperature", 0.7)
        max_tokens = data.get("max_tokens", 512)
        stop_words = data.get("stop", [])

        # Apply chat template
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        do_sample = temperature > 0.0
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "pad_token_id": tokenizer.eos_token_id,
            "do_sample": do_sample
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature

        # Generate completions
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        
        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_len:]
        response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Post-process stopping criteria at text level
        if isinstance(stop_words, str):
            stop_words = [stop_words]
        if stop_words:
            first_stop_idx = len(response_text)
            for stop_word in stop_words:
                idx = response_text.find(stop_word)
                if idx != -1 and idx < first_stop_idx:
                    first_stop_idx = idx
            response_text = response_text[:first_stop_idx]

        completion_id = f"chatcmpl-{uuid.uuid4()}"
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": data.get("model", "local-model"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": input_len,
                "completion_tokens": len(generated_tokens),
                "total_tokens": input_len + len(generated_tokens)
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model ID or local path")
    parser.add_argument("--port", type=int, default=1234, help="Server port")
    args = parser.parse_args()
    
    print(f"Loading model and tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Autodetect device
    if torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    elif torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32
        
    print(f"Using device: {device} with dtype: {dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True
    ).to(device)
    model.eval()
    print("Model loaded successfully.")
    
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")
