# llama_server/app.py
import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = os.getenv("MODEL_ID", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

app = FastAPI()

tokenizer = None
model = None

class GenReq(BaseModel):
    text: str
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.95

def build_prompt(user_text: str) -> str:
    return f"""Below is an instruction that describes a task.
Write a response that appropriately completes the request.

### Instruction:
{user_text}

### Response:
"""

def load_model():
    global tokenizer, model
    if model is not None:
        return

    dtype = torch.float16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, token=HF_TOKEN, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        token=HF_TOKEN,
    ).to("cuda")   # fp16/bf16에서는 OK

    model.eval()

@app.on_event("startup")
def startup():
    load_model()

@app.get("/health")
def health():
    return {"ok": True, "cuda": torch.cuda.is_available(), "model_loaded": model is not None}

@app.post("/llama/generate")
@torch.inference_mode()
def generate(req: GenReq):
    prompt = build_prompt(req.text)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    out = model.generate(
        **inputs,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        do_sample=req.temperature > 0,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    if "### Response:" in decoded:
        decoded = decoded.split("### Response:")[-1].strip()
    return {"answer": decoded}
