# app/nlp/pipelines.py
import os
import torch

# -----------------------------
# 0) 캐시 경로 (transformers import 전에!)
# -----------------------------
DEFAULT_HF_HOME = os.environ.get("HF_HOME", "/workspace/.cache/huggingface")
os.environ.setdefault("HF_HOME", DEFAULT_HF_HOME)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(DEFAULT_HF_HOME, "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(DEFAULT_HF_HOME, "transformers"))
os.environ.setdefault("TORCH_HOME", os.environ.get("TORCH_HOME", "/workspace/.cache/torch"))

from transformers import pipeline  # ✅ 캐시 설정 이후 import

# -----------------------------
# 1) 디바이스 결정 (pipeline용)
#   - CUDA 있으면 GPU
#   - 그 외는 CPU (MPS는 모델별로 불안정해서 일단 CPU 권장)
# -----------------------------
PIPELINE_DEVICE = 0 if torch.cuda.is_available() else -1

# -----------------------------
# 2) 파이프라인 싱글톤 (Lazy)
# -----------------------------
_policy_qa_model = None
_text_gen_model = None
_ko2en_model = None
_sentiment_model = None
_ner_model = None


def _get_policy_qa():
    global _policy_qa_model
    if _policy_qa_model is None:
        _policy_qa_model = pipeline(
            "question-answering",
            model="monologg/koelectra-base-v3-finetuned-korquad",
            tokenizer="monologg/koelectra-base-v3-finetuned-korquad",
            device=PIPELINE_DEVICE,
        )
    return _policy_qa_model


def _get_text_gen():
    global _text_gen_model
    if _text_gen_model is None:
        _text_gen_model = pipeline(
            "text-generation",
            model="skt/kogpt2-base-v2",
            tokenizer="skt/kogpt2-base-v2",
            device=PIPELINE_DEVICE,
        )
    return _text_gen_model


def _get_ko2en():
    global _ko2en_model
    if _ko2en_model is None:
        _ko2en_model = pipeline(
            "translation",
            model="Helsinki-NLP/opus-mt-ko-en",
            tokenizer="Helsinki-NLP/opus-mt-ko-en",
            device=PIPELINE_DEVICE,
        )
    return _ko2en_model


def _get_sentiment():
    global _sentiment_model
    if _sentiment_model is None:
        _sentiment_model = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            tokenizer="nlptown/bert-base-multilingual-uncased-sentiment",
            device=PIPELINE_DEVICE,
        )
    return _sentiment_model


def _get_ner():
    global _ner_model
    if _ner_model is None:
        # ⚠️ 가능하면 모델 명시 추천 (지금은 기존 동작 유지)
        _ner_model = pipeline(
            "ner",
            grouped_entities=True,
            device=PIPELINE_DEVICE,
        )
    return _ner_model


# -----------------------------
# 3) 외부에서 쓰는 함수들 (API)
# -----------------------------
def run_llama3(prompt: str) -> dict:
    if not prompt:
        return {"answer": ""}

    # ✅ LLaMA는 무거우니 필요할 때만 import (lazy)
    from .llama3_loader import llama3_generate
    return {"answer": llama3_generate(prompt)}


def run_policy_qa(context: str, question: str) -> dict:
    if not context or not question:
        return {"answer": "", "score": 0.0}
    qa = _get_policy_qa()
    return qa(question=question, context=context)


def generate_text(prompt: str, max_new_tokens: int = 80) -> str:
    if not prompt:
        return ""
    gen = _get_text_gen()
    outputs = gen(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.92,
        top_k=50,
        temperature=0.8,
        pad_token_id=gen.tokenizer.eos_token_id,
    )
    return outputs[0]["generated_text"].strip()


def translate_ko_to_en(text: str) -> str:
    if not text:
        return ""
    tr = _get_ko2en()
    result = tr(text)
    return result[0]["translation_text"]


def run_sentiment(text: str) -> dict:
    if not text:
        return {"label": "", "score": 0.0}
    s = _get_sentiment()
    return s(text)[0]


def run_ner(text: str):
    if not text:
        return []
    ner = _get_ner()
    return ner(text)
