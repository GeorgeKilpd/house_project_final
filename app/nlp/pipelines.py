# app/nlp/pipelines.py
import os
import torch
from transformers import pipeline

# -----------------------------
# 0) 캐시 경로 (RunPod /workspace 추천)
#   - runpod에서 export로 주는게 베스트지만,
#     코드에서도 안전하게 한번 더 잡아줌
# -----------------------------
DEFAULT_HF_HOME = os.environ.get("HF_HOME", "/workspace/.cache/huggingface")
os.environ.setdefault("HF_HOME", DEFAULT_HF_HOME)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(DEFAULT_HF_HOME, "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(DEFAULT_HF_HOME, "transformers"))
os.environ.setdefault("TORCH_HOME", os.environ.get("TORCH_HOME", "/workspace/.cache/torch"))

# -----------------------------
# 1) 디바이스 결정
#   - RunPod에서 GPU가 없으면 -1 (CPU)
#   - GPU가 있어도 VRAM 아끼려면 일부는 CPU로 돌리는 선택도 가능
# -----------------------------
DEVICE = 0 if torch.cuda.is_available() else -1

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
            device=DEVICE,
        )
    return _policy_qa_model


def _get_text_gen():
    global _text_gen_model
    if _text_gen_model is None:
        _text_gen_model = pipeline(
            "text-generation",
            model="skt/kogpt2-base-v2",
            tokenizer="skt/kogpt2-base-v2",
            device=DEVICE,
        )
    return _text_gen_model


def _get_ko2en():
    global _ko2en_model
    if _ko2en_model is None:
        _ko2en_model = pipeline(
            "translation",
            model="Helsinki-NLP/opus-mt-ko-en",
            tokenizer="Helsinki-NLP/opus-mt-ko-en",
            device=DEVICE,
        )
    return _ko2en_model


def _get_sentiment():
    global _sentiment_model
    if _sentiment_model is None:
        _sentiment_model = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            tokenizer="nlptown/bert-base-multilingual-uncased-sentiment",
            device=DEVICE,
        )
    return _sentiment_model


def _get_ner():
    """
    ⚠️ 주의: 지금 코드는 model을 안 적어서 기본값으로 잡히는데,
    환경에 따라 모델을 받다가 커지거나 불안정할 수 있어.
    가능하면 명시 모델 추천.
    (일단은 기존 동작 유지)
    """
    global _ner_model
    if _ner_model is None:
        _ner_model = pipeline(
            "ner",
            grouped_entities=True,
            device=DEVICE,
        )
    return _ner_model


# -----------------------------
# 3) 외부에서 쓰는 함수들 (API)
# -----------------------------
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
