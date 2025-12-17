# app/nlp/pipelines.py
from transformers import pipeline
import torch

# 정책 Q&A 모델 (앱 시작할 때 1번만 로드됨)
policy_qa_model = pipeline(
    "question-answering",
    model="monologg/koelectra-base-v3-finetuned-korquad",
    tokenizer="monologg/koelectra-base-v3-finetuned-korquad",
)

def run_policy_qa(context: str, question: str) -> dict:
    """
    정책 본문(context)과 사용자의 질문(question)을 넣으면
    HF Q&A 결과(dict)를 그대로 돌려줌.
    리턴 예: {"score": 0.92, "start": 10, "end": 15, "answer": "청년 전세자금대출"}
    """
    if not context or not question:
        return {"answer": "", "score": 0.0}

    result = policy_qa_model(
        question=question,
        context=context,
    )
    return result

# ✅ 새로 추가: 텍스트 생성 파이프라인 -------------------------
# 가벼운 한국어 GPT-2 모델 예시 (필요하면 다른 모델로 교체 가능)
text_gen_model = pipeline(
    "text-generation",
    model="skt/kogpt2-base-v2",
    tokenizer="skt/kogpt2-base-v2",
    device=0 if torch.cuda.is_available() else -1,
)

def generate_text(prompt: str, max_new_tokens: int = 80) -> str:
    if not prompt:
        return ""

    outputs = text_gen_model(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.92,
        top_k=50,
        temperature=0.8,
        pad_token_id=text_gen_model.tokenizer.eos_token_id,
    )

    full = outputs[0]["generated_text"].strip()

    # 프롬프트를 포함한 전체 문장을 반환
    return full



# --------------------------
# 3) 번역 파이프라인 (한국어 → 영어)
# --------------------------
ko2en_model = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-ko-en",
    tokenizer="Helsinki-NLP/opus-mt-ko-en",
    device=0 if torch.cuda.is_available() else -1,
)

def translate_ko_to_en(text: str) -> str:
    if not text:
        return ""
    result = ko2en_model(text)
    return result[0]["translation_text"]

# ---------------- 감성분석 ----------------
sentiment_model = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    tokenizer="nlptown/bert-base-multilingual-uncased-sentiment",
    device=0 if torch.cuda.is_available() else -1,
)

def run_sentiment(text: str) -> dict:
    """
    감성분석 결과 1개를 dict로 반환.
    예: {"label": "4 stars", "score": 0.87}
    """
    if not text:
        return {"label": "", "score": 0.0}

    result = sentiment_model(text)[0]
    return result

# ---------------- 개체명 인식(NER) ----------------
ner_model = pipeline(
    "ner",
    grouped_entities=True,  # 같은 엔티티는 묶어서 반환
    # model / tokenizer 안 적으면 기본 NER 모델 사용 (영어 위주지만 데모용 OK)
    device=0 if torch.cuda.is_available() else -1,
)

def run_ner(text: str):
    """
    개체명 인식 결과 리스트를 반환.
    예: [{"word": "Seoul", "entity_group": "LOC", "score": 0.99}, ...]
    """
    if not text:
        return []

    results = ner_model(text)
    return results