# app/views/nlq_views.py
import os
import json
import re
import requests
from flask import Blueprint, request, jsonify

from app.services.prediction_lookup import run_prediction_lookup

bp = Blueprint("nlq", __name__, url_prefix="")

SYSTEM_PROMPT = """
너는 부동산 예측 서비스의 JSON 생성기다.
반드시 JSON만 출력한다. 다른 텍스트/설명/마크다운 금지.

스키마:
{
  "contract": {"lease_type": "전세|월세"},
  "region": {"district_code": "string", "dong_name": "string(optional)"},
  "property": {"building_name": "string", "house_type": "string(optional)"},
  "db_context": {"district_code": "string(optional)", "building_name": "string(optional)"}
}
""".strip()

def _extract_json(text: str) -> str:
    if not text:
        raise ValueError("Empty LLM response")

    text = text.strip()
    # ```json ... ``` 제거
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"LLM output is not JSON: {text[:200]}")
    return text[start:end+1]

def call_llm_make_payload(prompt: str) -> dict:
    """
    1) SYSTEM_PROMPT + 사용자 prompt로 LLM 호출
    2) 응답에서 JSON만 추출
    3) json.loads 실패하면 'JSON만 다시' 1회 재시도
    """
    base = (os.getenv("RUNPOD_BASE_URL") or "").strip().rstrip("/")
    if not base:
        raise ValueError("RUNPOD_BASE_URL is required (e.g. https://<podid>-5000.proxy.runpod.net)")

    url = f"{base}/llama/generate"

    # ✅ LLM에게 'JSON만' 강하게 요구
    full_prompt = f"""{SYSTEM_PROMPT}

사용자 입력:
{prompt}

규칙:
- 반드시 JSON만 출력
- JSON 바깥 텍스트(설명/마크다운/코드블록) 금지
- 모든 키는 스키마 그대로 사용
- 문자열 값은 반드시 큰따옴표로 감싸기
"""

    body = {
        "text": full_prompt,
        "max_new_tokens": 256,
        "temperature": 0.0,   # ✅ JSON 안정성 위해 0.0 권장
        "top_p": 0.95
    }

    r = requests.post(url, json=body, timeout=120)
    r.raise_for_status()

    text = (r.json().get("answer") or "").strip()
    if not text:
        raise ValueError(f"RunPod response missing 'answer': {r.text[:200]}")

    json_str = _extract_json(text)

    # 1차 파싱 시도
    try:
        return json.loads(json_str)

    except json.JSONDecodeError:
        # ✅ 1회 자동 수정 재시도
        fix_prompt = f"""{SYSTEM_PROMPT}

아래 출력은 JSON이 깨져있다.
반드시 올바른 JSON "한 개"만 출력해라.
추가 텍스트/설명/마크다운/코드블록 금지.

깨진 출력:
{json_str}
"""
        body2 = {
            "text": fix_prompt,
            "max_new_tokens": 256,
            "temperature": 0.0,
            "top_p": 0.95
        }

        r2 = requests.post(url, json=body2, timeout=120)
        r2.raise_for_status()

        text2 = (r2.json().get("answer") or "").strip()
        if not text2:
            raise ValueError(f"RunPod response missing 'answer' (retry): {r2.text[:200]}")

        json_str2 = _extract_json(text2)
        return json.loads(json_str2)





@bp.post("/nlq")
def nlq():
    data = request.get_json(force=True) or {}
    prompt = (data.get("prompt") or "").strip()
    target_yq = (data.get("target_yq") or "2025Q1").strip()

    if not prompt:
        return jsonify({"ok": False, "error": "prompt is required"}), 400

    payload = call_llm_make_payload(prompt)
    result = run_prediction_lookup(payload, target_yq=target_yq)

    return jsonify({
        "ok": True,
        "target_yq": target_yq,
        "payload": payload,
        "result": result
    }), 200
