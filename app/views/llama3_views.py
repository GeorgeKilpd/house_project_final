import os
import requests
from flask import Blueprint, render_template, request, jsonify

bp = Blueprint("llama3", __name__)  # ✅ url_prefix 없음 (= 루트로 감)

LLAMA_URL = os.getenv("LLAMA_URL", "http://127.0.0.1:8000/llama/generate")

def call_llama3(text: str, max_new_tokens=256, temperature=0.2, top_p=0.95):
    r = requests.post(
        LLAMA_URL,
        json={
            "text": text,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json().get("answer", "")

# ✅ 페이지 (루트: /llama3)
@bp.get("/llama3")
def llama3_page():
    return render_template("llama3.html")

# ✅ API (루트: /api/llama3)
@bp.post("/api/llama3")
def llama3_api():
    data = request.get_json() or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text가 비어 있습니다."}), 400

    answer = call_llama3(text)
    return jsonify({"answer": answer})
