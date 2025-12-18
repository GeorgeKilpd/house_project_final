import os
import json
import requests
from flask import Blueprint, render_template, request, url_for, jsonify
from app.nlp.pipelines import run_policy_qa, run_sentiment, translate_ko_to_en, generate_text, run_ner
from app.model import SupportList


# ✅ LLaMA 서버 주소 (RunPod 외부 URL은 환경변수로 넣고, 없으면 로컬 기본값)
LLAMA_URL = os.getenv("LLAMA_URL", "http://127.0.0.1:8000/llama/generate").strip()

# ✅ requests 세션(커넥션 재사용)
_http = requests.Session()


def call_llama3(text: str, max_new_tokens: int = 256, temperature: float = 0.2, top_p: float = 0.95) -> str:
    # ✅ URL 미설정 방지
    if not LLAMA_URL:
        raise RuntimeError("LLAMA_URL이 비어있습니다. 환경변수 LLAMA_URL을 설정하세요.")

    payload = {
        "text": text,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }

    r = _http.post(
        LLAMA_URL,
        json=payload,
        timeout=180,  # ✅ LLaMA 응답이 느릴 수 있어 여유 있게
    )

    # ✅ 에러 본문을 같이 남겨서 디버깅 쉽게
    if r.status_code >= 400:
        raise RuntimeError(f"LLaMA server error {r.status_code}: {r.text[:800]}")

    data = r.json()
    return (data.get("answer") or "").strip()


bp = Blueprint("support", __name__, url_prefix="/support")


@bp.route("/search")
def support_search():
    all_items = SupportList.query.all()
    return render_template("support/support_search.html", items=all_items)


@bp.get("/<int:pid>")
def detail_view(pid: int):
    source = request.args.get("source", "main")
    return_url = url_for("index.index")

    if source == "list":
        target = request.args.get("target", "")
        biz = request.args.get("biz", "")
        page = request.args.get("page", "1")

        return_url = url_for("support.support_search") + f"?target={target}&biz={biz}&page={page}"

    item = SupportList.query.get_or_404(pid)

    raw = item.detail_json
    try:
        detail = json.loads(raw) if isinstance(raw, str) else raw
    except Exception as e:
        print(">>> JSON 변환 오류:", repr(e))
        return "detail_json 파싱 중 오류 발생", 500

    if item.source_type == "loan":
        template_name = "support/loan_detail.html"
    elif item.source_type == "policy":
        template_name = "support/policy_detail.html"
    else:
        return f"지원하지 않는 유형입니다: {item.source_type}", 400

    return render_template(template_name, data=detail, return_url=return_url)


@bp.route("/llama3")
def llama3_page():
    return render_template("llama3.html")


@bp.route("/api/genai-chat", methods=["POST"])
def genai_chat_api():
    """
    request JSON: { "task": "...", "text": "...", "context": "..." }
    task:
      - generate  : 텍스트 생성
      - translate : 번역
      - sentiment : 감성분석
      - ner       : 개체명 인식
      - qa        : 정책 Q&A (context 필수)
    response JSON: { "answer": "..." }
    """
    data = request.get_json(silent=True) or {}
    task = (data.get("task") or "").strip()
    text = (data.get("text") or "").strip()
    context = (data.get("context") or "").strip()

    if not task:
        return jsonify({"error": "task가 비어 있습니다."}), 400
    if not text:
        return jsonify({"error": "프롬프트(text)를 입력해주세요."}), 400

    if task == "qa":
        if not context:
            return jsonify({"error": "정책 Q&A는 context(정책 내용)가 필요합니다."}), 400
        result = run_policy_qa(context=context, question=text)
        answer = result.get("answer", "") or "적절한 답변을 찾지 못했습니다."
        return jsonify({"answer": answer})

    elif task == "translate":
        answer = translate_ko_to_en(text)
        return jsonify({"answer": answer})

    elif task == "sentiment":
        result = run_sentiment(text)
        label = result.get("label", "")
        score = float(result.get("score", 0.0))

        try:
            stars = int(label.split()[0])
        except Exception:
            stars = 3

        if stars <= 2:
            ko_label = "부정"
        elif stars == 3:
            ko_label = "중립"
        else:
            ko_label = "긍정"

        answer = f"예측 감성: {ko_label} ({label}, score={score:.3f})"
        return jsonify({"answer": answer})

    elif task == "ner":
        ents = run_ner(text)
        if not ents:
            answer = "인식된 개체명이 없습니다."
        else:
            lines = []
            for e in ents:
                word = e.get("word", "")
                label = e.get("entity_group") or e.get("entity") or "UNK"
                score = float(e.get("score", 0.0))
                lines.append(f"- {word} ({label}, score={score:.3f})")
            answer = "추출된 개체명 목록:\n" + "\n".join(lines)
        return jsonify({"answer": answer})

    elif task == "generate":
        answer = generate_text(text)
        return jsonify({"answer": answer})

    else:
        return jsonify({"error": f"지원하지 않는 task입니다: {task}"}), 400


@bp.route("/api/llama3", methods=["POST"])
def llama3_api():
    data = request.get_json(silent=True) or {}

    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text가 비어있습니다."}), 400

    # ✅ 프론트에서 옵션 넘기면 그대로 반영, 없으면 기본값
    max_new_tokens = int(data.get("max_new_tokens", 256))
    temperature = float(data.get("temperature", 0.2))
    top_p = float(data.get("top_p", 0.95))

    try:
        answer = call_llama3(
            text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return jsonify({"answer": answer})
    except Exception as e:
        print("LLAMA3 API ERROR:", repr(e))
        return jsonify({"error": str(e)}), 500
