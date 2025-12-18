import os
import json
import requests
from flask import Blueprint, render_template, request, url_for, jsonify
from sqlalchemy import or_, case
from app.nlp.pipelines import run_policy_qa, run_sentiment, translate_ko_to_en, generate_text, run_ner
from app.model import SupportList




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



bp = Blueprint('support', __name__, url_prefix='/support')


@bp.route('/search')
def support_search():
    all_items = SupportList.query.all()
    return render_template("support/support_search.html", items=all_items)


@bp.get("/<int:pid>")
def detail_view(pid: int):
    # 기본값은 항상 main
    source = request.args.get("source", "main")

    # 기본 뒤로가기: 메인 페이지
    return_url = url_for("index.index")

    # 목록에서 넘어온 경우 → 검색조건 포함해서 복원
    if source == "list":
        target = request.args.get("target", "")
        biz = request.args.get("biz", "")
        page = request.args.get("page", "1")

        return_url = (
            url_for("support.support_search")
            + f"?target={target}&biz={biz}&page={page}"
        )

    # DB 조회
    item = SupportList.query.get_or_404(pid)

    raw = item.detail_json
    try:
        detail = json.loads(raw) if isinstance(raw, str) else raw
    except Exception as e:
        print(">>> JSON 변환 오류:", e)
        return "detail_json 파싱 중 오류 발생", 500

    # 템플릿 분기
    if item.source_type == "loan":
        template_name = "support/loan_detail.html"
    elif item.source_type == "policy":
        template_name = "support/policy_detail.html"
    else:
        return f"지원하지 않는 유형입니다: {item.source_type}", 400

    return render_template(template_name, data=detail, return_url=return_url)

# 라마 3 페이지 이동
@bp.route("/llama3")
def llama3_page():
    return render_template("llama3.html")


# 🔹 생성형 AI 통합 챗봇 API  ---------------------------------
@bp.route("/api/genai-chat", methods=["POST"])
def genai_chat_api():

    """
    모달에서 호출하는 생성형 AI 통합 챗봇 API.
    request JSON: { "task": "...", "text": "...", "context": "..." }
    task:
      - generate  : 텍스트 생성
      - translate : 번역
      - sentiment : 감성분석
      - ner       : 개체명 인식
      - qa        : 정책 Q&A (context 필수)
    response JSON: { "answer": "..." }
    """
    data = request.get_json() or {}

    task = (data.get("task") or "").strip()
    text = (data.get("text") or "").strip()
    context = (data.get("context") or "").strip()

    if not task:
        return jsonify({"error": "task가 비어 있습니다."}), 400
    if not text:
        return jsonify({"error": "프롬프트(text)를 입력해주세요."}), 400

    # --- task별 분기 --------------------------------------
    # 1) 정책 Q&A
    if task == "qa":
        if not context:
            return jsonify({"error": "정책 Q&A는 context(정책 내용)가 필요합니다."}), 400

        result = run_policy_qa(context=context, question=text)
        answer = result.get("answer", "") or "적절한 답변을 찾지 못했습니다."
        return jsonify({"answer": answer})

    # 2) 번역
    elif task == "translate":
        answer = translate_ko_to_en(text)

    # 3) 감성분석
    elif task == "sentiment":
        result = run_sentiment(text)
        label = result.get("label", "")
        score = float(result.get("score", 0.0))

        try:
            stars = int(label.split()[0])
        except Exception:
            stars = 3  # 파싱 실패하면 중립

        if stars <= 2:
            ko_label = "부정"
        elif stars == 3:
            ko_label = "중립"
        else:
            ko_label = "긍정"

        answer = f"예측 감성: {ko_label} ({label}, score={score:.3f})"

    # 4) 개체명 인식
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

    # 5) 텍스트 생성
    elif task == "generate":
        answer = generate_text(text)


 


    else:
        return jsonify({"error": f"지원하지 않는 task입니다: {task}"}), 400

    # 공통 응답
    return jsonify({"answer": answer})

