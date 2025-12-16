from flask import Blueprint, render_template, request, url_for
from app.model import SupportList
from sqlalchemy import or_, case

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
                url_for("support.support_search") +
                f"?target={target}&biz={biz}&page={page}"
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

