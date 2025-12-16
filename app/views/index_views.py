from flask import Blueprint, render_template
from app.model import SupportList, HouseInfo
from sqlalchemy import func
import random

bp = Blueprint('index', __name__, url_prefix='/')

@bp.route('/')
def index():

    # 1) 정책 데이터 — 랜덤 셔플
    policy_items = SupportList.query.all()
    random.shuffle(policy_items)

    # 2) 전월세 매물 — DB에서 랜덤 10개
    house_items = (
        HouseInfo.query
        .order_by(func.random())     # 🔥 랜덤 섞기
        .limit(10)                   # 🔥 10개만 가져오기
        .all()
    )

    # 3) 템플릿 렌더링
    return render_template(
        "index/index.html",
        policy_cards=policy_items,
        house_cards=house_items
    )
