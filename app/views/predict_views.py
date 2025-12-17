from flask import Blueprint, render_template, request
from app.model import HouseInfo
from sqlalchemy import and_
import os

bp = Blueprint("predict", __name__, url_prefix="/predict")


# ---------------------------------------
# 헬퍼 함수들
# ---------------------------------------

def convert_gu_to_kor(gu):
    table = {
        "eunpyeong": "은평구",
        "guro": "구로구",
    }
    return table.get(gu, gu)


def convert_m2_to_pyeong(m2):
    if m2 is None:
        return None
    p = m2 / 3.305785
    return f"{int(round(p))}평"


def convert_yq_to_kor(yq):
    """
    2024Q1 → 2024년 1분기 계약
    """
    if not yq:
        return ""

    try:
        year = yq[:4]
        quarter = yq[-1]
        return f"{year}년 {quarter}분기 계약"
    except:
        return yq


def convert_floor(floor):
    """
    -1 → 지하 1층
    -2 → 지하 2층
     1 → 1층
    """
    if floor is None:
        return ""
    if floor < 0:
        return f"지하 {abs(floor)}층"
    return f"{floor}층"


@bp.route("/search")
def predict_search():
    # -------------------------
    # 1) GET 파라미터 읽기
    # -------------------------
    def get_param(name, default):
        val = request.args.get(name)
        if val is None or val.strip() == "":
            return default
        return val.strip()

    gu = get_param("gu", "eunpyeong")
    house_type = get_param("house_type", "빌라")
    lease_type = get_param("lease_type", "월세")
    area_range = get_param("area", "10-19")
    floor_range = get_param("floor", "low")

    query = HouseInfo.query

    # -------------------------
    # 2) 지역 필터
    # -------------------------
    query = query.filter(HouseInfo.district == gu)

    # -------------------------
    # 3) 주택유형
    # -------------------------
    query = query.filter(HouseInfo.house_type == house_type)

    # -------------------------
    # 4) 거래유형
    # -------------------------
    query = query.filter(HouseInfo.lease_type == lease_type)

    # -------------------------
    # 5) 면적 필터 (평 → ㎡)
    # -------------------------
    try:
        min_p, max_p = area_range.split("-")
        min_p = int(min_p)
        max_p = int(max_p)

        min_m2 = min_p * 3.305785
        max_m2 = max_p * 3.305785

        query = query.filter(
            HouseInfo.area_m2 >= min_m2,
            HouseInfo.area_m2 <= max_m2
        )
    except:
        pass

    # -------------------------
    # 6) 층수 필터
    # -------------------------
    if floor_range == "basement":
        query = query.filter(HouseInfo.floor < 0)

    elif floor_range == "low":
        query = query.filter(HouseInfo.floor >= 1, HouseInfo.floor <= 4)

    elif floor_range == "mid":
        query = query.filter(HouseInfo.floor >= 5, HouseInfo.floor <= 10)

    elif floor_range == "high":
        query = query.filter(HouseInfo.floor >= 11)

    # -------------------------
    # 7) 조회
    # -------------------------
    raw_items = query.all()

    # -------------------------
    # 8) 데이터 가공 + 모든 컬럼 전달
    # -------------------------
    items = []
    for item in raw_items:

        row = {
            # 기본 컬럼 (PK + 주요 정보)
            "building_name": item.building_name,
            "district": convert_gu_to_kor(item.district),
            "floor": convert_floor(item.floor),
            "floor_raw": item.floor,                 # JS에서 원본층 필요할 수도 있음
            "area_m2": item.area_m2,
            "area_p": convert_m2_to_pyeong(item.area_m2),
            "built_year": item.built_year,
            "house_type": item.house_type,
            "latitude": item.latitude,
            "longitude": item.longitude,

            # 최신 실거래
            "recent_yq": convert_yq_to_kor(item.recent_yq),
            "recent_yq_raw": item.recent_yq,
            "recent_deposit": item.recent_deposit,
            "recent_monthly": item.recent_monthly,

            # 기타 주소 정보
            "road_address": item.road_address,
            "jibun_address": item.jibun_address,
            "dong_name": item.dong_name,
            "lease_type": item.lease_type,

            # 현재 월세(기존 컬럼)
            "monthly_rent": item.monthly_rent,
        }

        # 전세 예측값 (2025~2030)
        for year in range(25, 31):     # 25,26,27,28,29,30
            for q in range(1, 5):      # q1~q4
                key = f"deposit_{year}q{q}"
                row[key] = getattr(item, key)

        # 월세 예측값 (2025~2030)
        for year in range(25, 31):
            for q in range(1, 5):
                key = f"monthly_rent_{year}q{q}"
                row[key] = getattr(item, key)

        items.append(row)

    # -------------------------
    # 9) 템플릿 렌더링
    # -------------------------
    return render_template(
        "predict/predict_search.html",
        items=items,
        init_filter={
            "gu": gu,
            "house_type": house_type,
            "lease_type": lease_type,
            "area": area_range,
            "floor": floor_range,
        }
    )
