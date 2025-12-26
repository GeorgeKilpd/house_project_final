from flask import Blueprint, render_template, request, jsonify
from app.model import HouseInfo
from app.services.input_builder import UserInput, build_prediction_input_json
from app.ml_model import run_prediction_lookup

bp = Blueprint("predict", __name__, url_prefix="/predict")


# -------------------------------------------------
# 공통: 입력 검증/변환 헬퍼
# -------------------------------------------------
ALLOWED_DISTRICTS = {"eunpyeong", "guro"}
ALLOWED_HOUSE_TYPES = {"빌라", "오피스텔"}  # 너 DB 기준


def _get_str(form, key, *, default=None, required=False):
    v = form.get(key)
    if v is None:
        if required:
            raise ValueError(f"'{key}' is required")
        return default
    v = v.strip()
    if v == "":
        if required:
            raise ValueError(f"'{key}' is required")
        return default
    return v


def _get_float(form, key, *, required=False, min_value=None, max_value=None):
    raw = _get_str(form, key, default=None, required=required)
    if raw is None:
        return None
    try:
        val = float(raw)
    except Exception:
        raise ValueError(f"'{key}' must be a number")
    if min_value is not None and val < min_value:
        raise ValueError(f"'{key}' must be >= {min_value}")
    if max_value is not None and val > max_value:
        raise ValueError(f"'{key}' must be <= {max_value}")
    return val


def _get_int(form, key, *, required=False, min_value=None, max_value=None):
    raw = _get_str(form, key, default=None, required=required)
    if raw is None:
        return None
    try:
        # "100,000,000" 같은 입력도 방어
        raw = raw.replace(",", "")
        val = int(raw)
    except Exception:
        raise ValueError(f"'{key}' must be an integer")
    if min_value is not None and val < min_value:
        raise ValueError(f"'{key}' must be >= {min_value}")
    if max_value is not None and val > max_value:
        raise ValueError(f"'{key}' must be <= {max_value}")
    return val


# -------------------------------------------------
# 1) 예측 입력 JSON 생성 API (Postman용 / ML·LLM 공용)
# -------------------------------------------------
@bp.route("/build-input", methods=["POST"])
def build_input():
    try:
        form = request.form

        district_code = _get_str(form, "district_code", default="eunpyeong", required=True)
        dong_name = _get_str(form, "dong_name", required=True)
        house_type = _get_str(form, "house_type", required=True)

        # ✅ 추가: 전세/월세 필수
        lease_type = _get_str(form, "lease_type", required=True)

        area_m2 = _get_float(form, "area_m2", required=True, min_value=1.0, max_value=1000.0)
        deposit_krw = _get_int(form, "deposit_krw", required=True, min_value=0)

        # ✅ 월세면 월세값도 받을 수 있게(선택/필수는 너희 정책)
        monthly_rent_krw = _get_int(form, "monthly_rent_krw", required=False, min_value=0)

        if district_code not in ALLOWED_DISTRICTS:
            raise ValueError(f"'district_code' must be one of {sorted(ALLOWED_DISTRICTS)}")
        if house_type not in ALLOWED_HOUSE_TYPES:
            raise ValueError(f"'house_type' must be one of {sorted(ALLOWED_HOUSE_TYPES)}")

        # ✅ lease_type 검증
        if lease_type not in ("전세", "월세"):
            raise ValueError("'lease_type' must be one of ['전세','월세']")

        # ✅ 전세/월세 입력 규칙 강제 (원하는 정책으로 조절)
        if lease_type == "전세":
            # 전세는 monthly_rent를 받으면 안 받는 게 깔끔
            monthly_rent_krw = None

        if lease_type == "월세":
            # 월세는 보증금+월세 둘 다 있어야 의미가 있으면 required=True로 바꿔도 됨
            # if monthly_rent_krw is None:
            #     raise ValueError("'monthly_rent_krw' is required when lease_type is '월세'")
            pass

        user = UserInput(
            district_code=district_code,
            dong_name=dong_name,
            house_type=house_type,
            lease_type=lease_type,              # ✅ UserInput에 필드 추가 필요
            area_m2=area_m2,
            deposit_krw=deposit_krw,
            monthly_rent_krw=monthly_rent_krw,  # ✅ UserInput에 필드 추가 필요(선택)
        )

        payload = build_prediction_input_json(user)  # ✅ 여기서 lease_type 기반 매칭하도록 수정 필요
        return jsonify(payload), 200

    except ValueError as e:
        return jsonify({
            "error": "validation_error",
            "message": str(e),
            "hint": {
                "required_keys": ["district_code", "dong_name", "house_type", "lease_type", "area_m2", "deposit_krw"],
                "allowed_lease_type": ["전세", "월세"],
            }
        }), 400

    except Exception:
        return jsonify({
            "error": "server_error",
            "message": "Failed to build prediction input json"
        }), 500

@bp.route("/run", methods=["POST"])
def run_prediction():
    payload = request.get_json(force=True)
    target_yq = request.args.get("target_yq", "2025Q1").strip()

    try:
        result = run_prediction_lookup(payload, target_yq=target_yq)
        return jsonify({
            "ok": True,

            "lease_type": result.get("selected_lease_type"),  # 👈 이 한 줄

            "debug": {
                "target_yq": target_yq,
                "payload_lease_type": (payload.get("contract", {}) or {}).get("lease_type"),
                "payload_building_name": (payload.get("db_context", {}) or {}).get("building_name"),
                "selected_rowid": result.get("selected_rowid"),
                "selected_lease_type": result.get("selected_lease_type"),
            },
            "result": result
        }), 200
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": "server_error"}), 500


# -------------------------------------------------
# 2) (기존) 검색 UI용 헬퍼
# -------------------------------------------------
def convert_gu_to_kor(gu):
    table = {"eunpyeong": "은평구", "guro": "구로구"}
    return table.get(gu, gu)


def convert_m2_to_pyeong(m2):
    if m2 is None:
        return None
    p = m2 / 3.305785
    return f"{int(round(p))}평"


def convert_yq_to_kor(yq):
    if not yq:
        return ""
    try:
        year = yq[:4]
        quarter = yq[-1]
        return f"{year}년 {quarter}분기 계약"
    except Exception:
        return yq


def convert_floor(floor):
    if floor is None:
        return ""
    if floor < 0:
        return f"지하 {abs(floor)}층"
    return f"{floor}층"


# -------------------------------------------------
# 3) (기존) 검색 화면
# -------------------------------------------------
@bp.route("/search")
def predict_search():
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
    query = query.filter(HouseInfo.district == gu)
    query = query.filter(HouseInfo.house_type == house_type)
    query = query.filter(HouseInfo.lease_type == lease_type)

    # 면적 필터
    try:
        min_p, max_p = area_range.split("-")
        min_p = int(min_p)
        max_p = int(max_p)
        min_m2 = min_p * 3.305785
        max_m2 = max_p * 3.305785
        query = query.filter(HouseInfo.area_m2 >= min_m2, HouseInfo.area_m2 <= max_m2)
    except Exception:
        pass

    # 층수 필터
    if floor_range == "basement":
        query = query.filter(HouseInfo.floor < 0)
    elif floor_range == "low":
        query = query.filter(HouseInfo.floor >= 1, HouseInfo.floor <= 4)
    elif floor_range == "mid":
        query = query.filter(HouseInfo.floor >= 5, HouseInfo.floor <= 10)
    elif floor_range == "high":
        query = query.filter(HouseInfo.floor >= 11)

    raw_items = query.all()

    items = []
    for item in raw_items:
        row = {
            "building_name": item.building_name,

            # display + code 둘 다 내려주기 (매우 중요)
            "district_code": item.district,
            "district": convert_gu_to_kor(item.district),

            "floor": convert_floor(item.floor),
            "floor_raw": item.floor,
            "area_m2": item.area_m2,
            "area_p": convert_m2_to_pyeong(item.area_m2),
            "built_year": item.built_year,
            "house_type": item.house_type,
            "latitude": item.latitude,
            "longitude": item.longitude,

            "recent_yq": convert_yq_to_kor(item.recent_yq),
            "recent_yq_raw": item.recent_yq,
            "recent_deposit": item.recent_deposit,
            "recent_monthly": item.recent_monthly,

            "road_address": item.road_address,
            "jibun_address": item.jibun_address,
            "dong_name": item.dong_name,
            "lease_type": item.lease_type,

            "monthly_rent": item.monthly_rent,
        }

        # 전세 예측값 (2025~2030) - 안전 getattr
        for year in range(25, 31):
            for q in range(1, 5):
                key = f"deposit_{year}q{q}"
                row[key] = getattr(item, key, None)

        # 월세 예측값 (2025~2030) - 안전 getattr
        for year in range(25, 31):
            for q in range(1, 5):
                key = f"monthly_rent_{year}q{q}"
                row[key] = getattr(item, key, None)

        items.append(row)

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
