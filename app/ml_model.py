# app/ml_model.py
from app.model import HouseInfo

def _normalize_yq(target_yq: str) -> str:
    """
    '2025Q1' -> '25q1'
    '25Q1'   -> '25q1'
    '25q1'   -> '25q1'
    """
    s = (target_yq or "").strip()
    if not s:
        raise ValueError("target_yq is required. ex) '2025Q1'")

    s = s.replace(" ", "").upper()

    if s.startswith("20") and "Q" in s:
        yy = s[2:4]           # 2025 -> 25
        q = s.split("Q")[-1]  # 1
        return f"{yy}q{q}"

    if len(s) == 4 and s[:2].isdigit() and s[2] == "Q" and s[3].isdigit():
        return f"{s[:2].lower()}q{s[3]}"

    if len(s) == 4 and s[:2].isdigit() and s[2] == "q" and s[3].isdigit():
        return s

    raise ValueError("Invalid target_yq format. Use '2025Q1' or '25q1'.")


def run_prediction_lookup(payload: dict, target_yq: str) -> dict:
    """
    payload(build-input 결과) + target_yq 를 받아
    DB에 이미 저장된 예측 컬럼에서 값을 꺼내 반환
    """
    contract = payload.get("contract", {})
    lease_type = contract.get("lease_type")  # '전세' or '월세'
    norm = _normalize_yq(target_yq)          # '25q1'

    # build-input에서 db_context에 building_name이 이미 들어오니까 그걸 기준으로 찾는 방식 추천
    db_ctx = payload.get("db_context", {})
    building_name = db_ctx.get("building_name")
    district_code = payload.get("location", {}).get("district_code") or payload.get("district_code")

    if not building_name:
        raise ValueError("db_context.building_name is missing. build-input matching failed.")

    # 실제 DB에서 해당 건물 row 가져오기
    q = HouseInfo.query.filter(HouseInfo.building_name == building_name)

    # district도 있으면 같이 묶어주는 게 안전(동명이 건물 방지)
    if district_code:
        q = q.filter(HouseInfo.district == district_code)

    item = q.first()
    if not item:
        raise ValueError("No matching building found in DB for prediction lookup.")

    # 전세/월세에 따라 컬럼명 결정
    if lease_type == "전세":
        col = f"deposit_{norm}"  # deposit_25q1
        value = getattr(item, col, None)
        return {
            "lease_type": lease_type,
            "target_yq": target_yq,
            "column": col,
            "predicted_deposit_krw": value,
        }

    elif lease_type == "월세":
        col = f"monthly_rent_{norm}"  # monthly_rent_25q1
        value = getattr(item, col, None)
        return {
            "lease_type": lease_type,
            "target_yq": target_yq,
            "column": col,
            "predicted_monthly_rent_krw": value,
        }

    else:
        raise ValueError(f"Unknown lease_type: {lease_type}")
