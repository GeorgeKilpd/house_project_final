# app/services/prediction_lookup.py
from app.model import HouseInfo

def _norm_yq(yq: str) -> str:
    return (yq or "").strip().upper().replace(" ", "")

def _col_for(prefix: str, target_yq: str) -> str:
    # prefix: "deposit" or "monthly_rent"
    # target_yq: "2025Q1"
    yq = _norm_yq(target_yq)
    if len(yq) != 6 or yq[4] != "Q":
        raise ValueError("target_yq must be like 2025Q1")
    yy = yq[2:4]   # "25"
    q = yq[-1]     # "1"
    return f"{prefix}_{yy}q{q}"

def _pick_best_row(rows, lease_type: str, target_yq: str):
    lease_type = (lease_type or "").strip()
    dep_col = _col_for("deposit", target_yq)
    mr_col  = _col_for("monthly_rent", target_yq)

    def score(r):
        dep = getattr(r, dep_col, None)
        mr  = getattr(r, mr_col, None)
        s = 0

        # lease_type 정확 일치 (여기서 이미 필터하지만 안전하게 가산)
        if getattr(r, "lease_type", None) == lease_type:
            s += 10000

        # 월세면: 해당 분기 월세값이 "양수"인 row 최우선
        if lease_type == "월세":
            if mr is not None and float(mr) > 0:
                s += 1000
            # 월세도 보증금이 의미가 있으면 가산
            if dep is not None and float(dep) >= 0:
                s += 10

        # 전세면: 보증금(전세금)이 양수인 row 우선
        if lease_type == "전세":
            if dep is not None and float(dep) > 0:
                s += 1000

        return s

    return sorted(rows, key=score, reverse=True)[0] if rows else None

def run_prediction_lookup(payload: dict, target_yq: str = "2025Q1") -> dict:
    target_yq = _norm_yq(target_yq)

    contract = payload.get("contract", {}) or {}
    lease_type = (contract.get("lease_type") or "").strip()
    if lease_type not in ("전세", "월세"):
        raise ValueError("payload.contract.lease_type must be '전세' or '월세'")

    region = payload.get("region", {}) or {}
    prop = payload.get("property", {}) or {}
    dbctx = payload.get("db_context", {}) or {}

    district = (region.get("district_code") or dbctx.get("district_code") or "").strip()
    building_name = (prop.get("building_name") or dbctx.get("building_name") or "").strip()
    house_type = (prop.get("house_type") or "").strip()
    dong_name = (region.get("dong_name") or "").strip()

    if not district or not building_name:
        raise ValueError("payload.region.district_code and payload.property.building_name are required")

    # ✅ 1) 후보 rows: lease_type을 WHERE로 강제 (전세 row가 끼는 걸 원천 차단)
    q = HouseInfo.query.filter(
        HouseInfo.district == district,
        HouseInfo.building_name == building_name,
    )

    if house_type:
        q = q.filter(HouseInfo.house_type == house_type)
    if dong_name:
        q = q.filter(HouseInfo.dong_name == dong_name)

    rows = q.all()

    # 너무 좁혀서 rows가 0이면, house_type/dong_name 조건을 풀고 다시 검색
    if not rows:
        q2 = HouseInfo.query.filter(
            HouseInfo.district == district,
            HouseInfo.building_name == building_name,
        )
        rows = q2.all()

    if not rows:
        raise ValueError("No matching rows found in DB for building/district")

    # ✅ 2) 같은 lease_type 내에서도 월세=0 같은 row 피하려면 스코어링
    chosen = _pick_best_row(rows, lease_type=lease_type, target_yq=target_yq)
    if chosen is None:
        raise ValueError("No usable row found after scoring")

    # 3) 컬럼 결정
    dep_col = _col_for("deposit", target_yq)
    mr_col  = _col_for("monthly_rent", target_yq)

    dep_val = getattr(chosen, dep_col, None)
    mr_val  = getattr(chosen, mr_col, None)

    # 4) 반환 규칙
    if lease_type == "전세":
        return {
            "lease_type": "전세",
            "target_yq": target_yq,
            "deposit_column": dep_col,
            "predicted_deposit_krw": float(dep_val) if dep_val is not None else None,
            "selected_rowid": getattr(chosen, "id", None) or getattr(chosen, "rowid", None),
            "selected_lease_type": getattr(chosen, "lease_type", None),
        }

    return {
        "lease_type": "월세",
        "target_yq": target_yq,
        "deposit_column": dep_col,
        "monthly_rent_column": mr_col,
        "predicted_deposit_krw": float(dep_val) if dep_val is not None else None,
        "predicted_monthly_rent_krw": float(mr_val) if mr_val is not None else None,
        "selected_rowid": getattr(chosen, "id", None) or getattr(chosen, "rowid", None),
        "selected_lease_type": getattr(chosen, "lease_type", None),
    }
