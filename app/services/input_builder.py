from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional, Dict, List, Tuple


KST = timezone(timedelta(hours=9))


# DB district_code -> 사람용 표시명 (LLM/화면용)
DISTRICT_DISPLAY_MAP = {
    "eunpyeong": "은평구",
    "guro": "구로구",
}


@dataclass(frozen=True)
class UserInput:
    district_code: str          # DB 값 그대로: 'eunpyeong' | 'guro'
    dong_name: str              # DB 값 그대로: 예) '불광동'
    house_type: str             # DB 값 그대로: 예) '아파트', '오피스텔' ...
    area_m2: float

    # 선택 입력 (없으면 DB 매칭 row에서 보강)
    built_year: Optional[int] = None
    floor: Optional[int] = None
    building_name: Optional[str] = None

    # 계약 입력 (예측 조건)
    lease_type: str = "월세"
    deposit_krw: Optional[int] = None
    monthly_rent_krw: Optional[int] = None  # 월세 예측 요청이면 None 유지 권장


def _project_root() -> Path:
    # app/services/input_builder.py 기준:
    # parents[0]=services, [1]=app, [2]=프로젝트 루트
    return Path(__file__).resolve().parents[2]


def _db_path() -> Path:
    # DB는 프로젝트 루트에 그대로 둔다고 했으니 여기로 고정
    return _project_root() / "realestate_v0.5.1.db"


def _parse_recent_yq(yq: Optional[str]) -> int:
    """
    recent_yq 예: '2025Q4'
    정렬용 점수로 변환: year*10 + quarter
    """
    if not yq:
        return -1
    try:
        yq = yq.strip().upper()
        year = int(yq[:4])
        q = int(yq[-1])
        return year * 10 + q
    except Exception:
        return -1


def _get_house_info_columns(conn: sqlite3.Connection) -> List[str]:
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(HOUSE_INFO);")
    cols = [row[1] for row in cur.fetchall()]
    if not cols:
        raise RuntimeError("HOUSE_INFO 테이블 컬럼을 읽지 못했습니다. DB 파일/테이블명을 확인하세요.")
    return cols


def _select_best_row(
    rows: List[sqlite3.Row],
    target_area: float,
    prefer_latest: bool = True,
) -> sqlite3.Row:
    """
    동일 조건에서 여러 row가 나오면:
    1) recent_yq 최신 우선
    2) 면적(area_m2) 차이 최소
    """
    def key_fn(r: sqlite3.Row) -> Tuple[int, float]:
        latest_score = _parse_recent_yq(r["recent_yq"]) if prefer_latest else 0
        area_val = float(r["area_m2"] or 0.0)
        return (latest_score, -abs(area_val - target_area))

    # max: latest_score 큰 것, area 차이 작은 것(-abs가 더 큰 것)
    return max(rows, key=key_fn)


def build_prediction_input_json(user: UserInput) -> Dict[str, Any]:
    """
    사용자 입력(폼) + DB(HOUSE_INFO) 조회로
    예측 입력 JSON(v1.0)을 완성해서 반환한다.

    - DB 조회 키: district, dong_name, house_type (+ building_name 있으면 활용)
    - 다중 매칭 시: recent_yq 최신 + area_m2 가장 가까운 row 선택
    - 사용자 입력 built_year/floor/building_name 있으면 우선, 없으면 DB 값으로 보강
    """

    # 0) 기본 JSON 뼈대
    payload: Dict[str, Any] = {
        "schema_version": "v1.0",
        "meta": {
            "request_id": None,
            "requested_at": datetime.now(KST).isoformat(),
            "source": "web_form",
        },
        "region": {
            "district_code": user.district_code,
            "district_display": DISTRICT_DISPLAY_MAP.get(user.district_code),
            "dong_name": user.dong_name,
        },
        "property": {
            "house_type": user.house_type,
            "area_m2": float(user.area_m2),
            "built_year": user.built_year,
            "floor": user.floor,
            "building_name": user.building_name,
        },
        "contract": {
            "lease_type": user.lease_type,
            "deposit_krw": user.deposit_krw,
            "monthly_rent_krw": user.monthly_rent_krw,
        },
        "location": {
            "latitude": None,
            "longitude": None,
        },
        "db_context": {
            "match_status": "not_checked",
            "matched_rowid": None,          # HOUSE_INFO는 id 컬럼이 없어서 rowid로 추적
            "recent_deposit_krw": None,
            "recent_monthly_rent_krw": None,
            "recent_yq": None,
            "road_address": None,
            "jibun_address": None,
            "building_name": None,
            # 분기 히스토리 (DB 컬럼 그대로)
            "deposit_history": {},
            "monthly_rent_history": {},
        },
        # 현재 DB에는 infra 테이블/컬럼이 없으니 null 유지(추후 확장)
        "infra_features": {
            "subway_distance_m": None,
            "bus_stop_count_500m": None,
            "mart_count_500m": None,
            "school_count_500m": None,
            "hospital_count_500m": None,
        },
    }

    db_file = _db_path()
    if not db_file.exists():
        payload["db_context"]["match_status"] = "db_not_found"
        payload["db_context"]["error"] = f"DB 파일을 찾을 수 없습니다: {db_file}"
        return payload

    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row

    try:
        cols = _get_house_info_columns(conn)

        # 1) 조회 쿼리 구성 (컬럼명은 DB 그대로 사용)
        # building_name이 들어오면 매칭 정확도 올리기 위해 조건 추가(선택)
        where_clauses = [
            "district = ?",
            "dong_name = ?",
            "house_type = ?",
        ]
        params: List[Any] = [user.district_code, user.dong_name, user.house_type]

        if user.building_name:
            where_clauses.append("building_name = ?")
            params.append(user.building_name)

        where_sql = " AND ".join(where_clauses)

        # rowid 포함해서 추적 가능하게 가져오기
        select_sql = f"""
            SELECT
                rowid AS _rowid,
                {", ".join(cols)}
            FROM HOUSE_INFO
            WHERE {where_sql}
        """

        cur = conn.cursor()
        cur.execute(select_sql, params)
        rows = cur.fetchall()

        if not rows:
            payload["db_context"]["match_status"] = "no_match"
            return payload

        # 2) 다중 매칭 해결
        best = _select_best_row(rows, target_area=float(user.area_m2), prefer_latest=True)

        # 3) DB 값으로 보강 (사용자 입력 우선)
        payload["db_context"]["match_status"] = "matched"
        payload["db_context"]["matched_rowid"] = best["_rowid"]

        # property 보강
        if payload["property"]["built_year"] is None:
            payload["property"]["built_year"] = best["built_year"]
        if payload["property"]["floor"] is None:
            payload["property"]["floor"] = best["floor"]
        if payload["property"]["building_name"] is None:
            payload["property"]["building_name"] = best["building_name"]

        # 위치
        payload["location"]["latitude"] = best["latitude"]
        payload["location"]["longitude"] = best["longitude"]

        # 최근값/주소
        payload["db_context"]["recent_deposit_krw"] = best["recent_deposit"]
        payload["db_context"]["recent_monthly_rent_krw"] = best["recent_monthly"]
        payload["db_context"]["recent_yq"] = best["recent_yq"]
        payload["db_context"]["road_address"] = best["road_address"]
        payload["db_context"]["jibun_address"] = best["jibun_address"]
        payload["db_context"]["building_name"] = best["building_name"]

        # 4) 분기 히스토리: deposit_25q1~30q4, monthly_rent_25q1~30q4
        deposit_hist: Dict[str, Any] = {}
        monthly_hist: Dict[str, Any] = {}

        for c in cols:
            if c.startswith("deposit_"):
                deposit_hist[c] = best[c]
            elif c.startswith("monthly_rent_"):
                monthly_hist[c] = best[c]

        payload["db_context"]["deposit_history"] = deposit_hist
        payload["db_context"]["monthly_rent_history"] = monthly_hist

        return payload

    finally:
        conn.close()
