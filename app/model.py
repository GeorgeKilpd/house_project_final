from datetime import datetime
from app import db
from sqlalchemy.dialects.sqlite import JSON

class Question(db.Model):
    __tablename__ = 'question'
    id = db.Column(db.Integer, primary_key=True)
    subject = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    create_date = db.Column(db.DateTime(), nullable=False, default=datetime.utcnow)


class Answer(db.Model):
    __tablename__ = 'answer'
    id = db.Column(db.Integer, primary_key=True)
    question_id = db.Column(db.Integer, db.ForeignKey('question.id', ondelete='CASCADE'))
    question = db.relationship('Question', backref=db.backref('answer_set', cascade='all, delete-orphan'))
    content = db.Column(db.Text, nullable=False)
    create_date = db.Column(db.DateTime(), nullable=False, default=datetime.utcnow)


class Users(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)

class SupportList(db.Model):
    __tablename__ = "SUPPORT_LIST"

    # Primary Key (CSV에서 가져오지 않고 새로 생성)
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)

    # ===== 기본 구분값 =====
    # loan / policy
    source_type = db.Column(db.String(20), nullable=False)

    # 지원 대상 구분 (전체 / 청년 / 신혼부부 등)
    target_type = db.Column(db.String(200), nullable=True)

    # ===== 비즈니스 타입 1,2 =====
    business_type1 = db.Column(db.String(200), nullable=True)
    business_type2 = db.Column(db.String(200), nullable=True)

    # 시행 기관
    implementing_agency = db.Column(db.String(200), nullable=True)

    # ===== 공통 정보 =====
    title = db.Column(db.String(500), nullable=False)
    homepage_url = db.Column(db.String(500), nullable=True)

    # ===== Loan 품목 전용 =====
    loan_target = db.Column(db.String(255), nullable=True)  # 대출대상
    loan_rate = db.Column(db.String(100), nullable=True)    # 대출금리
    loan_limit = db.Column(db.String(100), nullable=True)   # 대출한도
    loan_period = db.Column(db.String(100), nullable=True)  # 대출기간

    # ===== Policy 기준 정보 =====
    policy_income = db.Column(db.String(200), nullable=True)    # 소득기준
    policy_asset = db.Column(db.String(200), nullable=True)     # 자산기준
    policy_age = db.Column(db.String(200), nullable=True)       # 연령기준

    # ===== 상세 정보(JSON) =====
    detail_json = db.Column(JSON, nullable=True)

class HouseInfo(db.Model):
    __tablename__ = 'HOUSE_INFO'

    # 기본 정보
    building_name = db.Column(db.Text, primary_key=True)
    district = db.Column(db.Text, primary_key=True)
    floor = db.Column(db.Integer, primary_key=True)
    area_m2 = db.Column(db.Float, primary_key=True)
    built_year = db.Column(db.Integer)
    house_type = db.Column(db.Text, primary_key=True)
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)

    # 전세 예측값 2025~2030
    deposit_25q1 = db.Column(db.Float)
    deposit_25q2 = db.Column(db.Float)
    deposit_25q3 = db.Column(db.Float)
    deposit_25q4 = db.Column(db.Float)

    deposit_26q1 = db.Column(db.Float)
    deposit_26q2 = db.Column(db.Float)
    deposit_26q3 = db.Column(db.Float)
    deposit_26q4 = db.Column(db.Float)

    deposit_27q1 = db.Column(db.Float)
    deposit_27q2 = db.Column(db.Float)
    deposit_27q3 = db.Column(db.Float)
    deposit_27q4 = db.Column(db.Float)

    deposit_28q1 = db.Column(db.Float)
    deposit_28q2 = db.Column(db.Float)
    deposit_28q3 = db.Column(db.Float)
    deposit_28q4 = db.Column(db.Float)

    deposit_29q1 = db.Column(db.Float)
    deposit_29q2 = db.Column(db.Float)
    deposit_29q3 = db.Column(db.Float)
    deposit_29q4 = db.Column(db.Float)

    deposit_30q1 = db.Column(db.Float)
    deposit_30q2 = db.Column(db.Float)
    deposit_30q3 = db.Column(db.Float)
    deposit_30q4 = db.Column(db.Float)

    # 월세 예측값 2025~2030
    monthly_rent_25q1 = db.Column(db.Float)
    monthly_rent_25q2 = db.Column(db.Float)
    monthly_rent_25q3 = db.Column(db.Float)
    monthly_rent_25q4 = db.Column(db.Float)

    monthly_rent_26q1 = db.Column(db.Float)
    monthly_rent_26q2 = db.Column(db.Float)
    monthly_rent_26q3 = db.Column(db.Float)
    monthly_rent_26q4 = db.Column(db.Float)

    monthly_rent_27q1 = db.Column(db.Float)
    monthly_rent_27q2 = db.Column(db.Float)
    monthly_rent_27q3 = db.Column(db.Float)
    monthly_rent_27q4 = db.Column(db.Float)

    monthly_rent_28q1 = db.Column(db.Float)
    monthly_rent_28q2 = db.Column(db.Float)
    monthly_rent_28q3 = db.Column(db.Float)
    monthly_rent_28q4 = db.Column(db.Float)

    monthly_rent_29q1 = db.Column(db.Float)
    monthly_rent_29q2 = db.Column(db.Float)
    monthly_rent_29q3 = db.Column(db.Float)
    monthly_rent_29q4 = db.Column(db.Float)

    monthly_rent_30q1 = db.Column(db.Float)
    monthly_rent_30q2 = db.Column(db.Float)
    monthly_rent_30q3 = db.Column(db.Float)
    monthly_rent_30q4 = db.Column(db.Float)

    # 기타 컬럼
    monthly_rent = db.Column(db.Float)
    lease_type = db.Column(db.Text)
    road_address = db.Column(db.Text)
    jibun_address = db.Column(db.Text)
    dong_name = db.Column(db.Text)

    # 최신 실거래
    recent_deposit = db.Column(db.Float)
    recent_monthly = db.Column(db.Float)
    recent_yq = db.Column(db.Text)