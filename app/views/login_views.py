from flask import Blueprint, render_template, request, redirect, url_for, flash
from app.forms import SignupForm, LoginForm # 📌 방금 만든 폼 import
from datetime import date

bp = Blueprint('login', __name__, url_prefix='/auth')


@bp.route("/signup", methods=["GET", "POST"])
def signup():
    form = SignupForm()
    current_date = date.today().strftime("%Y-%m-%d")  # 오늘 날짜 문자열로 변환

    if form.validate_on_submit():
        # 회원가입 로직
        pass

    return render_template("login/signup.html", form=form, current_date=current_date)


@bp.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()  # ✅ 로그인 폼 객체 생성
    current_date = date.today().strftime("%Y-%m-%d")

    if form.validate_on_submit():
        # 로그인 검증 로직 작성 (예: DB 사용자 확인)
        return redirect(url_for("index.index"))  # 로그인 성공 시 이동

    # ✅ form을 템플릿으로 넘겨줘야 함!
    return render_template("login/login.html", form=form, current_date=current_date)