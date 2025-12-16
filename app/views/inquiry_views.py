from flask import Blueprint, render_template, request, redirect, url_for
from app.model import Question, Answer
from ..forms import QuestionForm, AnswerForm
from datetime import datetime
from werkzeug.utils import redirect
from .. import db

# ✅ Blueprint 이름을 'inquiry'로 수정
bp = Blueprint('inquiry', __name__, url_prefix='/inquiry')

# 문의 메인 (필요 시 제거 가능)
@bp.route('/')
def inquiry_main():
    return redirect(url_for('inquiry._list'))

# 문의 목록 페이지
# @bp.route('/list')
# def inquiry_list():
#     page = request.args.get('page', type=int, default=1)
#     question_list = Question.query.order_by(Question.create_date.desc())
#     question_list = question_list.paginate(page=page, per_page=10)
#     return render_template('inquiry/inquiry_list.html', question_list=question_list)


# 문의 목록 페이지-UI 확인용 더미데이터
@bp.route('/list')
def inquiry_list():
    # 더미 데이터 13개
    dummy_questions = [
        {'id':1, 'subject':'정부지원 문의', 'author':'alice', 'create_date':'2025-11-08'},
        {'id':2, 'subject':'검색 방법 문의', 'author':'bob', 'create_date':'2025-11-07'},
        {'id':3, 'subject':'로그인 문제', 'author':'charlie', 'create_date':'2025-11-06'},
        {'id':4, 'subject':'비밀번호 재설정 문의', 'author':'david', 'create_date':'2025-11-06'},
        {'id':5, 'subject':'계정 삭제 요청', 'author':'eve', 'create_date':'2025-11-05'},
        {'id':6, 'subject':'회원가입 오류', 'author':'frank', 'create_date':'2025-11-05'},
        {'id':7, 'subject':'서비스 이용 방법', 'author':'grace', 'create_date':'2025-11-04'},
        {'id':8, 'subject':'결제 관련 문의', 'author':'heidi', 'create_date':'2025-11-04'},
        {'id':9, 'subject':'공지사항 확인 문의', 'author':'ivan', 'create_date':'2025-11-03'},
        {'id':10, 'subject':'앱 오류 신고', 'author':'judy', 'create_date':'2025-11-03'},
        {'id':11, 'subject':'데이터 다운로드 문의', 'author':'kim', 'create_date':'2025-11-02'},
        {'id':12, 'subject':'기타 문의', 'author':'leo', 'create_date':'2025-11-02'},
        {'id':13, 'subject':'프로모션 관련', 'author':'mallory', 'create_date':'2025-11-01'},
    ]
    return render_template('inquiry/inquiry_list.html', question_list=dummy_questions)




# 문의 상세 페이지
@bp.route('/detail/<int:question_id>/')
def detail(question_id):
    form = AnswerForm()
    question = Question.query.get_or_404(question_id)
    return render_template('inquiry/inquiry_detail.html', question=question, form=form)

# 문의 작성 페이지
@bp.route('/create', methods=['GET', 'POST'])
def create():
    form = QuestionForm()

    if request.method == 'POST' and form.validate_on_submit():
        question = Question(
            subject=form.subject.data,
            content=form.content.data,
            create_date=datetime.now()
        )

        db.session.add(question)
        db.session.commit()

        # ✅ 수정: inquiry._list로 정확하게 연결
        return redirect(url_for('inquiry._list'))

    return render_template('inquiry/inquiry_form.html', form=form)

# 답변 작성
@bp.route('/answer/<int:question_id>', methods=('GET', 'POST'))
def q_answer(question_id):
    form = AnswerForm()
    question = Question.query.get_or_404(question_id)
    if form.validate_on_submit():
        content = request.form['content']
        answer = Answer(content=content, create_date=datetime.now())
        question.answer_set.append(answer)
        db.session.commit()
        # ✅ 수정: detail로 정확히 연결
        return redirect(url_for('inquiry.detail', question_id=question_id))

    return render_template('inquiry/inquiry_detail.html', question=question, form=form)
