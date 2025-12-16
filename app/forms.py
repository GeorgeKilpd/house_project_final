from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, DateField, TextAreaField
from wtforms.validators import DataRequired, Length, EqualTo, Email


class QuestionForm(FlaskForm):
    subject=StringField('제목',validators=[DataRequired('제목은 필수 입력 항목입니다.')])
    content=TextAreaField('내용', validators=[DataRequired('내용은 필수 입력 항목입니다.')])

class AnswerForm(FlaskForm):
        content = TextAreaField('내용', validators=[DataRequired('내용은 필수 입력 항목입니다.')])

class UserCreateForm(FlaskForm):
    username = StringField('사용자이름', validators=[DataRequired(), Length(min=3, max=25)])
    password1 = PasswordField('비밀번호', validators=[
        DataRequired(), EqualTo('password2', '비밀번호가 일치하지 않습니다.')])
    password2 = PasswordField('비밀번호확인', validators=[DataRequired()])
    email = StringField('이메일', validators=[DataRequired(), Email()])
    
class UserLoginForm(FlaskForm):
    username = StringField('사용자이름', validators=[DataRequired(), Length(min=3, max=25)])
    password = PasswordField('비밀번호', validators=[DataRequired()])

class SignupForm(FlaskForm):
    username = StringField("아이디", validators=[DataRequired(), Length(min=4, max=25)])
    password = PasswordField("비밀번호", validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField("비밀번호 확인", validators=[DataRequired(), EqualTo("password")])
    phone = StringField("휴대폰 번호", validators=[DataRequired()])
    name = StringField("이름", validators=[DataRequired()])
    email = StringField("이메일", validators=[DataRequired(), Email()])
    birth = DateField("생년월일", format='%Y-%m-%d')
    submit = SubmitField("회원가입")

class LoginForm(FlaskForm):
    username = StringField("아이디", validators=[DataRequired(), Length(min=4, max=25)])
    password = PasswordField("비밀번호", validators=[DataRequired(), Length(min=6)])
    submit = SubmitField("로그인")


