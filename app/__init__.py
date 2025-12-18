"""
DB 사용을 위한 config 설정
"""
from flask import Flask
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from app.config import Config

db = SQLAlchemy()
migrate = Migrate()

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # 🔥 🔥 HTML 템플릿 자동 리로드 활성화 🔥 🔥
    app.config["TEMPLATES_AUTO_RELOAD"] = True

    # 🔥 캐시 때문에 업데이트 안될 경우 해결
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

    db.init_app(app)
    migrate.init_app(app, db)

    from app import model, ml_model

    # Blueprint 등록
    from .views import index_views, predict_views, support_views, login_views, inquiry_views, llama3_views
    app.register_blueprint(index_views.bp)
    app.register_blueprint(predict_views.bp)
    app.register_blueprint(support_views.bp)
    app.register_blueprint(login_views.bp)
    app.register_blueprint(inquiry_views.bp)
    app.register_blueprint(llama3_views.bp)

    return app
