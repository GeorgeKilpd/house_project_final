from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = (BASE_DIR / "realestate_v0.5.1.db").resolve()

class Config:
    SQLALCHEMY_DATABASE_URI = f"sqlite:////{DB_PATH.as_posix().lstrip('/')}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = "dev"
