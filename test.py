from app import create_app, db
from sqlalchemy import text  # ← 반드시 import

app = create_app()

with app.app_context():
    with db.engine.connect() as conn:
        result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table';"))
        tables = [row[0] for row in result]
        print(tables)
