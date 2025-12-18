#!/bin/bash
set -e

# 현재 스크립트 위치(= llama_server)로 이동
cd "$(dirname "$0")"

# FastAPI 서버 실행 (llama_server/app.py 안의 app 객체)
python -m uvicorn app:app --host 0.0.0.0 --port 8000
