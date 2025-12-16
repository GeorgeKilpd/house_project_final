from app import create_app
import socket

app = create_app()

if __name__ == "__main__":
    # ============================
    # 🔍 서버 IP 자동 출력
    # ============================
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except:
        local_ip = "IP 탐지 실패"

    print("\n📌 서버 실행 중...")
    print("➡ Localhost  : http://127.0.0.1:5000")
    print(f"➡ Your IP    : http://{local_ip}:5000\n")

    # ============================
    # 🔥 Flask 서버 실행
    # ============================
    app.run(host="0.0.0.0", port=5000, debug=True)
