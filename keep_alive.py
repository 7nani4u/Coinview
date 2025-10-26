"""
Keep Alive Module for Streamlit Apps
이 모듈은 Flask 서버를 백그라운드에서 실행하여 외부에서 핑을 받을 수 있게 합니다.

⚠️ 주의사항:
1. Streamlit Community Cloud에서는 포트 8080이 외부에 노출되지 않을 수 있습니다.
2. 더 효과적인 방법은 UptimeRobot 같은 외부 서비스를 사용하는 것입니다.
3. Flask 추가로 인한 메모리 사용량 증가 (약 30-50MB)

사용 방법:
    from keep_alive import keep_alive
    
    # Streamlit 앱 시작 전에 호출
    keep_alive()
    
    # 이후 Streamlit 코드 실행
    st.title("My App")
    ...
"""

from flask import Flask
from threading import Thread
import logging

# Flask 로깅 레벨 설정 (불필요한 로그 최소화)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

@app.route('/')
def home():
    """
    헬스체크 엔드포인트
    외부 모니터링 서비스가 이 URL을 핑하여 앱이 살아있는지 확인
    """
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Crypto Prediction Bot Status</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                font-family: Arial, sans-serif;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }
            .status-card {
                background: white;
                padding: 40px;
                border-radius: 10px;
                box-shadow: 0 10px 25px rgba(0,0,0,0.2);
                text-align: center;
            }
            .status-icon {
                font-size: 64px;
                margin-bottom: 20px;
            }
            h1 {
                color: #333;
                margin: 0 0 10px 0;
            }
            p {
                color: #666;
                margin: 5px 0;
            }
            .version {
                margin-top: 20px;
                padding-top: 20px;
                border-top: 1px solid #eee;
                font-size: 12px;
                color: #999;
            }
        </style>
    </head>
    <body>
        <div class="status-card">
            <div class="status-icon">✅</div>
            <h1>Bot is Online</h1>
            <p>Crypto Prediction Streamlit App</p>
            <p>Status: <strong style="color: #4CAF50;">Running</strong></p>
            <div class="version">
                Version: 2.1.1 | Keep-Alive Module Active
            </div>
        </div>
    </body>
    </html>
    """, 200

@app.route('/health')
def health():
    """
    간단한 헬스체크 엔드포인트 (JSON 응답)
    """
    return {
        "status": "online",
        "service": "crypto-prediction-bot",
        "version": "2.1.1",
        "keep_alive": True
    }, 200

@app.route('/ping')
def ping():
    """
    Ping 엔드포인트 (최소 응답)
    """
    return "pong", 200

def run():
    """
    Flask 앱을 백그라운드에서 실행
    포트 8080에서 모든 인터페이스(0.0.0.0)로 리스닝
    """
    try:
        app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False)
    except Exception as e:
        # 포트가 이미 사용 중이거나 권한 문제 발생 시
        print(f"⚠️ Flask 서버 시작 실패: {e}")
        print("ℹ️  이는 정상적인 경우일 수 있습니다 (Streamlit Cloud 제한)")

def keep_alive():
    """
    Flask 서버를 별도 스레드에서 실행하여 keep-alive 기능 활성화
    
    Returns:
        bool: 서버 시작 성공 여부
    """
    try:
        t = Thread(target=run, daemon=True)
        t.start()
        print("✅ Keep-alive 서버 시작됨 (포트 8080)")
        print("ℹ️  외부에서 http://your-app-url:8080 으로 핑 가능")
        print("ℹ️  (단, Streamlit Cloud에서는 이 포트가 노출되지 않을 수 있음)")
        return True
    except Exception as e:
        print(f"⚠️ Keep-alive 서버 시작 실패: {e}")
        print("ℹ️  앱은 정상적으로 작동하지만 keep-alive는 비활성화됨")
        return False

# 직접 실행 시 테스트
if __name__ == "__main__":
    print("Keep-Alive 모듈 테스트")
    keep_alive()
    print("Flask 서버가 백그라운드에서 실행 중입니다...")
    print("http://localhost:8080 으로 접속하여 테스트하세요")
    
    # 메인 스레드 유지 (테스트용)
    import time
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n서버 종료")
