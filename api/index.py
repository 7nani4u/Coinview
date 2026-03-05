# api/index.py
# Vercel Python Runtime 진입점
# Vercel은 api/ 폴더의 파일을 자동으로 서버리스 함수로 인식합니다.
# main.py의 FastAPI app 인스턴스를 그대로 노출합니다.

import sys
import os

# 프로젝트 루트를 Python 경로에 추가 (main.py import를 위해)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app  # noqa: F401 - Vercel이 'app' 변수를 찾아 ASGI 앱으로 실행
