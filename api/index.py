# api/index.py
# Vercel Python Runtime 진입점
# - Vercel은 api/ 폴더의 .py 파일을 자동으로 서버리스 함수로 인식합니다.
# - 'app' 변수(ASGI)를 찾아 FastAPI 앱으로 실행합니다.
# - Python 버전은 루트의 .python-version 파일로 지정합니다.

import sys
import os

# 프로젝트 루트 경로를 sys.path에 추가 (main.py 임포트용)
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from main import app  # noqa: F401, E402
