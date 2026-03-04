from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Vercel에서 FastAPI 앱이 성공적으로 실행되었습니다."}

@app.get("/api/hello")
def hello_world():
    return {"message": "Hello, World!"}
