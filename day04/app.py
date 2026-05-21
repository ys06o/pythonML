# 파이썬과 통신을 주고 받을 수 있는 FastAPI
from fastapi import FastAPI
import controller

app = FastAPI()

# controller.py 라우터 연결
app.include_router(controller.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)