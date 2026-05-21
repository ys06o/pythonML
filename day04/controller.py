from fastapi import APIRouter, Request
from service import service

router = APIRouter(prefix='/api/model')

@router.post("/learn")
async def 학습요청(request: Request):
    car_list = await request.json()
    print(car_list)
    return service.학습요청(car_list)

@router.post("/predict")
async def 예측요청(request: Request): 
    car = await request.json()
    print(car)
    return service.예측요청(car)