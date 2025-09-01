from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
import uvicorn

templates = Jinja2Templates(directory="templates")

app = FastAPI()

@app.get("/add")
def add(request: Request, name : str, gender : str): # 변수를 정의하고 값이 들옴 : 값의 타입 정의 
    return templates.TemplateResponse('gettest.html', context = {"request":request, 'name': name, 'gender':gender})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port = 5100)