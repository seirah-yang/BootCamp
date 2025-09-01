#fastapi web 백엔드 프레임워크로 http://locallhost5000/alpaco를 입력하면 반응하는 함수정의
#그 함수는 이름(name), gender를 반환

#import 
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
import uvicorn

#directory 
templates = Jinja2Templates(directory="templates")

#web 
app = FastAPI()

#함수
#@app.get("/")
#def hello():
#    return{"meg":"hello, this is fastapi"}

@app.get("/alpaco")
def alpaco(request: Request):
    return templates.TemplateResponse("intro.html", context = {'request':request, 'name':'이름', 'gender': '성별'})

#서버실행구문 
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port = "5000")