from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
import uvicorn

templates = Jinja2Templates(directory="templates")
app=FastAPI()

@app.get("/")
def hello():
    return {"message":"안녕하세요. fastAPI입니다."}

@app.get("/user")
def user(request: Request):
    return templates.TemplateResponse("userinfo.html", {'request':request, 'uid':'아이디', 'upw':'비밀번호', 'uname':'이름','ugender':'성별'})

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=9000)
