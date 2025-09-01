# import 
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from typing_extensions import Annotated
from fastapi import Form
import uvicorn

# 초기화 
templates = Jinja2Templates(directory = "templates")

app = FastAPI()

#로그인화면
@app.get("/login_get")
def login_get(request : Request):    
    return templates.TemplateResponse("login.html", context = {"request":request})

# post를 받기 
@app.post("/login_post")
def login_post(request: Request, uname:Annotated[str,Form()], pwd:Annotated[str,Form()],
            uid:Annotated[str,Form()], gender:Annotated[str,Form()]):
    print(uid,pwd,uname,gender)
# return 값이 없어 
# 서버
if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port=3212)
    
         <td type="text">아이디</td><td><input type="text" name="uid"></td>
        </tr>
        
        <tr>
        <td type="text">비밀번호</td><td><input type="password" name="psw"></td>
        </tr>
                <tr>
        <td type="text">이름</td><td><input type="name" name="uname"></td>
        </tr>
                <tr>
        <td type="text">성별</td><td><input type="gender" name="gen"></td>