from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
import uvicorn

templates=Jinja2Templates(directory="templates")

app=FastAPI()

# 함수 
@app.get("/one")
def one(request: Request, a: int, b:int):
    return templates.TemplateResponse("one.html", context={"request":request, "a":a, "b":b})

@app.get("/two")
def two(request: Request, a: int, b:int):
    return templates.TemplateResponse("two.html", context={"request":request, "a":a, "b":b})

if __name__ =="__main__":
    uvicorn.run(app, host="0.0.0.0", port=3003)

