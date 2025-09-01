from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
import uvicorn

templates=Jinja2Templates(directory="templates")

app=FastAPI()

@app.get("/add")
def add(request: Request, a: int, b:int):
    return templates.TemplateResponse("cal.html", context={"request":request, "a":2, "b":1})


if __name__ =="__main__":
    uvicorn.run(app, host="0.0.0.0", port=3333)