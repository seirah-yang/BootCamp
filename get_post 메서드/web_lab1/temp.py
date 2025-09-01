@app.get("/test")
def test(request: Request):
    return templates.TemplateResponse("test.html", contedt = {'request':request, 'name':'이름', 'gender': '성별'})

#서버실행구문 
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port = "5000/test")