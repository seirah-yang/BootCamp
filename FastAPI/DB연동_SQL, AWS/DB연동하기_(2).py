
from sqlalchemy import create_engine
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
import uvicorn 

templates = Jinja2Templates(directory="templates")
app = FastAPI()

db_con = create_engine("mysql+pymysql://test:1234@localhost/test")

@app.get("/sql")
def sql(request:Request):
    query = db_con.execute("select * from player")
    result_db = query.fetchall()
    
    result = []
    
    for data in result_db:
        temp = {"player_name":data[1], 'player_height':data[-2], 'player_weight':data[-1]}
        result.append(temp)
    print(result)
    return templates.TemplateResponse("sql.html", {'request':request,"result_table":result})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port = 8383)
