from sqlalchemy import create_engine
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
import uvicorn 

templates = Jinja2Templates(directory="templates")
app = FastAPI()
#DB커넥터 객체            #DB 종류 + 커넥터://DB id : pw@ip/db명
db_con = create_engine("mysql+pymysql://test:1234@localhost/test")
        # execute : query 날리기 
# query = db_con.execute("select * from player")
# result = query.fetchall()

# for data in result: #result  for문 돌리면서 unpacking 
#     print(data)
@app.get("/mysqltest")
def mysqltest(request:Request):
    query = db_con.execute("select * from player")
    result_db = query.fetchall()
    
    result = []
    
    for data in result_db:
        temp = {"player_id":data[0], 'player_name':data[1]}
        result.append(temp)
    print(result)
    return templates.TemplateResponse("sqltest.html", {'request':request,"result_table":result})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port = 8000)
