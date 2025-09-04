from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import cv2
import uvicorn
app = FastAPI()
cap = cv2.VideoCapture(0)  # 웹캠
def generate_frames():  # 제너레이터 객체를 지칭하며, yield 키워드가 사용된 함수를 호출하면 제너레이터 객체가 반환
    while True:
        success, frame = cap.read()
        if not success:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'  #yield는 함수 실행을 일시 중지하고 값을 반환
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
@app.get("/video")
def video_feed():
    return StreamingResponse(generate_frames(),
                    media_type="multipart/x-mixed-replace; boundary=frame")
@app.get("/test")
def video_feedtest():
    return {"msg":"hi"}
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8500)
