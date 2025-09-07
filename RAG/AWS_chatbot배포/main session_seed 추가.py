# 새로 추가할 import
from fastapi import FastAPI, Request, Form, Response, Cookie
from pathlib import Path
import json, uuid
from datetime import datetime

app = FastAPI()
DATA_PATH = Path("data/chat_history.json")
DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

# 전역 메모리 대신, 세션별 대화를 담을 dict (메모리 캐시)
# 구조: {"<session_id>": [{"role": "user"/"bot", "content": "...", "ts": "..."}]}
_sessions: dict[str, list[dict]] = {}

def load_store():
    if DATA_PATH.exists():
        try:
            _sessions.update(json.loads(DATA_PATH.read_text(encoding="utf-8")))
        except Exception:
            pass  # 파일 깨졌을 때 서비스 중단 방지

def save_store():
    # 원자적 저장(임시파일 → 교체)까지 구현하면 더 안전
    DATA_PATH.write_text(json.dumps(_sessions, ensure_ascii=False, indent=2), encoding="utf-8")

load_store()

def get_or_create_session_id(session_id: str | None) -> str:
    if session_id:
        return session_id
    return uuid.uuid4().hex

@app.get("/", response_class=HTMLResponse)
async def get_chat(request: Request, session_id: str | None = Cookie(default=None)):
    sid = get_or_create_session_id(session_id)
    if sid not in _sessions:
        _sessions[sid] = []
        save_store()
    # 템플릿에 현재 세션의 히스토리만 전달
    response = templates.TemplateResponse("index.html", {"request": request,
                                                         "chat_history": _sessions[sid]})
    # 새 세션이면 쿠키 세팅
    if session_id is None:
        response.set_cookie("session_id", sid, httponly=True, samesite="lax")
    return response

@app.post("/", response_class=HTMLResponse)
async def post_chat(request: Request, user_input: str = Form(...), session_id: str | None = Cookie(default=None)):
    sid = get_or_create_session_id(session_id)
    if sid not in _sessions:
        _sessions[sid] = []

    # 사용자 메시지 저장
    _sessions[sid].append({"role": "user", "content": user_input, "ts": datetime.utcnow().isoformat()})

    # ---- LLM 호출 ----
    response = ollama.chat(
        model="exaone3.5:2.4b",
        messages=[{"role": "user", "content": user_input}],
    )
    answer = response["message"]["content"]

    # 봇 응답 저장
    _sessions[sid].append({"role": "bot", "content": answer, "ts": datetime.utcnow().isoformat()})

    # 디스크에 영속화
    save_store()

    # 렌더링
    resp = templates.TemplateResponse("index.html", {"request": request,
                                                     "chat_history": _sessions[sid]})
    if session_id is None:
        resp.set_cookie("session_id", sid, httponly=True, samesite="lax")
    return resp
#사용자마다 session_id 쿠키를 발급하여 세션별 대화 분리
#모든 대화를 data/chat_history.json에 지속 저장
#JSON 파일은 동시성·용량에 한계