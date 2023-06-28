from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_socketio import SocketManager
import time

app = FastAPI()
origins = ["*"]
sio = SocketManager(app=app, cors_allowed_origins=[])
app.add_middleware(CORSMiddleware,
                   allow_origins=origins,
                   allow_credentials=True,
                   allow_methods=['*'],
                   allow_headers=['*'])


@app.get("/hello")
def hello():
    return {"message": "Hello, World!"}


@app.get("/greet")
def greet(text: str):
    return {
        "content": text,
        "isUser": False,
        "knowledgeContext": "test"
    }


@sio.on('message')
async def on_message(sid, *args, **kwargs):
    user_message = args[0]['message']
    print("User connected", user_message)
    for index in range(0, 5):
        if index == 0:
            message_start = True
        else:
            message_start = False
        if index == 4:
            message_end = True
        else:
            message_end = False
        
        await sio.emit('assistant_response', {'message': 'No no ', 'message_start': message_start, 'message_end': message_end}, to=sid)
        time.sleep(1)


@sio.on('connect')
async def on_connect(sid, *args, **kwargs):
    print("User connected")
