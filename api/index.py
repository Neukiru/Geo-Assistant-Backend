from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_socketio import SocketManager
import time
import asyncio
     
import os
import tiktoken
from api.document_query_engine import DocumentQueryEngine
from api.utilities import get_service_context
from llama_index.callbacks import (
    CallbackManager, 
    TokenCountingHandler
)
                   
from llama_index.llms.openai import OpenAI
from langchain.memory import ChatMessageHistory
from api.milu_streaming_callback_handler import MiluStreamingCallbackHandler
from llama_index.agent import OpenAIAgent


doc_engine = DocumentQueryEngine(storage_path='./api/',model_of_choice="gpt-3.5-turbo")
token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
)
callback_manager = CallbackManager([token_counter])

service_context = get_service_context("gpt-3.5-turbo",temperature=0.00,callback_manager=callback_manager,streaming=True)
doc_engine.add_engine("banking_topics", "Provides information about Bank of Georgia products and services", service_context)
query_engine_tools = doc_engine.get_engine_tools()
    
  
app = FastAPI()
origins = ["*"]
sio = SocketManager(app=app, cors_allowed_origins=[])
app.add_middleware(CORSMiddleware,
                   allow_origins=origins,
                   allow_credentials=True,
                   allow_methods=['*'],
                   allow_headers=['*'])


preamble = """
As a Bank of Georgia AI Agent, your focus is on delivering courteous and effective customer service. 
Generate answers based on information provided in banking topics knowledge base.
Your responses should be as short and concise as possible. Only address user's question, do not provide additional information.
If a customer asks about topics unrelated to banking, or beyond the information you can infer from context, provide a brief response and politely steer the conversation back to banking-related matters.
If the knowledge base lacks a suitable information, inform the customer that you will redirect them to another operator for further help.
"""
     
history = ChatMessageHistory()
history.add_user_message(preamble)
openai_agent = None



# async def websocket_emitter():
#     global message_ready
#     while True:
#         await message_ready.wait()

#         # Reset the event for the next message
#         message_ready.clear()
#         message = await message_queue.get()
#         print('message',message)
#         await sio.emit("assistant_response", message, to=message["user_sid"])
#         message_queue.task_done()


@app.on_event("startup")
async def startup_event():
    pass
    # asyncio.create_task(websocket_emitter())


@sio.on('message')
async def on_message(sid, *args, **kwargs):
    user_message = args[0]['message']
    chat_generator = openai_agent.astream_chat(user_message)
    async for response in chat_generator:
        response_gen = response.response_gen
    
    for token in response_gen:
        await sio.emit("assistant_response", {'message': token, 'message_end': False}, to=sid)

    await sio.emit("assistant_response", {'message': '', 'message_end': True}, to=sid)

@sio.on('connect')
async def on_connect(sid, *args, **kwargs):
    print("User connected")


@sio.on("initialize_agent")
async def on_initialize_agent(sid, *args, **kwargs):
    meta_kwargs = {"user_sid": sid}
    global openai_agent
    handler = MiluStreamingCallbackHandler(sio,**meta_kwargs)
    custom_llm = OpenAI(model="gpt-3.5-turbo-0613",temperature=0, callbacks=[handler])
    openai_agent = OpenAIAgent.from_tools(
        query_engine_tools,
        max_function_calls=10,
        llm=custom_llm,
        verbose=False,
        callback_manager=callback_manager,
        system_prompt = preamble
    )

    print("Agent initialized")

    print("Initialization complete")

@sio.on("print_event")
async def print_event(sid, *args, **kwargs):
    print(args[0])
