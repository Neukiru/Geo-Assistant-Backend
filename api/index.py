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
                   
from llama_index.bridge.langchain import ChatOpenAI
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
As a Bank of Georgia AI Agent, your focus is on delivering courteous and effective customer service. Understand the customer's query attentively and leverage the banking topics knowledge base to provide answers. Maintain politeness and helpfulness throughout the interaction. If a customer asks about topics unrelated to banking, or beyond the information you can infer from context, provide a brief response and politely steer the conversation back to banking-related matters. If the knowledge base lacks a suitable solution, inform the customer that you will redirect them to another operator for further help.
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
    print("User messaged", user_message)
    await openai_agent.achat(user_message)


@sio.on('connect')
async def on_connect(sid, *args, **kwargs):
    print("User connected")


@sio.on("initialize_agent")
async def on_initialize_agent(sid, *args, **kwargs):
    meta_kwargs = {"user_sid": sid}
    global openai_agent
    handler = MiluStreamingCallbackHandler(sio,**meta_kwargs)
    custom_llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0613", callbacks=[handler], streaming=True)
    openai_agent = OpenAIAgent.from_tools(
        query_engine_tools,
        max_function_calls=10,
        llm=custom_llm,
        verbose=False,
        chat_history=history,
        callback_manager=callback_manager,
    )

    print("Agent initialized")

    print("Initialization complete")

