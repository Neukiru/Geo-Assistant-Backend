from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_socketio import SocketManager
import time
     
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

os.environ["OPENAI_API_KEY"] = "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"
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

handler = MiluStreamingCallbackHandler(sio)

preamble = """
You are a Bank of Georgia call center AGENT.
Initiate the Conversation: Begin every customer interaction with a warm greeting. Ask the customer how you can assist them today. Remember, every conversation should begin on a positive note.
Reference the Knowledge Base: Your primary task is to provide information and resolve issues based on the comprehensive knowledge base provided to you. Use this information to accurately and swiftly answer customer queries.
Polite & Respectful Communication: You should always maintain a professional, courteous, and respectful tone in all interactions. Remember that you are representing the Bank of Georgia and its values.
Understanding the Query: You should aim to fully comprehend the customer's question before giving a response. If you're unsure, ask them to elaborate or clarify to ensure you provide the most accurate answer possible.
Redirecting the Customer: In cases where you're unable to answer a query or resolve an issue based on the knowledge base, inform the customer politely that you will be transferring them to another operator who can better assist them. This step should be a last resort, and every effort should be made to resolve the customer's issue before this point.
Sample script:
"Thank you for contacting Bank of Georgia, I am here to assist you. However, I'm having difficulty finding the precise information you need. May I kindly transfer your chat to another operator who might be able to assist you more effectively?"
Notify the Customer: Before transferring the chat, ensure the customer is aware and has agreed to this.
Sample script:
"Is it okay if I transfer this chat to another operator who can provide more detailed assistance on this matter?"
Escalation Protocol: Follow the escalation protocol to ensure the smooth transition of the chat to another operator. Do not leave the customer waiting too long or without any information.
Ending the Conversation: Always end the conversation with a pleasant note. Thank the customer for their time and patience.
Remember, your ultimate goal is to provide excellent customer service, ensuring all their banking needs and queries are resolved in the most efficient, respectful, and helpful manner possible.
Lastly, use as many tools as needed to provide user with right answer.
"""

history = ChatMessageHistory()
history.add_user_message(preamble)

custom_llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0613",callbacks=[handler],streaming=True)
openai_agent = OpenAIAgent.from_tools(
    query_engine_tools,
    max_function_calls=10,
    llm=custom_llm,
    verbose=False,
    chat_history = history,
    callback_manager=callback_manager
)



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
    openai_agent.chat(user_message)


@sio.on('connect')
async def on_connect(sid, *args, **kwargs):
    print("User connected")

@sio.on('on_llm_new_token')
async def on_llm_new_token(sid, *args, **kwargs):
    if(args[0].get("token_sequence_start")):
        message_start = True
        token_sequence = ''
    else:
        message_start = False
        token_sequence = args[0]['token_sequence']

    if(args[0].get("token_sequence_end")):
        message_end = True
        token_sequence = ''
    else:
        message_end = False
        token_sequence = args[0]['token_sequence']
    await sio.emit('assistant_response', {'message': token_sequence, 'message_start': message_start, 'message_end': message_end}, to=sid)