from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.schema import (
    BaseMessage,
    LLMResult,
)
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

# class MiluStreamingCallbackHandler(StreamingStdOutCallbackHandler):
#     def __init__(self, sio, user_sid: str, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.sio = sio
#         self.user_sid = user_sid
#         self.queue = asyncio.Queue()
#         self.worker_task = asyncio.create_task(self.worker())

#     async def worker(self):
#         while True:
#             message = await self.queue.get()
#             await self.sio.emit("assistant_response", message, to=self.user_sid)
#             self.queue.task_done()

#     def on_llm_new_token(self, token: str, **kwargs: Any):
#         self.queue.put_nowait({'message': token, 'message_end': False})

#     def on_llm_end(self,response: LLMResult, **kwargs: Any):
#         self.queue.put_nowait({'message': '', 'message_end': True})
    
#     def cleanup(self):
#         self.worker_task.cancel()

class MiluStreamingCallbackHandler(AsyncCallbackHandler):

    def __init__(self,sio,**kwargs: Any) -> None:
        super().__init__()
        self.sio = sio
        self.user_sid = kwargs.get('user_sid')
        print('milu is connected!')
    

    async def on_chat_model_start(
    self,
    serialized: Dict[str, Any],
    messages: List[List[BaseMessage]],
    *,
    run_id: UUID,
    parent_run_id: Optional[UUID] = None,
    tags: Optional[List[str]] = None,
    **kwargs: Any,
    ) -> Any:
        pass


    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        await self.sio.emit("assistant_response",{'message': token, 'message_end': False},to=self.user_sid)
        # await self.message_queue.put({"message": token, "message_end": False, "user_sid": self.user_sid})
        # self.message_ready.set()
      
        # print('token was put!')
        

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        await self.sio.emit("assistant_response",{'message': '', 'message_end': True},to=self.user_sid)
        # self.message_queue.put_nowait({"message": "", "message_end": True, "user_sid": self.user_sid})
        # self.message_ready.set()
