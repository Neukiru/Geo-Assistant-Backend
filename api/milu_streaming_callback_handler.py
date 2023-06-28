from typing import Any, Dict, List, Union
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import LLMResult


class MiluStreamingCallbackHandler(StreamingStdOutCallbackHandler):

    def __init__(self,sio) -> None:
        super().__init__()
        self.sio = sio
    
    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        await self.sio.emit("on_llm_new_token",{"token_sequence":token})

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        await self.sio.emit("on_llm_new_token",{"token_sequence_end":True})

    async def on_llm_start(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        await self.sio.emit("on_llm_new_token",{"token_sequence_start":True})