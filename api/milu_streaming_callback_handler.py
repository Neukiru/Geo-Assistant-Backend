from typing import Any, Dict, List, Union
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import LLMResult


class MiluStreamingCallbackHandler(StreamingStdOutCallbackHandler):

    def __init__(self,sio) -> None:
        super().__init__()
        self.sio = sio
    
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        self.sio.emit("on_llm_new_token",{"token_sequence":token})

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        self.sio.emit("on_llm_new_token",{"token_sequence_end":True})

    def on_llm_start(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        self.sio.emit("on_llm_new_token",{"token_sequence_start":True})