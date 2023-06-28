import re
from langchain.chat_models import ChatOpenAI
from llama_index import (
    LLMPredictor, 
    ServiceContext
)


def get_service_context(model_name,temperature=0.0,request_timeout=120, callback_manager = None, streaming=False):
    llm_predictor_= LLMPredictor(llm=ChatOpenAI(temperature=temperature, model_name=model_name , request_timeout=request_timeout,streaming=streaming))
    service_context_ = ServiceContext.from_defaults(llm_predictor=llm_predictor_,callback_manager=callback_manager)

    return service_context_

def normalize_names(name):
    name = re.sub("_",' ',name.split('.')[0])
    return name