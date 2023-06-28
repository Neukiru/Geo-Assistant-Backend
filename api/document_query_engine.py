import os
import logging
import tiktoken
import re
from llama_index import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage, ServiceContext
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores import SimpleVectorStore
from llama_index.storage.index_store import SimpleIndexStore
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.callbacks import (
    CallbackManager, 
    TokenCountingHandler
)


class DocumentQueryEngine:
    def __init__(self,storage_path:str, model_of_choice:str):
        self.query_engine_tools = []
        self._storage_path = storage_path
        self._model = model_of_choice
        self.token_counter = TokenCountingHandler(
        tokenizer=tiktoken.encoding_for_model(self._model).encode
        )
        self._callback_manager = CallbackManager([self.token_counter])
        self._service_context = ServiceContext.from_defaults(callback_manager=self._callback_manager)
        # self._storage_context = StorageContext.from_defaults(persist_dir=storage_path)

    def add_engine(self, name, description, service_context, top_k=5):
        folder_name = re.sub('\s',' ',name.split('.')[0])
        index = load_index_from_storage(self._get_storage_context(folder_name))
        logging.info("Using cached storage context")
       

        engine = index.as_query_engine(similarity_top_k=top_k,service_context=service_context)

        query_engine_tool = QueryEngineTool(
            query_engine=engine,
            metadata=ToolMetadata(name=folder_name, description=description)
        )
        self.query_engine_tools.append(query_engine_tool)
    
    def _get_storage_context(self,folder_name):
        return StorageContext.from_defaults(
        docstore=SimpleDocumentStore.from_persist_dir(persist_dir=os.path.join(self._storage_path,folder_name)),
        vector_store=SimpleVectorStore.from_persist_dir(persist_dir=os.path.join(self._storage_path,folder_name)),
        index_store=SimpleIndexStore.from_persist_dir(persist_dir=os.path.join(self._storage_path,folder_name)),
        )



    def remove_engine(self, name):
        for tool in self.query_engine_tools:
            if tool.metadata.name == name:
                self.query_engine_tools.remove(tool)
                print(f'Removed engine: {name}')
                return
        print(f'No engine found with the name: {name}')
    
    def remove_all_engines(self):
        self.query_engine_tools.clear()
        print('All engines have been removed.')

    def get_engine_tools(self):
        return self.query_engine_tools