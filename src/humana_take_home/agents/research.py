import os
import typing as t
from logging import getLogger

from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.chat_engine.types import BaseChatEngine, ChatMode
from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer

from humana_take_home.prompts.research_paper import SYSTEM_PROMPT
from humana_take_home.utils import get_default_embedding_models, get_default_ollama_llm

logger = getLogger(__name__)


class ResearchAgent:
    """Agent to query the research paper using vector index."""

    def __init__(self, vector_index: VectorStoreIndex, similarity_top_k: int = 3) -> None:
        """constructor to initialize the agent with vector index and chat engine.

        Args:
            vector_index (VectorStoreIndex): vector index to query the research paper.
            similarity_top_k (int, optional): Number of nodes to use for context. Defaults to 3.
        """
        self.model_name = os.getenv('MODEL_NAME')
        self.__chat_engine = vector_index.as_chat_engine(
            chat_mode=ChatMode.CONTEXT,
            system_prompt=SYSTEM_PROMPT,
            llm=get_default_ollama_llm(temperature=0.0),
            similarity_top_k=similarity_top_k,
        )

    @classmethod
    def from_local_storage(cls, similarity_top_k: int = 3) -> 'ResearchAgent':
        """method to load the vector index from the local storage using env variables.

        Args:
            similarity_top_k (int, optional): Number of nodes to use for the context. Defaults to 3.

        Raises:
            EnvironmentError: if vector index path is not set or does not exists.

        Returns:
            ResearchAgent
        """
        _vector_index_path = os.getenv('VECTOR_INDEX_PATH')
        if _vector_index_path is None or not os.path.exists(_vector_index_path):
            raise EnvironmentError('valid `VECTOR_INDEX_PATH` is required load embeddings')

        storage_ctx = StorageContext.from_defaults(persist_dir=_vector_index_path)
        vector_index = load_index_from_storage(
            storage_context=storage_ctx,
            llm=get_default_ollama_llm(temperature=0.0),
            embed_model=get_default_embedding_models(),
        )

        return cls(vector_index=vector_index, similarity_top_k=similarity_top_k)

    def get_chat_engine(self) -> BaseChatEngine:
        """get the chat engine to query the research paper.
        used by streamlit and evaluator.

        Returns:
            BaseChatEngine defined chat engined in constructor.
        """
        return self.__chat_engine

    def query(self, query: str, chat_history: ChatMemoryBuffer | None = None) -> t.Any:
        """a method to quert chat engine with the query and chat history.
        Useful when creating an API.

        Args:
            query (str): user query to ask the question from chatbot.
            chat_history (ChatMemoryBuffer | None, optional): chat history for multi-turn chat. Defaults to None.

        Returns:
            t.Any: response from the chat engine.
        """
        response = self.__chat_engine.chat(
            message=query,
            chat_history=chat_history.get() if chat_history is not None else None,
        )

        return response
