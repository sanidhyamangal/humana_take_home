import os
import typing as t

from dotenv import load_dotenv
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.extractors import KeywordExtractor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

load_dotenv(override=True)


def get_default_ollama_llm(**options: dict[str, t.Any]) -> Ollama:
    """helper function to get opensource llm for all the tasks

    Returns:
        Ollama: Specialized subclass of LLM to perform all the llm related ops.
    """
    return Ollama(
        model=os.getenv("OLLAMA_MODEL"),
        request_timeout=120.0,  # longer timeout to give local system to run inference fully.
        **options,
    )


def get_default_embedding_models() -> HuggingFaceEmbedding:
    """util function to get default embedding model defined in config

    Returns:
        HuggingFaceEmbedding: Embedding model instance
    """

    return HuggingFaceEmbedding(model_name=os.getenv("EMBEDDINGS_MODEL"))


def get_default_transformations() -> list[t.Any]:
    """Define set of transformation to applied on data before data ingestion."""

    # define which llm to use for metadata extraction
    llm = get_default_ollama_llm(temperature=0.0)
    embed_model = get_default_embedding_models()

    # define transformations
    text_splitter = SemanticSplitterNodeParser(
        buffer_size=1, embed_model=embed_model, breakpoint_percentile_threshold=95
    )
    keyword_extractor = KeywordExtractor(llm=llm)

    return [text_splitter, keyword_extractor]
