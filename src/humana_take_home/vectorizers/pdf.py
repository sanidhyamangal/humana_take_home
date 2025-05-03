import os
import typing as t
from logging import getLogger

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex

from ..timer import Timer
from ..utils import get_default_embedding_models, get_default_ollama_llm, get_default_transformations

logger = getLogger(__name__)  # define logger


class PDFDataLoader:
    """
    Class to load pdf files and build vector index using llama_index.
    The class uses the following methods:
      - _load_data: method to load data from the given directory or files.
      - build_vector_index: method to build vector index from the loaded data.
      - __persist_vector_idx: method to persist the vector index to the local storage.
      - merge_new_documents: method to merge new documents into the existing vector index.
    """

    def __init__(self, vector_index_path: str | None = None) -> None:
        # define llm, embedding model and storage context
        self.llm = get_default_ollama_llm(temperature=0.0)  # zero temperature to minimize llm's creative thinking.
        self.embed_model = get_default_embedding_models()

        if vector_index_path is not None and os.path.exists(vector_index_path):
            self.storage_ctx = StorageContext.from_defaults(
                persist_dir=vector_index_path
            )  # useful if we want to merge new documents into existing vector index.

    def _load_data(
        self,
        input_dir: str | None,
        input_files: str | list[str] | None,
    ) -> t.Any:
        """method to load data or ingest pdfs into data loader

        Args:
            input_dir (str): input_dir to load all the files from a directory
            input_files (str | list[str]): either a single file to load or list of files to load

        Returns:
            t.Any: list of document object readed from files
        """

        if input_dir is None and input_files is None:
            raise ValueError("both `input_dir` or `input_files` can't be none")

        with Timer() as timer:
            documents = SimpleDirectoryReader(
                input_dir=input_dir,
                input_files=input_files,
                required_exts=['.pdf', '.docx'],
            ).load_data(show_progress=True)

        logger.info(f'Extracted {len(documents)=} in {timer.exec_time/60} mins')

        return documents

    def build_vector_index(
        self,
        input_dir: str | None = None,
        input_files: str | list[str] | None = None,
        persist_index_path: str | None = None,
    ) -> None:
        # """method to build vector index"""
        """method to build the vector index from the loaded data.

        Args:
            input_dir (str): input_dir to load all the files from a directory. Defaults to None.
            input_files (str | list[str]): either a single file to load or list of files to load. Defaults to None.
            persist_index_path (str | None, optional): Path to flush vectorized indexes. Defaults to None.

        Returns:
            _type_: _description_
        """

        # encapsulate into list if single file
        if isinstance(input_files, str):
            input_files = [input_files]

        # convert text files to document objects.
        documents = self._load_data(input_dir=input_dir, input_files=input_files)

        with Timer() as index_timer:
            vector_index = VectorStoreIndex.from_documents(
                documents=documents,
                show_progress=True,
                embed_model=self.embed_model,
                transformations=get_default_transformations(),
            )

        logger.info(f'Indexed all the documents in {index_timer.exec_time/60} mins')

        if persist_index_path is None:
            return vector_index

        self.__persist_vector_idx(vector_index, persist_index_path)  # flush the vector index to the local storage.

    def __persist_vector_idx(self, vector_index: VectorStoreIndex, persist_path: str) -> None:
        """method to persist the vector index to the local storage.

        Args:
            vector_index (VectorStoreIndex): Vector index to be persisted.
            persist_path (str): path to persist the vector index.
        """

        with Timer() as persist_timer:
            vector_index.storage_context.persist(persist_dir=persist_path)

        logger.info(f'Flushed vector index @ {persist_path} in {persist_timer.exec_time / 60}mins ')

    def merge_new_documents(
        self, input_dir: str | None = None, input_files: str | list[str] | None = None
    ) -> t.Any: ...

    """method to merge new documents into the existing vector index."""
    # TODO: implement this method in future, currently out of scope.
