import datetime
import os
from pyaws_s3 import S3Client
from typing import Literal, Any
from langchain_community.document_loaders import (CSVLoader,
                                                  PyPDFLoader,
                                                  JSONLoader)
from langchain_core.documents import Document
from .kgrag_graph import KGragGraph
from .kgrag_cache import MemoryRedisCacheRetriever
from .kgrag_utils import print_progress_bar
from log import get_logger, get_metadata

PathType = Literal["fs", "s3"]
FormatFile = Literal["pdf", "csv", "json"]


class KGragRetriever(KGragGraph):
    """
    Class for retrieving and processing documents
    from a memory store, either from a local filesystem or AWS S3.
    """
    aws_access_key_id: str | None
    aws_secret_access_key: str | None
    s3_bucket: str | None
    aws_region: str | None
    path_download: str | None
    path_type: PathType = "fs"
    format_file: FormatFile = "pdf"
    path_download: str | None = None
    fieldnames: list[str] = [
        "file_name",
        "updated_at",
        "ingested",
        "timestamp"
    ]
    memory_redis: MemoryRedisCacheRetriever
    logger = get_logger(
        name="MemoryStoreRetriever",
        loki_url=os.getenv("LOKI_URL")
    )

    def __init__(self, **kwargs: Any) -> None:
        """
        Initializes the MemoryStoreRetriever with the given parameters.
        Args:
            **kwargs: Keyword arguments for configuration.
                - format_file (str): The format of the file to process.
                - path_type (str): The type of path, either "fs" or "s3".
                - redis_host (str): Redis host.
                - redis_port (int): Redis port.
                - redis_db (int): Redis database number.
                - aws_access_key_id (str): AWS access key
                    ID (if path_type is "s3").
                - aws_secret_access_key (str): AWS secret
                    access key (if path_type is "s3").
                - s3_bucket (str): S3 bucket name (if path_type is "s3").
                - aws_region (str): AWS region (if path_type is "s3").
                - path_download (str): Local path for
                    downloading files from S3.
        """
        super().__init__(**kwargs)

        self.format_file = kwargs.get('format_file', 'pdf')
        self.path_type = kwargs.get('path_type', 'fs')

        host = kwargs.get('redis_host', 'localhost')
        port = kwargs.get('redis_port', 6379)
        db = kwargs.get('redis_db', 0)

        if not all([host, port, db]):
            msg: str = (
                "Redis connection parameters are not set. "
                "Please provide valid host, port, and db."
            )
            self.logger.error(msg,
                              extra=get_metadata(
                                    thread_id=str(self.thread)
                              ))
            raise ValueError(msg)

        self.memory_redis = MemoryRedisCacheRetriever(
            host=host,
            port=port,
            db=db,
            cache_key="memory_store_cache",
        )

        if self.path_type == "s3":
            self.aws_access_key_id = kwargs.get('aws_access_key_id', None)
            self.aws_secret_access_key = (
                kwargs.get('aws_secret_access_key', None)
            )
            self.s3_bucket = kwargs.get('s3_bucket', None)
            self.aws_region = kwargs.get('aws_region', None)
            self.path_download = kwargs.get('path_download', None)

            if not self.path_download:
                msg: str = (
                    "Path for downloading files from S3 is not set. "
                    "Please provide a valid path."
                )
                self.logger.error(
                    msg,
                    extra=get_metadata(
                        thread_id=str(self.thread)
                    )
                )
                raise ValueError(msg)

            self._create_download_dir(self.path_download)

            if not all([self.aws_access_key_id, self.aws_secret_access_key,
                        self.s3_bucket, self.aws_region]):
                msg: str = (
                    "Missing AWS credentials or bucket information "
                    "for S3 access. "
                    "Please provide valid aws_access_key_id, "
                    "aws_secret_access_key, s3_bucket, and aws_region."
                )
                self.logger.error(
                    msg,
                    extra=get_metadata(thread_id=str(self.thread))
                )
                raise ValueError(msg)

    async def load_cache(self) -> list[dict[str, Any]]:
        """
        Loads the cache from a CSV file and returns a list of dictionaries.

        Returns:
            list[dict]: List of dictionaries containing the cache data.
        """
        return await self.memory_redis.get_cache()

    async def _update_cache(
        self,
        file_name: str,
        error: bool = False
    ) -> None:
        """
        Updates or appends a row in a CSV file with file name, timestamp,
        and optional extra metadata.

        Args:
            file_name (str): Name of the file.
            error (bool): Whether an error occurred during processing.
            Defaults to False.
        Returns:
            None
        Raises:
            ValueError: If the file name is not provided.
        Raises:
            Exception: If there is an error while updating the cache.
        """

        # Read existing rows

        try:
            item = await self.memory_redis.get_cache_by(file_name=file_name)
            if item:
                msg: str = (
                    f"File {file_name} already exists in cache. "
                    "Updating existing entry."
                )
                self.logger.info(
                    msg,
                    extra=get_metadata(thread_id=str(self.thread))
                )
                # Update existing entry
                await self.memory_redis.update_cache_by(
                    file_name=file_name,
                    updates={
                        "ingested": 1,
                        "error": 1 if error else 0,
                    })
                return

            self.memory_redis.add_cache(data={
                "file_name": file_name,
                "ingested": 1,
                "error": 1 if error else 0,
                "update_at": (
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )
            })
        except Exception as e:
            self.logger.error(
                f"Error updating cache for file {file_name}: {e}",
                extra=get_metadata(thread_id=str(self.thread))
            )
            raise e

    def _create_download_dir(
        self,
        path: str,
        delete: bool = False
    ) -> str:
        """
        Creates a temporary local directory to store downloaded files.
        Args:
            path (str): The local path where files will be downloaded.
            delete (bool): Whether to delete existing files in the directory.
        """

        # create the directory if it doesn't exist
        if not os.path.exists(path):
            self.logger.info(f"Creating directory: {path}",
                             extra=get_metadata(
                                thread_id=str(self.thread)
                             ))
            os.makedirs(path, exist_ok=True)

        # Se esistono file nella cartella local_dir, cancellali tutti
        if delete:
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                try:
                    if os.path.isfile(file_path):
                        self.logger.info(
                            f"Deleting file: {file_path}",
                            extra=get_metadata(thread_id=str(self.thread))
                        )
                        os.remove(file_path)
                except Exception as e:
                    self.logger.error(
                        f"Errore durante la cancellazione di {file_path}: {e}",
                        extra=get_metadata(thread_id=str(self.thread))
                    )
                    raise e
        return path

    def _get_documents_from_path(
        self,
        path: str,
        format_file: FormatFile = "pdf",
        **kwargs_loaders: Any
    ) -> list[Document]:
        """
        Loads documents from a specified path based on the file format.
        Args:
        - path (str): The path to the document file.
        """

        if path is None:
            # Log the error message and raise a ValueError
            # to indicate that the path is not set.
            # This will help in debugging and ensure that the user
            # is aware of the issue.
            # Raise a ValueError to indicate that the path is not set.
            # Log the error message
            msg: str = (
                "Path is not set. Please provide a valid path."
            )
            self.logger.error(msg,
                              extra=get_metadata(
                                  thread_id=str(self.thread)
                              ))
            raise ValueError(msg)

        loader: Any = None
        if format_file == "pdf":
            loader = PyPDFLoader(path, **kwargs_loaders)
        elif format_file == "csv":
            loader = CSVLoader(path, **kwargs_loaders)
        elif format_file == "json":
            loader = JSONLoader(path, jq_schema='.', **kwargs_loaders)
        else:
            msg: str = (
                "Unsupported format"
                "Please provide either 'pdf' or 'csv' or 'json'."
            )
            # Log the error message and raise a ValueError
            # to indicate unsupported format
            # This will help in debugging and ensure that
            # the user is aware of the issue.
            # Raise a ValueError to indicate unsupported format
            # This will help in debugging and ensure that
            # the user is aware of the issue.
            # Log the error message
            self.logger.error(
                msg,
                extra=get_metadata(thread_id=str(self.thread))
            )
            raise ValueError(msg)

        if loader is None:
            msg: str = (
                "Loader is not set. "
                "Please provide a valid loader for the specified format."
            )
            self.logger.error(
                msg,
                extra=get_metadata(thread_id=str(self.thread))
            )
            raise ValueError(msg)

        return loader.load()

    async def _filter_files(
        self,
        files: list[str],
        refresh: bool = False
    ) -> list[str]:
        """
        Filters files based on a prefix.

        Args:
        - files (list[str]): List of file names.
        - prefix (str): The prefix to filter files.

        Returns:
        - list[str]: A list of filtered file names.
        """
        # Check if files already exist in the cache
        # Only keep files that are in the cache and have ingested == False
        files_to_process = []
        extra = get_metadata(thread_id=str(self.thread))

        if not refresh:
            index = 1
            for file in files:
                cached_file = await self.memory_redis.get_cache_by(
                    file_name=file
                )
                # stampa la stessa linea di log
                # con la percentuale nella stessa riga
                print_progress_bar(index, len(files),
                                   prefix=f"Checking file {file}",
                                   length=50)
                if cached_file is None:
                    files_to_process.append(file)
                    index += 1
                    continue

                if cached_file.get("ingested") == 0:
                    files_to_process.append(file)

                index += 1
        else:
            self.logger.info("Refreshing cache. All files will be processed.",
                             extra=extra
                             )
            files_to_process = files

        self.logger.info(
            f"Filtered files: {len(files_to_process)} files to process.",
            extra=extra
        )
        return files_to_process

    async def process_documents(self, **kwargs: Any):
        """
        Processes documents based on the specified path type and format.
        Args:
        - documents (list[Document]): List of Document objects to process.
        - path (str): The local path where files are located.
            If documents are provided, this is ignored.
        - force (bool): Whether to force the processing of files even
            if they already exist in the path.
        - bucket_name (str): The name of the S3 bucket.
            If documents are provided, this is ignored.
        - aws_region (str): The AWS region
            If documents are provided, this is ignored.
        """

        documents: list[Document] = kwargs.get("documents", [])
        extra: dict = get_metadata(thread_id=str(self.thread))
        force: bool = kwargs.get("force", False)

        if len(documents) == 0:
            path = kwargs.get("path", None)
            if path is None:
                msg: str = (
                    "Path is not set. "
                    "Please provide a valid path."
                )
                self.logger.error(msg,
                                  extra=extra)
                raise ValueError(msg)

            if os.path.exists(path) and not force:
                msg: str = (
                    f"Path '{path}' exist. "
                    "Ingestion not required."
                )
                self.logger.warning(
                    msg,
                    extra=extra
                )
                return

            file = os.path.basename(path)
            metadata = {
                "object_name": file,
                "local_path": path
            }

            bucket_name = kwargs.get("bucket_name", None)
            if bucket_name is not None:
                metadata["bucket_name"] = bucket_name

            aws_region = kwargs.get("aws_region", None)
            if aws_region is not None:
                metadata["aws_region"] = aws_region

            ext = os.path.splitext(path)[1].lower()
            if ext == ".pdf":
                self.format_file = "pdf"
            elif ext == ".csv":
                self.format_file = "csv"
            elif ext == ".json":
                self.format_file = "json"
            else:
                msg = f"Unsupported file extension: {ext}"
                self.logger.error(
                    msg,
                    extra=extra
                )
                raise ValueError(msg)

            docs = self._get_documents_from_path(
                path,
                headers=metadata,
                format_file=self.format_file
            )
            if not docs:
                self.logger.warning(
                    f"No documents found in {path}. Skipping.",
                    extra=extra
                )
                return

            for doc in docs:
                doc.metadata.update(metadata)
        else:
            docs = documents

        # Call the ingestion method from the parent class
        async for d in self._ingestion_batch(
            documents=docs,
            collection_name=self.collection_name,
            thread=self.thread
        ):
            yield d

    async def _process_documents_s3(self, **kwargs: Any):
        """
        Processes documents from S3 by downloading them to a
        local path and loading them.

        Args:
        - prefix (str): The S3 prefix to filter files.
        - limit (int): The maximum number of files to download.
        - path_download (str): The local path where files will be downloaded.
        - start (int): The index to start downloading files from.
        - refresh (bool): Whether to refresh the cache.
        - force (bool): Whether to force the download of files even
            if they already exist locally

        Returns:
        - list[Document]: A list of Document objects loaded
            from the downloaded files.
        """

        prefix = kwargs.get("prefix", None)
        limit = kwargs.get("limit", 0)
        start = kwargs.get("start", 0)
        path_download = kwargs.get("path_download", None)
        refresh = kwargs.get("refresh", False)

        self.logger.info("Processing documents from S3.")
        if not all([
            self.aws_access_key_id,
            self.aws_secret_access_key,
            self.s3_bucket,
            self.aws_region
        ]):
            msg: str = (
                "Missing AWS credentials or bucket information for S3 access. "
                "Please provide valid aws_access_key_id, "
                "aws_secret_access_key, "
                "s3_bucket, and aws_region."
            )
            # Log the error message and raise a ValueError
            # to indicate that the AWS credentials or bucket
            # information is missing.
            # This will help in debugging and ensure that the user is
            # aware of the issue.
            # Raise a ValueError to indicate that the AWS credentials or
            # bucket information is missing.
            # Log the error message
            self.logger.error(msg,
                              extra=get_metadata(thread_id=str(self.thread)))
            raise ValueError(msg)

        if prefix is None:
            msg: str = (
                "Prefix is not set. "
                "Please provide a valid prefix for S3 files."
            )
            # Log the error message and raise a ValueError
            # to indicate that the prefix is not set.
            # This will help in debugging and ensure that the user is
            # aware of the issue.
            # Raise a ValueError to indicate that the prefix is not set.
            # Log the error message
            self.logger.error(msg,
                              extra=get_metadata(thread_id=str(self.thread)))
            # Raise a ValueError to indicate that the prefix is not set.
            # This will help in debugging and ensure that the user is
            # aware of the issue.
            raise ValueError(msg)

        if path_download is None:
            msg: str = (
                "Path for downloading files from S3 is not set. "
                "Please provide a valid path."
            )
            self.logger.error(msg,
                              extra=get_metadata(thread_id=str(self.thread)))
            raise ValueError(msg)

        s3_client = S3Client(
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            bucket_name=self.s3_bucket,
            region_name=self.aws_region
        )
        list_files: list[str] = s3_client.list_files(prefix)
        list_files.sort()

        # filter files based on the cache
        list_files = await self._filter_files(list_files, refresh=refresh)

        if start > 0:
            self.logger.warning(f"Starting from index: {start}",
                                extra=get_metadata(thread_id=str(self.thread)))
            list_files = list_files[start:]

        # limit the number of files to download
        if limit > 0:
            self.logger.warning(
                f"Limiting the number of files to download to: {limit}",
                extra=get_metadata(thread_id=str(self.thread))
            )
            list_files = list_files[:limit]

        # create a temporary local directory to store downloaded files
        index: int = 1
        size = len(list_files)
        for file in list_files:
            self.logger.info(f"Downloading file: {file}",
                             extra=get_metadata(thread_id=str(self.thread)))
            # download the file into the temporary local folder
            local_file = f"{path_download}/{os.path.basename(file)}"
            if not os.path.exists(local_file):
                s3_client.download(file, local_path=local_file)
                self.logger.info(
                    f"File {file} downloaded to: {path_download}",
                    extra=get_metadata(thread_id=str(self.thread))
                )
            else:
                self.logger.info(
                    f"File {file} already exists locally in {path_download}",
                    extra=get_metadata(thread_id=str(self.thread))
                )

            async for d in self.process_documents(
                path=local_file,
                bucket_name=self.s3_bucket,
                aws_region=self.aws_region
            ):
                if d == "ERROR":
                    await self._update_cache(file_name=file, error=True)
                    index += 1
                    continue

            await self._update_cache(file_name=file)
            self.logger.info(
                f"Updating cache for file: {file}",
                extra=get_metadata(thread_id=str(self.thread))
            )
            yield index, size, file
            print_progress_bar(
                index,
                size,
                prefix=f"Ingesting documents {file}",
                length=50
            )
            index += 1

    async def _get_files_from_path(
        self,
        path: str,
        limit: int = 0,
        start: int = 0,
        refresh: bool = False
    ) -> list[str]:
        """
        Retrieves a list of files from a specified path.

        Args:
        - path (str): The local path where files are located.

        Returns:
        - list[str]: A list of file names in the specified path.
        """

        self.logger.info(
            "Processing documents from the local filesystem.",
            extra=get_metadata(thread_id=str(self.thread))
        )

        if path is None:
            msg: str = (
                "Path is not set. "
                "Please provide a valid path."
            )
            self.logger.error(msg,
                              extra=get_metadata(thread_id=str(self.thread)))
            raise ValueError(msg)

        pdf_files = [f for f in os.listdir(path) if f.lower().endswith(".pdf")]

        if start > 0:
            # Start from a specific index
            self.logger.warning(
                f"Starting from index: {start}",
                extra=get_metadata(thread_id=str(self.thread))
            )
            pdf_files = pdf_files[start:]

        if limit > 0:
            # Limit the number of files to process
            self.logger.warning(
                f"Limiting the number of files to process to: {limit}",
                extra=get_metadata(thread_id=str(self.thread))
            )
            pdf_files = pdf_files[:limit]

        return await self._filter_files(pdf_files, refresh=refresh)

    def _get_docs(self, path: str) -> list[Document]:
        """
        Loads documents from a specified path based on the file format.

        Args:
        - path (str): The path to the document file.

        Returns:
        - list[Document]: A list of Document objects
            loaded from the specified path.
        """

        metadata = {
            "object_name": path,
            "storage_type": "fs"
        }

        docs = self._get_documents_from_path(path)
        for doc in docs:
            doc.metadata.update(metadata)
        return docs

    async def _process_documents_fs(self, **kwargs: Any):
        """
        Processes documents from the local filesystem by loading
        them from a specified path.

        Args:
        - path (str): The local path where files are located.
        - limit (int): The maximum number of files to process.
        - start (int): The index to start processing files from.
        - refresh (bool): Whether to refresh the cache.
            if force is True, it will ignore the refresh
            flag and process all files.
        - force (bool): Whether to force the processing of files even
            if they already exist in the cache.

        Returns:
        - list[Document]: A list of Document objects loaded from
            the specified path.
        """

        try:
            path = kwargs.get("path", None)
            limit = kwargs.get("limit", 0)
            start = kwargs.get("start", 0)
            refresh = kwargs.get("refresh", False)

            self.logger.info(
                "Processing documents from the local filesystem.",
                extra=get_metadata(thread_id=str(self.thread))
            )

            if path is None:
                msg: str = (
                    "Path is not set. "
                    "Please provide a valid path."
                )
                # Log the error message and raise a ValueError
                # to indicate that the path is not set.
                # This will help in debugging and ensure that the user
                # is aware of the issue.
                self.logger.error(
                    msg,
                    extra=get_metadata(thread_id=str(self.thread))
                )
                raise ValueError(msg)

            pdf_files = await self._get_files_from_path(
                path=path,
                limit=limit,
                start=start,
                refresh=refresh
            )

            index: int = 1
            size = len(pdf_files)
            for pdf_file in pdf_files:
                file_path = os.path.join(path, pdf_file)
                # Call the ingestion method from the parent class
                async for d in self.process_documents(
                    path=file_path,
                ):
                    if d == "ERROR":
                        await self._update_cache(file_name=pdf_file,
                                                 error=True)
                        index += 1
                        continue

                    yield index, size, pdf_file
                    prefix: str = f"Ingesting documents {pdf_file}"
                    print_progress_bar(index,
                                       size,
                                       prefix=prefix,
                                       length=50)

                    self.logger.info(
                        f"Updating cache for file: {pdf_file}",
                        extra=get_metadata(thread_id=str(self.thread))
                    )
                    await self._update_cache(file_name=pdf_file)
                    yield index, size, pdf_file
                    index += 1

        except Exception as e:
            self.logger.error(
                f"Error processing documents from filesystem: {e}",
                extra=get_metadata(thread_id=str(self.thread))
            )
            raise e

    async def process_path(self, **kwargs: Any):
        """
        Processes a PDF document by loading it, splitting it into chunks,
        embedding them, and building a knowledge graph.

        Args:
            - pdf_path (str): The path to the PDF file to be processed.
            - path (str): The local path where files are located
                (if path_type is "fs").
            - path_type (str): The type of path, either "fs" for
                local filesystem or "s3" for AWS S3.
            - prefix (str): The S3 prefix to filter files
                (if path_type is "s3").
            - limit (int): The maximum number of files to download
                (if path_type is "s3").
            - path_download (str): The local path where files will be
                downloaded (if path_type is "s3").

        Returns:
        - None
        """
        try:
            path_type: PathType = kwargs.get("path_type", self.path_type)
            limit = kwargs.get("limit", 0)
            start = kwargs.get("start", 0)
            refresh = kwargs.get("refresh", False)

            if refresh:
                await self._refresh_graph()

            if path_type == "s3":
                prefix = kwargs.get("prefix", None)
                path_download = kwargs.get("path_download", self.path_download)
                async for index, size, file in self._process_documents_s3(
                        path_download=path_download,
                        prefix=prefix,
                        limit=limit,
                        start=start,
                        refresh=refresh
                ):
                    yield index, size, file
            elif path_type == "fs":
                path: str | None = kwargs.get("path", None)
                if path is None:
                    msg: str = (
                        "Path is not set. "
                        "Please provide a valid path."
                    )
                    self.logger.error(
                        msg,
                        extra=get_metadata(thread_id=str(self.thread))
                    )
                    raise ValueError(msg)
                async for index, size, file in self._process_documents_fs(
                        path=path,
                        limit=limit,
                        start=start,
                        refresh=refresh
                ):
                    yield index, size, file

        except Exception as e:
            self.logger.error(
                f"Error processing documents: {e}",
                extra=get_metadata(thread_id=str(self.thread))
            )
            raise e

    async def _refresh_graph(self):
        """
        Refreshes the graph by deleting all relationships and collections,
        then recreating the collection.
        """
        self.logger.debug(
            "Deleting cache",
            extra=get_metadata(thread_id=str(self.thread))
        )
        await self.memory_redis.delete_cache()
        # Delete all relationships in the graph
        self.logger.info(
            "Deleting all relationships in the graph.",
            extra=get_metadata(thread_id=str(self.thread))
        )
        self.delete_all_relationships()
        # Delete all collections in the graph
        self.logger.info(
            f"Deleting collection {self.collection_name} in the graph.",
            extra=get_metadata(thread_id=str(self.thread))
        )
        await self.delete_collection_async(self.collection_name)
        self.logger.info(
            f"Creating new collection {self.collection_name} in the graph.",
            extra=get_metadata(thread_id=str(self.thread))
        )
        await self.create_collection_async(
            self.collection_name,
            self.collection_dim
        )
        self.logger.info(
            "Graph refreshed successfully.",
            extra=get_metadata(thread_id=str(self.thread))
        )
